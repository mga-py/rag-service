import logging
import os
import gc
import time
import asyncio
import torch  # --- NUEVO ---
from threading import Lock  # --- NUEVO ---
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import List, Optional
from sentence_transformers.cross_encoder import CrossEncoder
from starlette.concurrency import run_in_threadpool

# --- 1. Configuración del Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- 2. Cargar Configuración (SIN CARGAR EL MODELO) ---
logger.info("Iniciando servidor de reclasificación...")

# Cargar configuración desde variables de entorno
model_name = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-large")
model_max_length = int(os.getenv("MODEL_MAX_LENGTH", "512"))
API_KEY = os.getenv("API_KEY", None)
# --- NUEVO: Timeout de inactividad en segundos (ej. 10 minutos) ---
IDLE_TIMEOUT = int(os.getenv("IDLE_TIMEOUT", "600"))

if API_KEY:
    logger.info("Protección con clave API HABILITADA.")
else:
    logger.warning("Protección con clave API DESHABILITADA (Servidor abierto).")

# --- 3. NUEVO: Gestor del Modelo (ModelManager) ---

class ModelManager:
    """
    Gestiona la carga y descarga del modelo para liberar VRAM
    cuando está inactivo.
    """

    def __init__(self, model_name, max_length, idle_timeout):
        self.model_name = model_name
        self.max_length = max_length
        self.idle_timeout = idle_timeout
        self.model: Optional[CrossEncoder] = None
        self.last_used: float = 0.0
        self.load_lock = Lock()  # Previene cargas/descargas simultáneas
        self.access_lock = Lock()  # Para actualizar self.last_used

        # Inicia el bucle de comprobación de inactividad
        asyncio.create_task(self.idle_check_loop())
        logger.info(f"Gestor de modelo iniciado. Timeout de inactividad: {self.idle_timeout}s")

    def _load_model(self):
        """Función interna BLOQUEANTE para cargar el modelo."""
        logger.info(f"Cargando modelo {self.model_name} en VRAM...")
        try:
            self.model = CrossEncoder(self.model_name, max_length=self.max_length)
            with self.access_lock:
                self.last_used = time.time()
            logger.info("Modelo cargado exitosamente.")
        except Exception as e:
            logger.error(f"Error fatal al cargar el modelo: {e}", exc_info=True)
            self.model = None  # Asegurarse de que esté None si falla
            raise  # Relanzar la excepción para que el endpoint falle

    def _unload_model(self):
        """Función interna BLOQUEANTE para descargar el modelo."""
        if self.model is None:
            return

        logger.info("Modelo inactivo. Descargando de la VRAM...")
        del self.model
        self.model = None
        gc.collect()  # Forzar recolección de basura
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Limpiar la caché de VRAM de PyTorch
        logger.info("VRAM liberada.")

    def get_model(self) -> CrossEncoder:
        """
        Obtiene el modelo. Carga si es necesario.
        Esta es una función BLOQUEANTE diseñada para run_in_threadpool.
        """
        # Patrón de bloqueo de doble verificación (DCLP)

        # 1. Comprobación rápida (sin bloqueo)
        if self.model is not None:
            with self.access_lock:
                self.last_used = time.time()
            return self.model

        # 2. Comprobación lenta (con bloqueo)
        with self.load_lock:
            # ¿Otro hilo lo cargó mientras esperábamos el bloqueo?
            if self.model is not None:
                with self.access_lock:
                    self.last_used = time.time()
                return self.model

            # No, realmente está descargado. Carguémoslo.
            self._load_model()
            if self.model is None:
                # La carga falló
                raise RuntimeError("No se pudo cargar el modelo.")

            return self.model

    async def idle_check_loop(self):
        """
        Bucle asíncrono que se ejecuta en segundo plano para
        comprobar la inactividad.
        """
        while True:
            # Espera un tiempo (ej. 60 segundos) antes de la próxima comprobación
            await asyncio.sleep(60)

            is_idle = False
            if self.model is not None:
                with self.access_lock:
                    idle_time = time.time() - self.last_used
                    if idle_time > self.idle_timeout:
                        is_idle = True

                if is_idle:
                    # Está inactivo. Usamos run_in_threadpool para la
                    # operación de descarga bloqueante.
                    await run_in_threadpool(self._unload_model_safe)

    def _unload_model_safe(self):
        """Wrapper seguro para descargar, usando el load_lock."""
        with self.load_lock:
            # Volver a comprobar el tiempo de inactividad *dentro* del bloqueo
            # para evitar una "race condition" donde una solicitud
            # justo acaba de usar el modelo.
            if self.model is None:
                return

            with self.access_lock:
                idle_time = time.time() - self.last_used

            if idle_time > self.idle_timeout:
                self._unload_model()
            else:
                logger.debug("Comprobación de inactividad cancelada (modelo usado recientemente).")


# --- Instanciar el gestor ---
manager = ModelManager(model_name, model_max_length, IDLE_TIMEOUT)

# --- 4. Dependencia de Seguridad (HTTP Bearer) ---
# (Sin cambios)
bearer_scheme = HTTPBearer(auto_error=False)


async def check_api_key(creds: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme)):
    """
    Verifica que la cabecera 'Authorization: Bearer <key>' coincida
    con la variable de entorno API_KEY.
    """
    if not API_KEY:
        return  # No hay clave configurada en el servidor, se permite el acceso

    if creds is None:
        logger.warning("Intento de acceso fallido: Cabecera Authorization faltante.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Cabecera 'Authorization' faltante o inválida"
        )
    if creds.scheme.lower() != "bearer":
        logger.warning(
            f"Intento de acceso fallido: Esquema incorrecto. Se esperaba 'Bearer', se recibió '{creds.scheme}'.")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Esquema de autenticación inválido. Se requiere 'Bearer'."
        )
    if creds.credentials != API_KEY:
        logger.warning("Intento de acceso fallido: Clave API incorrecta.")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Clave API incorrecta"
        )
    return


# --- 5. Inicializar FastAPI ---
app = FastAPI()


# --- 6. Definir Estructuras de la API (imitando a Cohere) ---
# (Sin cambios)
class RerankRequest(BaseModel):
    query: str
    documents: List[str]
    model: str = Field(default=model_name,
                       description="El nombre del modelo (ignorado por este servidor, pero parte de la API)")
    top_n: int = Field(default=None, description="Devuelve solo los N mejores resultados")


class RerankResult(BaseModel):
    index: int
    relevance_score: float


class RerankResponse(BaseModel):
    results: List[RerankResult]


# --- 7. Definir Endpoints ---

@app.get("/health")
async def health():
    """Endpoint de chequeo de salud."""
    # --- MODIFICADO: Informa el estado sin cargar el modelo ---
    model_status = "cargado" if manager.model is not None else "descargado (inactivo)"
    return {"status": "ok", "model": model_name, "model_status": model_status}


@app.post("/v1/rerank",
          response_model=RerankResponse,
          dependencies=[Depends(check_api_key)])
async def handle_rerank(request: RerankRequest):
    """
    Endpoint principal de reclasificación, compatible con la API Externa
    de Open Web UI (Cohere).
    """
    logger.info(f"Recibida solicitud de reclasificación para {len(request.documents)} documentos.")

    try:
        # --- MODIFICADO: Obtener el modelo del gestor ---
        # 1. Obtener el modelo.
        #    get_model() es bloqueante, así que lo ejecutamos en un threadpool
        #    para no bloquear el bucle de eventos de FastAPI.
        try:
            model = await run_in_threadpool(manager.get_model)
        except Exception as e:
            logger.error(f"Error al cargar el modelo bajo demanda: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Error al cargar el modelo: {e}"
            )

        logger.info("Modelo obtenido. Creando pares...")

        # 2. Crear los pares (consulta, documento) para el CrossEncoder
        pairs = [(request.query, doc) for doc in request.documents]

        # 3. Ejecutar el modelo en un hilo separado (para no bloquear el servidor)
        scores = await run_in_threadpool(model.predict, pairs, show_progress_bar=False)

        # 4. Formatear los resultados
        results = []
        for i, score in enumerate(scores):
            results.append(RerankResult(index=i, relevance_score=float(score)))

        # 5. Ordenar por puntuación (de mayor a menor)
        results.sort(key=lambda x: x.relevance_score, reverse=True)

        # 6. Aplicar top_n si se especificó
        if request.top_n and request.top_n > 0:
            results = results[:request.top_n]

        logger.info(f"Respuesta enviada con {len(results)} documentos reclasificados.")

        return RerankResponse(results=results)

    except Exception as e:
        # Si algo sale mal durante la predicción, lo registramos
        logger.error(f"Error procesando la solicitud: {e}", exc_info=True)
        # Y devolvemos un error HTTP claro
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error interno del servidor: {e}"
        )