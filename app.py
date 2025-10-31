import logging
import os
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import List, Optional
from sentence_transformers.cross_encoder import CrossEncoder
from starlette.concurrency import run_in_threadpool

# --- 1. Configuración del Logging ---
# Configurar el logging para que muestre mensajes INFO
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- 2. Cargar Configuración y Modelo ---
logger.info("Iniciando servidor de reclasificación...")

# Cargar configuración desde variables de entorno
model_name = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-large")
model_max_length = int(os.getenv("MODEL_MAX_LENGTH", "512"))
API_KEY = os.getenv("API_KEY", None)  # La clave API del servidor

try:
    model = CrossEncoder(model_name, max_length=model_max_length)
    logger.info(f"Modelo {model_name} (max_length: {model_max_length}) cargado exitosamente.")
    if API_KEY:
        logger.info("Protección con clave API HABILITADA.")
    else:
        logger.warning("Protección con clave API DESHABILITADA (Servidor abierto).")
except Exception as e:
    logger.error(f"Error fatal al cargar el modelo: {e}", exc_info=True)
    exit()  # Si el modelo no carga, no tiene sentido seguir

# --- 3. Dependencia de Seguridad (HTTP Bearer) ---
# Esto le dice a FastAPI que busque un header "Authorization: Bearer <token>"
bearer_scheme = HTTPBearer(auto_error=False)

async def check_api_key(creds: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme)):
    """
    Verifica que la cabecera 'Authorization: Bearer <key>' coincida
    con la variable de entorno API_KEY.
    """
    if not API_KEY:
        return  # No hay clave configurada en el servidor, se permite el acceso

    # Si creds es None, el cliente no envió la cabecera Authorization
    if creds is None:
        logger.warning("Intento de acceso fallido: Cabecera Authorization faltante.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Cabecera 'Authorization' faltante o inválida"
        )

    # Verifica que el esquema sea "Bearer"
    if creds.scheme.lower() != "bearer":
        logger.warning(f"Intento de acceso fallido: Esquema incorrecto. Se esperaba 'Bearer', se recibió '{creds.scheme}'.")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Esquema de autenticación inválido. Se requiere 'Bearer'."
        )

    # Compara la clave enviada (credentials) con la clave del servidor (API_KEY)
    if creds.credentials != API_KEY:
        logger.warning("Intento de acceso fallido: Clave API incorrecta.")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Clave API incorrecta"
        )

    return  # Clave correcta

# --- 4. Inicializar FastAPI ---
app = FastAPI()

# --- 5. Definir Estructuras de la API (imitando a Cohere) ---

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


# --- 6. Definir Endpoints ---

@app.get("/health")
async def health():
    """Endpoint de chequeo de salud."""
    return {"status": "ok", "model": model_name}


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
        # 1. Crear los pares (consulta, documento) para el CrossEncoder
        pairs = [(request.query, doc) for doc in request.documents]

        # 2. Ejecutar el modelo en un hilo separado (para no bloquear el servidor)
        scores = await run_in_threadpool(model.predict, pairs, show_progress_bar=False)

        # 3. Formatear los resultados
        results = []
        for i, score in enumerate(scores):
            results.append(RerankResult(index=i, relevance_score=float(score)))

        # 4. Ordenar por puntuación (de mayor a menor)
        results.sort(key=lambda x: x.relevance_score, reverse=True)

        # 5. Aplicar top_n si se especificó
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