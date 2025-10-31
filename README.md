# Rag-service
Rag service for Open WebUI servidor de reclasificación, modelo reranker
Este repositorio contiene un servicio de recuperación y generación (RAG) para Open WebUI, que permite la reclasificación y reranking de modelos de lenguaje. El servicio está diseñado para integrarse fácilmente con Open WebUI y mejorar la precisión de las respuestas generadas por los modelos de lenguaje.
## Características
- Integración con Open WebUI
- Soporte para múltiples modelos de lenguaje
- Reclasificación y reranking de respuestas generadas
- Fácil configuración y despliegue
## Requisitos
- Python 3.8 o superior
- Open WebUI
- Bibliotecas necesarias (ver requirements.txt)
## Instalación
  1. Clona este repositorio:
     ```bash
     git clone https://github.com/mga-py/rag-service.git
      cd rag-service
      ```
  2. Instala las dependencias:
     ```bash
     pip install -r requirements.txt
     ```
## Configuración
  Configura el servicio editando el archivo `docker-compose.yml` según tus necesidades. Asegúrate de especificar los modelos de lenguaje y las opciones de reclasificación.
## Uso
  Inicia el servicio ejecutando el siguiente comando:
     ```bash
     python rag_service.py
     ```
  Integra el servicio con Open WebUI siguiendo las instrucciones en la documentación de Open WebUI.
## Contribución
  Las contribuciones son bienvenidas. Por favor, abre un issue o envía un pull request para cualquier mejora o corrección.
## Licencia
  Este proyecto está licenciado bajo la Licencia Apache 2.0. Consulta el archivo LICENSE para más detalles.
