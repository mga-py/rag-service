# Usar una imagen base de Python
FROM python:3.11-slim

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar el archivo de requisitos e instalar dependencias
COPY requirements.txt .
# --no-cache-dir para mantener la imagen ligera
# torch se instala como dependencia de sentence-transformers
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el script del servidor
COPY app.py .

# Exponer el puerto en el que correrá FastAPI
EXPOSE 8000

# Comando para iniciar el servidor
# --host 0.0.0.0 para que sea accesible desde fuera del contenedor
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]