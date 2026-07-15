# --- Etapa 1: Constructor (Builder) ---
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV POETRY_NO_INTERACTION=1

WORKDIR /app

# Instalamos Poetry y el plugin para exportar (Ya no necesitamos gcc ni g++)
RUN pip install --no-cache-dir poetry \
    && poetry self add poetry-plugin-export

# Copiamos solo los archivos de configuración de dependencias
COPY pyproject.toml poetry.lock* /app/

# Exportamos las dependencias
RUN poetry export -f requirements.txt --output requirements.txt --without-hashes

# Descargamos los wheels oficiales ya precompilados. 
# Usamos --no-cache-dir para ahorrar espacio en disco durante el build.
RUN pip download --destination-directory /app/wheels -r requirements.txt


# --- Etapa 2: Runtime (Runtime definitivo) ---
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Instalamos supervisor y dependencias de sistema necesarias para OpenCV/Pillow
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        supervisor \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copiamos e instalamos los wheels del builder
COPY --from=builder /app/wheels /app/wheels
COPY --from=builder /app/requirements.txt /app/requirements.txt

# Instalamos limitando la memoria y usando los binarios descargados
RUN pip install --no-cache-dir --no-index --find-links=/app/wheels -r requirements.txt \
    && rm -rf /app/wheels

# Copiamos los archivos esenciales de la aplicación, el modelo y Streamlit
COPY src /app/src
COPY modelohongos /app/modelohongos
COPY app_streamlit.py /app/app_streamlit.py

# Copiamos la configuración de Supervisor
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Exponemos los puertos
EXPOSE 80
EXPOSE 8080

# Comando por defecto para arrancar ambos servicios en paralelo
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]