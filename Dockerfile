# --- Etapa 1: Constructor (Builder) ---
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV POETRY_NO_INTERACTION=1

WORKDIR /app

# Instalamos dependencias del sistema necesarias SOLO para compilar
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
    && rm -rf /var/lib/apt/lists/*

# Instalamos Poetry y el plugin para exportar
RUN pip install --no-cache-dir poetry \
    && poetry self add poetry-plugin-export

# Copiamos solo los archivos de configuración de dependencias
COPY pyproject.toml poetry.lock* /app/

# Exportamos las dependencias de producción sin hashes y construimos los wheels
RUN poetry export -f requirements.txt --output requirements.txt --without-hashes \
    && python -m pip wheel --wheel-dir=/app/wheels -r requirements.txt


# --- Etapa 2: Runtime (Runtime definitivo) ---
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Instalamos supervisor y las dependencias de imagen actualizadas para Debian Trixie
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        supervisor \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copiamos e instalamos los wheels del builder (mantiene la imagen ultra compacta)
COPY --from=builder /app/wheels /app/wheels
COPY --from=builder /app/requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --no-index --find-links=/app/wheels -r requirements.txt \
    && rm -rf /app/wheels

# Copiamos los archivos esenciales de la aplicación, el modelo y Streamlit[cite: 1]
COPY src /app/src
COPY modelohongos /app/modelohongos
COPY app_streamlit.py /app/app_streamlit.py

# Copiamos la configuración de Supervisor para controlar los dos procesos
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Exponemos el puerto 80 para Elastic Beanstalk y el 8080 para la API interna[cite: 2]
EXPOSE 80
EXPOSE 8080

# Comando por defecto para arrancar ambos servicios en paralelo
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]