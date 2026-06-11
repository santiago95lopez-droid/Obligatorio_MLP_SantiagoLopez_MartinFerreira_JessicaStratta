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


# --- Etapa 2: Runtime (Runtime) ---
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copiamos los wheels pre-compilados y el requirements del builder
COPY --from=builder /app/wheels /app/wheels
COPY --from=builder /app/requirements.txt /app/requirements.txt

# Instalamos las dependencias desde los wheels (sin compilar)
RUN pip install --no-cache-dir --no-index --find-links=/app/wheels -r requirements.txt \
    && rm -rf /app/wheels

# Copiamos SOLO los archivos necesarios para ejecutar la app
COPY src /app/src
COPY modelohongos /app/modelohongos

EXPOSE 8080

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8080"]
