# Use the official Python image as base
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY ./requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY ./src /app/

# Expose port 8080 to the outside world
EXPOSE 8080

# Command to run the application
CMD ["python", "-m", "src.api.app"]
