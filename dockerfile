# Use an official Python 3.11.9 runtime as a parent image.
# The 'slim-bookworm' variant is a good balance of size and functionality.
FROM python:3.11.9-slim-bookworm

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies. While the base image is robust,
# explicitly installing build-essential can prevent issues with
# packages that have complex C extensions.
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install dependencies first.
# This takes advantage of Docker's layer caching, so dependencies
# are only re-installed when requirements.txt changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application source code into the container
COPY . .

# The CMD is not strictly necessary as Render's startCommand will override it,
# but it's good practice for local testing.
CMD ["gunicorn", "-w", "2", "-k", "gthread", "-b", "0.0.0.0:8080", "data_ingestion.main:app"]
