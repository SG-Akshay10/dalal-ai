FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Setting python path so imports work correctly
ENV PYTHONPATH="${PYTHONPATH}:/app"
