FROM python:3.9-slim

WORKDIR /app

# Copy requirements from root (build context must be root)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy full src code (so src.api and src.metrics work)
COPY src/ /app/src/

# Set Python Path to include /app so "src.api.main" resolves
ENV PYTHONPATH=/app

# Expose Port
EXPOSE 8000

# Command
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
