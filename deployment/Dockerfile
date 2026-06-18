FROM python:3.10-slim

WORKDIR /app

# Install dependencies first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ /app/src/
COPY data/ /app/data/
COPY artifacts/ /app/artifacts/

# FIX: CMD was pointing to the root app.py (Streamlit) instead of src/app.py (FastAPI)
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]