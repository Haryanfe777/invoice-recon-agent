FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY recon/ recon/
# The dataset is deterministic (seed 42), so it is generated at build time, not copied.
RUN python -m recon.data_gen

RUN useradd --create-home app && chown -R app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# One process; FastAPI runs the synchronous agent in its thread pool.
CMD ["uvicorn", "recon.api:app", "--host", "0.0.0.0", "--port", "8000"]
