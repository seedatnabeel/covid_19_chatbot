FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7
COPY . /app/main
COPY requirements.txt /tmp/
RUN pip install --requirement /tmp/requirements.txt
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
