FROM python:3.10-slim
# FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

WORKDIR /app

COPY requirements_docker.txt requirements_docker.txt
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements_docker.txt

COPY . .

CMD ["python", "inference.py"]