FROM python:3.10-slim

WORKDIR /app

# Source: https://stackoverflow.com/questions/55036740/lightgbm-inside-docker-libgomp-so-1-cannot-open-shared-object-file
RUN apt-get update && apt-get install -y --no-install-recommends apt-utils
RUN apt-get -y install curl
RUN apt-get install libgomp1

COPY requirements_docker.txt requirements_docker.txt
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements_docker.txt

COPY . .

VOLUME ["/data", "/results"]

ENTRYPOINT ["python", "inference.py"]