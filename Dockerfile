FROM python:3.10-slim

WORKDIR /app

COPY . /app

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && apt-get upgrade -yq ca-certificates && \
    apt-get install -yq --no-install-recommends \
    prometheus-node-exporter

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 7860
EXPOSE 8000
EXPOSE 9100

# CMD ["python", "app.py"]
CMD bash -c "prometheus-node-exporter --web.listen-address=':9100' & python app.py"