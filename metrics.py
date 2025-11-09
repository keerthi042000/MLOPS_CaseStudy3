from prometheus_client import Counter, Gauge, Histogram, start_http_server
import time
import os

METRICS_PORT = int(os.environ.get("METRICS_PORT", 8000))
start_http_server(METRICS_PORT)

REQUEST_COUNT = Counter('app_requests_total', 'Total number of requests', ['endpoint'])
INPROGRESS = Gauge('app_inprogress_requests', 'In-progress requests')
REQUEST_LATENCY = Histogram('app_request_latency_seconds', 'Request latency seconds', ['endpoint'])
MODEL_CALLS = Counter('model_calls_total', 'How many times model was invoked')
IMAGE_SIZE_PIXELS = Gauge('last_image_pixels', 'Number of pixels in last processed image')
ERROR_COUNT = Counter('app_errors_total', 'Total number of errors')

def observe_endpoint(endpoint):
    def decorator(fn):
        def wrapper(*args, **kwargs):
            REQUEST_COUNT.labels(endpoint=endpoint).inc()
            INPROGRESS.inc()
            start = time.time()
            try:
                out = fn(*args, **kwargs)
                return out
            except Exception as e:
                ERROR_COUNT.inc()
                raise
            finally:
                duration = time.time() - start
                REQUEST_LATENCY.labels(endpoint=endpoint).observe(duration)
                INPROGRESS.dec()
        return wrapper
    return decorator
