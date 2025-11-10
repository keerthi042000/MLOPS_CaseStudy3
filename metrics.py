from prometheus_client import Counter, Gauge, Histogram, Summary, start_http_server
import time
import os

METRICS_PORT = int(os.environ.get("METRICS_PORT", 8000))
start_http_server(METRICS_PORT)

APP_MODEL_USAGE = Counter('app_model_usage_total', 'Number of times models are invoked', ['model'])
APP_STREAMING_CHUNKS = Counter('app_streaming_chunks_total', 'Number of streamed chunks received')

APP_ACTIVE_USERS = Gauge('app_active_users_current', 'Current number of active users')

REQUEST_COUNT = Counter('app_requests_total', 'Total number of requests', ['endpoint'])
REQUEST_LATENCY = Histogram('app_request_latency_seconds', 'Request latency seconds', ['endpoint'])
MODEL_CALLS = Counter('model_calls_total', 'How many times model was invoked')
IMAGE_SIZE_PIXELS = Gauge('last_image_pixels', 'Number of pixels in last processed image')
ERROR_COUNT = Counter('app_errors_total', 'Total number of errors')

APP_MESSAGE_LENGTH = Histogram('app_message_length_chars', 'User message length in chars', buckets=[10,50,100,200,500,1000])
APP_INFERENCE_TIME = Summary('app_inference_time_seconds', 'Time taken for inference')
APP_CARBON_FOOTPRINT = Summary('app_carbon_footprint_kg', 'Estimated carbon footprint per request')


def observe_endpoint(endpoint, model_name=None):
    def decorator(fn):
        def wrapper(*args, **kwargs):
            APP_ACTIVE_USERS.inc()
            REQUEST_COUNT.labels(endpoint=endpoint).inc()
            start = time.time()
            try:
                out = fn(*args, **kwargs)
                if model_name:
                    APP_MODEL_USAGE.labels(model=model_name).inc()
                if args and isinstance(args[0], str):
                    APP_MESSAGE_LENGTH.observe(len(args[0]))
                return out
            except Exception as e:
                ERROR_COUNT.inc()
                raise
            finally:
                duration = time.time() - start
                APP_INFERENCE_TIME.observe(duration)
                APP_CARBON_FOOTPRINT.observe(duration * 0.001)
                REQUEST_LATENCY.labels(endpoint=endpoint).observe(duration)
                # INPROGRESS.dec()
        return wrapper
    return decorator
