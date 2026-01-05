
from fastapi import FastAPI, HTTPException, Request
import time
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

# Try package-relative imports first; if running as a script, add project root to sys.path and import absolutely
try:
    from .producer import publish_event
    from .mock_kafka import broker
    from .consumer import start_consumer
except Exception as e:
    # Running as a script or from the debugger may not set the package context. Ensure project root is on sys.path.
    import sys, os, importlib.util
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    try:
        # Try normal absolute import now that project root is on sys.path
        from backend.producer import publish_event
        from backend.mock_kafka import broker
        from backend.consumer import start_consumer
    except Exception:
        # Final fallback: load modules directly from file paths (works even if package import fails)
        def load_module(name, rel_path):
            path = os.path.join(project_root, rel_path)
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module

        try:
            producer_mod = load_module('backend_producer', os.path.join('backend', 'producer.py'))
            publish_event = getattr(producer_mod, 'publish_event')

            mock_mod = load_module('backend_mock_kafka', os.path.join('backend', 'mock_kafka.py'))
            broker = getattr(mock_mod, 'broker')

            consumer_mod = load_module('backend_consumer', os.path.join('backend', 'consumer.py'))
            start_consumer = getattr(consumer_mod, 'start_consumer')
        except Exception as load_err:
            raise ImportError("Failed to import backend modules. Consider running with `python -m backend.main` or ensure project root is on PYTHONPATH.") from load_err

app = FastAPI(title="Titanic Survival API")


@app.get("/", tags=["health"])
def root():
    return {"status": "ok", "message": "POST JSON to /predict to receive a prediction."}


@app.post("/predict")
def predict(payload: dict):
    # Basic payload validation
    if not isinstance(payload, dict) or not payload:
        raise HTTPException(status_code=400, detail="Payload must be a non-empty JSON object")

    # Publish the event to the mock broker
    publish_event(payload)

    # Trigger consumer to process any pending messages (non-blocking)
    start_consumer(timeout=2.0)

    # Check for prediction result
    results = broker.topics.get("prediction_result", [])

    if not results:
        raise HTTPException(status_code=500, detail="No prediction available yet")

    # Pop the first result and return it
    result = results.pop(0)

    if 'error' in result:
        raise HTTPException(status_code=500, detail=result['error'])

    return {"prediction": int(result.get('prediction', -1))}


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    # Provide clearer message for 405, otherwise return original detail
    if getattr(exc, 'status_code', None) == 405:
        return JSONResponse(status_code=405, content={"detail": "Method Not Allowed. Please use POST /predict with a JSON payload."})
    return JSONResponse(status_code=getattr(exc, 'status_code', 500), content={"detail": getattr(exc, 'detail', 'Server Error')})


if __name__ == '__main__':
    # Allow running this module directly for debugging: `python backend/main.py`
    import uvicorn
    print('Starting uvicorn for debugging: http://127.0.0.1:8000')
    uvicorn.run('backend.main:app', host='127.0.0.1', port=8000, reload=False)
