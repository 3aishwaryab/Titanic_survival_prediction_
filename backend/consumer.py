from .mock_kafka import broker
import os, time
import joblib
from .utils import prepare_payload

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
# Load model safely
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Please train and place model here.")

model = joblib.load(MODEL_PATH)


def start_consumer(timeout: float = 1.0):
    """Process any queued passenger_data messages and publish prediction_result messages.
    This is non-blocking: it consumes existing messages and returns. If called repeatedly it will process new messages.
    """
    start = time.time()
    for msg in broker.subscribe("passenger_data"):
        try:
            df = prepare_payload(msg)
            # Model may be a sklearn pipeline expecting a DataFrame
            pred = int(model.predict(df)[0])
            broker.publish("prediction_result", {"prediction": pred})
        except Exception as e:
            # Publish an error result so the caller can be aware
            broker.publish("prediction_result", {"error": str(e)})
        # Respect timeout if subscriber is slow
        if time.time() - start > timeout:
            break


if __name__ == "__main__":
    start_consumer()
