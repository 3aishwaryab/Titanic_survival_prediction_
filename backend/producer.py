
from .mock_kafka import broker

def publish_event(data: dict):
    broker.publish("passenger_data", data)
