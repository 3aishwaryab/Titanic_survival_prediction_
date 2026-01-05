
class MockBroker:
    def __init__(self):
        self.topics = {}

    def publish(self, topic, message):
        self.topics.setdefault(topic, []).append(message)

    def subscribe(self, topic):
        while self.topics.get(topic):
            yield self.topics[topic].pop(0)

broker = MockBroker()
