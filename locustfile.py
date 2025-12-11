# locustfile.py
from locust import HttpUser, task, between
import random
import time

FEATURES = ["Time"] + [f"V{i}" for i in range(1,29)] + ["Amount"]

def random_transaction():
    # Produce plausible random values; adapt ranges to your data distribution.
    tx = {}
    tx["Time"] = random.uniform(0, 1e6)   # change to realistic time range if needed
    for i in range(1, 29):
        # V1..V28 are PCA-like, use small gaussian
        tx[f"V{i}"] = random.normalvariate(0, 1)
    tx["Amount"] = round(random.expovariate(1/50.0), 2)  # skewed amount distribution
    return {"transaction": tx}

class FraudUser(HttpUser):
    wait_time = between(0.5, 1.5)  # adjust concurrency rate
    @task
    def predict(self):
        payload = random_transaction()
        headers = {"Content-Type": "application/json"}
        # replace host:port with your LoadBalancer IP or domain
        self.client.post("/predict", json=payload, headers=headers, timeout=30)
