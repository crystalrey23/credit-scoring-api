import pytest
from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)

@pytest.fixture(scope="module")
def sample_payload():
    return {"feature1": 0.5, "feature2": 1.2}

def test_predict_endpoint(sample_payload):
    response = client.post("/predict", json=sample_payload)
    assert response.status_code == 200
    # ganti sesuai format respons sebenarnya
    assert "prediction" in response.json()
