
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_home():
    response = client.get("/")

    assert response.status_code == 200
    assert "message" in response.json()


def test_predict():
    test_data = {
        "pclass": 3,
        "sex": "male",
        "age": 25.0,
        "sibsp": 0,
        "parch": 0,
        "fare": 7.25,
        "embarked": "S",
        "who": "man",
        "adult_male": True,
        "embark_town": "Southampton",
        "alone": True
    }

    response = client.post("/predict", json=test_data)

    assert response.status_code == 200

    result = response.json()

    assert "prediction" in result
    assert "prediction_label" in result
    assert "probability" in result
