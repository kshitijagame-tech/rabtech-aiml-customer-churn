
# Real-Time ML Inference REST API

## Project Overview

This project deploys a trained Machine Learning model as a real-time REST API using FastAPI.

## Model

The API uses the trained Titanic classification champion model from Task 4.

## Technologies Used

- Python
- FastAPI
- Uvicorn
- Scikit-learn
- Pandas
- Joblib
- Pydantic
- Docker

## API Endpoints

### GET /
Checks whether the API is running.

### POST /predict
Accepts passenger information in JSON format and returns:

- Prediction
- Prediction label
- Prediction probability

## Example Input

```json
{
  "pclass": 3,
  "sex": "male",
  "age": 25.0,
  "sibsp": 0,
  "parch": 0,
  "fare": 7.25,
  "embarked": "S",
  "who": "man",
  "adult_male": true,
  "embark_town": "Southampton",
  "alone": true
}
