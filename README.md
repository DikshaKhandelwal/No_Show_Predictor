# No-Show Prediction API
1. Project Overview \
This project provides a FastAPI-based ML service that predicts whether a patient will miss an appointment using demographic and appointment features. \
Output: \
1 → No-show \
0 → Show 

# Run Locally
git clone <repo-url> \
cd No_Show_Predictor \
pip install -r requirements.txt \
uvicorn app.main:app --reload

# Docker Commands

docker pull dikshakh1010/no-show-predictor:latest

docker run -p 8000:8000 dikshakh1010/no-show-predictor:latest

# Curl Requests

Health check:
`
curl -X 'GET' \
  'http://127.0.0.1:8000/health' \
  -H 'accept: application/json'
`

Prediction:
`
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "age": 10,
  "gender": "M",
  "scholarship": 0,
  "diabetes": 0,
  "hipertension": 0,
  "sms_received": 0,
  "handicap": 0,
  "scheduled_day": "2026-02-11T11:05:25.220Z",
  "appointment_day": "2026-02-12T11:05:25.220Z"
}'
`
\
Response: 
`
{
  "no_show_probability": 0
}
`
Request:
`
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "age": 23,
  "gender": "F",
  "scholarship": 1,
  "diabetes": 0,
  "hipertension": 0,
  "sms_received": 1,
  "handicap": 0,
  "scheduled_day": "2026-03-31T12:09:00.277Z",
  "appointment_day": "2026-04-29T12:09:00.278Z"
}'
`

Response: 
`
{
  "no_show_probability": 1
}
`


# CI/CD
GitHub Actions pipeline: \
Ruff → code quality check \
Bandit → security scan (no HIGH issues) \
Pytest + Coverage ≥ 80% \
Docker build & push automatically on push to main \

