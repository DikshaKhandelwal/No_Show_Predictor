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

# Docker
Run: \
docker pull <your-username>/no-show-predictor:latest \
docker run -p 8000:8000 <your-username>/noshow-predictor:latest

# CI/CD
GitHub Actions pipeline: \
Ruff → code quality check \
Bandit → security scan (no HIGH issues) \
Pytest + Coverage ≥ 80% \
Docker build & push automatically on push to main \

