# Docker & CI/CD Setup

## Docker Usage

### Building the Image Locally
```bash
docker build -t no-show-predictor .
```

### Running the Container
```bash
docker run -p 8000:8000 no-show-predictor
```

The API will be available at `http://localhost:8000`

### Health Check
```bash
curl http://localhost:8000/health
```

## GitHub Actions CI/CD

The pipeline automatically:

1. **Code Quality Checks**
   - Runs `ruff check` for linting
   - Runs `bandit -r app/` for security analysis
   - Runs tests with coverage reporting

2. **Docker Build & Push** (on main/master push)
   - Builds multi-architecture Docker image
   - Pushes to Docker Hub
   - Tests the built image

### Required GitHub Secrets

Add these secrets to your GitHub repository:

- `DOCKERHUB_USERNAME`: Your Docker Hub username
- `DOCKERHUB_TOKEN`: Your Docker Hub access token

### Setting up Docker Hub Token

1. Go to Docker Hub → Account Settings → Security
2. Create a new Access Token
3. Add it as `DOCKERHUB_TOKEN` secret in GitHub

## Local Development

### Install Dependencies
```bash
pip install -r requirements.txt
pip install ruff bandit
```

### Run Quality Checks
```bash
# Linting
ruff check .

# Security analysis
bandit -r app/

# Tests with coverage
pytest --cov=app --cov-report=term-missing
```

### Start Development Server
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```