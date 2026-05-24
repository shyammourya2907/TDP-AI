# Setup Instructions

## 1. Create a Python Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

## 2. Install Dependencies
```bash
pip install -r requirements.txt
```

## 3. Environment Variables
Copy `.env.example` to `.env` and add your API keys.
```bash
cp .env.example .env
```

## 4. Running Scripts
You can run any script directly:
```bash
python machine-learning/supervised/linear_regression.py
```

## 5. Running the API
```bash
cd api
uvicorn app:app --reload
```
