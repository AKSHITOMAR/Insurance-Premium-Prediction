# 🏥 Insurance Premium Category Prediction

An end-to-end Machine Learning project that predicts the insurance premium category based on a user's demographic, lifestyle, and financial information.

The application is built using **FastAPI** for the backend, **Streamlit** for the frontend, and deployed using **Docker** on **AWS EC2**.

---

## 🚀 Live Demo

- **Frontend:** *(Add your Streamlit URL after deployment)*
- **Backend API:** `http://51.20.76.52:8000`
- **API Docs:** `http://51.20.76.52:8000/docs`

---

## 📌 Features

- Predicts Insurance Premium Category
- FastAPI REST API
- Interactive Streamlit User Interface
- Input validation using Pydantic
- Dockerized application
- AWS EC2 deployment
- Health check endpoint
- Modular project structure

---

## 🛠️ Tech Stack

### Machine Learning
- Python
- Scikit-learn
- Pandas
- NumPy
- Joblib

### Backend
- FastAPI
- Uvicorn
- Pydantic

### Frontend
- Streamlit

### Deployment
- Docker
- AWS EC2
- GitHub

---

## 📂 Project Structure

```text
Insurance-Premium-Prediction/
│
├── Model/
│   ├── predict.py
│   └── ...
│
├── schema/
│   ├── user_input.py
│   └── prediction_response.py
│
├── config/
│
├── app.py
├── frontend.py
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/AKSHITOMAR/Insurance-Premium-Prediction.git
```

Go to the project folder

```bash
cd Insurance-Premium-Prediction
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run FastAPI

```bash
uvicorn app:app --reload
```

API will run on

```
http://127.0.0.1:8000
```

Swagger Documentation

```
http://127.0.0.1:8000/docs
```

---

## ▶️ Run Streamlit

```bash
streamlit run frontend.py
```

Application will open at

```
http://localhost:8501
```

---

## 🐳 Docker

Build Docker image

```bash
docker build -t insurance-premium-api .
```

Run Docker container

```bash
docker run -p 8000:8000 insurance-premium-api
```

---

## 🌐 API Endpoint

### POST `/predict`

### Sample Request

```json
{
    "age": 30,
    "weight": 65,
    "height": 1.70,
    "income_lpa": 10,
    "smoker": false,
    "city": "Mumbai",
    "occupation": "private_job"
}
```

### Sample Response

```json
{
    "response": {
        "predicted_category": "Medium",
        "confidence": 0.91,
        "class_probabilities": {
            "Low": 0.04,
            "Medium": 0.91,
            "High": 0.05
        }
    }
}
```

---

## ❤️ Health Check

```
GET /health
```

Response

```json
{
    "status": "OK",
    "version": "1.0",
    "model_loaded": true
}
```

---

## 📸 Screenshots

### Streamlit Frontend

*(Add screenshot after deployment)*

### FastAPI Swagger UI

*(Add screenshot of `/docs` endpoint)*

---

## 📈 Future Improvements

- User Authentication
- Database Integration
- CI/CD Pipeline using GitHub Actions
- HTTPS with Nginx and Let's Encrypt
- Model Monitoring
- Cloud Storage Integration

---

## 👩‍💻 Author

**Akshi Tomar**

GitHub: https://github.com/AKSHITOMAR

---

## ⭐ If you like this project

Please consider giving it a ⭐ on GitHub!
