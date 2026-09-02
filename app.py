from fastapi import FastAPI, HTTPException
from schema.user_input import UserInput
from schema.prediction_response import PredictionResponse
from Model.predict import (
    predict_output,
    get_model_info,
    get_feature_importance,
    MODEL_VERSION,
    model
)

app = FastAPI(
    title="Insurance Premium Prediction API",
    description="ML-powered insurance premium category prediction API",
    version="1.0.0"
)


@app.get("/")
def home():
    return {
        "message": "Insurance Premium Prediction API",
        "version": MODEL_VERSION
    }


@app.get("/health")
def health_check():
    return {
        "status": "OK",
        "version": MODEL_VERSION,
        "model_loaded": model is not None
    }

@app.get("/model-info")
def model_info():

    return get_model_info()


@app.get("/feature-importance")
def feature_importance():

    return get_feature_importance()

@app.post("/predict", response_model=PredictionResponse)
def predict_premium(data: UserInput):

    user_input = {
        "bmi": data.bmi,
        "age_group": data.age_group,
        "lifestyle_risk": data.lifestyle_risk,
        "city_tier": data.city_tier,
        "income_lpa": data.income_lpa,
        "occupation": data.occupation,
    }

    try:
        prediction = predict_output(user_input)

        return prediction

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )




