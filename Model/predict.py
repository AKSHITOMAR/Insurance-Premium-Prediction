import pickle
import pandas as pd

# import the ml model
with open('Model/model.pkl', 'rb') as f:
    model = pickle.load(f)

# MLFlow
MODEL_VERSION = '1.0.0'

# Get class labels from model (important for matching probabilities to class names)
class_labels = model.classes_.tolist()

def predict_output(user_input: dict):

    df = pd.DataFrame([user_input])

    # Predict the class
    predicted_class = model.predict(df)[0]

    # Get probabilities for all classes
    probabilities = model.predict_proba(df)[0]
    confidence = max(probabilities)
    
    # Create mapping: {class_name: probability}
    class_probs = dict(zip(class_labels, map(lambda p: round(p, 4), probabilities)))

    return {
        "predicted_category": predicted_class,
        "confidence": round(confidence, 4),
        "class_probabilities": class_probs
    }


def get_model_info():

    classifier = model.named_steps["classifier"]

    return {
        "model_type": type(classifier).__name__,
        "model_version": MODEL_VERSION,
        "classes": class_labels,
        "features": list(model.feature_names_in_)
        if hasattr(model, "feature_names_in_")
        else [],
        "number_of_features": len(model.feature_names_in_)
        if hasattr(model, "feature_names_in_")
        else 0,
        "pipeline_steps": [
            {
                "name": name,
                "type": type(step).__name__
            }
            for name, step in model.steps
        ],
        "feature_importance_available": hasattr(
            classifier,
            "feature_importances_"
        )
    }

def get_feature_importance():

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    feature_names = preprocessor.get_feature_names_out()

    importances = classifier.feature_importances_

    feature_importance = dict(
        sorted(
            zip(feature_names, importances),
            key=lambda x: x[1],
            reverse=True
        )
    )

    return {
        "feature_importance": {
            name: round(float(value), 4)
            for name, value in feature_importance.items()
        }
    }