from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

app = Flask(__name__)

# Load trained model
model = tf.keras.models.load_model(
    "student_performance_dl_model.keras"
)

# Load preprocessor
preprocessor = joblib.load(
    "student_preprocessor.pkl"
)


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    performance = None

    # Store entered values
    student_data = {}

    if request.method == "POST":

        student_data = {
            "gender": request.form["gender"],
            "race": request.form["race"],
            "parental_education": request.form["parental_education"],
            "lunch": request.form["lunch"],
            "test_preparation": request.form["test_preparation"],
            "reading_score": float(request.form["reading_score"]),
            "writing_score": float(request.form["writing_score"])
        }

        # Create DataFrame
        new_student = pd.DataFrame({
            "Gender": [student_data["gender"]],
            "Race": [student_data["race"]],
            "Parental_Education": [
                student_data["parental_education"]
            ],
            "Lunch": [student_data["lunch"]],
            "Test_Preparation": [
                student_data["test_preparation"]
            ],
            "Reading_Score": [
                student_data["reading_score"]
            ],
            "Writing_Score": [
                student_data["writing_score"]
            ]
        })

        # Preprocess
        processed_data = preprocessor.transform(
            new_student
        )

        processed_data = np.asarray(
            processed_data
        )

        # Prediction
        predicted_score = model.predict(
            processed_data,
            verbose=0
        )

        prediction = float(
            predicted_score[0][0]
        )

        # Performance level
        if prediction >= 90:
            performance = "Excellent"
        elif prediction >= 75:
            performance = "Very Good"
        elif prediction >= 60:
            performance = "Good"
        elif prediction >= 50:
            performance = "Average"
        else:
            performance = "Needs Improvement"

    return render_template(
        "index.html",
        prediction=prediction,
        performance=performance,
        student_data=student_data
    )


if __name__ == "__main__":
    app.run(debug=True)