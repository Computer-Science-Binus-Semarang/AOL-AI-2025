import gradio as gr
import pandas as pd
import joblib

# ======================
# LOAD MODEL
# ======================
model = joblib.load("workout_model.pkl")

MEAL_PLAN = {
    "Yoga": "Light & plant-based meals",
    "HIIT": "High protein meals",
    "Strength": "High calorie & protein meals",
    "Cardio": "Balanced high-energy meals"
}

# ======================
# PREDICTION FUNCTION
# ======================
def predict_workout(
    age, gender, weight, height,
    duration, calories, frequency, experience
):
    input_data = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Weight (kg)": weight,
        "Height (m)": height,
        "Session_Duration (hours)": duration,
        "Calories_Burned": calories,
        "Workout_Frequency (days/week)": frequency,
        "Experience_Level": experience
    }])

    workout = model.predict(input_data)[0]
    meal = MEAL_PLAN.get(workout, "Balanced meal")

    return f"""
### 🏋 Workout Type
*{workout}*

### 🍽 Recommended Meal Plan
*{meal}*
"""

# ======================
# CUSTOM CSS
# ======================
custom_css = """
body {
    background-color: #071925;
}

/* Header */
h1, h2 {
    color: #FFFFFF !important;
    text-align: center;
}

/* Main Input Card */
.input-card {
    background-color: #071925;
    padding: 30px;
    border-radius: 18px;
}

/* Input fields */
input, select {
    background-color: #F4EAD5 !important;
    color: #000000 !important;
    border-radius: 12px !important;
    border: none !important;
    padding: 14px !important;
    font-size: 16px !important;
}

/* Labels */
label {
    color: #FFFFFF !important;
    font-weight: 600;
}

/* Button */
button {
    background-color: #FF6A2A !important;
    color: #FFFFFF !important;
    border-radius: 14px !important;
    font-size: 18px !important;
    padding: 14px !important;
}

/* Result box */
.result-box {
    background-color: #F4EAD5;
    color: #000000;
    padding: 28px;
    border-radius: 18px;
    margin-top: 30px;
    font-size: 18px;
}
"""

# ======================
# UI
# ======================
with gr.Blocks(css=custom_css, title="FitAI") as demo:

    gr.Markdown("# 💪 FitAI")
    gr.Markdown("## Generate Your Workout Plan")

    with gr.Column(elem_classes="input-card"):
        age = gr.Number(label="Age")
        gender = gr.Dropdown(["Male", "Female"], label="Gender")
        weight = gr.Number(label="Weight (kg)")
        height = gr.Number(label="Height (m)")
        duration = gr.Number(label="Session Duration (hours)")
        calories = gr.Number(label="Target Calories Burned")
        frequency = gr.Number(label="Workout Frequency (days/week)")
        experience = gr.Number(
            label="Experience Level (1=Beginner, 2=Intermediate, 3=Advanced)",
            minimum=1, maximum=3
        )

        submit = gr.Button("Generate Workout Plan")

    output = gr.Markdown(elem_classes="result-box")

    submit.click(
        predict_workout,
        inputs=[
            age, gender, weight, height,
            duration, calories, frequency, experience
        ],
        outputs=output
    )

demo.launch()