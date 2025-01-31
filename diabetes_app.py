import gradio as gr
import joblib
import numpy as np

def predict_diabetes(pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age):
    try:
        model = joblib.load("diabetes_logistic_model.pkl")
        input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1] * 100
        result = "Diabetic" if prediction == 1 else "Not Diabetic"
        return f"Prediction: {result} (Confidence: {probability:.2f}%)"
    except Exception as e:
        return f"Error: {str(e)}"

def calculate_bmi(weight, height):
    if height > 0:
        bmi = weight / ((height / 100) ** 2)
        return f"Your BMI: {bmi:.2f}"
    return "Invalid height value!"

# Custom CSS for a modern and clean UI
custom_css = """
body { 
    font-family: 'Arial', sans-serif; 
    background-color: #f7fafc; 
    color: #2d3748; 
}
h1, h2, h3 { 
    color: #2b6cb0; 
}
.gr-box { 
    background-color: #ffffff; 
    border-radius: 8px; 
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); 
    padding: 20px; 
    margin-bottom: 20px; 
}
.gr-button { 
    background-color: #4299e1; 
    color: white; 
    border-radius: 6px; 
    padding: 10px 20px; 
    border: none; 
    cursor: pointer; 
    transition: background-color 0.3s ease; 
}
.gr-button:hover { 
    background-color: #3182ce; 
}
.gr-number-input, .gr-textbox { 
    border-radius: 6px; 
    border: 1px solid #e2e8f0; 
    padding: 10px; 
    width: 100%; 
}
.gr-textbox { 
    background-color: #edf2f7; 
}
"""

with gr.Blocks(css=custom_css) as app:
    gr.Markdown("""
    # 🩺 **Diabetes Prediction & BMI Calculator**
    Enter your health details below to check your diabetes risk and calculate BMI.
    """)
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🧪 **Health Metrics**")
            pregnancies = gr.Number(label="Pregnancies", value=0)
            glucose = gr.Number(label="Glucose Level (mg/dL)", value=120)
            blood_pressure = gr.Number(label="Blood Pressure (mmHg)", value=80)
            skin_thickness = gr.Number(label="Skin Thickness (mm)", value=20)
            insulin = gr.Number(label="Insulin Level (mu U/ml)", value=85)
            bmi = gr.Number(label="BMI", value=25.0)
            dpf = gr.Number(label="Diabetes Pedigree Function (DPF)", value=0.5, interactive=True)
            age = gr.Number(label="Age", value=30)
            predict_btn = gr.Button("🔍 Predict Diabetes")
            prediction_output = gr.Textbox(label="Prediction Result", interactive=False)
        
        with gr.Column():
            gr.Markdown("### ⚖️ **BMI Calculator**")
            weight = gr.Number(label="Weight (kg)", value=70)
            height = gr.Number(label="Height (cm)", value=170)
            bmi_btn = gr.Button("⚖️ Calculate BMI")
            bmi_output = gr.Textbox(label="Your BMI", interactive=False)
            
            gr.Markdown("""
            ### ℹ️ **What is Diabetes Pedigree Function (DPF)?**
            - It estimates your **genetic risk** of diabetes based on family history.
            - If unsure, start with **0.5 (average risk)**.
            - A doctor or genetic test can provide an accurate value.
            """)
    
    # Add informational details for Glucose, Skin Thickness, and Insulin
    gr.Markdown("""
    ### 📊 **Understanding Your Health Metrics**
    Here’s what each input means and how it affects your diabetes risk:
    
    #### **1. Glucose Level (mg/dL)**
    - Glucose is the sugar in your blood that provides energy.
    - **Normal Range**: 70–100 mg/dL (fasting).
    - **High Glucose (>126 mg/dL fasting)**: May indicate diabetes or prediabetes.
    
    #### **2. Skin Thickness (mm)**
    - Skin thickness is measured using a skinfold caliper.
    - It’s used to estimate body fat percentage.
    - **Higher Values**: May indicate higher body fat, which is a risk factor for diabetes.
    
    #### **3. Insulin Level (mu U/ml)**
    - Insulin is a hormone that regulates blood sugar.
    - **Normal Range**: 2.6–24.9 µU/mL (fasting).
    - **High Insulin**: May indicate insulin resistance, a precursor to diabetes.
    
    #### **4. BMI (Body Mass Index)**
    - BMI is a measure of body fat based on height and weight.
    - **Normal Range**: 18.5–24.9.
    - **Higher BMI**: Increases the risk of diabetes and other health issues.
    """)
    
    predict_btn.click(predict_diabetes, inputs=[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age], outputs=prediction_output)
    bmi_btn.click(calculate_bmi, inputs=[weight, height], outputs=bmi_output)
    
app.launch(share=True)