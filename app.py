import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# --- Función de validación de entradas ---
def validate_inputs(data):
    if not (10 <= data["age"] <= 100):
        raise ValueError("Edad fuera de rango (10-100 años).")
    if data["gender"] not in ["Male", "Female", "Other"]:
        raise ValueError("Género inválido.")
    if not (0.0 <= data["study_hours_per_day"] <= 10.0):
        raise ValueError("Horas de estudio fuera de rango (0-10).")
    if not (0.0 <= data["social_media_hours"] <= 10.0):
        raise ValueError("Horas en redes sociales fuera de rango (0-10).")
    if not (0.0 <= data["netflix_hours"] <= 10.0):
        raise ValueError("Horas en Netflix fuera de rango (0-10).")
    if data["part_time_job"] not in ["Yes", "No"]:
        raise ValueError("Valor inválido para trabajo de medio tiempo.")
    if not (0.0 <= data["attendance_percentage"] <= 100.0):
        raise ValueError("Porcentaje de asistencia fuera de rango (0-100%).")
    if not (0.0 <= data["sleep_hours"] <= 12.0):
        raise ValueError("Horas de sueño fuera de rango (0-12).")
    if data["diet_quality"] not in ["Poor", "Fair", "Good"]:
        raise ValueError("Valor inválido para calidad de dieta.")
    if not (0 <= data["exercise_frequency"] <= 14):
        raise ValueError("Frecuencia de ejercicio fuera de rango (0-14).")
    if data["parental_education_level"] not in ["None", "Primary", "Secondary", "Higher"]:
        raise ValueError("Nivel educativo de los padres inválido.")
    if data["internet_quality"] not in ["Poor", "Fair", "Good"]:
        raise ValueError("Valor inválido para calidad del internet.")
    if not (1 <= data["mental_health_rating"] <= 10):
        raise ValueError("Estado de salud mental fuera de rango (1-10).")
    if data["extracurricular_participation"] not in ["Yes", "No"]:
        raise ValueError("Valor inválido para participación extracurricular.")

# --- Cargar el modelo ---
model = joblib.load("model/gb_model.pkl")

# --- Interfaz de usuario ---
st.title("📊 Predicción del Puntaje de Examen")
st.markdown("""
Esta aplicación utiliza un modelo de inteligencia artificial para predecir el puntaje de examen de un estudiante, 
basado en factores como hábitos de estudio, sueño, uso de redes sociales y más.
""")

# --- Formulario de entrada ---
with st.form("student_form"):
    st.header("📝 Ingrese los datos del estudiante")

    age = st.number_input("Edad", min_value=10, max_value=100, value=20)
    gender = st.selectbox("Género", ["Male", "Female", "Other"])
    study_hours_per_day = st.slider("Horas de estudio por día", 0.0, 10.0, 2.0)
    social_media_hours = st.slider("Horas en redes sociales", 0.0, 10.0, 2.0)
    netflix_hours = st.slider("Horas en Netflix", 0.0, 10.0, 1.0)
    part_time_job = st.selectbox("¿Tiene trabajo de medio tiempo?", ["Yes", "No"])
    attendance_percentage = st.slider("Porcentaje de asistencia", 0.0, 100.0, 90.0)
    sleep_hours = st.slider("Horas de sueño por noche", 0.0, 12.0, 7.0)
    diet_quality = st.selectbox("Calidad de la dieta", ["Poor", "Fair", "Good"])
    exercise_frequency = st.slider("Frecuencia de ejercicio (veces por semana)", 0, 14, 3)
    parental_education_level = st.selectbox("Nivel educativo de los padres", ["None", "Primary", "Secondary", "Higher"])
    internet_quality = st.selectbox("Calidad del internet", ["Poor", "Fair", "Good"])
    mental_health_rating = st.slider("Estado de salud mental (1-10)", 1, 10, 5)
    extracurricular_participation = st.selectbox("¿Participa en actividades extracurriculares?", ["Yes", "No"])

    submitted = st.form_submit_button("🎯 Predecir puntaje")

# --- Procesamiento al enviar el formulario ---
if submitted:
    try:
        input_data_dict = {
            "age": age,
            "gender": gender,
            "study_hours_per_day": study_hours_per_day,
            "social_media_hours": social_media_hours,
            "netflix_hours": netflix_hours,
            "part_time_job": part_time_job,
            "attendance_percentage": attendance_percentage,
            "sleep_hours": sleep_hours,
            "diet_quality": diet_quality,
            "exercise_frequency": exercise_frequency,
            "parental_education_level": parental_education_level,
            "internet_quality": internet_quality,
            "mental_health_rating": mental_health_rating,
            "extracurricular_participation": extracurricular_participation
        }

        # Validar los datos
        validate_inputs(input_data_dict)

        # Crear DataFrame
        input_data = pd.DataFrame([input_data_dict])

        # Predicción
        prediction = model.predict(input_data)[0]
        st.success(f"✅ El puntaje estimado del examen es: **{round(prediction, 2)}**")

        # Visualización
        st.subheader("📈 Comparación de tiempo en actividades")
        fig, ax = plt.subplots()
        activities = ["Estudio", "Redes Sociales", "Netflix", "Sueño"]
        hours = [study_hours_per_day, social_media_hours, netflix_hours, sleep_hours]
        ax.bar(activities, hours, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"])
        ax.set_ylabel("Horas por día")
        ax.set_ylim(0, max(hours) + 1)
        ax.set_title("Distribución diaria de actividades")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ Error en la predicción: {str(e)}")