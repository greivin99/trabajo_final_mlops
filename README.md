## 👥 Autores
- Greivin Alejandro Pérez Conejo | Sección 2 (Jueves)
- Erick Rodríguez Arguedas | Sección 1 (Lunes)

# 📊 Predicción del Puntaje de Examen Estudiantil

Este proyecto utiliza un modelo de machine learning para predecir el puntaje de examen de un estudiante en función de sus hábitos de estudio, sueño, ejercicio, uso de redes sociales, calidad del internet, entre otras variables personales y contextuales.

## 🧠 ¿Cómo funciona?
- El modelo está basado en un algoritmo de **Gradient Boosting** entrenado sobre un conjunto de datos sintético de estudiantes.
- Recibe como entrada características del estudiante a través de una interfaz web desarrollada con **Streamlit**.
- Realiza el preprocesamiento automáticamente (mediante un pipeline) y genera una predicción numérica del puntaje esperado en el examen.
- También incluye una visualización interactiva de las horas dedicadas a diferentes actividades.

## 🌐 App en línea (API)
Puedes usar la API para hacer predicciones accediendo a la siguiente dirección desde cualquier navegador o herramienta como Postman:  
🔗 [http://3.145.1.22:8001/docs](http://3.145.1.22:8001/docs)  
Esta instancia está alojada en **Amazon EC2**.

## 🚀 ¿Cómo ejecutarlo localmente?
1. Clona el repositorio.
2. Instala dependencias desde `requirements.txt`.
3. Ejecuta la app con Streamlit:
   ```bash
   streamlit run app.py
   ```
4. O usa Docker:
   ```bash
   docker build -t greivin99/trabajo_final_mlops .
   docker run -p 8001:8000 greivin99/trabajo_final_mlops
   ```

## 🧪 Ejemplo de entrada
- Edad: 20  
- Horas de estudio por día: 2  
- Horas de redes sociales: 3  
- Calidad del internet: Buena  
- Participación extracurricular: Sí  
- Salud mental (1-10): 6  

