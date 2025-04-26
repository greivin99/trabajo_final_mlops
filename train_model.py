import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import os
import matplotlib.pyplot as plt


# --------------------------------------------
# CARGA DE DATOS
# --------------------------------------------
df = pd.read_csv("data/student_habits_performance.csv")

# Eliminar columnas irrelevantes
df = df.drop(columns=["student_id"])  # No aporta a la predicción

X = df.drop(columns=["exam_score"])
y = df["exam_score"]

# --------------------------------------------
# PREPROCESAMIENTO
# --------------------------------------------
categorical_cols = X.select_dtypes(include="object").columns.tolist()
numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numerical_cols),
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols)
])

# --------------------------------------------
# PIPELINE DEL MODELO
# --------------------------------------------
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42))
])

# --------------------------------------------
# ENTRENAMIENTO Y EVALUACIÓN
# --------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# --------------------------------------------
# GUARDAR MÉTRICAS
# --------------------------------------------
with open("metrics.txt", "w") as f:
    f.write(f"MAE: {mae:.2f}\n")
    f.write(f"RMSE: {rmse:.2f}\n")
    f.write(f"R2 Score: {r2:.2f}\n")

# --------------------------------------------
# GUARDAR MODELO
# --------------------------------------------
os.makedirs("model", exist_ok=True)
joblib.dump(pipeline, "model/gb_model.pkl")

feature_names = numerical_cols + list(
    preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_cols)
)

importances = pipeline.named_steps["model"].feature_importances_

# Crear gráfico
plt.figure(figsize=(10, 6))
plt.barh(feature_names, importances)
plt.xlabel("Importancia")
plt.title("Importancia de las características")
plt.tight_layout()

# Guardar imagen
plt.savefig("feature_importance.png")

print("✅ Modelo entrenado y guardado correctamente.")
