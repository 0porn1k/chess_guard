import joblib
import pandas as pd
from pathlib import Path

# Абсолютный путь
model_path = Path(r"C:\Users\portn\PycharmProjects\PythonProject3\models\lgbm.joblib")
model = joblib.load(model_path)

perfect = pd.DataFrame([{
    "match_top1": 1,
    "match_top3": 1,
    "cp_loss": 0
}])

print("P(чит | идеальный ход):", model.predict_proba(perfect)[0, 1])