import pandas as pd
import joblib

model = joblib.load("models/lgbm.joblib")

# Идеальный ход
perfect = pd.DataFrame([{
    "match_top1": 1,
    "match_top3": 1,
    "cp_loss": 0
}])

print(model.predict_proba(perfect))
# Ожидаемо: [[0.58, 0.42]] или около того
# Если второе число (класс 1) < 0.7 → модель недообучена