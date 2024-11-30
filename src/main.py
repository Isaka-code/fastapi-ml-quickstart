"""src/main.py"""

import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# モデルを読み込む
model = joblib.load("models/model.joblib")


# 入力データのスキーマ定義
class InputData(BaseModel):
    value: float


@app.post("/predict/")
def predict(data: InputData):
    # 入力データをモデルに渡す
    prediction = model.predict(np.array([[data.value]]))
    return {"input": data.value, "prediction": prediction[0]}
