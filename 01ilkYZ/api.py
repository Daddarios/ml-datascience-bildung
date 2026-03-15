import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io

from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, accuracy_score

# ---------------- DATA ----------------
df = pd.read_csv("glass.csv")

X = df.drop("Type", axis=1)
y = df["Type"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=9)

# ---------------- MODEL ----------------
model = RandomForestClassifier()
model.fit(X_train, y_train)

cam_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, cam_pred))
print(classification_report(y_test, cam_pred))

# ---------------- FASTAPI ----------------
app = FastAPI()

# 📊 Confusion Matrix endpoint
@app.get("/confusion-matrix")
def confusion_matrix_endpoint():

    fig, ax = plt.subplots(figsize=(6,6))
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax)

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)

    return StreamingResponse(buf, media_type="image/png")

# 📊 Classification report endpoint
@app.get("/classification-report")
def classification_report_endpoint():
    report = classification_report(y_test, cam_pred, output_dict=True)
    return JSONResponse(report)