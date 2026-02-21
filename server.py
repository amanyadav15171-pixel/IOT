from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

# ---------- LOAD MODEL ----------
model = tf.keras.models.load_model("keras_model.h5")

# ---------- LOAD LABELS ----------
labels = []
with open("labels.txt") as f:
    for line in f:
        # remove index safely
        clean = line.strip().split(" ", 1)[-1]
        labels.append(clean)

# ---------- SCIENTIFIC NAMES ----------
plant_info = {
    "MoneyPlant": "Epipremnum aureum",
    "Pepper": "Capsicum annuum",
    "Potato": "Solanum tuberosum",
    "Tomato": "Solanum lycopersicum"
}

# ---------- HOME ----------
@app.route('/')
def home():
    return render_template("index.html")


# ---------- PREDICTION ----------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # ===== RECEIVE IMAGE =====
        file = request.files['image']
        file.save("leaf.jpg")

        # ===== PREPROCESS IMAGE =====
        img = Image.open("leaf.jpg").convert("RGB")
        img = img.resize((224, 224))

        img = np.array(img, dtype=np.float32)
        img = (img / 127.5) - 1
        img = np.expand_dims(img, axis=0)

        # ===== MODEL PREDICTION =====
        prediction = model.predict(img, verbose=0)[0]

        # ===== GET TOP 2 PREDICTIONS =====
        sorted_preds = np.sort(prediction)[::-1]

        top_confidence = float(sorted_preds[0])
        second_confidence = float(sorted_preds[1])

        index = int(np.argmax(prediction))
        confidence_gap = top_confidence - second_confidence

        # ===== SMART VALIDATION =====
        CONFIDENCE_THRESHOLD = 0.75
        GAP_THRESHOLD = 0.20

        # Reject uncertain predictions
        if top_confidence < CONFIDENCE_THRESHOLD or confidence_gap < GAP_THRESHOLD:
            return jsonify({
                "plant": "No Plant Detected",
                "scientific": "-",
                "status": "Unknown",
                "disease": "Model not confident",
                "confidence": round(top_confidence * 100, 2)
            })

        # ===== GET LABEL =====
        result = labels[index]   # Example: Tomato_Healthy

        # Split safely
        parts = result.split("_")
        plant = parts[0].strip()
        condition = "_".join(parts[1:])

        # ===== CHECK NoPlant CLASS =====
        if "noplant" in plant.lower():
            return jsonify({
                "plant": "No Plant Detected",
                "scientific": "-",
                "status": "Unknown",
                "disease": "No leaf visible",
                "confidence": round(top_confidence * 100, 2)
            })

        # ===== HEALTH STATUS =====
        if "healthy" in condition.lower():
            status = "Healthy"
            disease = "None"
        else:
            status = "Diseased"
            disease = condition.replace("_", " ")

        # ===== CLEAN PLANT NAME =====
        plant_clean = plant.split()[-1]

        scientific = plant_info.get(plant_clean, "Unknown")

        # ===== FINAL RESPONSE =====
        return jsonify({
            "plant": plant_clean,
            "scientific": scientific,
            "status": status,
            "disease": disease,
            "confidence": round(top_confidence * 100, 2)
        })

    except Exception as e:
        print("ERROR:", e)
        return jsonify({"error": str(e)}), 500


# ---------- RUN SERVER ----------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)