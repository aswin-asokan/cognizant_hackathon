from flask import Flask, request, jsonify
import pandas as pd
from isolation_test import run_isolation_forest  # your function

app = Flask(__name__)

@app.route("/api/analyze-fraud", methods=["POST"])
def analyze_fraud():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if not file.filename.endswith(".csv"):
        return jsonify({"error": "Invalid file format"}), 400

    try:
        # Read CSV
        df = pd.read_csv(file)

        # Call your isolation_test.py function
        result = run_isolation_forest(df)

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
