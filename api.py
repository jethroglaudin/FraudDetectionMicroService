from flask import Flask, jsonify
import fraud_detection_model

app = Flask(__name__)

@app.route("/api/test")
def test():
    return "Hello user, this is a test"

@app.route("/api/fraud-detection/run")
def run_model():
    result = fraud_detection_model.run_model()
    return jsonify(result), 200

if __name__ == "__main__":
    app.run(host="localhost", port='8090')