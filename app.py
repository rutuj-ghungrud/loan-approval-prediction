from flask import Flask, request, render_template, redirect, url_for
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

# Store the history of predictions
prediction_history = []

# Root now renders start.html directly
@app.route('/')
def start():
    return render_template("start.html")

# Main form page
@app.route('/form')
def form():
    return render_template("index.html", prediction_text="", history=prediction_history)

# Handle prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_features = [float(x) for x in request.form.values()]
        final_features = np.array([input_features])
        prediction = model.predict(final_features)

        result = "Loan Approved" if prediction[0] == 1 else "Loan Rejected"

        prediction_history.append({
            "input": input_features,
            "result": result
        })

        return render_template("index.html", prediction_text=result, history=prediction_history)

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}", history=prediction_history)

if __name__ == "__main__":
    app.run(debug=True)
