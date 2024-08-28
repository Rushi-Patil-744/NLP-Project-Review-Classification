from flask import Flask, request, render_template, jsonify
import pickle

app = Flask(__name__)

# Load the trained model (Replace this with actual model loading if it's saved separately)
model = pickle.load(open('review.pkl', 'rb'))

# Endpoint to render the HTML page
@app.route('/')
def index():
    return render_template('review.html')

# Endpoint to handle form submission and return sentiment prediction
@app.route('/predict', methods=['POST'])
def predict():
    review_text = request.form.get('review')

    # Assuming model is trained and we use a feature extraction pipeline here
    features = np.array([len(review_text.split())]).reshape(1, -1)

    prediction = model.predict(features)

    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
