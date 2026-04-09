from flask import Flask, render_template, request, jsonify
import model

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        question = request.form['question']
        if not question.strip():
            return render_template('index.html', prediction="Please enter a question.", question=question)
        
        prediction = model.predict_question(question)
        return render_template('index.html', prediction=f"Category: {prediction}", question=question)
    
    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}", question=question)

if __name__ == '__main__':
    app.run(debug=True)

