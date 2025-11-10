from flask import Flask, request, render_template
from img_convert import convert_image_to_data
import joblib
import sqlite3

model_path = r'C:\python_test\ML_project\svm_model4.joblib'
svm = joblib.load(model_path)

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return render_template('home1.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No image part in the request", 400
    image = request.files['image']
    if image.filename == '':
        return "No image selected for uploading", 400

    if image:
        image_data = convert_image_to_data(image)

        if image_data is not None:
            image_data = image_data.reshape(1, -1)
            prediction = svm.predict(image_data)

            conn = sqlite3.connect('SVM.db')
            cursor = conn.cursor()
            select_query = 'SELECT name FROM users where id =?'
            cursor.execute(select_query,(prediction[0],))
            rows = cursor.fetchone()
            conn.close()
            
            prediction = rows
            return render_template('result1.html', prediction=prediction[0])

    return render_template('home1.html')

if __name__ == '__main__':
    app.run(debug=True)
