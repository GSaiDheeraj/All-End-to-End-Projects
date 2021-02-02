from flask import Flask,render_template, url_for ,flash , redirect
import joblib
from flask import request
import numpy as np

import os
from flask import send_from_directory
from keras.preprocessing import image

app=Flask(__name__,template_folder='template')

app.config['SECRET_KEY'] = '5791628bb0b13ce0c676dfde280ba245'

dir_path = os.path.dirname(os.path.realpath(__file__))

UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'

from keras.models import load_model
model = load_model('model5.h5')
model222=load_model("model6.h5")

#FOR THE FIRST MODEL
def api(full_path):
    data = image.load_img(full_path, target_size=(50, 50, 3))
    data = np.expand_dims(data, axis=0)
    data = data * 1.0 / 255
    predicted = model.predict(data)
    return predicted

#FOR THE SECOND MODEL
def api1(full_path):
    data = image.load_img(full_path, target_size=(64, 64, 3))
    data = np.expand_dims(data, axis=0)
    data = data * 1.0 / 255
    predicted = model222.predict(data)
    return predicted

@app.route('/upload', methods=['POST','GET'])
def upload_file():
    if request.method == 'GET':
        return render_template('malaria.html')
    else:
        try:
            file = request.files['image']
            full_name = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(full_name)

            indices = {0: 'PARASITIC', 1: 'Uninfected', 2: 'Invasive carcinomar', 3: 'Normal'}
            result = api(full_name)
            print(result)

            predicted_class = np.asscalar(np.argmax(result, axis=1))
            accuracy = round(result[0][predicted_class] * 100, 2)
            label = indices[predicted_class]
            return render_template('predict.html', image_file_name = file.filename, label = label, accuracy = accuracy)
        except:
            flash("Please select the image first !!", "danger")      
            return redirect(url_for("Malaria"))

@app.route('/upload11', methods=['POST','GET'])
def upload11_file():

    if request.method == 'GET':
        return render_template('pneumonia.html')
    else:
        try:
            file = request.files['image']
            full_name = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(full_name)
            indices = {0: 'Normal', 1: 'Pneumonia'}
            result = api1(full_name)
            if(result>50):
                label= indices[1]
                accuracy= result
            else:
                label= indices[0]
                accuracy= 100-result
            return render_template('predict1.html', image_file_name = file.filename, label = label, accuracy = accuracy)
        except:
            flash("Please select the image first !!", "danger")      
            return redirect(url_for("Pneumonia"))

@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route("/")

@app.route("/home")
def home():
    return render_template("home.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/cancer")
def cancer():
    return render_template("cancer.html")

@app.route("/diabetes")
def diabetes():
    #if form.validate_on_submit():
    return render_template("diabetes.html")

@app.route("/heart")
def heart():
    return render_template("heart.html")

@app.route("/liver")
def liver():
    #if form.validate_on_submit():
    return render_template("liver.html")

@app.route("/kidney")
def kidney():
    #if form.validate_on_submit():
    return render_template("kidney.html")

@app.route("/Malaria")
def Malaria():
    return render_template("malaria.html")

@app.route("/Pneumonia")
def Pneumonia():
    return render_template("pneumonia.html")

def ValuePredictor(to_predict_list, size):
    to_predict = np.array(to_predict_list).reshape(1,size)
    if(size==8):#Diabetes
        loaded_model = joblib.load("model1")
        result = loaded_model.predict(to_predict)
    elif(size==30):#Cancer
        loaded_model = joblib.load("model")
        result = loaded_model.predict(to_predict)
    elif(size==12):#Kidney
        loaded_model = joblib.load("model3")
        result = loaded_model.predict(to_predict)
    elif(size==10):
        loaded_model = joblib.load("model4")
        result = loaded_model.predict(to_predict)
    elif(size==11):#Heart
        loaded_model = joblib.load("model2")
        result =loaded_model.predict(to_predict)
    return result[0]

@app.route('/result',methods = ["POST"])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list=list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
        if(len(to_predict_list)==30):#Cancer
            result = ValuePredictor(to_predict_list,30)
        elif(len(to_predict_list)==8):#Daiabtes
            result = ValuePredictor(to_predict_list,8)
        elif(len(to_predict_list)==12):
            result = ValuePredictor(to_predict_list,12)
        elif(len(to_predict_list)==11):
            result = ValuePredictor(to_predict_list,11)
            #if int(result)==1:
        elif(len(to_predict_list)==10):
            result = ValuePredictor(to_predict_list,10)
    if(int(result)==1):
        prediction='You are Suffering'
    else:
        prediction='Congrats! you are Healthy'
    return(render_template("result.html", prediction=prediction))

if __name__ == "__main__":
    app.run(debug=True)
