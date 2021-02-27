# Important Modules
from flask import Flask, render_template, url_for, flash, redirect
# from forms import RegistrationForm, LoginForm
import joblib
from flask import request
import numpy as np
import tensorflow


import os
from flask import send_from_directory

import tensorflow as tf

app = Flask(__name__, template_folder='template')

# RELATED TO THE SQL DATABASE
app.config['SECRET_KEY'] = '5791628bb0b13ce0c676dfde280ba245'

import keras


dir_path = os.path.dirname(os.path.realpath(__file__))

UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'


from keras.models import load_model

global graph
graph = tf.get_default_graph()
model = load_model('model5.h5')
model222 = load_model("model6.h5")

# model = load_model('model5.h5')
# model222 = load_model("model6.h5")


# FOR THE FIRST MODEL

# call model to predict an image
def api(full_path):
    with graph.as_default():
        data = keras.preprocessing.image.load_img(full_path, target_size=(50, 50, 3))
        data = np.expand_dims(data, axis=0)
        data = data * 1.0 / 255

        # with graph.as_default():
        predicted = model.predict(data)
        return predicted

# FOR THE SECOND MODEL
def api1(full_path):
    with graph.as_default():
        data = keras.preprocessing.image.load_img(full_path, target_size=(64, 64, 3))
        data = np.expand_dims(data, axis=0)
        data = data / 255

    # with graph.as_default():
        predicted = model222.predict(data)
        return predicted





# procesing uploaded file and predict it
@app.route('/upload', methods=['POST', 'GET'])
def upload_file():
    with graph.as_default():

        if request.method == 'GET':
            return render_template('index.html')
        else:
            #try:
            file = request.files['image']
            full_name = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(full_name)

            indices = {0: 'PARASITIC', 1: 'Uninfected', 2: 'Invasive carcinomar', 3: 'Normal'}
            result = api(full_name)
            print(result)

            predicted_class = np.asscalar(np.argmax(result, axis=1))
            accuracy = round(result[0][predicted_class] * 100, 2)
            label = indices[predicted_class]
            return render_template('predict.html', image_file_name=file.filename, label=label, accuracy=accuracy)
        # except:
        #     flash("Please select the image first !!", "danger")
        #     return redirect(url_for("Malaria"))


@app.route('/upload11', methods=['POST', 'GET'])
def upload11_file():
    with graph.as_default():

        if request.method == 'GET':
            return render_template('index2.html')
        else:
            #try:
            file = request.files['image']
            full_name = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(full_name)
            indices = {0: 'Normal', 1: 'Pneumonia'}
            result = api1(full_name)
            if (result > 50):
                label = indices[1]
                accuracy = result
            else:
                label = indices[0]
                accuracy = 100 - result
            return render_template('predict1.html', image_file_name=file.filename, label=label, accuracy=accuracy)
        # except:
        #     flash("Please select the image first !!", "danger")
        #     return redirect(url_for("Pneumonia"))



    # with graph.as_default():
    #    ...
    #    # here you can use your model as you want
    #    preds = model.predict(...)
    #    # If you want you can convert your predit results to json
    #    data = toDict(preds)
    # # return a response in json format
    # return flask.jsonify(data)





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
    # if form.validate_on_submit():
    return render_template("diabetes.html")


@app.route("/heart")
def heart():
    return render_template("heart.html")


@app.route("/liver")
def liver():
    # if form.validate_on_submit():
    return render_template("liver.html")


@app.route("/kidney")
def kidney():
    # if form.validate_on_submit():
    return render_template("kidney.html")


@app.route("/Malaria")
def Malaria():
    return render_template("index.html")


@app.route("/Pneumonia")
def Pneumonia():
    return render_template("index2.html")


"""
@app.route("/register", methods=["GET", "POST"])
def register():
    form =RegistrationForm()
    if form.validate_on_submit():
        #flash("Account created for {form.username.data}!".format("success"))
        flash("Account created","success")      
        return redirect(url_for("home"))
    return render_template("register.html", title ="Register",form=form )
@app.route("/login", methods=["POST","GET"])
def login():
    form =LoginForm()
    if form.validate_on_submit():
        #if form.email.data =="sho" and form.password.data=="password":
        flash("You Have Logged in !","success")
        return redirect(url_for("home"))
    #else:
    #   flash("Login Unsuccessful. Please check username and password","danger")
    return render_template("login.html", title ="Login",form=form )
def ValuePredictor1(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1,30)
    loaded_model = joblib.load("model")
    result = loaded_model.predict(to_predict)
    return result[0]

@app.route('/result1',methods = ["GET","POST"])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list=list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
        result = ValuePredictor(to_predict_list)
        if int(result)==1:
            prediction='cancer'
        else:
            prediction='Healthy'       
    return(render_template("result.html", prediction=prediction))"""


def ValuePredictor(to_predict_list, size):
    to_predict = np.array(to_predict_list).reshape(1, size)
    if (size == 8):  # Diabetes
        loaded_model = joblib.load("model1")
        result = loaded_model.predict(to_predict)
    elif (size == 30):  # Cancer
        loaded_model = joblib.load("model")
        result = loaded_model.predict(to_predict)
    elif (size == 12):  # Kidney
        loaded_model = joblib.load("model3")
        result = loaded_model.predict(to_predict)
    elif (size == 10):
        loaded_model = joblib.load("model4")
        result = loaded_model.predict(to_predict)
    elif (size == 11):  # Heart
        loaded_model = joblib.load("model2")
        result = loaded_model.predict(to_predict)
    return result[0]


@app.route('/result', methods=["POST"])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
        if (len(to_predict_list) == 30):  # Cancer
            result = ValuePredictor(to_predict_list, 30)
        elif (len(to_predict_list) == 8):  # Daiabtes
            result = ValuePredictor(to_predict_list, 8)
        elif (len(to_predict_list) == 12):
            result = ValuePredictor(to_predict_list, 12)
        elif (len(to_predict_list) == 11):
            result = ValuePredictor(to_predict_list, 11)
            # if int(result)==1:
            #   prediction ='diabetes'
            # else:
            #   prediction='Healthy' 
        elif (len(to_predict_list) == 10):
            result = ValuePredictor(to_predict_list, 10)
    if (int(result) == 1):
        prediction = 'Sorry ! Suffering'
    else:
        prediction = 'Congrats ! you are Healthy'
    return (render_template("result.html", prediction=prediction))


if __name__ == "__main__":
    app.run(debug=True)