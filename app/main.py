from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/linear')
def linear():
    return render_template('linear.html')

@app.route('/binary')
def binary():
    return render_template('binary.html')

@app.route('/objectdetect')
def object_detection():
    return render_template('object.html')