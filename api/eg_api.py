# coding=utf-8
import os
import shutil
import sys
import time

import cv2
import numpy as np
import tensorflow as tf

sys.path.append(os.getcwd())
from nets import model_train as model
from utils.rpn_msr.proposal_layer import proposal_layer
from utils.text_connector.detectors import TextDetector

import pytesseract
import unidecode

from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from main import demo_al

def delete_prev(path):
    
    for the_file in os.listdir(path):
        file_path = os.path.join(path, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)
            continue

app = Flask(__name__)

app.root_path = os.path.join(os.getcwd(), 'api')
print(app.root_path)
app.template_folder = os.path.join(os.getcwd(), 'api/templates')
print(app.template_folder)
app.static_folder = os.path.join(os.getcwd(), 'api/static')

print(os.path.isdir(app.template_folder))
print(os.path.isdir(app.static_folder))    

print()
print(os.listdir(app.template_folder))
print(os.listdir(app.static_folder))

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'api/uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def hello_world():
    return render_template('home_al.html', filename='#')

@app.route('/upload', methods=  ['POST'])
def upload_file():

    if request.method == 'POST':
        file = request.files['image']
        filename = file.filename

        # prepare directory for processing
        delete_prev(app.config['UPLOAD_FOLDER'])
        f = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        # add your custom code to check that the uploaded file is a valid image and not a malicious file (out-of-scope for this post)
        file.save(f)

        demo_al.main()

        return render_template('home_al.html', filename=filename)
    else:

        print('No request')
        return render_template('home_al.html', filename='#')

@app.route('/<filename>')
def send_file(filename):

    print(filename)
    print(os.listdir(app._static_folder))
    filename = os.path.basename(filename)
    
    return send_from_directory(os.path.join(app._static_folder, 'res'), filename)

app.run(debug=True, threaded=True)