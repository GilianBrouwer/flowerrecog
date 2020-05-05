# -*- coding: utf-8 -*-
"""
Created on Tue May  5 15:15:16 2020

@author: Gebruiker
"""

import os
from flask import Flask, render_template, request, redirect, flash, after_this_request
from werkzeug.utils import secure_filename
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import random

#load model
flowermodel=load_model('flower.h5')

#set image upload requirements
UPLOAD_FOLDER= 'image'
ALLOWED_EXTENSION= set(['png', 'jpg', 'jpeg'])

#flask app

app=Flask(__name__)
app.config['UPLOAD_FOLDER']=UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSION

@app.route('/', methods=['GET'])
def my_form():
    return render_template('index.html')


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            path=os.path.join(app.config['UPLOAD_FOLDER'], filename)

    def flowerpredict(flowerimage):
        test_image = image.load_img(flowerimage, target_size = (128, 128))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = flowermodel.predict(test_image)
        results=np.where(result == 1)
        results=int(results[1])
        
        #checks int results and adds correct class
        if results == 0:
            flowerclass='daisy'
        elif results == 1:
            flowerclass='dandelion'
        elif results == 2:
            flowerclass='rose'
        elif results == 3:
            flowerclass='sunflower'
        elif results == 4:
            flowerclass = 'tulip'
        else: flowerclass= 'ERROR'    
        
        pretty=['pretty', 'beautiful', 'charming', 'lovely', 'enchanting', 'alluring', 
                'sweet', 'splendid', 'brilliant', 'gorgeous']
        pretty2=random.choice(pretty)
        if pretty2.startswith('e') or pretty2.startswith('a'):
            prop='an'
        else: prop='a'
        prettyflower='It\'s '+prop+' '+pretty2+' '+flowerclass+'!'
        
        return results, prettyflower
    results, prettyflower=flowerpredict(flowerimage=path)
    
    @after_this_request 
    def remove_file(response): 
      os.remove(path) 
      return response 
    # stack up images list to pass for prediction
    return render_template('index.html', prettyflower=prettyflower, results=results)

if __name__ == '__main__':
    app.run()

