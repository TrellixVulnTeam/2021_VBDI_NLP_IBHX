import os
from datetime import datetime
from flask import Flask, request, redirect, url_for, send_from_directory, render_template
from werkzeug.utils import secure_filename
import numpy as np
import pandas as pd
from application_extractor import *
from evaluator_final import *
import json
from translate import Translator
translator = Translator(secret_access_key = 'b0caa63c2caa4e9a1291',
                        from_lang="japanese",
                        to_lang="english")
                        
secret_access_key = 'b0caa63c2caa4e9a1291'
ALLOWED_EXTENSIONS = set(['pdf'])
UPLOAD_FOLDER = 'uploads'

#%%
f0 = open(r'D:\VBDI_NLP\jbddata\jbddata_test_data.json', 'rb')
data = json.load(f0, encoding = 'utf-8')["data"]
# Re arrange the json data
data_rearrange = {}
for dat in range(len(data)):
    key = data[dat]['title'].split('.')[0]
    data_rearrange[key] = data[dat]
    
#%%
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

def build_input_file(file_name):
    file_json = {"version": "v2.0",
                 "data":[data_rearrange[file_name]]}
    print('Writing data to json file')
    with open('./uploads/{}.json'.format(file_name), 'w', encoding = 'utf-8') as fp:
        json.dump(file_json, fp, indent=2, ensure_ascii=False)
    print('Finish dump json file')
    return

def read_predicted_file():
    jp = open('{}\predictions.json'.format(output_dir), 'rb')
    data = json.load(jp, encoding = 'utf-8')
    return data
    
def predict(file_name): # name of article file
    build_input_file(file_name)
    extractor(r'D:\VBDI_NLP\bert\uploads\{}.json'.format(file_name))
    output = read_predicted_file()
    return output

def compare_result(file_name, output):
    gt = {}
    for i in range(len(data_rearrange[file_name]['paragraphs'][0]['qas'])):
        if len(data_rearrange[file_name]['paragraphs'][0]['qas'][i]['answers']) != 0:
            gt[data_rearrange[file_name]['paragraphs'][0]['qas'][i]['id']] = \
                {'question': data_rearrange[file_name]['paragraphs'][0]['qas'][i]['question'], 
                 'answer': data_rearrange[file_name]['paragraphs'][0]['qas'][i]['answers'][0]['text']}
        else:
            gt[data_rearrange[file_name]['paragraphs'][0]['qas'][i]['id']] = \
                {'question': data_rearrange[file_name]['paragraphs'][0]['qas'][i]['question'], 
                 'answer': 'no info'}
    headings = ('Extraction Tags', 'Extractor Result', 'Ground Truth', 'F1-score')
    tr = []
    print('Building table and translate to English')
    for q_id in output.keys():
        f1 = f1_score(output[q_id], gt[q_id]['answer'])
        tr.append((gt[q_id]['question'], 
                   output[q_id], 
                   gt[q_id]['answer'], 
                   f1))
                   
    tr = tuple(tr)
    return headings, tr

#%%
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route("/")
def template_test():
    return render_template('home.html', label='')


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        
        if file and allowed_file(file.filename):
            start_time = datetime.now()
            filename = secure_filename(file.filename)
            title_article = filename.split('.')[0]
            output = predict(title_article) # title_acrticle here is the name of title in json
            headings, data = compare_result(title_article, output)
            end_time = datetime.now()
            spend_time = end_time - start_time
    return render_template("home.html", headings=headings, data=data, spend_time = spend_time)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

if __name__ == "__main__":
    app.run(debug=False, threaded=False, port=600)

