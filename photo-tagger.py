import os
import sys; sys.path.append('image-similarity-clustering')

from flask import (Flask, flash, redirect, render_template, request,
                   send_from_directory, url_for, render_template_string)
from werkzeug.utils import secure_filename

from features import extract_features
from predictor import predict_location
import numpy as np 

UPLOAD_FOLDER = 'temp'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    # check if the file extension is allowed
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Open the route for the upload_file function
@app.route('/', methods=['GET', 'POST'])
def upload_file():

    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No files part')
            return redirect(request.url)
        files = request.files.getlist('file')
        # if user does not select file, browser also
        # submit an empty part without filename
        safe_files = []
        for file in files:
            if file.filename == '':
                flash('No selected file')
                return redirect(request.url)
            if file and allowed_file(file.filename):
                # check if the filename is malicious
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                safe_files.append(filename)
        return results(safe_files)
            #redirect(url_for('results',filename='/Uploads/'+filename))
    return render_template('upload.html')


# Open the route for the results function
@app.route('/results', methods=['GET', 'POST'])
def results(filenames):
    #full_filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    print(filenames)
    features_df = extract_features('temp', filenames=filenames)
    prediction_conf = predict_location(features_df, model='Model_04.h5')
    cities = ['Coimbra', 'Lisboa', 'Porto']
    position_max = [np.argmax(p) for p in prediction_conf]

    photo_urls = [url_for('uploaded_file',filename=filename) for filename in filenames]

    html_code = ""
    for i, photo in enumerate(photo_urls):

        html_code += (f'''<div class="image_box">
        <img src={photo} width='224' height='224' class="image_thumbnail" />
        <div class="label">
            <table>
                <tr>
                    <td class="city">{cities[position_max[i]]}</td>
                    <td class="pred">{prediction_conf[i][position_max[i]] * 100:.2f} %</td>
                </tr>
            </table>
            </div>
        </div>''')

    return render_template('upload.html', html_code=html_code)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],filename)

# running the flask app defined above
if __name__ == '__main__':
    app.run(debug=True)

