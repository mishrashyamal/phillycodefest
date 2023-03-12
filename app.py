from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/Users/geetakukreja/Documents/Masters/Q2/smarthire/phillycodefest/resume'
app = Flask(__name__, template_folder='/Users/geetakukreja/Documents/Masters/Q2/smarthire/phillycodefest/templates')


# app.config['MAX_CONTENT_PATH'] =

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        f.save(secure_filename(f.filename))
        return 'file uploaded successfully'


if __name__ == '__main__':
    app.run(host='localhost', port=8082, debug=True)
