import os
from flask import Flask, request, redirect, render_template, flash
from werkzeug.utils import secure_filename
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing import image
from sklearn.preprocessing import normalize

import numpy as np

import librosa as lb


labels = ['jazz', 'classical', 'metal', 'disco', 'reggae', 'country', 'rock', 'hiphop', 'blues', 'pop']

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = set(['wav'])

app = Flask(__name__)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#音声データからNumPy配列とサンプリングレートを返す関数
def read_sample(file):
    y, sr = lb.load(file)
    return y, sr

#メルスペクトログラム化したNumPy配列をモデルのinput_shapeに対応したNumPy配列として返す関数
def pad_mel(mel, input_shape):

    input_length = input_shape[2]
    if mel.shape[1] == input_length:
        return mel

    elif mel.shape[1] > input_length:
        mel = mel[:,:input_length]
    else:
        mel = np.pad(mel, (0, max(0, mel- len(mel))))
    return mel

#学習済みモデルをロード
model = load_model('./model.h5', compile=False)


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('ファイルがありません')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('ファイルがありません')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(UPLOAD_FOLDER, filename))
            filepath = os.path.join(UPLOAD_FOLDER, filename)

            #音声データをメルスペクトログラムに変換
            sample, sr = read_sample(filepath)
            mel = lb.feature.melspectrogram(y=sample, sr=sr)
            spect = lb.power_to_db(mel, ref=1.0)
            norm_spect = normalize(spect)

            #学習済みモデルを元にジャンルを予測
            input_shape = model.layers[0].output_shape[0]

            x = pad_mel(norm_spect, input_shape)
            x = np.reshape(x, (1, input_shape[1], input_shape[2], input_shape[3]))

            pred_index = np.argmax(model.predict(x))
            pred_answer = "予想されたジャンル" +  labels[pred_index]

            return render_template("index.html",answer=pred_answer)

    return render_template("index.html",answer="")


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    app.run(host ='0.0.0.0',port = port)