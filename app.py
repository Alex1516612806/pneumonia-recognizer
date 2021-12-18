# https://www.geeksforgeeks.org/deploying-a-tensorflow-2-1-cnn-model-on-the-web-with-flask/

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from tensorflow import keras
import tensorflow as tf
# from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import numpy as np
import os


try:
    import shutil
    print()
except:
    pass

model = tf.keras.models.load_model('pneumonia_CNN.h5')
app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploaded/image'


@app.route('/')
def upload_f():
    return render_template('upload.html')


def finds(filename):
    vals = ['Normal', 'Pneumonia']
    img_path = 'uploaded/image/'+filename
    img = image.load_img(img_path, target_size=(180, 180))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 180.

    pred = model.predict_generator(img_tensor)
    classes_x = np.argmax(pred, axis=1)
    # print([np.argmax(pred)])
    # return str(vals[np.argmax(pred)])
    return vals[int(classes_x)]


@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        f.save(os.path.join(
            app.config['UPLOAD_FOLDER'], secure_filename(f.filename)))

        val = finds(f.filename)
        return render_template('pred.html', ss=val)


if __name__ == '__main__':
    app.run()
