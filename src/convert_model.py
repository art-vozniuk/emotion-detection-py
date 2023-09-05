import os
from keras.models import load_model
import tensorflow as tf

src_dir = os.getcwd() + '/src/'
src_model_path = src_dir + 'model.h5'

keras_model = load_model(src_model_path)
keras_model.save('model_tf', save_format='tf')

loaded_model = tf.keras.models.load_model('model_tf')
loaded_model.summary()