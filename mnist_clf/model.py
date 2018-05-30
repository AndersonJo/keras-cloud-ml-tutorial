from typing import Tuple, List

import keras.backend as K
import tensorflow as tf
from keras import Sequential, Model, Input
from keras.backend.tensorflow_backend import set_session
from keras.layers import Dense, Activation, Dropout
from tensorflow.python.saved_model import builder as tf_model_builder, tag_constants, signature_constants
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def
from tensorflow.python.training.momentum import MomentumOptimizer


# Model
def create_model() -> Tuple[Model, tf.Tensor]:
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.1
    set_session(tf.Session(config=config))

    image_input = Input(shape=(784,))

    h = Dense(512, activation='relu', input_shape=(784,))(image_input)
    h = Dropout(0.2, seed=0)(h)
    h = Dense(256, activation='relu')(h)
    h = Dropout(0.2, seed=0)(h)
    h = Dense(10)(h)
    output = Activation('softmax')(h)
    arg_max = K.argmax(output)

    model = Model(image_input, [output])
    model.compile(loss='categorical_crossentropy',
                  optimizer=MomentumOptimizer(0.01, momentum=0.9),
                  metrics=['accuracy'])

    return model, arg_max


# Save Model
def save_as_tensorflow(model: Model, export_path: str, arg_max):
    """
    Convert the Keras HDF5 model into TensorFlow SavedModel
    export_path: either local path or Google Cloud Storage's bucket path
                 (ex. "checkpoints", "gs://anderson-mnist/mnist_train_20180530_145010")
    """

    builder = tf_model_builder.SavedModelBuilder(export_path)
    signature = predict_signature_def(inputs={'image': model.inputs[0]},
                                      outputs={'probabilities': model.outputs[0],
                                               'class': arg_max})
    sess = K.get_session()
    builder.add_meta_graph_and_variables(
        sess=sess,
        tags=[tag_constants.SERVING],
        signature_def_map={signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature}
    )
    builder.save()
