import tensorflow as tf
from keras import Sequential, Model
from keras.backend.tensorflow_backend import set_session
from keras.layers import Dense, Activation, Dropout
from keras.models import load_model
from tensorflow.python.saved_model import builder as tf_model_builder, tag_constants, signature_constants
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def
from tensorflow.python.training.momentum import MomentumOptimizer

import keras.backend as K


# Model
def create_model():
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.1
    set_session(tf.Session(config=config))

    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(784,)))
    model.add(Dropout(0.2, seed=0))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2, seed=0))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=MomentumOptimizer(0.01, momentum=0.9),
                  metrics=['accuracy'])
    return model


# Save Model
def save_model(model: Model, export_path: str):
    """
    Convert the Keras HDF5 model into TensorFlow SavedModel
    """
    model.save('/tmp/model.h5')

    # Reset Session
    K.clear_session()
    sess = tf.Session()
    K.set_session(sess)

    model = load_model('/tmp/model.h5')
    config = model.get_config()
    weights = model.get_weights()
    new_model = Sequential.from_config(config)
    new_model.set_weights(weights)

    builder = tf_model_builder.SavedModelBuilder(export_path)
    signature = predict_signature_def(inputs={'image': new_model.inputs[0]},
                                      outputs={'output': new_model.outputs[0]})
    with K.get_session() as sess:
        builder.add_meta_graph_and_variables(
            sess=sess,
            tags=[tag_constants.SERVING],
            signature_def_map={signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature}
        )
        builder.save()
