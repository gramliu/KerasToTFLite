import os

from keras import backend as K
from keras.models import load_model

import tensorflow as tf

K.set_learning_phase(False)


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.graph_util.convert_variables_to_constants(session, input_graph_def, output_names,                                                               freeze_var_names)
        return frozen_graph


def create_lite_model_from_saved_model(saved_model_dir, tf_lite_path):
    converter = tf.contrib.lite.TocoConverter.from_saved_model(saved_model_dir)
    tf_lite_model = converter.convert()
    open(tf_lite_path, "wb").write(tf_lite_model)


def save_model(keras_model, session, pb_model_path):
    x = keras_model.input
    y = keras_model.output
    prediction_signature = tf.saved_model.signature_def_utils.predict_signature_def({"inputs": x}, {"prediction": y})
    builder = tf.saved_model.builder.SavedModelBuilder(pb_model_path)
    legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
    signature = {tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: prediction_signature, }
    builder.add_meta_graph_and_variables(session, [tf.saved_model.tag_constants.SERVING], signature_def_map=signature,                                         legacy_init_op=legacy_init_op)
    builder.save()


def run():
    sess = K.get_session()
    keras_model_name = 'cnn_model_keras2.h5'
    lite_model_name = 'lite_model_file.tflite'
    keras_model_file_path = keras_model_name
    lite_model_file_path = lite_model_name
    pb_model_path = "saveBuilder"
    model = load_model(keras_model_file_path)
    output_names = [node.op.name for node in model.outputs]
    _ = freeze_session(sess, output_names=output_names)
    save_model(keras_model=model, session=sess, pb_model_path=pb_model_path)
    create_lite_model_from_saved_model(saved_model_dir=pb_model_path, tf_lite_path=lite_model_file_path)


if __name__ == "__main__":
    run()