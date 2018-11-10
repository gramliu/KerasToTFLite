import tensorflow as tf

converter = tf.contrib.lite.TocoConverter.from_saved_model("saveBuilder")
converter.post_training_quantize = True
quantized = converter.convert()
open("quantized_model.tflite", "wb").write(quantized)