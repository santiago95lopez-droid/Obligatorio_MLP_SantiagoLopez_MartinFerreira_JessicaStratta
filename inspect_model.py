import tensorflow as tf
from tensorflow.keras.applications import mobilenet_v2

model_path = 'modelohongos/modelo_hongos_mobilenet.keras'
model = tf.keras.models.load_model(model_path, compile=False, custom_objects={'preprocess_input': mobilenet_v2.preprocess_input})
print('type', type(model))
print('input attr', getattr(model, 'input', None))
print('inputs attr', getattr(model, 'inputs', None))
print('layers count', len(model.layers))
for i, layer in enumerate(model.layers):
    if i < 40:
        print(i, layer.name, type(layer).__name__, getattr(layer, 'output_shape', None))
