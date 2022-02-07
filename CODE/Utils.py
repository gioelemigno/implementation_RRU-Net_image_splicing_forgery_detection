import pickle
import os 
import tensorflow as tf
# ----------------------------------------------------------------------
def save_weights(model, folder):
  if not os.path.isdir(folder):
    os.makedirs(folder)

  for i, layer in enumerate(model.layers):
    filename = os.path.join(folder, "layer_" + str(i))
    layer.save_weights(filename)


def load_weights(model, folder):
  if not os.path.isdir(folder):
    print("<folder> doesn't exist!")
    return 

  for i, layer in enumerate(model.layers):
    filename = os.path.join(folder, "layer_" + str(i))
    layer.load_weights(filename)


# ----------------------------------------------------------------------
# https://stackoverflow.com/questions/49503748/save-and-load-model-optimizer-state
def save_optimizer(model, folder):
  if not os.path.isdir(folder):
    os.makedirs(folder)

  symbolic_weights = getattr(model.optimizer, 'weights')
  weight_values = tf.keras.backend.batch_get_value(symbolic_weights)

  filename = os.path.join(folder, 'optimizer.pkl')
  with open(filename, 'wb') as f:
    pickle.dump(weight_values, f)


def load_optimizer(model, folder):
  if not os.path.isdir(folder):
    print("<folder> doesn't exist!")
    return
  '''
  # Build train function (to get weight updates).
  if isinstance(model, tf.keras.Sequential):
      model._make_train_function()
  else:
      model._make_train_function() 
  '''
  filename = os.path.join(folder, 'optimizer.pkl')
  with open(filename, 'rb') as f:
    weight_values = pickle.load(f)
    model.optimizer.set_weights(weight_values)

# ----------------------------------------------------------------------
def save_weights_and_optimizer(model, folder):
  save_weights(model, folder)
  save_optimizer(model, folder)
  
def load_weights_and_optimizer(model, folder):
  load_weights(model, folder)
  load_optimizer(model, folder)