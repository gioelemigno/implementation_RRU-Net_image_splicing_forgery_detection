import tensorflow as tf

def dice_loss(y_true, y_pred):
  y_true = tf.cast(y_true, dtype=tf.float32)
  y_pred = tf.cast(y_pred, dtype=tf.float32)
  eps = 0.0001


  inter = tf.math.reduce_sum(tf.multiply(y_true, y_pred)) + eps
  union = tf.math.reduce_sum(y_true) + tf.math.reduce_sum(y_pred) + eps

  return (2*inter)/union

def dice_loss_flatten(y_true, y_pred):
  y_true = tf.cast(y_true, dtype=tf.float32)
  y_pred = tf.cast(y_pred, dtype=tf.float32)

  true_mask_flatten = tf.reshape(y_true, [-1])
  pred_mask_flatten = tf.reshape(y_pred, [-1])

  eps = 0.0001
  inter = tf.math.reduce_sum(tf.multiply(true_mask_flatten, pred_mask_flatten)) + eps
  union = tf.math.reduce_sum(true_mask_flatten) + tf.math.reduce_sum(pred_mask_flatten) + eps

  return (2*inter)/union


def binary_cross_entropy_flatten(y_true, y_pred):
  true_mask_flatten = tf.reshape(y_true, [-1])
  pred_mask_flatten = tf.reshape(y_pred, [-1])

  return tf.keras.losses.binary_crossentropy(true_mask_flatten, pred_mask_flatten)
