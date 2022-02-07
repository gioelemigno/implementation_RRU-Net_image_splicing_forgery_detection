import tensorflow as tf
import shutil
import Losses
import os
import Utils

def evaluation(model, tf_dataset, dataset_size, batch_size) -> dict:
  tf_recall = tf.keras.metrics.Recall()
  tf_precision = tf.keras.metrics.Precision()
  
  dice = 0
  iterations = 0
  for image, mask in tf_dataset:
    #print(iterations)
    y_pred = model.predict(image)
    y_pred = tf.reshape(y_pred, [-1])

    y_true = tf.reshape(mask, [-1])

    tf_recall.update_state(y_true, y_pred)
    tf_precision.update_state(y_true, y_pred)

    dice += Losses.dice_loss(y_true, y_pred)
    iterations += 1

  dice = dice / iterations
  
# for step in range(dataset_size // batch_size):
  #   for image, mask in tf_dataset.take(1):
  #     y_pred = model.predict(image)
  #     y_pred = tf.reshape(y_pred, [-1])

  #     y_true = tf.reshape(mask, [-1])

  #     tf_recall.update_state(y_true, y_pred)
  #     tf_precision.update_state(y_true, y_pred)

  #     dice += Losses.dice_loss(y_true, y_pred)
  #dice = dice / (step+1)



  recall = tf_recall.result().numpy()
  precision = tf_precision.result().numpy()

  den = (precision+recall)
  if den != 0:
    f1_score = 2*(precision*recall)/den
  else:
    f1_score = 0.0
    
  return {'recall':recall, 'precision': precision, 'f1_score': f1_score, 'dice_loss':dice.numpy()}




class evaluation_save_best_weigths(tf.keras.callbacks.Callback):
    #https://developpaper.com/tf-keras-implements-f1-score-precision-recall-and-other-metrics/
    def __init__(self, _valid_data, _val_size, _batch_size, _SETTING, _BEST_WEIGHTS, _CHECKPOINT):
        super(evaluation_save_best_weigths, self).__init__()
        self.validation_data = _valid_data
        self.val_size = _val_size
        self.batch_size = _batch_size

        self.last_best_weights_folder = None 
        self.last_best_f1_score = -1

        self.BEST_WEIGHTS = _BEST_WEIGHTS
        self.CHECKPOINT = _CHECKPOINT
        self.SETTING = _SETTING

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        res = evaluation(self.model, self.validation_data, self.val_size, self.batch_size)

        logs['val_f1'] = res['f1_score']
        logs['val_recall'] = res['recall']
        logs['val_precision'] = res['precision']
        logs['val_dice_loss'] = res['dice_loss']

        tf.summary.scalar('val_f1', data=res['f1_score'], step=epoch)
        tf.summary.scalar('val_recall', data=res['recall'], step=epoch)
        tf.summary.scalar('val_precision', data=res['precision'], step=epoch)
        tf.summary.scalar('val_dice_loss', data=res['dice_loss'], step=epoch)


        if res['f1_score'] > self.last_best_f1_score:
          self.last_best_f1_score = res['f1_score']

          # delete old weights
          if self.last_best_weights_folder != None:
            shutil.rmtree(self.last_best_weights_folder)

          folder_name = "f1_score=" + str(res['f1_score']) + "__epoch=" + str(epoch)
          folder = os.path.join(self.BEST_WEIGHTS, folder_name)
          Utils.save_weights(self.model, folder)

          self.last_best_weights_folder = folder

        if epoch == self.SETTING['EPOCHS']-1 or not epoch % 5:
          folder = os.path.join(self.CHECKPOINT, 'epoch=' + str(epoch).zfill(4))
          Utils.save_weights_and_optimizer(self.model, folder)
          
        print(" — val_f1: %f — val_precision: %f — val_recall: %f - val_dice_loss: %f" % (res['f1_score'], res['precision'], res['recall'], res['dice_loss']))
        return
