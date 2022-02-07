import tensorflow as tf
import tensorflow_io as tfio
from glob import glob
import os

class Dataset:
  def __init__(self, 
               training_image_folder, 
               validation_image_folder,
               batch_size,
               random_seed,
               img_height,
               img_width
               ):
    
    self.img_height = img_height
    self.img_width = img_width
    
    self.training_image_folder = training_image_folder
    self.validation_image_folder = validation_image_folder
    self.batch_size = batch_size
    self.random_seed = random_seed


    self.training_set_size = len(glob(self.training_image_folder + "/*"))
    self.validation_set_size = len(glob(self.validation_image_folder + "/*"))


    train_dataset = tf.data.Dataset.list_files(self.training_image_folder + "/*", seed=self.random_seed)
    train_dataset = train_dataset.map(self.parse_image)

    val_dataset = tf.data.Dataset.list_files(self.validation_image_folder + "/*", seed=self.random_seed)
    val_dataset =val_dataset.map(self.parse_image)

    self.dataset = {"train": train_dataset, "val": val_dataset}


    # For more information about autotune:
    # https://www.tensorflow.org/guide/data_performance#prefetching
    self.AUTOTUNE = tf.data.experimental.AUTOTUNE


    # for reference about the BUFFER_SIZE in shuffle:
    # https://stackoverflow.com/questions/46444018/meaning-of-buffer-size-in-dataset-map-dataset-prefetch-and-dataset-shuffle
    self.buffer_size = 100#1000

    # -- Train Dataset --#
    self.dataset['train'] = self.dataset['train'].map(self.load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    self.dataset['train'] = self.dataset['train'].shuffle(buffer_size=self.buffer_size, seed=self.random_seed)
    self.dataset['train'] = self.dataset['train'].repeat()
    self.dataset['train'] = self.dataset['train'].batch(self.batch_size)
    self.dataset['train'] = self.dataset['train'].prefetch(buffer_size=self.AUTOTUNE)

    #-- Validation Dataset --#
    self.dataset['val'] = self.dataset['val'].map(self.load_image_test)
    #self.dataset['val'] = self.dataset['val'].repeat()
    self.dataset['val'] = self.dataset['val'].batch(self.batch_size)
    self.dataset['val'] = self.dataset['val'].prefetch(buffer_size=self.AUTOTUNE)






  # SOURCE: https://yann-leguilly.gitlab.io/post/2019-12-14-tensorflow-tfdata-segmentation/
  def parse_image(self, img_path: str) -> dict:
      '''
        Load image and mask starting from path of image
      '''
      image = tf.io.read_file(img_path)
    

      print(tf.get_static_value(img_path))
      extension = '???'
      
      def jpg_loader():
        return {'tensor':tf.image.decode_jpeg(image, channels=3), 'extension':"jpg"}
      def tif_loader():
        return {'tensor': tfio.experimental.image.decode_tiff(image)[:,:,:3], 'extension': "tif"}# (RGBA -> RGB)
      def default_loader():
        return {'tensor': tf.constant(0, shape=(0, 0, 0), dtype=tf.uint8), 'extension':"unknown"}

      result = tf.case([
                (tf.strings.regex_full_match(img_path, ".*[.]jpg"), jpg_loader), 
                (tf.strings.regex_full_match(img_path, ".*[.]tif"), tif_loader)
            ], default=default_loader, exclusive=True)


      image = result['tensor']
      extension = result['extension']

      print(extension)
      image = tf.image.convert_image_dtype(image, tf.uint8)
    
      # obtain mask path from image path
      mask_path = tf.strings.regex_replace(img_path, "images", "masks")
      mask_path = tf.strings.regex_replace(mask_path, extension, "png")

      
      mask = tf.io.read_file(mask_path)
      mask = tf.image.decode_png(mask, channels=1)

      return {'image': image, 'segmentation_mask': mask}


  @tf.function
  def normalize(self, input_image: tf.Tensor, input_mask: tf.Tensor) -> tuple:
      '''
        Normalize image and its mask
      '''
      input_image = tf.cast(input_image, tf.float32) / 255.0
      input_mask = tf.cast( tf.cast(input_mask, tf.float32) / 255.0, tf.uint8)
      return input_image, input_mask


  @tf.function
  def load_image_train(self, datapoint: dict) -> tuple:
      # Images of the dataset are already of right size

      #Altough images's size is already fixed, these lines of codes are necessary
      #instead, removing them, tf cannot see the size (i.e. model get in input (none, none, none, none))
      input_image = tf.image.resize(datapoint['image'], (self.img_height, self.img_width))
      input_mask = tf.image.resize(datapoint['segmentation_mask'], (self.img_height, self.img_width))
      
      input_image, input_mask = self.normalize(input_image, input_mask)
    
      return input_image, input_mask

  @tf.function
  def load_image_test(self, datapoint: dict) -> tuple:
      # in this case, since load_image_train() does not perform any image augmentation
      # load_image_test() is equal to load_image_test()
      input_image = tf.image.resize(datapoint['image'], (self.img_height, self.img_width))
      input_mask = tf.image.resize(datapoint['segmentation_mask'], (self.img_height, self.img_width))

      input_image, input_mask = self.normalize(input_image, input_mask)

      return input_image, input_mask