from tensorflow import keras
from tensorflow.keras import layers
import random
from tqdm import tqdm
import tensorflow as tf
from tensorflow import io,image
import numpy as np
from  keras.applications import vgg16,inception_v3

CNN_model_name = "vgg16"
def save_features(feature, filename, path='', modelName="default"):
  path_of_feature = path + '/' + modelName + '/' + filename
  np.save(path_of_feature , feature)

def parse_image(image_path):
  if CNN_model_name=="vgg16":
    shape=(224,224)
  elif CNN_model_name=="resnet":
    shape= (224,224)
  elif CNN_model_name=="InceptionV3":
    shape=(299,299)

  img = io.read_file(image_path)
  img = image.decode_jpeg(img, channels=3)
  img = image.resize(img, shape)
  return img, image_path

def get_feature_extraction_model(input_shape):
  if CNN_model_name == 'vgg16':
    conv_base =keras.applications.vgg16.VGG16(weights = "imagenet" , include_top=False )
    feature_input = conv_base.input
    feature_output = conv_base.layers[-1].output
    image_feature_model = tf.keras.Model(feature_input, feature_output)
  elif CNN_model_name=="InceptionV3":
    conv_base = keras.applications.inception_v3.InceptionV3(include_top=False)
    feature_input = conv_base.input
    feature_output = conv_base.layers[-1].output
    image_feature_model = tf.keras.Model(feature_input, feature_output)
  elif CNN_model_name=="resnet":
    conv_base = keras.applications.resnet.ResNet50(weights = "imagenet", include_top=False,  input_shape=input_shape)
    feature_input = conv_base.input
    feature_output = conv_base.layers[-1].output
    image_feature_model = tf.keras.Model(feature_input, feature_output)
   
  return conv_base

def persist_features_from_batch(batch_features, batch_path, CNN_model_name):
  feature_dict = {}
  feature_persist_path= "C:/Users/test/Downloads/modularized/data"

  for feature , path in zip(batch_features, batch_path):
    feature_path = str(path.numpy().decode("utf-8"))
    feature_path = feature_path.split('/')[2]
    feature_dict[feature_path]= feature.numpy()
    save_features(feature.numpy(), feature_path, feature_persist_path, CNN_model_name)
  return feature_dict

def extract_image_features(base_model, image_file_list=None):
  CNN_model_name = base_model
  if CNN_model_name == 'vgg16':
    input_img_shape=(224,224,3)
  
    feature_extract_model = get_feature_extraction_model(input_img_shape)
    image_dataset = tf.data.Dataset.from_tensor_slices(image_file_list)
    image_dataset = image_dataset.map(parse_image,num_parallel_calls=4).batch(64)

    for image_parsed,image_paths in tqdm(image_dataset):

      features = feature_extract_model(image_parsed)  
      features = tf.reshape(features,(features.shape[0],-1,features.shape[3]))
      
      dict_path_to_feature = persist_features_from_batch(features,image_paths, CNN_model_name)

  elif CNN_model_name=="InceptionV3":
    input_img_shape=(299,299,3)
  
    feature_extract_model =get_feature_extraction_model(input_img_shape)
    image_dataset = tf.data.Dataset.from_tensor_slices(image_file_list)
    image_dataset = image_dataset.map(parse_image,num_parallel_calls=4).batch(64)

    for image_parsed,image_paths in tqdm(image_dataset):
      
      features = feature_extract_model(image_parsed)
      features = tf.reshape(features,(features.shape[0],-1,features.shape[3]))

      dict_path_to_feature = persist_features_from_batch(features,image_paths, CNN_model_name)

  elif CNN_model_name=="resnet":
    input_img_shape=(224,224,3)
  
    feature_extract_model = get_feature_extraction_model(input_img_shape)
    image_dataset = tf.data.Dataset.from_tensor_slices(image_file_list)
    image_dataset = image_dataset.map(parse_image,num_parallel_calls=4).batch(64)

    for image_parsed,image_paths in tqdm(image_dataset):
      features = feature_extract_model(image_parsed)
      features = tf.reshape(features,(features.shape[0],-1,features.shape[3]))

      dict_path_to_feature = persist_features_from_batch(features,image_paths, CNN_model_name)

    return dict_path_to_feature