import os
import subprocess
import sys
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import random
import pandas as pd
import numpy as np

def download_data():
    if(not os.path.isfile("/content/gdrive/Shareddrives/projects_data/image_caption/flickr8k.zip")):
        os.system("kaggle datasets download -d adityajn105/flickr8k")
        os.system("unzip \*.zip  && rm *.zip")
    
    os.system("mkdir -p data")
    image_captions_df = pd.read_csv("C:/Users/test/Downloads/modularized/captions.txt",sep=",",header=0)
    return image_captions_df

def train_test_split(annotate_dict,split_fraction=.2,shuffle=False):
  image_path = list(annotate_dict.keys())
  if shuffle:
    random.shuffle(image_path)

  l = len(image_path)
  train_idx = int(l * (1-split_fraction))
  
  train_img_path = image_path[:train_idx]
  test_img_path = image_path[train_idx:]

  train_annotate_dict = { k : annotate_dict[k] for k in train_img_path }
  test_annotate_dict = { k : annotate_dict[k] for k in test_img_path }
  
  return train_annotate_dict, test_annotate_dict

def captions_annote_dict(image_captions_df):
    image_captions_df['image_path'] = os.path.abspath('C:/Users/test/Downloads/modularized')+'/'+'Images'+'/'+image_captions_df['image']
    image_captions_df['annotation'] = "[start] "+image_captions_df['caption']+" [end]"
    image_annotate_agg_df = image_captions_df[['image_path','annotation']].groupby(['image_path'],as_index=False).agg({'annotation':list}).reset_index()
    image_annotate_dict={}
    for key,val in zip(image_annotate_agg_df['image_path'],image_annotate_agg_df['annotation']):
          image_annotate_dict[key] = val

    return image_annotate_dict

def explode_dict_to_list(annotate_dict):
  key_list =[]
  value_list =[]
  
  for k,v in annotate_dict.items():
    for val in v:
      key_list.append(k)
      value_list.append(val) 
      
  return key_list, value_list