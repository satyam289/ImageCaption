import numpy as np
import tensorflow as tf
import pickle

def cleanup_text(caption):

  excl_list = ['[start]' , '[end]']
  for sub in excl_list:

    caption = caption.replace( sub , ' ')

  return caption
  
def standardize(inputtext):
  lower = tf.strings.lower(inputtext)
  final_text = tf.strings.regex_replace(lower,
                                  r"""[.,.,#,$,!,\\,",\*,(,),\/,:,;,=,+,?,^,<,>,_,`,\-,%,&,@,|,{,},~]""", "") # Excluding '[', ']' as they are part of masks
  
  final_text = tf.strings.regex_replace(final_text,r"\s[b-z]{1}\s", "") #  removing one letter words 

  final_text = tf.strings.regex_replace(final_text,r"[a-z]*[0-9]+[a-z]*", "")

  return final_text
  

def preprocessCaptions(): 
  max_length =17 
  vocabulary_size = 5000
  tokenizer = tf.keras.layers.TextVectorization(
      max_tokens=vocabulary_size,
      standardize=standardize,
      output_sequence_length=max_length)
  return tokenizer