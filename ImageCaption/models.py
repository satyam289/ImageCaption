import os
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import random
import pandas as pd

class CNN_Encoder(tf.keras.Model):

    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        # shape after fc == (batch_size,  7 * 7 , embedding_dim)
        self.fc = tf.keras.layers.Dense(embedding_dim)

    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x

class RNN_Decoder_Without_Attention(tf.keras.Model):
  def __init__(self, embedding_dim, units, vocab_size):
    super(RNN_Decoder, self).__init__()
    self.units = units

    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.LSTM(self.units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    self.dense1 = tf.keras.layers.Dense(self.units)
    self.dense2 = tf.keras.layers.Dense(vocab_size)

    self.attention = visualAttention(self.units)

  def call(self, x, features, hidden):
    x = self.embedding(x)
    context_vector, attention_weights = self.attention(features, hidden)
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)    
    output, state = self.gru(x)    
    x = self.dense1(output)
    x = tf.reshape(x, (-1, x.shape[2]))    
    x = self.dense2(x)
    return x, state, attention_weights

  def reset_state(self, batch_size):
    return tf.zeros((batch_size, self.units))

class RNN_Decoder(tf.keras.Model):
  def __init__(self, embedding_dim, units, vocab_size):
    super(RNN_Decoder, self).__init__()
    self.units = units

    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    self.dense1 = tf.keras.layers.Dense(self.units)
    self.dense2 = tf.keras.layers.Dense(vocab_size)

    self.attention = visualAttention(self.units)

  def call(self, x, features, hidden):
    x = self.embedding(x)
    context_vector, attention_weights = self.attention(features, hidden)
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)    
    output, state = self.gru(x)    
    x = self.dense1(output)
    x = tf.reshape(x, (-1, x.shape[2]))    
    x = self.dense2(x)
    return x, state, attention_weights

  def reset_state(self, batch_size):
    return tf.zeros((batch_size, self.units))

class visualAttention(tf.keras.Model):
  def __init__(self, units):
    super(visualAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    

  def call(self, features, hidden):
    
    hidden_with_time_axis = tf.expand_dims(hidden, 1)

    score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))
    
    attention_weights = tf.nn.softmax(score, axis=1)
   
    context_vector = attention_weights * features
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights

class encoder(tf.keras.Model):
    def __init__(self, embedding_dim):
        super(encoder, self).__init__()
        self.flattenLayer = tf.keras.layers.Dense(embedding_dim)

    def call(self, x):
        x = self.flattenLayer(x)
        x = tf.nn.relu(x)
        return x


class CustomEmbedding(layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)

        self.token_embeddings = layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim)
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=embed_dim)
        self.sequence_length = sequence_length
        self.input_dim = vocab_size
        self.output_dim = embed_dim

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

class CNNEncoder(layers.Layer):
  def __init__(self, embed_dim, **kwargs):
      super().__init__(**kwargs)
      self.embed_dim = embed_dim
      self.pooling = keras.Sequential(
          [layers.GlobalAveragePooling1D()])  
      self.dense_proj = keras.Sequential(
          [layers.Dense(embed_dim, activation="relu")]) 
                              

  def call(self, inputs):
    x = self.pooling(inputs)
    return self.dense_proj(x)      

class RnnDecoderWithoutAttention(layers.Layer):
    def __init__(self, units, embed_dim, dense_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.units = units
        
        self.LSTM = tf.keras.layers.LSTM(self.units, 
                                         return_sequences = True,
                                       recurrent_initializer='glorot_uniform')
        self.gru = tf.keras.layers.GRU(self.units,
                                       return_sequences = True,
                                       recurrent_initializer='glorot_uniform')
        
        self.dense_proj = keras.Sequential(
            [layers.Dense(dense_dim, activation="relu"),
             layers.Dense(embed_dim),]
        )
    def call(self, inputs, encoder_outputs, mask=None):
        x = tf.concat([inputs, tf.expand_dims(encoder_outputs, -1)], axis=-1)
        x = self.LSTM(x)
        output = self.dense_proj(x)
        return output

def get_model_without_attention():
  embed_dim = 49
  dense_dim = 256
  sequence_length = 50
  vocab_size = 5000

  encoder_inputs = keras.Input(shape=(49,512), dtype="int64", name="image_features") 
  encoder_outputs = CNNEncoder(embed_dim)(encoder_inputs)

  decoder_inputs = keras.Input(shape=(None,), dtype="int64", name="captions") 
  x = CustomEmbedding(sequence_length, vocab_size, embed_dim)(decoder_inputs) 
  x = RnnDecoderWithoutAttention(256, embed_dim, dense_dim)(x, encoder_outputs)

  x = layers.Dropout(0.5)(x)
  
  decoder_outputs = layers.Dense(vocab_size, activation="softmax")(x)
  transformer = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
  transformer.compile(
    optimizer="rmsprop",
    loss = "sparse_categorical_crossentropy")
  return transformer