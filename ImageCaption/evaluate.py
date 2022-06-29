import evaluate
from torchmetrics.functional.text.rouge import rouge_score
from nltk.translate.meteor_score import meteor_score
import nltk
import numpy as np

from pprint import pprint
import tensorflow as tf

def evaluate(image, encoder, decoder, max_length,
             attention_features_shape, tokenizer,
             word_to_index, index_to_word):
    attention_plot = np.zeros((max_length, attention_features_shape))

    hidden = decoder.reset_state(batch_size=1)

    
    img_tensor_val = np.load(image+'.npy')


    features = encoder(img_tensor_val)

    dec_input = tf.expand_dims([word_to_index('[start]')], 0)
    result = []

    for i in range(max_length):
        predictions, hidden, attention_weights = decoder(dec_input,
                                                         features,
                                                         hidden)

        # attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()

        predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
        predicted_word = tf.compat.as_text(index_to_word(predicted_id).numpy())
        result.append(predicted_word)

        if predicted_word == '[end]':
            return result, attention_plot

        dec_input = tf.expand_dims([predicted_id], 0)

    # attention_plot = attention_plot[:len(result), :]
    return result, None

# BLEU

def corpus_bleu(corpus_predictions, corpus_references):
    '''
        corpus_predictions: list of prediction strings
        corpus_references: list of list reference strings
    '''

    bleu = evaluate.load("bleu")
    results = bleu.compute(predictions=corpus_predictions, references=corpus_references)

    return results

# METEOR
def corpus_meteor(corpus_predictions, corpus_references):
    '''
        corpus_predictions: list of prediction strings
        corpus_references: list of list reference strings
    '''

    corpus_meteor_score = []
    for (references, prediction) in zip(corpus_references, corpus_predictions):
        references = [nltk.word_tokenize(reference) for reference in references]
        prediction = nltk.word_tokenize(prediction)
        corpus_meteor_score.append(meteor_score(references, prediction))

    corpus_meteor_score = np.mean(np.array(corpus_meteor_score))
    return corpus_meteor_score

# ROUGE
def corpus_rouge(corpus_predictions, corpus_references):
    '''
        corpus_predictions: list of prediction strings
        corpus_references: list of list reference strings
    '''

    results = rouge_score(corpus_predictions, corpus_references)
    return results