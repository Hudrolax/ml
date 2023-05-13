import tensorflow as tf
from sklearn.metrics import balanced_accuracy_score
from keras import backend as K


def balanced_acc(y_true, y_pred):
    y_true = K.argmax(y_true, axis=-1)
    y_pred = K.argmax(y_pred, axis=-1)
    return tf.py_function(balanced_accuracy_score, (y_true, y_pred), tf.double)

def f1_metric(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = K.cast(true_positives, 'float32') / (K.cast(predicted_positives, 'float32') + K.epsilon())
    recall = K.cast(true_positives, 'float32') / (K.cast(possible_positives, 'float32') + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val