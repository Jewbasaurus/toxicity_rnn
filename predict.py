import argparse
import json
import logging
import numpy as np
import pandas as pd
import os
import tensorflow as tf
import tensorflow.keras as keras
import youtokentome as yttm
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Bidirectional, Dense, Dropout, Embedding, LSTM, Masking, SpatialDropout1D
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tensorflow.keras.models import Sequential, model_from_json


def read_data(file, model_dir):
    """
    Reads a file and turns it into a tf.data.Dataset of subword units

    Args:
      file : a string path to a .txt file
      model_dir : a directory with model files

    Returns:
      tf.data.Dataset of subwords

    """
    with open(file, 'r') as f:
        lines = [line for line in f if line.strip()]
    bpe_model = yttm.BPE(f'{model_dir}/bpe_10000.model')  
    test_data = np.array(bpe_model.encode(lines, dropout_prob=0.1))
    test_data = tf.ragged.constant(test_data)
    test_data = tf.data.Dataset.from_tensor_slices(test_data)
    return test_data


def load_model(args):
    """
    Loads or creates a model

    Args:
      args : arguments passed to argparse.Parser

    Returns:
      a Keras model

    """
    logger.warning("***** Model *****")
    # If there is no model or we want to overwrite the current one, we create a new model
    logger.warning("***** Loading a model *****")
    with open(f'{args.model_dir}/config.json', 'r') as json_file:
        config = json.load(json_file)
        model = model_from_json(config)
    logger.warning("***** Loading model weights *****")
    model.load_weights(f'{args.model_dir}/model_weights.h5')
    logger.info(model.summary())
    return model


@tf.function(experimental_relax_shapes=True)
def predict(model, x):
    """
    Does a forward pass and returns a class prediction for x.
    Separate function to wrap into tf.function

    Args:
      model : a Keras model
      x : [batch_size, seq_len] input data

    Returns:
      predictions : [batch_size] class labels in a tensor

    """
    predictions = tf.math.argmax(model(x, training=False), axis=1)
    return predictions


def main(description='Train a Bi-LSTM to classify toxic commentaries'):
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--input_file", default=None, type=str, required=True,
                        help="The input .txt file. Example per line")
    parser.add_argument("--model_dir", default='model', type=str, required=True,
                        help="The directory with model files (bpe model, config and weights).")
    parser.add_argument("--output_file", default=None, type=str, required=True,
                        help="THe output .txt file. Label per line")

    ## Other parameters
    parser.add_argument("--batch_size", default=32, type=int,
                        help="Batch size. Defaults to 32.")

    args = parser.parse_args()

    model = load_model(args)
    test_data = read_data(args.input_file, args.model_dir)

    BATCH_SIZE = args.batch_size
    test_bar = tf.keras.utils.Progbar(np.floor(int(tf.data.experimental.cardinality(test_data)/BATCH_SIZE))+1)
    # Predict
    with open(f'{args.output_file}', 'w') as target: 
        for batch in test_data.batch(BATCH_SIZE):
            # Optimize the model
            X = batch.to_tensor()
            predictions = predict(model, X)
            for p in predictions.numpy():
                target.write(str(p)+'\n')
            # Track progress 
            test_bar.add(1)


logger = logging.getLogger(__name__)
logger.setLevel(10)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
physical_devices = tf.config.list_physical_devices()
logger.warning(f'Physical devices: {physical_devices}')
tf.config.experimental.set_memory_growth(physical_devices[2], True)
if __name__ == "__main__":
    main()
