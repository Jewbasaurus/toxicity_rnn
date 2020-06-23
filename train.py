import argparse
import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import tensorflow as tf
import tensorflow.keras as keras
import youtokentome as yttm
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Bidirectional, Dense, Dropout, Embedding, LSTM, GRU, Masking, SpatialDropout1D
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tensorflow.keras.models import Sequential, model_from_json


def read_data(file, model_dir):
    """
    Reads a file and turns it into a train and test tf.data.Datasets of ((subword_units, label))

    Args:
      file : a string path to a .csv file
      model_dir : a directory with model files

    Returns:
      train_data : tf.data.Dataset with (bpe_encoded_example, label) elements
      test_data : tf.data.Dataset with (bpe_encoded_example, label) elements
      weights_table : tf HashTable for looking up class weights

    """
    data = pd.read_csv(f'{file}').drop(columns=['uuid'])
    data = data[data.comment_text.notnull()] # not empty comments
    data = data[data.comment_text.map(len) > 1]
    train, test = train_test_split(data, test_size=0.1)
    train.toxicity = train.loc[:, 'toxicity'].astype(np.int32)
    test.toxicity = test.loc[:, 'toxicity'].astype(np.int32)
    # Load BPE model (10k)
    bpe_model = yttm.BPE(f'{model_dir}/bpe_5000.model')
    # Convert train data to bpe, remove looooong instances
    train_bpe = train.comment_text.apply(lambda row: np.array(bpe_model.encode(row, dropout_prob=0.1)))
    lens = train_bpe.apply(lambda row: len(row))
    # 3 STD should be enough, but why not 4
    max_len = np.ceil(np.mean(lens) + 3 * np.std(lens))
    train_bpe = train_bpe[train_bpe.apply(lambda row: len(row)) <= max_len]
    train_data = tf.ragged.constant(train_bpe)
    train_labels = train.toxicity[train_bpe.index]
    train_data = tf.data.Dataset.from_tensor_slices((train_data, train_labels))

    # Convert test data to bpe, remove looooong instances
    test_bpe = test.comment_text.apply(lambda row: bpe_model.encode(row))
    lens = test_bpe.apply(lambda row: len(row))
    # 3 STD should be enough, but why not 4
    max_len = np.ceil(np.mean(lens) + 3 * np.std(lens))
    test_bpe = test_bpe[test_bpe.apply(lambda row: len(row)) <= max_len]
    test_data = tf.ragged.constant(test_bpe)
    test_labels = test.toxicity[test_bpe.index]
    test_data = tf.data.Dataset.from_tensor_slices((test_data, test_labels))
    logger.info(f"Train data: {tf.data.experimental.cardinality(train_data)}\n Test data: {tf.data.experimental.cardinality(test_data)}")

    class_weights = {}
    # Opt 1
    # for c in train_labels.unique():
    #     class_weights[c] = 1. / train_labels.value_counts()[c]
    # Opt 2
    for c in train_labels.unique():
        class_weights[c] = len(train_labels) / (len(train_labels.unique()) * train_labels.value_counts()[c])
    # Opt 3
    # for c in train_labels.unique():
    #     class_weights[c] = len(train_labels) / train_labels.value_counts()[c] - 1
    print(class_weights)
    keys_tensor = tf.constant(list(class_weights.keys()))
    vals_tensor = tf.constant(list(class_weights.values()))
    weights_table = tf.lookup.StaticHashTable(
                        tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor), -1)

    return train_data, test_data, weights_table


def load_model(args):
    """
    Loads or creates a model

    Args:
      args : arguments passed to argparse.Parser

    Returns:
      a Keras model

    """
    logger.warning("***** Model *****\n")
    physical_devices = tf.config.list_physical_devices()
    logger.warning(f'Physical devices: {physical_devices}\n')
    tf.config.experimental.set_memory_growth(physical_devices[2], True)
    if args.mixed_precision:
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_policy(policy)
    # If there is no model or we want to overwrite the current one, we create a new model
    if args.overwrite or not os.path.exists(f'{args.model_dir}/config.json'):
        logger.warning("***** Creating a new model *****")
        model = Sequential()
        model.add(Embedding(input_dim=5000,
                            output_dim=256,
                            embeddings_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None),
                            trainable=True,
                            mask_zero=True,
                            name='Embeddings'))
        model.add(SpatialDropout1D(0.15, name='Emb_drop'))
        model.add(Bidirectional(GRU(256,
                                    recurrent_regularizer=tf.keras.regularizers.l2(1e-4),
                                    kernel_regularizer=tf.keras.regularizers.l2(2e-3),
                                    dropout=0.2,
                                    recurrent_dropout=0.2, 
                                    return_sequences=True),
                                name='Bi-LSTM_1'))
        model.add(Bidirectional(GRU(256,
                                    recurrent_regularizer=tf.keras.regularizers.l2(1e-4),
                                    kernel_regularizer=tf.keras.regularizers.l2(2e-3),
                                    dropout=0.2,
                                    recurrent_dropout=0.2, 
                                    return_sequences=True),
                                name='Bi-LSTM_2'))
        model.add(Bidirectional(GRU(128,
                                    # recurrent_regularizer=tf.keras.regularizers.l2(1e-4),
                                    # kernel_regularizer=tf.keras.regularizers.l2(2e-3),
                                    recurrent_dropout=0.2,
                                    dropout=0.2),
                                name='Bi-LSTM_3'))        
        model.add(Dropout(0.25, name='Dropout_2'))
        model.add(Dense(512, name='Dense_1'))
        # model.add(Dropout(0.3, name='Dropout_3'))
        model.add(Dense(256, name='Dense_2'))
        model.add(Dense(6, activation='sigmoid', dtype='float32', name='Output'))
    else:
        logger.warning("***** Loading a model *****")
        
        with open(f'{args.model_dir}/config.json', 'r') as json_file:
            config = json.load(json_file)
            model = model_from_json(config)
    if not args.overwrite and os.path.exists(f'{args.model_dir}/model_weights.h5'):
        logger.warning("***** Loading model weights *****")
        model.load_weights(f'{args.model_dir}/model_weights.h5')

    logger.info(model.summary())

    return model

@tf.function(experimental_relax_shapes=True)
def train_step(model, loss_object, weights_table, optimizer, X, y, mixed_precision):
    """
    Perform a train step

    Args:
      X : a [batch_size, sequence_length] tensor with token ids
      y : a [batch_size] tensor with real class values for each sample

    Returns:
      loss : a [1] tensor with fp32 loss
      predictions : a [batch_size] tensor with a number prediction for each sample

    """
    sample_weights = weights_table.lookup(y)
    with tf.GradientTape() as tape:
        # training=True is needed only if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        logits = model(X, training=True)
        loss = loss_object(y, logits, sample_weight=sample_weights)
        if mixed_precision:
            scaled_loss = optimizer.get_scaled_loss(loss)
    if mixed_precision:
        scaled_gradients = tape.gradient(scaled_loss, model.trainable_variables)
        gradients = optimizer.get_unscaled_gradients(scaled_gradients)
    else:
        gradients = tape.gradient(loss, model.trainable_variables)
    gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in zip(scaled_gradients, model.trainable_variables)]
    optimizer.apply_gradients(gradients)
    # optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss, logits


@tf.function(experimental_relax_shapes=True)
def test_step(model, loss_object, x, y):
    """
    Perform a test step

    Args:
      x : a [batch_size, sequence_length] tensor with token ids
      y : a [batch_size] tensor with real class values for each sample

    Returns:
      loss : a [1] tensor with fp32 loss
      predictions : a [batch_size] tensor with a number prediction for each sample

    """
    logits = model(x, training=False)
    loss = loss_object(y, logits)
    return loss, logits


def main(description='Train a Bi-LSTM to classify toxic commentaries'):
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--input_file", default=None, type=str, required=True,
                        help="The input .csv file. It is split into train and test.")

    ## Other parameters
    parser.add_argument("--model_dir", default='model', type=str,
                        help="The directory with model files (bpe model, config and weights).")
    parser.add_argument("--lr", default=2e-3, type=float,
                        help="Learning rate. Defaults to 2e-3.")
    parser.add_argument("--epochs", default=10, type=int,
                        help="How many epochs to train. Defaults to 10.")
    parser.add_argument("--batch_size", default=32, type=int,
                        help="Batch size. Defaults to 32.")
    parser.add_argument("--overwrite", action="store_true",
                        help="If the model should start from scratch and overwrite the existing model")
    parser.add_argument("--mixed_precision", action="store_true",
                        help="Use mixed precision training.")

    args = parser.parse_args()


    model = load_model(args)
    train_data, test_data, weights_table = read_data(args.input_file, args.model_dir)

    # Train!
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    if args.mixed_precision:
        logger.warning('***** Using Mixed Precision *****')
        optimizer = mixed_precision.LossScaleOptimizer(optimizer, loss_scale='dynamic')
        
    # Keep results for plotting
    train_loss_results = []
    train_accuracy_results = []
    test_loss_results = []
    test_accuracy_results = []

    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    logger.warning(f'Batch Size: {BATCH_SIZE}')

    metrics_names = ['loss','acc'] 

    for epoch in range(args.epochs):
        train_bar = tf.keras.utils.Progbar(np.floor(int(tf.data.experimental.cardinality(train_data)/BATCH_SIZE))+1,
                                           stateful_metrics=metrics_names)
        test_bar = tf.keras.utils.Progbar(np.floor(int(tf.data.experimental.cardinality(test_data)/BATCH_SIZE))+1,
                                          stateful_metrics=metrics_names)
        train_loss_avg = tf.keras.metrics.Mean() # Avg loss
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy() # Avg accuracy
        test_loss_avg = tf.keras.metrics.Mean()
        test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        
        logger.warning(f'Epoch {epoch+1}/{EPOCHS}')

        # Training loop - using batches of BATCH_SIZE
        for batch in train_data.shuffle(buffer_size=20000, reshuffle_each_iteration=True).batch(BATCH_SIZE): # Dataset (features, label)
            X = batch[0].to_tensor() # RaggedTensor -> Sparse Tensor, Post Pad by the longest element
            y = batch[1]
            # Optimize the model
            loss, logits = train_step(model, loss_object, weights_table, optimizer, X, y, args.mixed_precision)
            train_loss_avg.update_state(loss)  # Add current batch loss
            
            # Compare predicted label to actual label
            train_accuracy.update_state(y, logits)
            values=[('loss', train_loss_avg.result()), ('acc', train_accuracy.result())]
            train_bar.add(1, values=values)
        
        # Test 
        preds = []
        gts = []    
        for batch in test_data.batch(BATCH_SIZE):
            # Optimize the model
            X = batch[0].to_tensor()
            y = batch[1]

            loss, logits = test_step(model, loss_object, X, y)
            # Track progress
            gts.extend(y)
            preds.extend(tf.math.argmax(logits, axis=1))
            test_loss_avg.update_state(loss)  # Add current batch loss
            test_accuracy.update_state(y, logits)
            values=[('loss', test_loss_avg.result()), ('acc', test_accuracy.result())]
            test_bar.add(1, values=values)
        
        # End epoch
        train_loss_results.append(train_loss_avg.result())
        train_accuracy_results.append(train_accuracy.result())
        test_loss_results.append(test_loss_avg.result())
        test_accuracy_results.append(test_accuracy.result())
        logger.warning(f'Epoch {epoch+1}/{EPOCHS}.\n'\
                       f'Train loss: {train_loss_avg.result()}, train acc: {train_accuracy.result()}\n'\
                       f'Test loss: {test_loss_avg.result()}, test acc: {test_accuracy.result()}')
        if (epoch+1) % 10 == 0:
            cm = tf.math.confusion_matrix(gts, preds)
            logger.warning(cm)

    model.save_weights(f'{args.model_dir}/model_weights.h5')
    config = model.to_json()
    with open(f'{args.model_dir}/config.json', 'w') as f:
        json.dump(config, f)


logger = logging.getLogger(__name__)
logger.setLevel(10)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
pd.options.mode.chained_assignment = None
if __name__ == "__main__":
    main()
