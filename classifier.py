# coding:utf8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import numpy as np

import tensorflow as tf
from tensorflow.contrib import rnn

import inputs

# 训练输入数据参数
tf.flags.DEFINE_string("model_type", None,
                       "Model type can be: fasttext, rnn, cnn.")
tf.flags.DEFINE_string("train_records", None,
                       "Training file pattern for TFRecords.")
tf.flags.DEFINE_string("eval_records", None,
                       "Evaluation file pattern for TFRecords.")
tf.flags.DEFINE_string("gold_label_file", None,
                       "The gold label of eval records。")
tf.flags.DEFINE_string("label_file", None, "The label dict file")
tf.flags.DEFINE_string("model_dir", None,
                       "Output directory for checkpoints and summaries.")
tf.flags.DEFINE_integer("label_size", None, "Number of labels.")
tf.flags.DEFINE_integer("vocab_size", None, "Number of words in vocabulary.")
tf.flags.DEFINE_integer("num_reader_threads", 1, "Number of reader threads.")

# 共有模型参数
tf.flags.DEFINE_integer("embedding_dimension", 16,
                        "Dimension of word embedding.")
tf.flags.DEFINE_integer("attention_dimension", 16, "Dimension of attention.")
tf.flags.DEFINE_integer("batch_size", 128, "Training mini batch size.")
tf.flags.DEFINE_integer("num_epochs", 10, "Number of training data epochs.")
tf.flags.DEFINE_boolean("use_attention", False, "Whether to use attention.")
tf.flags.DEFINE_float("learning_rate", 0.001, "Learning rate for training.")
tf.flags.DEFINE_float("per_process_gpu_memory_fraction", 0.5,
                      "Fraction per process gpu memory to use")

# rnn模型参数
tf.flags.DEFINE_integer("rnn_dimension", 16, "Dimension of rnn.")
tf.flags.DEFINE_string("cell_type", "gru", "Rnn cell type can be: gru, lstm")

# cnn模型参数
tf.flags.DEFINE_string("filter_sizes", "1,2,3,4,5,6,7",
                       "Comma-separated filter sizes.")
tf.flags.DEFINE_integer("num_filters", 256,
                        "Number of filters per filter size.")
tf.flags.DEFINE_integer("sequence_length", 512,
                        "Max sequence length per sample")

# debug参数
tf.flags.DEFINE_boolean("use_hook", False,
                        "Whether to add hook while training. "
                        "If use hook, then time line json file will be saved")

FLAGS = tf.flags.FLAGS


def get_run_config():
    gpu_option = tf.GPUOptions(
        per_process_gpu_memory_fraction=FLAGS.per_process_gpu_memory_fraction)
    session_config = tf.ConfigProto(
        gpu_options=gpu_option)
    config = tf.contrib.learn.RunConfig(
        session_config=session_config)
    return config


def attention(text_embedding, attention_dim):
    attention_matrix \
        = tf.contrib.layers.fully_connected(inputs=text_embedding,
                                            num_outputs=attention_dim,
                                            activation_fn=tf.nn.tanh)
    attention_vector = tf.Variable(
        tf.random_uniform([FLAGS.attention_dimension],
                          -1.0 / FLAGS.attention_dimension,
                          1.0 / FLAGS.attention_dimension))

    alpha = tf.nn.softmax(
        tf.reduce_sum(tf.multiply(attention_matrix, attention_vector), axis=2,
                      keep_dims=True), dim=1)
    attention_embedding = tf.reduce_sum(tf.multiply(text_embedding, alpha),
                                        axis=1, keep_dims=False)
    return attention_embedding


def get_word_embedding(features):
    word_embedding_table = tf.Variable(tf.random_uniform(
        [FLAGS.vocab_size, FLAGS.embedding_dimension],
        -1.0 / FLAGS.embedding_dimension, 1.0 / FLAGS.embedding_dimension))
    word_embeddings = tf.nn.embedding_lookup(word_embedding_table, features)
    return word_embeddings


def get_estimator_spec(hidden_layer, mode, labels):
    logits = tf.contrib.layers.fully_connected(
        inputs=hidden_layer, num_outputs=FLAGS.label_size,
        biases_initializer=None,
        activation_fn=None)
    logits = tf.expand_dims(logits, -2)
    predictions = tf.argmax(logits, axis=-1)
    loss, train_op = None, None
    metrics = {}
    if mode == tf.estimator.ModeKeys.TRAIN:
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
                                                           logits=logits))
        opt = tf.train.AdamOptimizer(FLAGS.learning_rate)
        train_op = opt.minimize(loss,
                                global_step=tf.train.get_global_step())
        labels = tf.squeeze(labels, -1)
        metrics = {
            "accuracy": tf.metrics.accuracy(labels, predictions)
        }
    exports = {}
    probability = tf.nn.softmax(logits)
    return tf.estimator.EstimatorSpec(
        mode, predictions=probability, loss=loss, train_op=train_op,
        eval_metric_ops=metrics, export_outputs=exports)


def rnn_estimator(model_dir):
    def model_fn(features, labels, mode):
        word_embeddings = get_word_embedding(features)
        rnn_fw_cell, rnn_bw_cell = None, None
        if FLAGS.cell_type == "lstm":
            rnn_fw_cell = rnn.BasicLSTMCell(FLAGS.rnn_dimension)
            rnn_bw_cell = rnn.BasicLSTMCell(FLAGS.rnn_dimension)
        elif FLAGS.cell_type == "gru":
            rnn_fw_cell = rnn.GRUCell(FLAGS.rnn_dimension)
            rnn_bw_cell = rnn.GRUCell(FLAGS.rnn_dimension)
        else:
            print("unknown cell type: %s" % FLAGS.cell_type)
            exit(-1)

        outputs, _ = tf.nn.bidirectional_dynamic_rnn(rnn_fw_cell, rnn_bw_cell,
                                                     word_embeddings,
                                                     dtype=tf.float32)
        text_embedding = tf.concat(outputs, 2)
        if FLAGS.use_attention:
            hidden_layer = attention(text_embedding, FLAGS.attention_dimension)
        else:
            hidden_layer = tf.reduce_mean(text_embedding, axis=-2)
        return get_estimator_spec(hidden_layer, mode, labels)

    return tf.estimator.Estimator(model_fn=model_fn, model_dir=model_dir,
                                  config=get_run_config())


def fasttext_estimator(model_dir):
    def model_fn(features, labels, mode):
        word_embeddings = get_word_embedding(features)

        if FLAGS.use_attention:
            hidden_layer = attention(word_embeddings, FLAGS.attention_dimension)
        else:
            hidden_layer = tf.reduce_mean(word_embeddings, axis=-2)

        return get_estimator_spec(hidden_layer, mode, labels)

    return tf.estimator.Estimator(model_fn=model_fn, model_dir=model_dir,
                                  config=get_run_config())


def cnn_estimator(model_dir):
    def model_fn(features, labels, mode):
        word_embeddings = get_word_embedding(features)
        word_embeddings = tf.expand_dims(word_embeddings, -1)
        filter_sizes = [int(x) for x in FLAGS.filter_sizes.split(",")]
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("convolution-max_pooling-%d" % filter_size):
                filter_shape = [filter_size, FLAGS.embedding_dimension, 1,
                                FLAGS.num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1),
                                name="W-%d" % filter_size)
                b = tf.get_variable("b-%d" % filter_size, [FLAGS.num_filters])
                convolution = tf.nn.conv2d(
                    word_embeddings,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="convolution")
                h = tf.nn.relu(tf.nn.bias_add(convolution, b), name="relu")
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, FLAGS.sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="max_pooling")
                pooled_outputs.append(pooled)

        num_filters_total = FLAGS.num_filters * len(filter_sizes)
        h_pool = tf.concat(pooled_outputs, 3)
        hidden_layer = tf.reshape(h_pool, [-1, num_filters_total])

        return get_estimator_spec(hidden_layer, mode, labels)

    return tf.estimator.Estimator(model_fn=model_fn, model_dir=model_dir,
                                  config=get_run_config())


def calculate_prf(gold_count, predict_count, right_count):
    precision, recall, f_score = 0, 0, 0
    if predict_count > 0:
        precision = right_count / predict_count
    if gold_count > 0:
        recall = right_count / gold_count
    if precision + recall > 0:
        f_score = precision * recall * 2 / (precision + recall)

    return precision, recall, f_score


def calculate_performance(predict_probability, gold, num_class, other_class,
                          threshold):
    confusion_matrix = np.zeros((num_class, num_class), dtype=np.int64)
    # filter the sample that max predict prob less than threshold
    filtered_confusion_matrix = np.zeros((num_class, num_class), dtype=np.int64)
    line_count = 0
    for predict in predict_probability:
        gold_label = gold[line_count]
        predict_np = np.array(predict)
        predict_label = predict_np.argmax()
        confusion_matrix[gold_label][predict_label] += 1
        if predict_np.max() > threshold:
            filtered_confusion_matrix[gold_label][predict_label] += 1
        line_count += 1
    gold_count_category = filtered_confusion_matrix.sum(axis=1)
    predict_count_category = filtered_confusion_matrix.sum(axis=0)
    # erase 'other' category count
    for other_id in other_class:
        gold_count_category[other_id] = 0
        predict_count_category[other_id] = 0
    gold_count = 0
    predict_count = 0
    right_count = 0

    category_performance = []
    for i in range(0, num_class):
        if i in other_class:
            continue
        category_performance.append(
            calculate_prf(gold_count_category[i], predict_count_category[i],
                          filtered_confusion_matrix[i][i]))
        gold_count += gold_count_category[i]
        predict_count += predict_count_category[i]
        right_count += filtered_confusion_matrix[i][i]
    sys.stdout.write(
        "gold count: %d\t,predict count: %d\t,right_count: %d\n" % (
            gold_count, predict_count, right_count))
    return calculate_prf(gold_count, predict_count, right_count)


def train():
    gold_label = [int(x.strip("\n")) for x in
                  open(FLAGS.gold_label_file).readlines()]
    labels = [line.split("\t") for line in open(FLAGS.label_file).readlines()]
    other_labels = filter(lambda x: x[1].find("其他") != -1, labels)
    other_labels = set([int(x[0]) for x in other_labels])
    print(other_labels)
    sys.stdout.write("init")
    if FLAGS.use_hook:
        hook = [tf.train.ProfilerHook(save_steps=100, output_dir='./timeline')]
    else:
        hook = None
    if FLAGS.model_type == "rnn":
        estimator = rnn_estimator(FLAGS.model_dir)
    elif FLAGS.model_type == "fasttext":
        estimator = fasttext_estimator(FLAGS.model_dir)
    elif FLAGS.model_type == "cnn":
        estimator = cnn_estimator(FLAGS.model_dir)
    else:
        print(
            "unknown model type: %s, "
            "support model type: cnn, rnn, fasttext" % FLAGS.model_type)

    input_fn = inputs.batch_reader_input_fn
    for i in range(0, FLAGS.num_epochs):
        train_input = input_fn(tf.estimator.ModeKeys.TRAIN, FLAGS.train_records,
                               FLAGS.batch_size, 1, FLAGS.num_reader_threads)
        sys.stdout.write("start training epoch %d\n" % i)
        estimator.train(input_fn=train_input, hooks=hook)
        sys.stdout.write("start evaluate at epoch %d\n" % i)
        predict_input = input_fn(tf.estimator.ModeKeys.PREDICT,
                                 FLAGS.eval_records, FLAGS.batch_size, 1,
                                 FLAGS.num_reader_threads)
        predict = estimator.predict(input_fn=predict_input, hooks=None)
        sys.stdout.write("eval done\n")
        sys.stdout.write("epoch %d performance:\n" % i)
        sys.stdout.write("%f\t%f\t%f\n" % (
            calculate_performance(predict, gold_label, len(labels),
                                  other_labels, 0)))
        sys.stdout.flush()


def export_fn():
    features = {
        "features": tf.placeholder(dtype=tf.string, shape=[None],
                                   name='features')
    }
    return tf.estimator.export.build_raw_serving_input_receiver_fn(features)


def main(_):
    train()


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.DEBUG)
    tf.app.run()
