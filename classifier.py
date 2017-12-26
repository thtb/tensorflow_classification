# coding:utf8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import numpy as np

import tensorflow as tf

import inputs

tf.flags.DEFINE_string("train_records", None,
                       "Training file pattern for TFRecords, can use wildcards")
tf.flags.DEFINE_string("eval_records", None,
                       "Evaluation file pattern for TFRecords, can use wildcards")
tf.flags.DEFINE_string("gold_label_file", None, "")
tf.flags.DEFINE_string("predict_records", None,
                       "File pattern for TFRecords to predict, can use wildcards")
tf.flags.DEFINE_string("label_file", None, "File containing output labels")
tf.flags.DEFINE_string("vocab_file", None, "Vocabulary file, one word per line")
tf.flags.DEFINE_integer("vocab_size", None, "Number of words in vocabulary")
tf.flags.DEFINE_integer("num_oov_vocab_buckets", 0,
                        "Number of hash buckets to use for OOV words")
tf.flags.DEFINE_string("model_dir", ".",
                       "Output directory for checkpoints and summaries")
tf.flags.DEFINE_string("export_dir", None, "Directory to store savedmodel")

tf.flags.DEFINE_integer("embedding_dimension", 15,
                        "Dimension of word embedding")
tf.flags.DEFINE_integer("attention_dimension", 16, "Dimension of attention")

tf.flags.DEFINE_float("learning_rate", 0.001, "Learning rate for training")
tf.flags.DEFINE_float("clip_gradient", 5.0, "Clip gradient norm to this ratio")
tf.flags.DEFINE_integer("batch_size", 128, "Training minibatch size")
tf.flags.DEFINE_integer("num_epochs", None, "Number of training data epochs")
tf.flags.DEFINE_integer("checkpoint_steps", 1000,
                        "Steps between saving checkpoints")
tf.flags.DEFINE_integer("num_threads", 1, "Number of reader threads")
tf.flags.DEFINE_boolean("log_device_placement", False,
                        "log where ops are located")
tf.flags.DEFINE_float("per_process_gpu_memory_fraction", 0.5,
                      "per_process_gpu_memory_fraction")
tf.flags.DEFINE_boolean("debug", False, "Debug")
FLAGS = tf.flags.FLAGS


def fasttext_estimator(model_dir):
    params = {
        "learning_rate": FLAGS.learning_rate,
    }

    def model_fn(features, labels, mode, params):
        num_classes = len(open(FLAGS.label_file).readlines())
        text_ids = features
        text_embedding_w = tf.Variable(tf.random_uniform(
            [FLAGS.vocab_size + FLAGS.num_oov_vocab_buckets,
             FLAGS.embedding_dimension],
            -1.0 / FLAGS.embedding_dimension, 1.0 / FLAGS.embedding_dimension))
        # text_embedding = tf.nn.embedding_lookup(text_embedding_w, text_ids)
        # attention_matrix = tf.contrib.layers.fully_connected(inputs=text_embedding, num_outputs=FLAGS.attention_dimension, activation_fn=tf.nn.tanh)
        # attention_vector = tf.Variable(tf.random_uniform([FLAGS.attention_dimension], -1.0 / FLAGS.attention_dimension, 1.0 / FLAGS.attention_dimension))
        ##attention_vector = tf.Variable(tf.zeros([FLAGS.attention_dimension]))
        # alpha = tf.nn.softmax(tf.reduce_sum(tf.multiply(attention_matrix, attention_vector), axis=2, keep_dims=True), dim=1)
        # attention_embedding = tf.reduce_sum(tf.multiply(text_embedding, alpha), axis=1, keep_dims=False)
        # input_layer = attention_embedding

        text_embedding = tf.reduce_mean(tf.nn.embedding_lookup(
            text_embedding_w, text_ids), axis=-2)
        input_layer = text_embedding

        logits = tf.contrib.layers.fully_connected(
            inputs=input_layer, num_outputs=num_classes,
            biases_initializer=None,
            activation_fn=None)
        logits = tf.expand_dims(logits, -2)
        predictions = tf.argmax(logits, axis=-1)
        loss, train_op = None, None
        metrics = {}
        if mode != tf.estimator.ModeKeys.PREDICT:
            loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
                                                               logits=logits))
            labels = tf.squeeze(labels, -1)
            opt = tf.train.AdamOptimizer(params["learning_rate"])
            opt_sgd = tf.train.GradientDescentOptimizer(learning_rate=0.01)
            train_op = opt.minimize(loss,
                                        global_step=tf.train.get_global_step())
            metrics = {
                "accuracy": tf.metrics.accuracy(labels, predictions)
            }
        exports = {}
        # if FLAGS.export_dir:
        probs = tf.nn.softmax(logits)
        # exports["probs"] = tf.estimator.export.ClassificationOutput(
        #     scores=probs)
        # exports["embedding"] = tf.estimator.export.RegressionOutput(
        #     value=text_embedding)
        # exports[
        #     tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = \
        #     tf.estimator.export.ClassificationOutput(scores=probs)
        return tf.estimator.EstimatorSpec(
            mode, predictions=probs, loss=loss, train_op=train_op,
            eval_metric_ops=metrics, export_outputs=exports)

    gpu_option = tf.GPUOptions(
        per_process_gpu_memory_fraction=FLAGS.per_process_gpu_memory_fraction)
    session_config = tf.ConfigProto(
        log_device_placement=FLAGS.log_device_placement,
        gpu_options=gpu_option)
    config = tf.contrib.learn.RunConfig(
        session_config=session_config)
    return tf.estimator.Estimator(model_fn=model_fn, model_dir=model_dir,
                                  params=params, config=config)


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
                          threshold, debug_file="label_debug.txt"):
    debug_f = None
    if debug_file != None:
        debug_f = open(debug_file, "w")
    confusion_matrix = np.zeros((num_class, num_class), dtype=np.int64)
    # filter max predict prob less than threshold
    filtered_confusion_matrix = np.zeros((num_class, num_class), dtype=np.int64)
    gold_label_line_count = len(gold)
    line_count = 0
    for predict in predict_probability:
        gold_label = gold[line_count]
        predict_np = np.array(predict)
        predict_label = predict_np.argmax()
        if debug_f != None:
            debug_f.write("%d\t%d\n" % (predict_label, gold_label))
        confusion_matrix[gold_label][predict_label] += 1
        if predict_np.max() > threshold:
            filtered_confusion_matrix[gold_label][predict_label] += 1
        line_count += 1
    print(gold_label_line_count, line_count)
    # erase 'other' category count
    print(filtered_confusion_matrix)
    gold_count_category = filtered_confusion_matrix.sum(axis=1)
    predict_count_category = filtered_confusion_matrix.sum(axis=0)
    for other_id in other_class:
        gold_count_category[other_id] = 0
        predict_count_category[other_id] = 0
    print(gold_count_category)
    print
    print(predict_count_category)
    gold_count = 0
    predict_count = 0
    right_count = 0

    category_performance = []
    for i in xrange(0, num_class):
        if i in other_class:
            continue
        category_performance.append(
            calculate_prf(gold_count_category[i], predict_count_category[i],
                          filtered_confusion_matrix[i][i]))
        gold_count += gold_count_category[i]
        predict_count += predict_count_category[i]
        right_count += filtered_confusion_matrix[i][i]
    sys.stdout.write("%d\t%d\t%d\n" % (gold_count, predict_count, right_count))
    return calculate_prf(gold_count, predict_count, right_count)


def train():
    gold_label = [int(x.strip("\n")) for x in
                  open(FLAGS.gold_label_file).readlines()]
    labels = [line.split("\t") for line in open(FLAGS.label_file).readlines()]
    other_labels = filter(lambda x: x[1].find("其他") != -1, labels)
    other_labels = set([int(x[0]) for x in other_labels])
    print(other_labels)
    sys.stdout.write("init")
    hook = tf.train.ProfilerHook(save_steps=100, output_dir='./timeline')
    estimator = fasttext_estimator(FLAGS.model_dir)
    for i in xrange(0, FLAGS.num_epochs):
        train_input = inputs.fasttext_input_fn(tf.estimator.ModeKeys.TRAIN,
                                               FLAGS.train_records,
                                               FLAGS.batch_size)
        sys.stdout.write("start training epoch %d\n" % i)
        # estimator.train(input_fn=train_input, steps=10000, hooks=None)
        estimator.train(input_fn=train_input, hooks=None)
        sys.stdout.write("start evaluate at epoch %d\n" % i)
        predict_input = inputs.fasttext_input_fn(tf.estimator.ModeKeys.PREDICT,
                                                 FLAGS.eval_records, 1)
        predict = estimator.predict(input_fn=predict_input, hooks=None)
        eval_input = inputs.fasttext_input_fn(tf.estimator.ModeKeys.PREDICT,
                                                 FLAGS.eval_records, 1)
        estimator.evaluate(input_fn=eval_input, hooks=None)
        train_eval_input = inputs.fasttext_input_fn(tf.estimator.ModeKeys.PREDICT,
                                                 FLAGS.train_records, 1)
        estimator.evaluate(input_fn=train_eval_input, hooks=None)
        sys.stdout.write("eval done\n")
        sys.stdout.write("epoch %d precision recall f_score:\n" % i)
        sys.stdout.write("%f\t%f\t%f\t" % (calculate_performance(predict, gold_label, len(labels),
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
    if FLAGS.debug:
        tf.logging.set_verbosity(tf.logging.DEBUG)
    tf.app.run()
