"""Train simple fastText-style classifier.

Inputs:
  words - text to classify
  ngrams - n char ngrams for each word in words
  labels - output classes to classify

Model:
  word embedding
  ngram embedding
  LogisticRegression classifier of embeddings to labels
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import inputs
import tensorflow as tf
from tensorflow.contrib.layers import feature_column
from tensorflow.contrib.learn.python.learn import learn_runner
from tensorflow.contrib.learn.python.learn.estimators.run_config import RunConfig
from tensorflow.contrib.learn.python.learn.metric_spec import MetricSpec

from neg_loss import NegLoss


tf.flags.DEFINE_string("train_records", None,
                       "Training file pattern for TFRecords, can use wildcards")
tf.flags.DEFINE_string("eval_records", None,
                       "Evaluation file pattern for TFRecords, can use wildcards")
tf.flags.DEFINE_string("predict_records", None,
                       "File pattern for TFRecords to predict, can use wildcards")
tf.flags.DEFINE_string("label_file", None, "File containing output labels")
tf.flags.DEFINE_string("vocab_file", None, "Vocabulary file, one word per line")
tf.flags.DEFINE_integer("vocab_size", None, "Number of words in vocabulary")
tf.flags.DEFINE_integer("num_oov_vocab_buckets", 20,
                        "Number of hash buckets to use for OOV words")
tf.flags.DEFINE_string("model_dir", ".",
                       "Output directory for checkpoints and summaries")
tf.flags.DEFINE_string("export_dir", None, "Directory to store savedmodel")

tf.flags.DEFINE_integer("embedding_dimension", 10, "Dimension of word embedding")
tf.flags.DEFINE_integer("attention_dimension", 16, "Dimension of attention")
tf.flags.DEFINE_boolean("use_ngrams", False, "Use character ngrams in embedding")
tf.flags.DEFINE_integer("num_ngram_buckets", 1000000,
                        "Number of hash buckets for ngrams")
tf.flags.DEFINE_integer("ngram_embedding_dimension", 10, "Dimension of word embedding")

tf.flags.DEFINE_boolean("fast", False, "Run fastest training without full experiment")
tf.flags.DEFINE_float("learning_rate", 0.001, "Learning rate for training")
tf.flags.DEFINE_float("clip_gradient", 5.0, "Clip gradient norm to this ratio")
tf.flags.DEFINE_integer("batch_size", 128, "Training minibatch size")
tf.flags.DEFINE_integer("train_steps", 1000,
                        "Number of train steps, None for continuous")
tf.flags.DEFINE_integer("eval_steps", 100, "Number of eval steps")
tf.flags.DEFINE_integer("num_epochs", None, "Number of training data epochs")
tf.flags.DEFINE_integer("checkpoint_steps", 1000,
                        "Steps between saving checkpoints")
tf.flags.DEFINE_integer("num_threads", 1, "Number of reader threads")
tf.flags.DEFINE_boolean("log_device_placement", False, "log where ops are located")
tf.flags.DEFINE_float("per_process_gpu_memory_fraction", 0.5, "per_process_gpu_memory_fraction")
tf.flags.DEFINE_boolean("debug", False, "Debug")
FLAGS = tf.flags.FLAGS

neg_loss = NegLoss("probs.txt")
def FeatureColumns(include_target):
    return inputs.FeatureColumns(
        include_target, FLAGS.use_ngrams, FLAGS.vocab_file, FLAGS.vocab_size,
        FLAGS.embedding_dimension, FLAGS.num_oov_vocab_buckets,
        FLAGS.ngram_embedding_dimension, FLAGS.num_ngram_buckets)


def InputFn(mode, input_file, num_epochs):
    return inputs.InputFn(
        mode, FLAGS.use_ngrams, input_file, FLAGS.vocab_file, FLAGS.vocab_size,
        FLAGS.embedding_dimension, FLAGS.num_oov_vocab_buckets,
        FLAGS.ngram_embedding_dimension, FLAGS.num_ngram_buckets,
        FLAGS.batch_size, num_epochs, FLAGS.num_threads)


def ContribEstimator(model_dir, config=None):
    num_classes = len(open(FLAGS.label_file).readlines())
    features = FeatureColumns(False)
    model = tf.contrib.learn.LinearClassifier(
        features, model_dir, n_classes=num_classes,
        optimizer=tf.train.AdamOptimizer(FLAGS.learning_rate),
        gradient_clip_norm=FLAGS.clip_gradient,
        config=config)
    return model


def BasicEstimator(model_dir, config=None):
    params = {
        "learning_rate": FLAGS.learning_rate,
    }
    def model_fn(features, labels, mode, params):
        text_lookup_table = tf.contrib.lookup.index_table_from_file(
            FLAGS.vocab_file, FLAGS.num_oov_vocab_buckets, FLAGS.vocab_size)
        text_ids = text_lookup_table.lookup(features["text"])
        text_embedding_w = tf.Variable(tf.random_uniform(
            [FLAGS.vocab_size + FLAGS.num_oov_vocab_buckets, FLAGS.embedding_dimension],
            -1.0 / FLAGS.embedding_dimension, 1.0 / FLAGS.embedding_dimension))

        #text_embedding = tf.nn.embedding_lookup(text_embedding_w, text_ids)
        #attention_matrix = tf.contrib.layers.fully_connected(inputs=text_embedding, num_outputs=FLAGS.attention_dimension, activation_fn=tf.nn.tanh)
        #attention_vector = tf.Variable(tf.random_uniform([FLAGS.attention_dimension], -1.0 / FLAGS.attention_dimension, 1.0 / FLAGS.attention_dimension))
        ##attention_vector = tf.Variable(tf.zeros([FLAGS.attention_dimension]))
        #alpha = tf.nn.softmax(tf.reduce_sum(tf.multiply(attention_matrix, attention_vector), axis=2, keep_dims=True), dim=1)
        #attention_embedding = tf.reduce_sum(tf.multiply(text_embedding, alpha), axis=1, keep_dims=False)
        #input_layer = attention_embedding

        text_embedding = tf.reduce_mean(tf.nn.embedding_lookup(
            text_embedding_w, text_ids), axis=-2)
#        text_embedding = tf.expand_dims(text_embedding, -2)
        input_layer = text_embedding

        if FLAGS.use_ngrams:
            ngram_hash = tf.string_to_hash_bucket(features["ngrams"],
                                                  FLAGS.num_ngram_buckets)
            ngram_embedding_w = tf.Variable(tf.random_uniform(
            [FLAGS.num_ngram_buckets, FLAGS.ngram_embedding_dimension], -1.0, 1.0))
            ngram_embedding = tf.reduce_mean(tf.nn.embedding_lookup(
                ngram_embedding_w, ngram_hash), axis=-2)
            ngram_embedding = tf.expand_dims(ngram_embedding, -2)
            input_layer = tf.concat([text_embedding, ngram_embedding], -1)
        num_classes = len(open(FLAGS.label_file).readlines())
        logits = tf.contrib.layers.fully_connected(
            inputs=input_layer, num_outputs=num_classes,
            activation_fn=None)
#        W = tf.get_variable(name="W", shape=[FLAGS.embedding_dimension, num_classes], initializer=tf.contrib.layers.xavier_initializer())
#        bias = tf.get_variable(name="bias", shape=[num_classes], initializer=tf.zeros_initializer)
#        logits = tf.matmul(input_layer, W) + bias
        logits = tf.expand_dims(logits, -2)
        predictions = tf.argmax(logits, axis=-1)
        loss, train_op = None, None
        metrics = {}
        if mode != tf.estimator.ModeKeys.PREDICT:
            sampled_values = tf.nn.fixed_unigram_candidate_sampler(
              true_classes=labels,
              num_true=1,
              num_sampled=6,
              unique=False,
              range_max=num_classes,
              vocab_file="probs.txt",)
            #labels = labels - 1
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
            #loss = tf.reduce_mean(tf.nn.nce_loss(
            #loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(
            #loss = tf.reduce_mean(neg_loss.neg_loss(
            #                      weights=tf.transpose(W),
            #                      labels=labels,
            #                      biases=bias,
            #                      inputs=input_layer,
            #                      num_sampled=6,
            #                      num_classes=num_classes,
            #                      sampled_values=sampled_values,
            #                      remove_accidental_hits=True))
            #labels_one_hot = tf.one_hot(labels, num_classes)
            #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            #    labels=labels_one_hot, logits=logits))
            # Squeeze dimensions from labels and switch to 0-offset
            labels = tf.squeeze(labels, -1)
            opt = tf.train.AdamOptimizer(params["learning_rate"])
            # opt = tf.train.AdagradOptimizer(params["learning_rate"])
            train_op = opt.minimize(loss, global_step=tf.train.get_global_step())
            metrics = {
                "accuracy": tf.metrics.accuracy(labels, predictions)
            }
        exports = {}
        if FLAGS.export_dir:
            probs = tf.nn.softmax(logits)
            exports["proba"] = tf.estimator.export.ClassificationOutput(scores=probs)
            exports["embedding"] = tf.estimator.export.RegressionOutput(value=text_embedding)
            exports[tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = \
                    tf.estimator.export.ClassificationOutput(scores=probs)
        return tf.estimator.EstimatorSpec(
            mode, predictions=predictions, loss=loss, train_op=train_op,
            eval_metric_ops=metrics, export_outputs=exports)
    gpu_option =tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.per_process_gpu_memory_fraction)
    session_config = tf.ConfigProto(
        log_device_placement=FLAGS.log_device_placement,
        gpu_options=gpu_option)
    config = tf.contrib.learn.RunConfig(
        save_checkpoints_secs=None,
        save_checkpoints_steps=FLAGS.checkpoint_steps,
        session_config=session_config)
    return tf.estimator.Estimator(model_fn=model_fn, model_dir=model_dir,
                                  params=params, config=config)


def Experiment(output_dir):
    """Construct an experiment for training and evaluating a model.
    Saves checkpoints and exports the model for tf serving.
    """
    mode = tf.estimator.ModeKeys.TRAIN
    config = tf.contrib.learn.RunConfig(
        save_checkpoints_secs=None,
        save_checkpoints_steps=FLAGS.checkpoint_steps)
    train_input = InputFn(mode, FLAGS.train_records)
    eval_input = InputFn(mode, FLAGS.eval_records)
    estimator = ContribEstimator(output_dir, config)
#    export_strategies = []
#    if FLAGS.export_dir:
#        export_strategies.append(MakeExportStrategy())
    experiment = tf.contrib.learn.Experiment(
        estimator=estimator,
        train_input_fn=train_input,
        eval_input_fn=eval_input,
        train_steps=FLAGS.train_steps,
        eval_steps=FLAGS.eval_steps,
        eval_delay_secs=0,
        eval_metrics=None,
        continuous_eval_throttle_secs=10,
        min_eval_frequency=1000,
        train_monitors=None)
    return experiment


def FastTrain():
    sys.stderr.write("Testing")
    estimator = BasicEstimator(FLAGS.model_dir)
    train_input = InputFn(tf.estimator.ModeKeys.TRAIN, FLAGS.train_records, FLAGS.num_epochs)
    sys.stderr.write("STARTING TRAIN")
    #estimator.train(input_fn=train_input, steps=10000, hooks=None)
    estimator.train(input_fn=train_input, hooks=None)
    sys.stderr.write("TRAIN COMPLETE")
    sys.stderr.write("EVALUATE")
    #eval_metrics = { "accuracy": tf.metrics.accuracy(labels, predictions) }
    #result = estimator.evaluate(input_fn=eval_input, steps=FLAGS.train_steps, hooks=None)
    #result = estimator.evaluate(input_fn=eval_input, hooks=None)
    #print(result)
    predict_input = InputFn(tf.estimator.ModeKeys.PREDICT, FLAGS.eval_records, 1)
    predict = estimator.predict(input_fn=predict_input, hooks=None)
    for n in predict:
        print(n)
    eval_input = InputFn(tf.estimator.ModeKeys.EVAL, FLAGS.eval_records, 1)
    result = estimator.evaluate(input_fn=eval_input, hooks=None)
    sys.stderr.write(result)
    sys.stderr.write("DONE")
    if FLAGS.export_dir:
        sys.stderr.write("EXPORTING")
        estimator.export_savedmodel(FLAGS.export_dir, ExportFn())


def ExportFn():
    features = {
        "text": tf.placeholder(dtype=tf.string, shape=[None], name='text')
    }
    if FLAGS.use_ngrams:
        features["ngrams"] = tf.placeholder(
            dtype=tf.string, shape=[None], name='ngrams')
    return tf.estimator.export.build_raw_serving_input_receiver_fn(features)


def main(_):
    if not FLAGS.vocab_size:
        FLAGS.vocab_size = len(open(FLAGS.vocab_file).readlines())
    if FLAGS.fast:
        FastTrain()
    elif FLAGS.train_records:
        if FLAGS.export_dir:
            tf.logging.warn(
                "Exporting savedmodels not supported for contrib experiment, --nofast")
        learn_runner.run(experiment_fn=Experiment, output_dir=FLAGS.model_dir)

if __name__ == '__main__':
    if FLAGS.debug:
        tf.logging.set_verbosity(tf.logging.DEBUG)
    tf.app.run()
