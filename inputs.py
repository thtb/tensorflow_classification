# coding:utf8

import tensorflow as tf


# 只能处理定长数据，1.4trunk已经支持sparse，但用pip安装的还是只能定长
def dataset_input_fn(mode,
                     input_file,
                     batch_size,
                     num_epochs=1):
    def input_fn():
        def parser(record):
            keys_to_sample = {
                "features": tf.VarLenFeature(dtype=tf.int64),
                "label": tf.FixedLenFeature(shape=(1,), dtype=tf.int64,
                                            default_value=None),
                "real_len": tf.FixedLenFeature(shape=(1,), dtype=tf.int64,
                                               default_value=None)
            }
            parsed = tf.parse_single_example(record, keys_to_sample)
            features = tf.sparse_tensor_to_dense(parsed["features"],
                                                 default_value=0)
            return features, parsed["label"], parsed["real_len"]

        dataset = tf.data.TFRecordDataset(input_file)
        dataset = dataset.map(parser)
        if mode == tf.estimator.ModeKeys.TRAIN:
            dataset = dataset.shuffle(buffer_size=1000000)
        dataset = dataset.batch(batch_size)
        dataset = dataset.repeat(num_epochs)
        iterator = dataset.make_one_shot_iterator()

        features, labels, real_len = iterator.get_next()
        return features, labels

    return input_fn


# 可以处理变长和定长数据
def batch_reader_input_fn(mode, input_file, batch_size, num_epochs=1,
                          num_threads=1):
    def input_fn():
        keys_to_sample = {
            "features": tf.VarLenFeature(dtype=tf.int64),
            "label": tf.FixedLenFeature(shape=(1,),
                                        dtype=tf.int64,
                                        default_value=None)
        }
        randomize_input = False
        if mode == tf.estimator.ModeKeys.TRAIN:
            randomize_input = True
        sample = tf.contrib.learn.read_batch_features(
            input_file, batch_size, keys_to_sample, tf.TFRecordReader,
            num_epochs=num_epochs, reader_num_threads=num_threads,
            queue_capacity=1000000, randomize_input=randomize_input)
        features = tf.sparse_tensor_to_dense(sample["features"])
        label = sample["label"]
        return features, label

    return input_fn
