import tensorflow as tf


def disordered_input_fn(mode,
                        input_file,
                        batch_size,
                        num_epochs=1):
    if num_epochs <= 0:
        num_epochs = 1

    def input_fn():
        def parser(record):
            keys_to_features = {
                "features": tf.VarLenFeature(dtype=tf.int64),
                "label": tf.FixedLenFeature(shape=(1,), dtype=tf.int64,
                                            default_value=None),
                "real_len": tf.FixedLenFeature(shape=(1,), dtype=tf.int64,
                                               default_value=None)
            }
            parsed = tf.parse_single_example(record, keys_to_features)
            return tf.sparse_tensor_to_dense(parsed["features"],
                                             default_value=" "), parsed["label"]

        dataset = tf.data.TFRecordDataset(input_file)
        dataset = dataset.map(parser)
        dataset = dataset.shuffle(buffer_size=1000000)
        dataset = dataset.batch(batch_size)
        dataset = dataset.repeat(num_epochs)
        iterator = dataset.make_one_shot_iterator()

        features, labels = iterator.get_next()
        return features, labels

    return input_fn


def fasttext_input_fn(mode, input_file, batch_size, num_epochs=None,
                      num_threads=3):
    if num_epochs <= 0:
        num_epochs = 1

    def input_fn():
        keys_to_sample = {"features": tf.VarLenFeature(dtype=tf.int64)}
        keys_to_sample["label"] = tf.FixedLenFeature(shape=(1,),
                                                       dtype=tf.int64,
                                                       default_value=None)

        randomize_input = False
        if mode == tf.estimator.ModeKeys.TRAIN :
            randomize_input = True
        sample = tf.contrib.learn.read_batch_features(
            input_file, batch_size, keys_to_sample, tf.TFRecordReader,
            num_epochs=num_epochs, reader_num_threads=num_threads,
            queue_capacity=10000, randomize_input=randomize_input)
        features = tf.sparse_tensor_to_dense(sample["features"])
        label = sample["label"]
        return features, label

    return input_fn
