from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import tensorflow as tf
from collections import Counter

tf.flags.DEFINE_string("feature_file_prefix", "", "Feature file prefix")
tf.flags.DEFINE_string("output_dir", ".",
                       "Directory to store tfrecord, vocab and label.")
tf.flags.DEFINE_boolean("is_fixed_length", False, "If set to be true, than ")
tf.flags.DEFINE_integer("max_length", 2000,
                        "Max length of features for one sample")
tf.flags.DEFINE_integer("num_shards", 1,
                        "Number of output_files to create.")
tf.flags.DEFINE_integer("label_level", 1,
                        "Level of label to use")
FLAGS = tf.flags.FLAGS


def generate_vocab_and_label_map(train_feature_file, vocab_file, label_file,
                                 is_fixed_length=False, max_length=2000,
                                 level=1):
    """Parse features
    Input format: label\t(feature )+\tcomments
    Label format: cate1--cate2--cate3
    """
    print("parsing feature file %s" % train_feature_file)
    category_separator = "--"
    feature_count = Counter()
    label_count = Counter()
    sample_size = 0
    label_index = 0
    label_map = {}
    id_label_map = {}
    with open(train_feature_file) as f:
        for line in f:
            content = line.split('\t')
            label_taxonomy = content[0].split(category_separator)
            if len(label_taxonomy) < int(level):
                continue

            sample_size += 1
            label_string = category_separator.join(label_taxonomy[0:level])
            if label_string in label_map:
                label = label_map[label_string]
            else:
                label = label_index
                label_map[label_string] = label_index
                id_label_map[label_index] = label_string
                label_index += 1
            features = content[1].split(" ")

            if is_fixed_length and len(features) >= max_length:
                features = features[0:max_length]

            feature_count.update(features)
            label_count.update([label])

            if is_fixed_length and len(features) < max_length:
                for i in xrange(0, max_length - len(features)):
                    features.append("_PAD")

    print("sample size: %d" % sample_size)
    print("label size: %d" % label_index)

    feature_map = {}
    feature_index = 0
    feature_list = feature_count.most_common()
    if is_fixed_length:
        feature_list = [("_UNK", 0), ("_PAD", 0)] + feature_list
    with open(vocab_file, "w") as f:
        for feature in feature_list:
            feature_map[feature[0]] = feature_index
            feature_index += 1
            f.write(feature[0] + '\n')
    with open(label_file, "w") as f:
        for label in label_count.most_common():
            f.write("%s\t%s\n" % (label[0], id_label_map[label[0]]))

    return feature_map, label_map


def to_tf_record(feature_file, feature_map, label_map, max_vocab_size=-1,
                 is_fixed_length=False, max_length=2000, level=1):
    """Parse features
    Input format: label\t(feature )+\tconments
    Label format: cate1--cate2--cate3
    """
    print("parsing feature file %s" % feature_file)
    category_separator = "--"
    sample_size = 0
    samples = []
    with open(feature_file) as f:
        for line in f:
            content = line.split('\t')
            label_taxonomy = content[0].split(category_separator)
            if len(label_taxonomy) < int(level):
                continue
            label_string = category_separator.join(label_taxonomy[0:level])
            label = -1
            if label_string not in label_map:
                print("wrong label of line: " + line)
                continue

            sample_size += 1
            label = label_map[label_string]
            features = content[1].split(" ")

            # for rnn length should be fix
            if is_fixed_length:
                if len(features) > max_length:
                    features = features[0:max_length]
                features = [
                    feature_map[x] if x in feature_map else feature_map["_UNK"]
                    for x in features]
                real_len.append(len(features))
                if len(features) < max_length:
                    features = features + [feature_map["_PAD"]] * (
                            max_length - len(features))

            # for fasttext only keep feature in feature_map
            else:
                features = filter(lambda x: x in feature_map, features)

                features = [feature_map[x] for x in features]
            real_len = len(features)
            samples.append({
                "features": features,
                "label": label,
                "real_len": real_len
            })

    print("feature file %s has sample %d" % (feature_file, sample_size))
    return samples


def write_tf_record(samples, output_file, num_shards=1):
    """write samples in TFRecord format.
    """
    shard = 0
    num_per_shard = len(samples) / num_shards + 1
    for n, sample in enumerate(samples):
        if n % num_per_shard == 0:
            shard += 1
            writer = tf.python_io.TFRecordWriter(output_file + '-%d-of-%d' %
                                                 (shard, num_shards))
        tf_record = tf.train.Example()
        for i in xrange(0, len(sample["features"])):
            tf_record.features.feature["features"].int64_list.value.append(
                sample["features"][i])
        tf_record.features.feature["label"].int64_list.value.append(
            sample["label"])
        tf_record.features.feature["real_len"].int64_list.value.append(
            sample["real_len"])
        writer.write(tf_record.SerializeToString())


def main(_):
    train_feature_file = FLAGS.feature_file_prefix + ".train"
    test_feature_file = FLAGS.feature_file_prefix + ".test"
    label_level = ".level" + str(FLAGS.label_level)

    vocab_file = os.path.join(FLAGS.output_dir,
                              FLAGS.feature_file_prefix
                              + label_level + ".vocab")
    label_file = os.path.join(FLAGS.output_dir,
                              FLAGS.feature_file_prefix
                              + label_level + ".labels")
    feature_map, label_map = generate_vocab_and_label_map(train_feature_file,
                                                          vocab_file,
                                                          label_file,
                                                          FLAGS.is_fixed_length,
                                                          FLAGS.max_length,
                                                          FLAGS.label_level)

    train_samples = to_tf_record(train_feature_file, feature_map, label_map, -1,
                                 FLAGS.is_fixed_length, FLAGS.max_length,
                                 FLAGS.label_level)
    output_file = os.path.join(FLAGS.output_dir,
                               train_feature_file + label_level + ".tfrecord")
    write_tf_record(train_samples, output_file, FLAGS.num_shards)

    test_samples = to_tf_record(test_feature_file, feature_map, label_map, -1,
                                FLAGS.is_fixed_length, FLAGS.max_length,
                                FLAGS.label_level)
    output_file = os.path.join(FLAGS.output_dir,
                               test_feature_file + label_level + ".tfrecord")
    write_tf_record(test_samples, output_file, 1)

    with open(test_feature_file + ".gold_label", "w") as f:
        for sample in test_samples:
            f.write("%d\n" % sample["label"])


if __name__ == '__main__':
    tf.app.run()
