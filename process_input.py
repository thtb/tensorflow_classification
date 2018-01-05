# coding:utf8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
from collections import Counter

import tensorflow as tf

# 输入数据参数
tf.flags.DEFINE_string("feature_file_prefix", "", "Feature file prefix.")
tf.flags.DEFINE_string("tfrecord_output_dir", ".",
                       "Directory to store tfrecord, vocab and label.")
tf.flags.DEFINE_integer("min_feature_count", 0, "min count of feature occur.")
tf.flags.DEFINE_integer("feature_field", 1,
                        "feature file in file, separated by '\t'")

# cnn模型参数
tf.flags.DEFINE_integer("sequence_length", 2000,
                        "Max length of features for one sample.")
FLAGS = tf.flags.FLAGS


def generate_vocab_and_label_map(train_feature_file, vocab_file, label_file,
                                 is_fixed_length=False, max_length=2000,
                                 min_feature_count=1):
    """Parse features
    Input format: label\t(feature )+\tcomments
    Label format: cate1--cate2--cate3
    """
    print("parsing feature file %s" % train_feature_file)
    feature_count = Counter()
    label_count = Counter()
    sample_size = 0
    label_index = 0
    label_map = {}
    id_label_map = {}
    feature_field = int(FLAGS.feature_field)
    with open(train_feature_file) as f:
        for line in f:
            content = line.split('\t')

            sample_size += 1
            label_string = content[0]
            if label_string in label_map:
                label = label_map[label_string]
            else:
                label = label_index
                label_map[label_string] = label_index
                id_label_map[label_index] = label_string
                label_index += 1
            features = content[feature_field].split(" ")

            if is_fixed_length and len(features) >= max_length:
                features = features[0:max_length]

            feature_count.update(features)
            label_count.update([label])

    print("sample size: %d" % sample_size)
    print("label size: %d" % label_index)

    feature_map = {}
    feature_index = 0
    feature_list = feature_count.most_common()
    if is_fixed_length:
        feature_list = [("_UNK", 10000), ("_PAD", 10000)] + feature_list
    with open(vocab_file, "w") as f:
        for feature in feature_list:
            if feature[1] > min_feature_count:
                feature_map[feature[0]] = feature_index
                feature_index += 1
                f.write("%s\t%d\n" % (feature[0], feature[1]))
    with open(label_file, "w") as f:
        for label in label_count.most_common():
            f.write("%s\t%s\n" % (label[0], id_label_map[label[0]]))

    return feature_map, label_map


def get_feature_list(feature_map, features, is_fixed_length,
                     max_length):
    # for rnn length should be fix
    if is_fixed_length:
        if len(features) > max_length:
            features = features[0:max_length]
        features = [
            feature_map[x] if x in feature_map else feature_map["_UNK"]
            for x in features]
        real_len = len(features)
        if len(features) < max_length:
            features = features + [feature_map["_PAD"]] * (
                    max_length - len(features))
    # for fasttext only keep feature in feature_map
    else:
        features = filter(lambda x: x in feature_map, features)
        features = [feature_map[x] for x in features]
        real_len = len(features)
    return features, real_len


def to_tf_record(feature_file, feature_map, label_map, max_vocab_size=-1,
                 is_fixed_length=False, max_length=2000):
    """Parse features
    Input format: label\t(feature )+\tconments
    Label format: cate1--cate2--cate3
    """
    print("parsing feature file %s" % feature_file)
    sample_size = 0
    samples = []
    feature_field = int(FLAGS.feature_field)
    with open(feature_file) as f:
        for line in f:
            content = line.split('\t')
            label_string = content[0]
            if label_string not in label_map:
                print("wrong label of line: " + line)
                continue

            sample_size += 1
            label = label_map[label_string]
            features = content[feature_field].split(" ")
            features, real_len = get_feature_list(feature_map, features,
                                                  is_fixed_length, max_length)
            samples.append({
                "features": features,
                "label": label,
                "real_len": real_len
            })

    print("feature file %s has sample %d" % (feature_file, sample_size))
    return samples


def write_tf_record(samples, output_file):
    """write samples in TFRecord format.
    """
    writer = tf.python_io.TFRecordWriter(output_file)
    for sample in samples:
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

    vocab_file = os.path.join(FLAGS.output_dir,
                              FLAGS.feature_file_prefix + ".vocab")
    label_file = os.path.join(FLAGS.output_dir,
                              FLAGS.feature_file_prefix + ".labels")
    feature_map, label_map = generate_vocab_and_label_map(train_feature_file,
                                                          vocab_file,
                                                          label_file,
                                                          FLAGS.is_fixed_length,
                                                          FLAGS.max_length,
                                                          FLAGS.min_feature_count)

    train_samples = to_tf_record(train_feature_file, feature_map, label_map, -1,
                                 FLAGS.is_fixed_length, FLAGS.max_length)
    output_file = os.path.join(FLAGS.output_dir,
                               train_feature_file + ".tfrecord")
    write_tf_record(train_samples, output_file)
    with open("article_train.gold_label", "w") as f:
        for sample in train_samples:
            str_sample = [str(x) for x in sample["features"]]
            f.write("%d\t%s\n" % (sample["label"], "_".join(str_sample)))

    test_samples = to_tf_record(test_feature_file, feature_map, label_map, -1,
                                FLAGS.is_fixed_length, FLAGS.max_length)
    output_file = os.path.join(FLAGS.output_dir,
                               test_feature_file + ".tfrecord")
    write_tf_record(test_samples, output_file)

    with open(test_feature_file + ".gold_label", "w") as f:
        for sample in test_samples:
            f.write("%d\n" % sample["label"])


if __name__ == '__main__':
    tf.app.run()
