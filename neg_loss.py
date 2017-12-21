import tensorflow as tf
import random as rnd
import numpy as np
from tensorflow.python.ops import nn_impl

class NegLoss():
    def __init__(self, probs_file_name):
        self.probs = self.read_probs_files(probs_file_name);
    def read_probs_files(self, file_name):
        probs = []
        with open(file_name) as f:
            while 1:
                line = f.readline().strip("\n")
                if not line:
                    break
                id_weight = line.split(",")
                probs.append(float(id_weight[1]))
        return probs
    def neg_loss(self,
             weights,
             biases,
             labels,
             inputs,
             num_sampled,
             num_classes,
             num_true=1,
             sampled_values=None,
             remove_accidental_hits=True,
             partition_strategy="mod",
             name="neg_loss"):
        logits, labels = nn_impl._compute_sampled_logits(
            weights=weights,
            biases=biases,
            labels=labels,
            inputs=inputs,
            num_sampled=num_sampled,
            num_classes=num_classes,
            num_true=num_true,
            sampled_values=sampled_values,
            subtract_log_q=False,
            remove_accidental_hits=remove_accidental_hits,
            partition_strategy=partition_strategy,
            name=name)
        sampled_losses = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=labels, logits=logits, name="sampled_losses")
        # sampled_losses is batch_size x {true_loss, sampled_losses...}
        # We sum out true and sampled losses.
        return nn_impl._sum_rows(sampled_losses)


#neg_loss = NegLoss("probs.txt")
#
#num_classes = 18
#class_name = tf.Variable(np.arange(4), dtype=tf.int64)
#classtest = tf.reshape(class_name, [4, 1])
#weight = tf.get_variable(name="weight", shape=[num_classes, 16], initializer=tf.contrib.layers.xavier_initializer())
#bias = tf.get_variable(name="bias", shape=[num_classes], initializer=tf.zeros_initializer)
#inputs = tf.get_variable(name="inputs", shape=[4, 16], initializer=tf.contrib.layers.xavier_initializer())
#with tf.Session() as sess:
#    sess.run(tf.global_variables_initializer())
#    print(classtest.eval())
#    print(weight.eval())
#    print()
#    loss = neg_loss.neg_loss(weight, bias, classtest, inputs, 5, num_classes)
#    print(loss.eval())
