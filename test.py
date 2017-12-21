import tensorflow as tf
import numpy as np
class_name = tf.Variable(np.arange(18), dtype=tf.int64)
classtest = tf.reshape(class_name, [18, 1])
one = tf.constant(1)
zero = tf.constant(0)
with tf.Session() as sess: 
    sess.run(tf.global_variables_initializer())
    for i in xrange(100):
	sampled = tf.nn.log_uniform_candidate_sampler(
	    classtest,
	    1,
	    1,
	    False,
	    18,
	)
        print(sampled)
        #print(sampled.sampled_candidates.eval())
        #print(sampled.true_expected_count.eval())
        #print(sampled.sampled_expected_count.eval())
        #print()
