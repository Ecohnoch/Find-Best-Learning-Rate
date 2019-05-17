import tensorflow as tf 

kernel_size = [3, 3]
input_channels  = 32
output_channels = 64

conv_shape = [kernel_size[0], kernel_size[1], input_channels, output_channels]

initializer_random_normal = tf.random_normal_initializer()
initializer_truncated_normal = tf.truncated_normal_initializer()

initializer_random_uniform = tf.random_uniform_initializer(minval=-10, maxval=10)
initializer_unit_uniform = tf.uniform_unit_scaling_initializer()
msra = tf.variance_scaling_initializer()
orth = tf.orthogonal_initializer() 
glorot_uniform = tf.glorot_uniform_initializer()
glorot_normal = tf.glorot_normal_initializer()


weight_random_normal    = tf.get_variable('weight_random_normal', shape=conv_shape, initializer=initializer_random_normal)
weight_truncated_normal = tf.get_variable('weight_truncated_normal', shape=conv_shape, initializer=initializer_truncated_normal)
weight_random_uniform   = tf.get_variable('weight_random_uniform', shape=conv_shape, initializer=initializer_random_uniform)
weight_unit_uniform     = tf.get_variable('weight_uniform_unit_scaling_initializer', shape=conv_shape, initializer=initializer_unit_uniform)
weight_msra              = tf.get_variable('weight_variance_scaling_initializer', shape=conv_shape, initializer=msra)
weight_glorot_uniform = tf.get_variable('weight_glorot_uniform', shape=conv_shape, initializer=glorot_uniform)
weight_glorot_normal = tf.get_variable('weight_glorot_normal', shape=conv_shape, initializer=glorot_normal)




tf.summary.histogram('cnn/weight_random_normal', weight_random_normal)
tf.summary.histogram('cnn/weight_truncated_normal', weight_truncated_normal)
tf.summary.histogram('cnn/weight_random_uniform', weight_random_uniform)
tf.summary.histogram('cnn/weight_uniform_unit_scaling_initializer', weight_unit_uniform)
tf.summary.histogram('cnn/weight_variance_scaling_initializer', weight_msra)
tf.summary.histogram('cnn/weight_glorot_uniform', weight_glorot_uniform)
tf.summary.histogram('cnn/weight_glorot_normal', weight_glorot_normal)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter('./summary')
    summaries = tf.summary.merge_all()

    sum_val = sess.run(summaries)
    writer.add_summary(sum_val)
