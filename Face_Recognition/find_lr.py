import tensorflow as tf 
import numpy as np 
import os


from resnet50 import resnet50
from loss import arcface_loss

import matplotlib
matplotlib.use('TKAgg')
from matplotlib import pyplot as plt 


def parse_function(example_proto):
	features = {'image_raw': tf.FixedLenFeature([], tf.string),
		'label': tf.FixedLenFeature([], tf.int64)}
	features = tf.parse_single_example(example_proto, features)

	img = tf.image.decode_jpeg(features['image_raw'])
	img = tf.reshape(img, shape=(112, 112, 3))
	r, g, b = tf.split(img, num_or_size_splits=3, axis=-1)
	img = tf.concat([b, g, r], axis=-1)
	img = tf.cast(img, dtype=tf.float32)
	img = tf.subtract(img, 127.5)
	img = tf.multiply(img, 0.0078125)
	img = tf.image.random_flip_left_right(img)
	label = tf.cast(features['label'], tf.int64)
	return img, label

def generate_lr_boundaries():
	iterations = list(range(1, 101, 1))
	lr = []
	lr_start   = 1e-5
	lr_mult    = (1/1e-5) ** (1/100)
	lr.append(lr_start)
	for i in iterations:
		lr_start = lr_start * lr_mult
		lr.append(lr_start)
	print(len(iterations), len(lr))
	return iterations, lr


def train():
    num_classes = 85742   # 85164
    batch_size  = 64
    # ckpt_save_dir = '/data/ChuyuanXiong/backup/face_real403_ckpt'
    w_init_method = tf.contrib.layers.xavier_initializer(uniform=False)

    tfr = '/data/ChuyuanXiong/up/face/tfrecords/tran.tfrecords'
    dataset = tf.data.TFRecordDataset(tfr)
    dataset = dataset.map(parse_function)
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    images = tf.placeholder(tf.float32, [None, 112, 112, 3], name='image_inputs')
    labels = tf.placeholder(tf.int64,   [None, ], name='labels_inputs')

    emb = resnet50(images, is_training=True)

    logit = arcface_loss(embedding=emb, labels=labels, w_init=w_init_method, out_num=num_classes)
    inference_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=labels))

    global_step = tf.Variable(name='global_step', initial_value=0, trainable=False)
    inc_op = tf.assign_add(global_step, 1, name='increment_global_step')
    lr_steps, values = generate_lr_boundaries()
    lr = tf.train.piecewise_constant(global_step, lr_steps, values, name='lr_schedule')
    opt = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9)

    grads = opt.compute_gradients(inference_loss)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = opt.apply_gradients(grads, global_step=global_step)


    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True

    loss_vals = []

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(iterator.initializer)
        for i in range(100):
            image_train, label_train = sess.run(next_element)
            _, loss_val, _ = sess.run([train_op, inference_loss, inc_op], feed_dict={images: image_train, labels: label_train})
            loss_vals.append(loss_val)

        plt.figure()
        plt.xticks(np.log([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]), (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1))
        plt.xlabel('learning rate')
        plt.ylabel('loss')
        plt.plot(np.log(values[:len(loss_vals)]), loss_vals)
        plt.show()
        plt.savefig('output.jpg')


if __name__ == '__main__':
    train()
    # generate_lr_boundaries()