import tensorflow as tf 
import numpy as np 

import matplotlib
matplotlib.use('TKAgg')
from matplotlib import pyplot as plt 

class DataLoader:
    def __init__(self, config=None):
        '''
        DataLoader: Initialize the data loader and do some preprocessing.
        '''
        from tensorflow.examples.tutorials.mnist import input_data
        self.mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    def generate_batch(self, batch_size):
        return self.mnist.train.next_batch(batch_size)


class Model:
    def __init__(self, config=None):
        '''
        Model: Put your network here.
        '''
        pass
    
    def network(self, input_tensor):
        print('input_tensor :', input_tensor)
        x = tf.layers.dense(input_tensor, 256, name='Hidden')
        x = tf.layers.dense(x, 10, name='Output')
        print('network :', x)
        return x

class LearningRate:
    def __init__(self, config=None):
        pass
    
    def lr_schedule(self):
        iterations = list(range(1, 101, 1))
        lr = []
        lr_start   = 1e-5
        lr_mult    = (1/1e-5) ** (1/100)
        lr.append(lr_start)
        for i in iterations:
            lr_start = lr_start * lr_mult
            lr.append(lr_start)
        return iterations, lr
    
    def plot_lr_schedule(self, save_path='lr_schedule.jpg'):
        iterations, lr = self.lr_schedule()
        plt.figure()
        plt.xlabel('Iterations')
        plt.ylabel('Learning Rate')
        plt.plot(iterations, lr[:len(iterations)])
        plt.savefig(save_path)


class Main:
    def __init__(self, config=None):
        '''
        Main: Initialize the sess and run the computation graph.
        Steps:
            Input->Placeholder
            Model->Network
            Loss ->Loss
            Sess ->Session
            Batch->Generate Batch
            Compute->Compute the loss value
        '''

        self.x_input = tf.placeholder(tf.float32, [None, 784], 'x_input')
        self.y_input = tf.placeholder(tf.int64, [None, 10], 'y_input')

        self.config = {
            'batch_size': 64
        }

    def loss(self, logits, labels):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))


    def main(self):
        config = tf.ConfigProto()
        config.allow_soft_placement = True
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            dl        = DataLoader()
            model     = Model().network(self.x_input)
            lr_method = LearningRate()

            
            loss = self.loss(model, self.y_input)

            global_step = tf.Variable(name='global_step', initial_value=0, trainable=False)
            inc_op = tf.assign_add(global_step, 1, name='increment_global_step')
            lr_steps, values = lr_method.lr_schedule()
            lr_method.plot_lr_schedule()
            lr = tf.train.piecewise_constant(global_step, lr_steps, values, name='lr_schedule')
            opt = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9)
            train_op = opt.minimize(loss)

            sess.run(tf.global_variables_initializer())
            loss_vals = []
            for i in range(100):
                batch_x, batch_y = dl.generate_batch(self.config['batch_size'])
                print(batch_x.shape, batch_y.shape)
                _, loss_val, _ = sess.run([train_op, loss, inc_op], feed_dict={self.x_input: batch_x, self.y_input: batch_y})
                loss_vals.append(loss_val)
            
            plt.figure()
            plt.xticks(np.log([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]), (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1))
            plt.xlabel('learning rate')
            plt.ylabel('loss')
            plt.plot(np.log(values[:len(loss_vals)]), loss_vals)
            plt.show()
            plt.savefig('output.jpg')

if __name__ == '__main__':
    main = Main()
    main.main()



