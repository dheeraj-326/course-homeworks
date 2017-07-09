'''
Created on 08-Jul-2017

@author: Dheeraj
'''
# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
import pickle

pickle_file = 'notMNIST.pickle'

with open(pickle_file, 'rb') as f:
        save = pickle.load(f)
        
        train_dataset = save['train_dataset']
        train_labels = save['train_labels']
        valid_dataset = save['valid_dataset']
        valid_labels = save['valid_labels']
        test_dataset = save['test_dataset']
        test_labels = save['test_labels']
        del save        # hint to help gc free up memory
        print('Training set', train_dataset.shape, train_labels.shape)
        print('Validation set', valid_dataset.shape, valid_labels.shape)
        print('Test set', test_dataset.shape, test_labels.shape)
        

image_size = 28
num_labels = 10

def reformat(dataset, labels):
        dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
        # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
        labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
        return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


# With gradient descent training, even this much data is prohibitive.
# Subset the training data for faster turnaround.
train_subset = 10000
num_hidden = 512

graph = tf.Graph()

logits = weights1 = biases1 = weights2 = biases2 = relulayer = None


def getoperation(dataset, graf):
    # Variables.
    # These are the parameters that we are going to be training. The weight
    # matrix will be initialized using random values following a (truncated)
    # normal distribution. The biases1 get initialized to zero.
    pass
    
#     return out_oper

with graph.as_default() as graf:

        # Input data.
        # Load the training, validation and test data into constants that are
        # attached to the graph.
        tf_train_dataset = tf.constant(train_dataset[:train_subset, :])
        tf_train_labels = tf.constant(train_labels[:train_subset])
        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)
        
        
        
        
        # Training computation.
        # We multiply the inputs with the weight matrix, and add biases1. We compute
        # the softmax and cross-entropy (it's one operation in TensorFlow, because
        # it's very common, and it can be optimized). We take the average of this
        # cross-entropy across all training examples: that's our loss.
        
        weights1 = tf.Variable(
        tf.truncated_normal([image_size * image_size, num_hidden]))        
        biases1 = tf.Variable(tf.zeros([num_hidden]))
            
        weights2 = tf.Variable(
            tf.truncated_normal([num_hidden, num_labels]))
        biases2 = tf.Variable(tf.zeros([num_labels]))
        
        logits = tf.matmul(tf_train_dataset, weights1) + biases1
        relulayer = tf.nn.relu(logits)
        out_oper = tf.matmul(relulayer, weights2) + biases2
#         out_oper = getoperation(tf_train_dataset, graf)
        loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=out_oper))
        
        # Optimizer.
        # We are going to find the minimum of this loss using gradient descent.
        optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
        
        # Predictions for the training, validation, and test data.
        # These are not part of training, but merely here so that we can report
        # accuracy figures as we train.
        train_prediction = tf.nn.softmax(out_oper)
        layer1_valid = tf.nn.relu(tf.matmul(tf_valid_dataset, weights1) + biases1)
        valid_out = tf.matmul(layer1_valid, weights2) + biases2
        valid_prediction = tf.nn.softmax(valid_out)
        layer1_test = tf.nn.relu(tf.matmul(tf_test_dataset, weights1) + biases1)
        test_out = tf.matmul(layer1_test, weights2) + biases2
        test_prediction = tf.nn.softmax(test_out)

num_steps = 801

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
                    / predictions.shape[0])

with tf.Session(graph=graph) as session:
    # This is a one-time operation which ensures the parameters get initialized as
    # we described in the graph: random weights1 for the matrix, zeros for the
    # biases1. 
    tf.global_variables_initializer().run()
    print('Initialized')
    for step in range(num_steps):
        # Run the computations. We tell .run() that we want to run the optimizer,
        # and get the loss value and the training predictions returned as numpy
        # arrays.
        
        _, l, predictions = session.run([optimizer, loss, train_prediction])
        
        if (step % 100 == 0):                  
#             print("Logits ", logts)
#             print("Relu ", myrelu)
#             print("Loss ", l)
            print('Loss at step %d: %f' % (step, l))
            print('Training accuracy: %.1f%%' % accuracy(
                predictions, train_labels[:train_subset, :]))
            # Calling .eval() on valid_prediction is basically like calling run(), but
            # just to get that one numpy array. Note that it recomputes all its graph
            # dependencies.
            print('Validation accuracy: %.1f%%' % accuracy(
                valid_prediction.eval(), valid_labels))
    print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))