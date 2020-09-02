# One-Class Neural Network Algorithm

import numpy as np
#import tensorflow as tf
from tempfile import TemporaryFile

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

value = TemporaryFile()
def init_weights(shape):
    """ Weight initialization """
    weights = tf.random_normal(shape,mean=0, stddev=0.01)
    return tf.Variable(weights)

def forwardprop(X, w_1, w_2):
    """
    Forward-propagation.
    IMPORTANT: yhat is not softmax since TensorFlow's softmax_cross_entropy_with_logits() does that internally.
    """
    X = tf.cast(X, tf.float32)
    w_1 = tf.cast(w_1, tf.float32)
    w_2 = tf.cast(w_2, tf.float32)
    h    = tf.nn.sigmoid(tf.matmul(X, w_1))  # The \sigma function
    yhat = tf.matmul(h, w_2)  # The \varphi function
    return yhat


def nnScore(X, w, V):
    g   = lambda x : 1/(1 + tf.exp(-x))
    X = tf.cast(X, tf.float32)
    w = tf.cast(w, tf.float32)
    V = tf.cast(V, tf.float32)
    return tf.matmul(g((tf.matmul(X, w))), V)

def relu(x):
    y = x
#     y[y < 0] = 0
    return y

def ocnn_obj(theta, X, nu, w1, w2 ,r):
    w = w1
    V = w2
    
    X = tf.cast(X, tf.float32)
    w = tf.cast(w1, tf.float32)
    V = tf.cast(w2, tf.float32)

    term1 = 0.5  * tf.reduce_sum(w**2)
    term2 = 0.5  * tf.reduce_sum(V**2)
    term3 = 1/nu * tf.reduce_mean(relu(r - nnScore(X, w, V)))
    term4 = -r

    return term1 + term2 + term3 + term4

def tf_OneClass_NN_sigmoid_Train(data_train):

    tf.reset_default_graph()
    train_X = data_train

    RANDOM_SEED = 42
    tf.set_random_seed(RANDOM_SEED)

     # Layer's sizes
    x_size = train_X.shape[1]   # Number of input nodes: 4 features and 1 bias
    h_size = 100                # Number of hidden nodes
    y_size = 1   # Number of outcomes (3 iris flowers)
    D = x_size
    K = h_size

    theta = np.random.normal(0, 1, K + K*D + 1)
    rvalue = np.random.normal(0,1,(len(train_X),y_size))
    nu = 0.01
    
    # Symbols
    X = tf.placeholder("float32", shape=[None, x_size])
    r = tf.get_variable("r", dtype=tf.float32,shape=(),trainable=False)

    # Weight initializations
    w_1 = init_weights((x_size, h_size))
    w_2 = init_weights((h_size, y_size))
    saver = tf.train.Saver()
    # Forward propagation
    yhat    = forwardprop(X, w_1, w_2)
    predict = tf.argmax(yhat, axis=1)

    # Backward propagation
    cost    = ocnn_obj(theta, X, nu, w_1, w_2, r)
    updates = tf.train.GradientDescentOptimizer(0.001).minimize(cost)

    # Run SGD
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    rvalue = 0.01
    for epoch in range(300):
            # Train with each example
                sess.run(updates, feed_dict={X: train_X,r:rvalue})
                rvalue = nnScore(train_X, w_1, w_2)
                with sess.as_default():
                    rvalue = rvalue.eval()
                    rvalue = np.percentile(rvalue,q=100*0.04)
#                print("Epoch = %d, r = %f"
#                  % (epoch + 1,rvalue))
                
    train = nnScore(train_X, w_1, w_2)
    with sess.as_default():
        arrayTrain = train.eval()
    rstar =rvalue
#    print(sess.run(w_1))
#    print(rvalue)
    save_path = saver.save(sess, "one_class/NL.ckpt")
    sess.close()
#    print ("Session Closed!!!")
    np.save('value',rstar)
    
    pos_decisionScore = arrayTrain-rstar

    return [pos_decisionScore]


def tf_OneClass_NN_sigmoid_Test(data_test):

    tf.reset_default_graph()
    test_X = data_test
     # Layer's sizes
    x_size = test_X.shape[1]   # Number of input nodes: 4 features and 1 bias
    h_size = 100                # Number of hidden nodes
    y_size = 1   # Number of outcomes (3 iris flowers)
    
#    rvalue = np.random.normal(0,1,(len(test_X),y_size))
#    np.load(outfile)
    # Weight initializations
    w_1 = init_weights((x_size, h_size))
    w_2 = init_weights((h_size, y_size))


    saver = tf.train.Saver()
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    saver.restore(sess, "one_class/NL.ckpt")
    test = nnScore(test_X, w_1, w_2)
    with sess.as_default():
        arrayTest = test.eval()
        
#    print(sess.run(w_1))    
#    print(outfile)
    sess.close()
#    print ("Session Closed!!!")
    value = np.load('value.npy')
#    print(value)
    rstar = value
    neg_decisionScore = arrayTest-rstar

    return [neg_decisionScore]