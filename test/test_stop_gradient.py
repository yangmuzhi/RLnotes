import tensorflow as tf
import numpy as np

x = tf.placeholder(tf.float32,[3,2])
y = tf.placeholder(tf.float32,[3,4])
w1 = tf.Variable(tf.ones([2,3]))
w2 = tf.Variable(tf.ones([3,4]))
hidden = tf.stop_gradient(tf.matmul(x,w1))
output = tf.matmul(hidden,w2)
loss = output - y
optimizer = tf.train.GradientDescentOptimizer(1).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("*****before gradient descent*****")
    print("w1---\n",w1.eval(),"\n","w2---\n",w2.eval())
    w1_,w2_,_ = sess.run([w1,w2,optimizer],feed_dict = {x:np.random.normal(size = (3,2)),y:np.random.normal(size = (3,4))})
    print("*****after gradient descent*****")
    print("w1---\n",w1_,"\n","w2---\n",w2_)
