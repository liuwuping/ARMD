import csv
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
from sklearn.model_selection import train_test_split

tf.disable_v2_behavior()

# Read the data.
raw = []
with open('path') as f:
    reader = csv.reader(f)
    for row in reader:
        raw.append(row)
        lenth = len(row)
    raw_data = pd.DataFrame(raw[1:])
raw_features = raw_data.drop([0], axis=1)
dataset = raw_data.values
dataset = dataset.astype(np.float64)
totalX = raw_features.values.astype(np.float64)
totalY1 = raw_data[0].values.astype(np.int64)
totalY = []
for i in totalY1:
    # HC dataset.
    if i == 0:
        totalY.append([1, 0, 0, 0])
    # AMI dataset.
    elif i == 1:
        totalY.append([0, 1, 0, 0])
    # AMICVD dataset.
    elif i == 2:
        totalY.append([0, 0, 1, 0])
    # AMICVDD dataset.
    elif i == 3:
        totalY.append([0, 0, 0, 1])
print(totalY)
totalY = np.array(totalY)
X_train, X_test, y_train, y_test = train_test_split(totalX, totalY, test_size=0.25, random_state=0, stratify=totalY)
x = tf.placeholder(tf.float32, [None, lenth - 1], name='X')
y = tf.placeholder(tf.float32, [None, 4], name='Y')
# Number of neuron nodes in the first layer.
H1_NN = 64
# Number of neuronal nodes in layer 2.
H2_NN = 256
# Number of neuron nodes in layer 3.
H3_NN = 128
# First layer.
W1 = tf.Variable(tf.truncated_normal([lenth - 1, H1_NN], stddev=0.1))
b1 = tf.Variable(tf.zeros(H1_NN))
# Second layer.
W2 = tf.Variable(tf.truncated_normal([H1_NN, H2_NN], stddev=0.1))
b2 = tf.Variable(tf.zeros(H2_NN))
# Third layer.
W3 = tf.Variable(tf.truncated_normal([H2_NN, H3_NN], stddev=0.1))
b3 = tf.Variable(tf.zeros(H3_NN))
# Output layer.
W4 = tf.Variable(tf.truncated_normal([H3_NN, 4], stddev=0.1))
b4 = tf.Variable(tf.zeros(4))
# Outcome.
Y1 = tf.nn.relu(tf.matmul(x, W1) + b1)
Y2 = tf.nn.relu(tf.matmul(Y1, W2) + b2)
Y3 = tf.nn.relu(tf.matmul(Y2, W3) + b3)
forward = tf.matmul(Y3, W4) + b4
pred = tf.nn.softmax(forward)
# The loss function uses cross entropy.
loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=forward, labels=y))
# Setting training parameters.
train_epochs = 1000

total_batch = int(len(X_train))
learning_rate = 0.0005
display_step = 50
# Optimizer.
opimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss_function)
# Defining accuracy.
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(pred, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# Start training
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
for epochs in range(train_epochs):
    for batch in range(1):
        xs, ys = X_train, y_train
        sess.run(opimizer, feed_dict={x: xs, y: ys})
    loss, acc = sess.run([loss_function, accuracy],
                         feed_dict={
                             x: X_train,
                             y: y_train})
    if (epochs + 1) % display_step == 0:
        epochs += 1
        print("Train Epoch:", epochs, "Loss=", loss, "Accuracy=", acc)
ret = sess.run(accuracy, feed_dict={x: X_test, y: y_test})
print('DNN Accuracy:%.3f' % ret)
