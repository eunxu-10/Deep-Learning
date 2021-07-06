import tensorflow as tf
import numpy as np
from sklearn.metrics import f1_score

learning_rate = 0.001
training_epochs = 50
batch_size = 128

keep_prob = tf.placeholder(tf.float32)
KEEP_PROB = 0.7

def batch_data(shuffled_idx, batch_size, data, labels, start_idx):
    idx = shuffled_idx[start_idx:start_idx+batch_size]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

def build_CNN_classifier(x):
    x_image = x

    W1 = tf.get_variable(name="W1", shape=[5, 5, 3, 64], initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable(name="b1", shape=[64], initializer=tf.contrib.layers.xavier_initializer())
    c1 = tf.nn.conv2d(x_image, W1, strides=[1, 1, 1, 1], padding='SAME')
    l1 = tf.nn.bias_add(c1, b1)
    l1 = tf.layers.batch_normalization(l1)
    l1 = tf.nn.relu(l1)
    l1_pool = tf.nn.max_pool(l1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    W2 = tf.get_variable(name="W2", shape = [5,5,64,128], initializer= tf.contrib.layers.xavier_initializer())
    b2 = tf.get_variable(name="b2", shape = [128], initializer = tf.contrib.layers.xavier_initializer())
    c2 = tf.nn.conv2d(l1_pool, W2, strides=[1, 1, 1, 1], padding='SAME')
    l2 = tf.nn.bias_add(c2, b2)
    l2 = tf.layers.batch_normalization(l2)
    l2 = tf.nn.relu(l2)
    l2 = tf.nn.dropout(l2,keep_prob = KEEP_PROB)
    l2_pool = tf.nn.max_pool(l2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    
    W3 = tf.get_variable(name="W3", shape = [5,5,128,256], initializer= tf.contrib.layers.xavier_initializer())
    b3 = tf.get_variable(name="b3", shape = [256], initializer = tf.contrib.layers.xavier_initializer())
    c3 = tf.nn.conv2d(l2_pool, W3, strides=[1, 1, 1, 1], padding='SAME')
    l3 = tf.nn.bias_add(c3, b3)
    l3 = tf.layers.batch_normalization(l3)
    l3 = tf.nn.relu(l3)
    l3 = tf.nn.dropout(l3, keep_prob = KEEP_PROB)
    l3_pool = tf.nn.max_pool(l3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    W4 = tf.get_variable(name="W4", shape = [5,5,256,256], initializer= tf.contrib.layers.xavier_initializer())
    b4 = tf.get_variable(name="b4", shape = [256], initializer = tf.contrib.layers.xavier_initializer())
    c4 = tf.nn.conv2d(l3_pool, W4, strides=[1, 1, 1, 1], padding='SAME')
    l4 = tf.nn.bias_add(c4, b4)
    l4 = tf.layers.batch_normalization(l4)
    l4 = tf.nn.relu(l4)
    l4_pool = tf.nn.max_pool(l4, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    
    print("L4***********************", l4_pool)

    l4_flat = tf.reshape(l4_pool, [-1, 256*2*2])

    W1_fc = tf.get_variable(name="W1_fc", shape=[256*2*2, 256*4], initializer=tf.contrib.layers.xavier_initializer())
    b1_fc = tf.get_variable(name="b1_fc", shape=[256*4], initializer=tf.contrib.layers.xavier_initializer())
    l1_fc = tf.nn.relu(tf.nn.bias_add(tf.matmul(l4_flat, W1_fc), b1_fc))
    l1_fc = tf.nn.dropout(l1_fc, keep_prob=KEEP_PROB)

    W2_fc = tf.get_variable(name="W2_fc", shape=[256*4, 256], initializer=tf.contrib.layers.xavier_initializer())
    b2_fc = tf.get_variable(name="b2_fc", shape=[256], initializer=tf.contrib.layers.xavier_initializer())
    l2_fc = tf.nn.relu(tf.nn.bias_add(tf.matmul(l1_fc, W2_fc), b2_fc))
    l2_fc = tf.nn.dropout(l2_fc, keep_prob=KEEP_PROB)

    W3_fc = tf.get_variable(name="W3_fc", shape=[256, 10], initializer=tf.contrib.layers.xavier_initializer())
    b3_fc = tf.get_variable(name="b3_fc", shape=[10], initializer=tf.contrib.layers.xavier_initializer())
    logits = tf.nn.bias_add(tf.matmul(l2_fc, W3_fc), b3_fc)
    
    hypothesis = tf.nn.softmax(logits)

    return hypothesis, logits

ckpt_path = "output/"

x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
y = tf.placeholder(tf.float32, shape=[None, 10])

x_train = np.load("data/x_train.npy")
y_train = np.load("data/y_train.npy")

dev_num = len(x_train) // 4

x_dev = x_train[:dev_num]
y_dev = y_train[:dev_num]

x_train = x_train[dev_num:]
y_train = y_train[dev_num:]

y_train_one_hot = tf.squeeze(tf.one_hot(y_train, 10),axis=1)
y_dev_one_hot = tf.squeeze(tf.one_hot(y_dev, 10),axis=1)

y_pred, logits = build_CNN_classifier(x)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits))
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cost)

total_batch = int(len(x_train)/batch_size) if len(x_train)%batch_size == 0 else int(len(x_train)/batch_size) + 1


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #saver = tf.train.Saver()
    #saver.restore(sess, ckpt_path)

    print("학습시작")

    for epoch in range(training_epochs):
        print("Epoch", epoch+1)
        start = 0
        average_cost = 0
        shuffled_idx = np.arange(0, len(x_train))
        np.random.shuffle(shuffled_idx)

        for i in range(total_batch):
            batch = batch_data(shuffled_idx, batch_size, x_train, y_train_one_hot.eval(), i*batch_size)
            c,_ = sess.run([cost,train_step], feed_dict={x: batch[0], y: batch[1]})
            average_cost += c /total_batch
        print("Epoch:", '%d'%(epoch+1), 'cost = ', '{:.9f}'.format(average_cost))
    saver = tf.train.Saver()
    saver.save(sess, ckpt_path)
    saver.restore(sess, ckpt_path)

    y_prediction = np.argmax(y_pred.eval(feed_dict={x: x_dev}), 1)
    y_true = np.argmax(y_dev_one_hot.eval(), 1)
    dev_f1 = f1_score(y_true, y_prediction, average="weighted") # f1 스코어 측정
    print("dev 데이터 f1 score: %f" % dev_f1)

    # 밑에는 건드리지 마세요
    x_test = np.load("data/x_test.npy")
    test_logits = y_pred.eval(feed_dict={x: x_test})
    np.save("result", test_logits)