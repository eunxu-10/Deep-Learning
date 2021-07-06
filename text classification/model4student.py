import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn_cell

ATTENTION_UNITS = 100
SEQUENCE_LENGTH = 500
EMBEDDING_DIM = 100
HIDDEN_SIZE = 150
BATCH_SIZE = 128
NUM_EPOCHS = 10
learning_rate = 0.001

keep_prob = tf.placeholder(tf.float32)
KEEP_PROB = 0.75

def batch_data(shuffled_idx, batch_size, data, labels, start_idx):
    idx = shuffled_idx[start_idx:start_idx+batch_size]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

def get_vocabulary_size(X):
    return max([max(x) for x in X]) + 1 # plus the 0th word

def fit_in_vocabulary(X, voc_size):
    return [[w for w in x if w < voc_size] for x in X]

def zero_pad(X, seq_len):
    return np.array([x[:seq_len - 1] + [0] * max(seq_len - len(x), 1) for x in X])

def attention(inputs, attention_size, time_major= False, return_alphas = False):
    if isinstance(inputs,tuple):
        input = tf.concat(inputs,2)

    if time_major:
        inputs = tf.array_ops.transpose(inputs, [1,0,2])
    
    hidden_size = inputs.shape[2].value

    w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
    b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

    with tf.name_scope('v'):
        v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)

    vu = tf.tensordot(v, u_omega, axes=1, name='vu') 
    alphas = tf.nn.softmax(vu, name='alphas')

    output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

    if not return_alphas:
        return output
    else:
        return output, alphas

def build_classifier(x, vocabulary_size, EMBEDDING_DIM, HIDDEN_SIZE):
    # Embedding layer
    embeddings_var = tf.Variable(tf.random_uniform([vocabulary_size, EMBEDDING_DIM], -1.0, 1.0), trainable=True)
    batch_embedded = tf.nn.embedding_lookup(embeddings_var, x)

    # RNN layer
    rnn_outputs, states = tf.nn.dynamic_rnn(rnn_cell.GRUCell(HIDDEN_SIZE),batch_embedded, dtype=tf.float32)
    
    # Attention layer
    attention_output, alphas = attention(rnn_outputs, ATTENTION_UNITS, return_alphas=True)

    #Fully
    W1_fc = tf.get_variable(name="W1_fc", shape=[HIDDEN_SIZE, 64], initializer=tf.contrib.layers.xavier_initializer())
    b1_fc = tf.get_variable(name="b1_fc", shape=[64], initializer=tf.contrib.layers.xavier_initializer())
    l1_fc = tf.nn.relu(tf.nn.bias_add(tf.matmul(attention_output, W1_fc), b1_fc))
    l1_fc = tf.nn.dropout(l1_fc, keep_prob=KEEP_PROB)

    W2_fc = tf.get_variable(name="W2_fc", shape=[64, 32], initializer=tf.contrib.layers.xavier_initializer())
    b2_fc = tf.get_variable(name="b2_fc", shape=[32], initializer=tf.contrib.layers.xavier_initializer())
    l2_fc = tf.nn.relu(tf.nn.bias_add(tf.matmul(l1_fc, W2_fc), b2_fc))
    l2_fc = tf.nn.dropout(l2_fc, keep_prob=KEEP_PROB)

    W3_fc = tf.get_variable(name="W3_fc", shape=[32, 2], initializer=tf.contrib.layers.xavier_initializer())
    b3_fc = tf.get_variable(name="b3_fc", shape=[2], initializer=tf.contrib.layers.xavier_initializer())
    
    logits = tf.nn.bias_add(tf.matmul(l2_fc, W3_fc), b3_fc)
    hypothesis = tf.nn.softmax(logits)

    return hypothesis, logits

ckpt_path = "output/"

# Load the data set
np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)


x_train = np.load("data/x_train.npy")
y_train = np.load("data/y_train.npy")
x_test = np.load("data/x_test.npy")

np.load = np_load_old


dev_num = len(x_train) // 4

x_dev = x_train[:dev_num]
y_dev = y_train[:dev_num]

x_train = x_train[dev_num:]
y_train = y_train[dev_num:]

y_train_one_hot = tf.squeeze(tf.one_hot(y_train, 2))
y_dev_one_hot = tf.squeeze(tf.one_hot(y_dev, 2))


# Sequences pre-processing
vocabulary_size = get_vocabulary_size(x_train)
x_dev = fit_in_vocabulary(x_dev, vocabulary_size)
x_train = zero_pad(x_train, SEQUENCE_LENGTH)
x_dev = zero_pad(x_dev, SEQUENCE_LENGTH)

batch_ph = tf.placeholder(tf.int32, [None, SEQUENCE_LENGTH], name='batch_ph')
target_ph = tf.placeholder(tf.float32, [None, 2], name='target_ph')


y_pred, logits = build_classifier(batch_ph, vocabulary_size, EMBEDDING_DIM, HIDDEN_SIZE)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=target_ph, logits=logits))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# Accuracy metric
is_correct = tf.equal(tf.argmax(y_pred, 1), tf.argmax(target_ph, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

total_batch = int(len(x_train)/BATCH_SIZE) if len(x_train)%BATCH_SIZE == 0 else int(len(x_train)/BATCH_SIZE) + 1

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    print("학습시작")

    for epoch in range(NUM_EPOCHS):
        print("Epoch", epoch + 1)
        start = 0
        shuffled_idx = np.arange(0, len(x_train))
        np.random.shuffle(shuffled_idx)

        for i in range(total_batch):
            batch = batch_data(shuffled_idx, BATCH_SIZE, x_train, y_train_one_hot.eval(), i * BATCH_SIZE)
            sess.run(optimizer, feed_dict={batch_ph: batch[0], target_ph: batch[1]})
        saver = tf.train.Saver()
        saver.save(sess, ckpt_path)
        saver.restore(sess, ckpt_path)

    dev_accuracy = accuracy.eval(feed_dict={batch_ph: x_dev, target_ph: np.asarray(y_dev_one_hot.eval())})
    print("dev 데이터 Accuracy: %f" % dev_accuracy)

    # 밑에는 건드리지 마세요
    x_test = fit_in_vocabulary(x_test, vocabulary_size)
    x_test = zero_pad(x_test, SEQUENCE_LENGTH)

    test_logits = y_pred.eval(feed_dict={batch_ph: x_test})
    np.save("result", test_logits)