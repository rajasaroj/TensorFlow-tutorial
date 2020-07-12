import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


def visualize():

    x = tf.constant([3, 6, 7])
    y = tf.constant([1, 2, 3])
    z1 = tf.add(x, y)
    z2 = x * y
    z3 = z2 - z1

    with tf.Session() as sess:
        with tf.summary.FileWriter('summaries', sess.graph) as writer:
            writer.close()
        print(sess.run(z1))

# Visualize graph command(conda prompt) tensorboard --logdir=C:\Users\Raja\PycharmProjects\tensorflow_tutorial\summaries

x2 = tf.constant([1, 2, 3])
x3 = tf.constant([4, 5, 6])
x4 = tf.stack([x2,x3])
x5 = x4[1, :]

with tf.Session() as sess:
    print(x5.eval())



