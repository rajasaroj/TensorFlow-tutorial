import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def forward_pass(w, x):
    return tf.matmul(w, x)


def  train_loop(x, iter=5):

    with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
        w = tf.get_variable("weights",
                            shape=(1, 2),
                            initializer=tf.truncated_normal_initializer(),
                            trainable=True)
        pred = []


        for i in range(iter):
            print("in")

            pred.append(forward_pass(w, x))
            w = w + 0.1

        return pred


def make_it_run():

    x = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    pred = train_loop(x)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()



        for i in range(len(pred)):
            print("{}:{}".format(i, pred[i].eval()))


make_it_run()