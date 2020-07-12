import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


def areaFinder(sides):
    a = sides[:, 0]
    b = sides[:, 1]
    c = sides[:, 2]

    s = (a + b + c) * 0.5
    area = s * (s - a) * (s - b) * (s - c)

    return tf.sqrt(area)


def helper():
    sides = tf.placeholder(dtype=tf.float32, shape=(2, None), name="sides")
    area = areaFinder(sides)

    with tf.Session() as sess:
        result = sess.run(area, feed_dict={"sides:0": [
                                                 [4.0, 4.0, 4.0],
                                                 [2.0, 4.0, 5.0]
                                              ]
                                           })
        print(result)
        print(area)


helper()
