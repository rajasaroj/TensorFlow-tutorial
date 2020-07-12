import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
tf.compat.v1.disable_v2_behavior()


def transform(rows):
    print("------------------------------")
    tf.print(rows)
    print(type(rows))
    str1 = tf.strings.split(",")
    tf.print(str1)

    # 26:21.0	4.5	2009-06-15 17:26:21 UTC	-73.844311	40.721319	-73.84161	40.712278	1
    columns = ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude", "passenger_count"]
    record_default = [["26:21.0"], [4.5], ["2009-06-15"], [-73.844311], [40.721319], [-73.84161], [40.712278], [1]]
    reco = [[""], [4.5], [""], [0.0], [0.0], [0.0], [0.0], [0]]
    cols = tf.compat.v1.decode_csv(records=rows,
                                   record_defaults=reco)

    features = {}
    features["pickup_longitude"] = cols[3]
    features["pickup_latitude"] = cols[4]
    features["dropoff_longitude"] = cols[5]
    features["dropoff_latitude"] = cols[6]
    features["passenger_count"] = cols[7]
    label = cols[1]
    return features, label

def get_data():

    # dataset = tf.compat.v1.data.TextLineDataset(
    #     r"C:\Users\Raja\PycharmProjects\tensorflow_tutorial\Data\train-001.csv").skip(3).batch(128).map(
    #     transform)
    # iterator_helper1 = dataset.make_one_shot_iterator()
    #print(iterator_helper1.get_next())


    file_data = tf.compat.v1.data.Dataset.list_files(r"C:\Users\Raja\PycharmProjects\tensorflow_tutorial\train-*").flat_map(tf.compat.v1.data.TextLineDataset).skip(3).batch(128).map(transform)
    # data = file_data.flat_map(tf.compat.v1.data.TextLineDataset).skip(3).batch(128).map(
    #     transform)

    #file_data = file_data.shuffle(1000)
    #feature, label = dataset.make_one_shot_iterator().get_next()

    # for line in file_data.take(10):
    #     print(line)


    iterator_helper2 = file_data.make_one_shot_iterator()

    #print(iterator_helper2.get_next())
    #print(iterator_helper2.get_next())

    with tf.compat.v1.Session() as sess:
        filename_temp = iterator_helper2.get_next()
        print(sess.run([filename_temp]))

get_data()


# dataset = tf.compat.v1.data.TextLineDataset(r"C:\Users\Raja\PycharmProjects\tensorflow_tutorial\Data\train-000.csv")
#
#
# iterator_helper = data.make_one_shot_iterator()
#
# it_dataset = dataset.make_one_shot_iterator()
#
# print(data)
# print(dataset)
# print(iterator_helper.get_next())
# print(iterator_helper.get_next())
# print(it_dataset.get_next())

# with tf.Session() as sess:
#     filename_temp = iterator_helper.get_next()
#
#     print(sess.run[filename_temp])
#     print(filename_temp)
