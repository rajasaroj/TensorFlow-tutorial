import tensorflow as tf
import numpy as np
import pandas as pd

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
tf.compat.v1.enable_eager_execution()

Columns = ['key', 'fare_amount', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude',
           'dropoff_latitude', 'passenger_count']

Features = Columns[2:(len(Columns))]
Labels = Columns[1]


# Load train data with pandas
def train_df():
    return pd.read_csv(r"C:\Users\Raja\PycharmProjects\tensorflow_tutorial\Data\train-000.csv", ).dropna()


# Load load validation data with pandas
def valid_df():
    return pd.read_csv(r"C:\Users\Raja\PycharmProjects\tensorflow_tutorial\Data\train-002.csv").dropna()


# Helper parsing function for Data Ingestion nodes to create feature and label data objects from  csv rows
def transform(rows):
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


# Helper data filter function for Data Ingestion nodes to filter rows with header data
def filter_head(row):
    # print("row1 "+ str(type(row)))
    column_values = tf.strings.split(row, ",")
    # stringed = tf.strings.as_string(row)
    # tf.print("---------------")
    # tf.print(column_values.values.shape)
    # tf.print(type(column_values))

    # tf.print(column_values[1])
    # tf.print(type(stringed))
    # ab = tf.strings.substr(row, 0, 1)
    # tf.print(row)
    # tf.print(row)
    # tf.print(ab)
    # tf.print(ab.shape)

    # tf.print(type(tf.strings.substr(row, 0, 3)))
    # tf.print(tf.strings.substr(row, 0, 3))
    # tf.print("---------------")
    key = tf.constant("fare_amount")

    if tf.equal(column_values[1], key):
        tf.print(column_values[1])
        return False
    else:
        return True

    # return tf.not_equal(tf.strings.substr(row, 0, 3), "key")
    return True


# Creates Data Ingestion Node to pump a batched and transformed/parsed data into Execution graph (requires single big csv file)
def get_data_iteratively():
    dataset = tf.compat.v1.data.TextLineDataset(
        r"C:\Users\Raja\PycharmProjects\tensorflow_tutorial\Data\train-000.csv").skip(1).batch(128).map(
        transform)
    dataset = dataset.shuffle(1000).repeat(15)
    feature, label = dataset.make_one_shot_iterator().get_next()
    return feature, label

# Creates Data Ingestion Node to pump a bbatched and transformed/parsed data into Execution graph (can work with multiple big csv files)
def get_Bunchoff_data_iteratively():
    dataset = tf.compat.v1.data.Dataset.list_files(
        r"C:\Users\Raja\PycharmProjects\tensorflow_tutorial\train-*").flat_map(tf.compat.v1.data.TextLineDataset) \
        .filter(filter_head) \
        .batch(128) \
        .map(transform)

    dataset = dataset.repeat(15)
    feature, label = dataset.make_one_shot_iterator().get_next()

    return feature, label


# data input function (if you're reading data from pandas)
def input_train_fn(df, epoch):
    return tf.compat.v1.estimator.inputs.pandas_input_fn(
        x=df[Features],
        y=df[Labels],
        num_epochs=epoch,
        queue_capacity=1000,
        shuffle=True,
        num_threads=1
    )


# create list of columns with there dtypes tensors feeded to model while initilizing it so that model  knows what to expect from data
def featCols():
    return [tf.feature_column.numeric_column(feature) for feature in Features]


# Initalize model and train with connecting Data ingestion Nodes to the graph
def train_model():
    model = tf.estimator.LinearRegressor(featCols())
    model.train(get_data_iteratively)
    return model

# Initalize model and train with connecting Data ingestion Nodes to the graph
def train_DNN_model():
    model = tf.estimator.DNNRegressor(hidden_units=[32, 8, 2],
                                      feature_columns=featCols())
    model.train(get_Bunchoff_data_iteratively)
    return model

# print error
def print_rmse(name):
    model = train_DNN_model()
    metrics = model.evaluate(input_fn=input_train_fn(valid_df(), 1))
    rootmse_value = np.sqrt(metrics['average_loss'])
    print("RMSE on {} dataset {}".format(name, rootmse_value))


print_rmse("validation")
# print(train_df().dtypes)
# print(Features)
