import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)


def train_input_fn():
    features = {
        "sq_footage": [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000],
        "type": ["house", "house", "house", "house", "apt", "apt", "apt", "apt"]
    }

    labels = [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000]
    return features, labels


def build_and_train():
    feat_cols = [tf.feature_column.numeric_column("sq_footage"),
                 tf.feature_column.categorical_column_with_vocabulary_list("type", ["house", "apt"])]

    model = tf.estimator.LinearRegressor(feat_cols)
    model.train(train_input_fn, steps=4000)

    return model


def predict_data():
    features = {
        "sq_footage": [5000, 5500],
        "type": ["house", "apt"]
    }

    return features


def predict_sales():
    model = build_and_train()

    predictions = model.predict(predict_data)

    print(next(predictions))
    print(next(predictions))


predict_sales()
