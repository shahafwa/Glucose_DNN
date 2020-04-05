import pandas as pd
from scipy.stats import pearsonr

from wis_dnn_challenge import Predictor, val


def compute_mean_pearson(y_true, y_pred, individual_index_name='id', n_future_time_points=8):
    """
    This function takes the true glucose values and the predicted ones, flattens the data per individual and then
    computed the Pearson correlation between the two vectors per individual.

    **This is how we will evaluate your predictions, you may use this function in your code**

    :param y_true: an M by n_future_time_points data frame holding the true glucose values
    :param y_pred: an M by n_future_time_points data frame holding the predicted glucose values
    :param individual_index_name: the name of the individual's indeces, default is 'id'
    :param n_future_time_points: number of future time points to predict, default is 8
    :return: the mean Pearson correlation
    """
    # making sure y_true and y_pred are of the same size
    assert y_true.shape == y_pred.shape
    # making sure y_true and y_pred share the same exact indeces and index names
    assert (y_true.index == y_pred.index).all() and y_true.index.names == y_pred.index.names
    # making sure that individual_index_name is a part of the index of both dataframes
    assert individual_index_name in y_true.index.names and individual_index_name in y_pred.index.names

    # concat data frames
    joined_df = pd.concat((y_true, y_pred), axis=1)
    return joined_df.groupby(individual_index_name) \
        .apply(lambda x: pearsonr(x.iloc[:, :n_future_time_points].values.ravel(),
                                  x.iloc[:, n_future_time_points:].values.ravel())[0]).mean()


if __name__ == '__main__':
    # creating a Predictor instance
    predictor = Predictor()
    # load the GlucoseValues that you got for training
    X_glucose = Predictor.load_data_frame(val / 'GlucoseValues.csv')
    X, y_true = predictor.build_features(X_glucose, None)
    print('shape of X:', X.shape)
    print('shape of y_true:', y_true.shape)

    X_meals = Predictor.load_data_frame(val / 'Meals.csv')
    y_pred = predictor.predict(X_glucose, X_meals)
    assert (y_true.index == y_pred.index).all() and y_true.index.names == y_pred.index.names
    compute_mean_pearson(y_true, y_pred)
