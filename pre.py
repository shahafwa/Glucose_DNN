import pickle
from pathlib import Path

import numpy as np
import pandas as pd

CATEGORICAL = ['food_id', 'meal_type', 'unit_id']
DATA_RESOLUTION_MIN = 15

_data_dir = Path(__file__).parent / 'our_data'

with (_data_dir / 'norm_stats.pickle').open('rb') as f:
    norm_stats = pickle.load(f)

with (_data_dir / 'categories.pickle').open('rb') as f:
    cat = pickle.load(f)
    cat = {k: pd.api.types.CategoricalDtype(categories=v) for k, v in cat.items()}


def normalize_column(df, col_name):
    with_mean = False
    mean, std = norm_stats[col_name]
    df[col_name] = df[col_name].fillna(mean)
    df[col_name] = ((df[col_name] - mean * with_mean) / std)


def normalize_glucose_meals(cgm, meals):
    normalize_column(cgm, 'GlucoseValue')
    for col_name in meals.columns:
        if col_name not in CATEGORICAL + ['id', 'date']:
            normalize_column(meals, col_name)


def to_cat(meals):
    for col_name in CATEGORICAL:
        meals[col_name] = meals[col_name].astype(cat[col_name])


def preprocess(cgm, meals):
    to_cat(meals)
    normalize_glucose_meals(cgm, meals)


def extract_y(df, n_future_time_points=8):
    """
    Extracting the m next time points (difference from time zero)
    :param n_future_time_points: number of future time points
    :return:
    """
    for g, i in zip(
            range(DATA_RESOLUTION_MIN, DATA_RESOLUTION_MIN * (n_future_time_points + 1), DATA_RESOLUTION_MIN),
            range(1, (n_future_time_points + 1), 1)):
        df['Glucose difference +%0.1dmin' % g] = df.GlucoseValue.shift(-i) - df.GlucoseValue
    return df.dropna(how='any', axis=0).drop('GlucoseValue', axis=1)


def create_shifts(df, n_previous_time_points=48):
    """
    Creating a data frame with columns corresponding to previous time points
    :param df: A pandas data frame
    :param n_previous_time_points: number of previous time points to shift
    :return:
    """
    for g, i in zip(
            range(DATA_RESOLUTION_MIN, DATA_RESOLUTION_MIN * (n_previous_time_points + 1), DATA_RESOLUTION_MIN),
            range(1, (n_previous_time_points + 1), 1)):
        df['GlucoseValue -%0.1dmin' % g] = df.GlucoseValue.shift(i)
    return df.dropna(how='any', axis=0)


def build_cgm(X_glucose, drop=True):
    # using X_glucose and X_meals to build the features
    # get the past 48 time points of the glucose
    X = X_glucose.reset_index().groupby('id').apply(create_shifts, ).set_index(['id', 'Date'])

    # this implementation of extracting y is a valid one.
    y = X_glucose.reset_index().groupby('id').apply(extract_y).set_index(['id', 'Date'])
    if drop:
        index_intersection = X.index.intersection(y.index)
        X = X.loc[index_intersection]
        y = y.loc[index_intersection]
    return X, y


def get_dfs(data_dir, normalize=True):
    cgm = pd.read_csv(data_dir / 'GlucoseValues.csv', index_col=[0, 1], parse_dates=['Date']).sort_index()
    meals = pd.read_csv(data_dir / 'Meals.csv', index_col=[0, 1], parse_dates=['Date']).sort_index()
    cgm = filter_no_meals_data(cgm, meals)
    if normalize:
        preprocess(cgm, meals)
    return cgm, meals


def filter_no_meals_data(cgm_df, meals_df):
    cgm_patients = cgm_df.index.get_level_values('id').unique()
    meals_patients = meals_df.index.get_level_values('id').unique()
    removal_patients = np.setdiff1d(cgm_patients, meals_patients, assume_unique=True)
    cgm_df = cgm_df.drop(index=removal_patients, level='id')
    return cgm_df
