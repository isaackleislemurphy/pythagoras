"""Fits the models for you"""
import os
import random
import itertools
import datetime as dt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# from scipy.stats import norm
from tqdm import trange

import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
import tensorflow_probability as tfp

from etl import load_model_data

_SEED = 2022
os.environ["PYTHONHASHSEED"] = "0"
np.random.seed(_SEED)
tf.random.set_seed(_SEED)
pd.set_option("max_columns", None)

### shorthands
tfkl = tf.keras.layers
tfko = tf.keras.optimizers
tfkr = tf.keras.regularizers
tfki = tf.keras.initializers
tfpl = tfp.layers
tfpd = tfp.distributions


#########################################################################
## Tensorflow odds/ends
#########################################################################


def weibull_log_loss(ytrue, net_output):
    """
    Computes log likelihood according to Weibull distribution
    """
    ### split net outputs
    k, lambda_ = tf.unstack(net_output, num=2, axis=-1)
    ### reshape, you have more than 1
    ### also exponentiate, gotta keep Weibull params > 0
    k_exp = tf.math.exp(tf.expand_dims(k, -1))
    lambda_exp = tf.math.exp(tf.expand_dims(lambda_, -1))
    ### average likelihood
    log_lik = -tf.math.reduce_mean(
        tfpd.Weibull(concentration=k_exp, scale=lambda_exp).log_prob(ytrue)
    )
    return log_lik


def make_forward_block(
    dim_in,
    architecture=(32, 16),
    dropout=(
        0.0,
        0.0,
        0.0,
    ),
    l2=1e-3,
    eta=1e-3,
    compile=True,
):
    """
    Makes (and optionally compiles) a block of linear layers, for use in your model.

    Args:
        dim_in : int
            Column dimension of design matrix passed into layers
        architecture : tuple[int]
            Each element corresponds to a hidden layer, with the value of the element
            the number of units in the layer.
        dropout : tuple[float]
            Dropout rate to apply to each hidden layer
        l2 : float
            L2 regularization applied over all hidden layers.
            TODO: tie this into final layer as well
        eta : float
            Learning rate for ADAM optimizer, if compiled
        compile : bool
            Whether or not to compile the model.
    """
    x_in = tfkl.Input(dim_in)
    for i, units in enumerate(architecture):
        x = tfkl.Dense(
            units=units,
            bias_initializer="zeros",
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=_SEED + i),
            kernel_regularizer=tf.keras.regularizers.L2(l2),
        )(x_in if i == 0 else x)
        x = tfkl.LeakyReLU()(x)
        x = tfkl.Dropout(dropout[i], seed=_SEED + 2 * i)(x)
    x_out = tfkl.Dense(
        units=2,
        bias_initializer="zeros",
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=_SEED),
    )(x if len(architecture) > 0 else x_in)
    model = tf.keras.Model(inputs=x_in, outputs=x_out, name="pythag_model")
    if compile:
        model.compile(loss=weibull_log_loss, optimizer=tfko.Adam(eta))
    return model


class PythagRunModel:
    """Class to manage your pythagorean run model"""

    def __init__(self):
        self.data = {}
        self.val_data = {}
        self.features = None
        self.scaler = MinMaxScaler()
        self.is_fit = False
        self.fit_params = {}

    def process_data_train(self, train_data_df, features):
        """
        Scales and scores training data.

        Args:
            train_data_df : pd.DataFrame
                A dataframe containing your data
            features : list[str]
                Columns in data that will serve as model features.
        """
        self.features = features
        self.data["X"] = self.scaler.fit_transform(
            train_data_df[features].values.astype("float32")
        )
        self.data["Y"] = train_data_df.runs.values.astype("float32") + 0.5
        self.data["df"] = train_data_df.copy()

    def process_data_predict(self, data_df):
        """
        Applies scaling to new data for prediction
        """
        X = self.scaler.transform(data_df[self.features].values)
        if "runs" in data_df.columns:
            Y = data_df.runs.values + 0.5
        else:
            Y = None
        return X, Y

    def configure_data_train(self, train_data_df, features, val_data_df=None):
        """Configures training data ahead of fit"""
        self.process_data_train(train_data_df, features)
        if val_data_df is not None:
            val_x, val_y = self.process_data_predict(val_data_df)
            self.val_data["X"] = val_x.astype("float32")
            self.val_data["Y"] = val_y.astype("float32") + 0.5
            self.val_data["df"] = val_data_df.copy()

    def configure_model(self, **kwargs):
        """
        Sets up and compiles model
        """
        self.model = make_forward_block(
            dim_in=self.data["X"].shape[1], compile=True, **kwargs
        )

    def fit(self, **kwargs):
        """
        Standard fit method
        """
        if len(self.val_data):
            val_data = self.val_data["X"], self.val_data["Y"]
        else:
            val_data = None
        self.model.fit(
            self.data["X"], self.data["Y"], validation_data=val_data, **kwargs
        )

    def predict_params(self, X, as_numpy=False):
        """
        Given a design matrix, returns gamma|X and alpha|X
        """
        if as_numpy:
            return np.exp(self.model(X).numpy())
        return tf.math.exp(self.model(X))

    def predict_tfpd(self, X):
        """
        Passes X through network, and returns the resulting tfpd distribution
        as the prediction

        Args:
          X : np.array
                A design matrix of prediction data
        Returns : tfpd.Weibull
          A distribution Y|X
        """
        return tfpd.Weibull(
            concentration=self.predict_params(X)[:, 0],
            scale=self.predict_params(X)[:, 1],
        )

    def predict(self, X):
        """
        Predicts mean runs for X.

        Args:
          X : np.array(N, P)
                A design matrix of prediction data
        Returns : np.array(N, )
                Mean run predictions.
        """
        return self.predict_tfpd(X).mean().numpy() - 0.5


def train_test_repeated(
    train_data, val_data, features, model_params, fit_params, n_ensemble=100
):
    """
    Trains and validates over a single hyperparameter configuration n_runs times,
    and then scores the average over the validation set.
    """
    train_histories, val_histories = [], []
    for run in trange(n_ensemble):
        model = PythagRunModel()
        model.configure_data_train(train_data, features, val_data_df=val_data)
        model.configure_model(**model_params)
        model.fit(**fit_params)
        ### save scoring
        train_histories.append(model.model.history.history["loss"])
        val_histories.append(model.model.history.history["val_loss"])
        del model
    return np.stack(train_histories), np.stack(val_histories)


def plot_repeated_tuning(val_loss):
    """"""
    for item in val_loss:
        plt.plot(item, alpha=0.2, color="red", linewidth=1.0)
    plt.plot(
        val_loss.mean(axis=0), alpha=1.0, linewidth=3.0, color="maroon", label="Val"
    )
    plt.ylabel("Loss")
    plt.xlabel("Iter")
    plt.legend()
    plt.show()


def main(tune=False, train=True):
    """
    Tunes (val on '19-20) and/or tests (test on '21) the Pythagorean run projection model.
    """
    model_data_df = load_model_data()
    N_EPOCHS = 237
    LAYERS = (256, 128)
    L2 = 1e-3
    ETA = 1e-3

    if tune:
        ##################################################
        # Tune Model (Optional)
        # TODO: put this in a loop
        ##################################################

        model = PythagRunModel()
        model.configure_data_train(
            model_data_df.query("season < 2019"),
            features=FEATURES,
            val_data_df=model_data_df.query("season == 2019 | season == 2020"),
        )
        model.configure_model(architecture=(256, 128), l2=1e-3, eta=1e-3)
        model.fit(epochs=250, batch_size=model.data["X"].shape[0])
        # plot convergence
        plt.plot(model.model.history.history["val_loss"], label="val")
        plt.plot(model.model.history.history["loss"], label="train")
        plt.legend()
        plt.show()
        # loss quantiles
        print("-" * 50)
        print("*Min/Q2.5/Q5 Val Loss:")
        print(np.quantile(model.model.history.history["val_loss"], [0.0, 0.025, 0.05]))
        print("* Opt Epochs.:")
        print(
            np.min(
                np.where(
                    model.model.history.history["val_loss"]
                    < np.quantile(model.model.history.history["val_loss"], 0.05)
                )[0]
            )
        )
        del model  # cleanup

    ##################################################
    # Fit Model
    ##################################################

    ### Hyperparams hard coded here
    model = PythagRunModel()
    model.configure_data_train(
        model_data_df.query("season < 2021"),
        features=FEATURES,
        val_data_df=model_data_df.query("season == 2021"),
    )
    model.configure_model(architecture=LAYERS, l2=L2, eta=ETA)
    model.fit(
        epochs=N_EPOCHS, batch_size=model.data["X"].shape[0]
    )  # full batch, should modify!

    ##################################################
    # Predict
    ##################################################
    test_df = model.val_data["df"].copy()[
        ["game_pk", "date", "pitching_team", "batting_team", "runs"]
    ]
    test_df["runs_hat"] = model.predict(model.val_data["X"])

    test_wide = pd.merge(test_df, test_df, on=["date", "game_pk"]).query(
        "pitching_team_x != pitching_team_y"
    )
    test_wide["p_y"] = np.divide(
        np.power(test_wide.runs_hat_y.values, 2),
        np.add(
            np.power(test_wide.runs_hat_y.values, 2),
            np.power(test_wide.runs_hat_x.values, 2),
        ),
    )
    result = []
    for team_name in test_wide.batting_team_y.unique():
        tm_szn_df = test_wide.query(f"batting_team_y == '{team_name}'")
        result.append(tm_szn_df.p_y.sum())

    pecota_standings = pd.DataFrame(
        {"team": list(test_wide.batting_team_y.unique()), "mean_wins": result}
    ).sort_values("mean_wins", ascending=False)

    pecota_standings.to_csv("pecota_standings.csv")
