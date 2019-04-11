# coding:utf-8
import warnings
import gc
import time
import dill
from pathlib import Path

from sklearn.metrics import r2_score
import scipy.optimize as optimize
import numpy as np
from sklearn.model_selection import train_test_split, ShuffleSplit, learning_curve

import matplotlib.pyplot as plt


class Predictor():

    def __init__(self, name, func, guess, diverging = False):
        """ Create a Predictor with a given {name}, {function} and initial parameters: {guess}."""
        self.name = name
        self.func = func
        self.guess = guess
        self.params = self.guess
        self.score = None
        self.cov = {}
        self.diverging = diverging

    def __call__(self, x, *args):
        return self.func(x, *args) if len(args) > 1 else self.func(x, *self.params)

    def __repr__(self):
        return f"({self.name} [params:{self.params}][score:{self.score}])"


class LearningCurve():

    def __init__(self, predictors=[], scoring=r2_score):

        defaults_predictors = [
            Predictor("pow",        lambda x, a, b, c, d    : a - (b*x+d)**c,                [1, 1.7, -.5, 1e-3]),
            Predictor("pow_log",    lambda x, a, b, c, m, n : a - b*x**c + m*np.log(x**n),   [1, 1.7, -.5, 1e-3, 1e-3], True),
            Predictor("pow_log_2",  lambda x, a, b, c       : a / (1 + (x/np.exp(b))**c),    [1, 1.7, -.5]),
            Predictor("inv_log",    lambda x, a, b          : a - b/np.log(x),               [1, 1.6])
        ]
        
        self.predictors = self.get_unique_list(defaults_predictors+predictors)
        self.recorder = {}
        self.scoring = scoring

    def save(self, path="./lc_data.pkl"):
        """ Save the LearningCurve object as a pickle object in disk. Use dill to save because the object contains lambda functions. """
        with open(path, 'wb') as f:
            dill.dump(self, f)

    @staticmethod
    def load(path="./lc_data.pkl"):
        """ Load a LearningCurve object from disk. """
        with open(path, 'rb') as f:
            return dill.load(f)

    def get_lc(self, estimator, X, Y, **kwargs):
        """ Compute and plot the learning curve. See train and plot functions for parameters."""

        self.train(estimator, X, Y, **kwargs)
        return self.plot(**kwargs)

    def train(self, estimator, X, Y, train_sizes=None, test_size=0.2, n_splits=3, verbose=1, n_jobs=-1, **kwargs):
        """ Compute the learning curve of an estimator over a dataset. Returns an object that can then be passed to plot_lc function.
            Parameters:
                - estimator: object type that implements the “fit” and “predict” methods.
                - train_sizes: See sklearn learning_curve function documentation. Default is 
                - n_splits: int. Number of random cross validation calculated for each train size
                - verbose: int. The higher, the more verbose.
                - n_jobs: See sklearn learning_curve function documentation.
        """

        if train_sizes is None:
            # [0.001, 0.003, 0.005, 0.007, 0.01, 0.05, 0.09, 0.13, 0.17, 0.2, 0.28, 0.36, 0.44, 0.52, 0.6, 0.68, 0.76, 0.84, 0.92]
            train_sizes = [i/1000 for i in range(1, 9, 2)] + [i/100 for i in range(1, 20, 4)] + [i/50 for i in range(10, 47, 4)]

        cv = ShuffleSplit(n_splits=n_splits, test_size=test_size)
        t_start = time.perf_counter()
        train_sizes, train_scores, test_scores = learning_curve(estimator, X, Y,
                                                                cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, verbose=verbose)
        self.recorder["data"] = {
            "total_size": len(X),
            "train_sizes": train_sizes,
            "train_scores_mean": np.mean(train_scores, axis=1),
            "train_scores_std": np.std(train_scores, axis=1),
            "test_scores_mean": np.mean(test_scores, axis=1),
            "test_scores_std": np.std(test_scores, axis=1),
            "time": time.perf_counter() - t_start
        }

        gc.collect()

        return self.recorder["data"]

    def get_predictor(self, pred):
        """ Get the first predictor with matching {pred}. Returns None if no predictor matches. 
            pred can be: a string (Predictor name, "best" or "all"), a Predictor, a list of string (Predictor names), a list of Predictors"""

        if isinstance(pred, str):
            if pred == "best":
                return self.best_predictor()
            elif pred == "all":
                return self.predictors 
            else:
                matches = [P for P in self.predictors if P.name == pred]
                if len(matches) > 0: return matches[0]

        elif isinstance(pred, Predictor): return pred

        elif isinstance(pred, list):
            if "best" in pred or "all" in pred: raise ValueError("A list of predictors can not contain 'best' or 'all'.")
            return [self.get_predictor(P) for P in pred]
            
        raise ValueError(f"Predictor {pred} could not be found.")

    def fit_all(self, x, y):
        """ Fit a curve with all the predictors and retrieve r2 score if y_pred is finite.
            Returns an array of predictors with the updated params and score."""
        return self.fit_all_cust(x, y, self.predictors)

    def fit_all_cust(self, x, y, predictors):
        """ Fit a curve with all the predictors and retrieve r2 score if y_pred is finite.
            Returns an array of predictors with the updated params and score."""

        results = []
        for p in predictors:
            try:
                results.append(self.fit(p, x, y))
            except RuntimeError:
                warnings.warn(f"{p.name}: Impossible to fit the learning curve (change initial gess).")
        return results  # [self.fit(p,x,y) for p in self.predictors] # No error handling

    def fit(self, predictor, x, y):
        """ Fit a curve with a predictor and retrieve score (default:R2) if y_pred is finite.
            Returns the predictor with the updated params and score."""

        assert isinstance(predictor, Predictor), "The given Predictor is not a Predictor object."

        predictor.params, predictor.cov = optimize.curve_fit(predictor, x, y, predictor.params)
        # try:
        y_pred = predictor(x)
        predictor.score = self.scoring(y, y_pred) if np.isfinite(y_pred).all() else np.nan
        # except ValueError:
        #    print(x, P(x))
        return predictor

    def threshold(self, P="best", **kwargs):
        """ See threshold_cust documentation. This function calls threshold_cust with the LearningCurve data points."""
        if not 'data' in self.recorder: raise RuntimeError("recorder is empty. You must first compute learning curve data points using the train method.")
        if isinstance(P, str) : P = self.get_predictor(P)
        return self.threshold_cust(P, self.recorder["data"]["train_sizes"], **kwargs)

    def threshold_cust(self, P, x, threshold=0.99, max_scaling=3, force=False, **kwargs):
        """ Find the training set size providing the highest accuracy up to a predefined threshold.
            P(x) = y and for x -> inf, y -> saturation value.
            This method approximates x_thresh such as P(x_thresh) = threshold * saturation value
            Returns (saturation value, x_thresh, y_thresh)
            max_scaling is use if the Predictor is diverging. It defines the order of magnitude for determining the saturation value.
            max_scaling is added to the order of magnitude of the maximum value of x.
        """               
        if max_scaling > 5 and not force:
            raise ValueError("max_scaling > 5: this will consume a lot of memory. Use Force=True to continue.")

        x_max_scale = LearningCurve.get_scale(x[-1])
        max_val = 10 ** (x_max_scale + max_scaling)
        step = 5 * (max_scaling - 2)
        # using result results in bad approximation of opt_trn_size because the step if proportional to max_scaling
        # np.linspace(x[0], max_val, 1e3, dtype=np.uintc) 
        # using np.arange we can specify a step and be more precise, but we need to adjust the step with max_scaling
        # otherwise we run out of memory
        x = np.arange(x[0], max_val, step, dtype=np.uintc) 
        y = P(x)

        if P.diverging:
            warnings.warn("""Using a diverging Predictor. Because no saturation value exists for such Predictor, max_scaling will be use to determine the saturation value. You should adapt max_scaling to fit the maximum number of data you can fit.""")
            sat_val = y[-1]
        else:
            sat_val = P.params[0]

        desired_acc = sat_val * threshold
        i = np.argmax(y >= desired_acc)
        opt_trn_size, opt_acc = x[i], y[i]

        return round(sat_val,4), round(opt_trn_size,4), round(opt_acc,4)

    @staticmethod
    def get_scale(val):
        """ Returns the scale of a value. Eg: get_scale(1e-15) = -15 """
        return np.floor(np.log10(np.abs(val)))

    def best_predictor(self, **kwargs):
        """ Find the best predictor of the LearningCurve data for the test score learning curve."""

        if not 'data' in self.recorder:
            raise RuntimeError("recorder is empty. You must first compute learning curve data points using the train method.")
        return self.best_predictor_cust(self.predictors, self.recorder["data"]["train_sizes"], self.recorder["data"]["test_scores_mean"], **kwargs)

    def best_predictor_cust(self, predictors, x, y, fit=True, **kwargs):
        """ Find the best predictor for the test score learning curve."""

        best_p = predictors[0]
        if fit is True : predictors = self.fit_all_cust(x, y, predictors)
        for P in predictors:
            if P.score is not None and P.score > best_p.score:
                best_p = P
        return best_p

    def plot(self, predictor=None, **kwargs):
        """ Plot the training and test learning curve of the LearningCurve data, and optionally a fitted function. 
            - predictor: The name of the predictor or predictor to use for fitting the learning curve. Can also be "all" or "best".
        """
        if not 'data' in self.recorder:
            raise RuntimeError("recorder is empty. You must first compute learning curve data points using the train method.")
        return self.plot_cust(predictor=predictor, **self.recorder["data"], **kwargs)

    def plot_cust(self, train_sizes, train_scores_mean, train_scores_std, test_scores_mean, test_scores_std,
                  predictor=None, ylim=None, figsize=None, title=None, saturation=None, **kwargs):
        """ Plot any training and test learning curve, and optionally a fitted function. """

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        if 'title' is not None:
            ax.set_title(title)
        if ylim is not None:
            ax.ylim(*ylim)
        ax.set_xlabel("Training examples")
        ax.set_ylabel("Score")
        ax.grid()
        ax.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color='#1f77b4')
        ax.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.15, color="g")
        ax.plot(train_sizes, train_scores_mean, 'o-', color='#1f77b4', label="Training score")
        ax.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

        x_values = train_sizes

        predictors = []
        if predictor is not None: 
            to_add = self.get_predictor(predictor)
            predictors += to_add if isinstance(to_add, list) else [to_add]

        if saturation is not None:
            saturation = self.get_predictor(saturation)
            predictors += saturation if isinstance(saturation, list) else [saturation]

        predictors = self.get_unique_list(predictors) # Remove duplicates
        self.fit_all_cust(train_sizes, test_scores_mean, predictors)

        # Plot saturation
        if isinstance(saturation, Predictor):
            ax, optx = self.plot_saturation(ax, saturation, **kwargs)
            if(train_sizes[-1] < optx):
                x_scale = LearningCurve.get_scale(train_sizes[-1])
                extra_vals = np.linspace(x_scale+1, optx+100, 10)
                x_values = np.concatenate((x_values, extra_vals), axis=None)

        for P in predictors:
            if P is not None: ax = self.plot_fitted_curve(ax, P, x_values, **kwargs)

        ax.legend(loc="best")
        plt.close(fig)
        return fig

    def plot_fitted_curve(self, ax, P, x, scores=True, **kwargs):
        """ Add to figure ax a fitted curve. """
        trialX = np.linspace(x[0], x[-1], 500)
        label = P.name + f" ({round(P.score,4)})" if scores is True and P.score is not None else ""
        ax.plot(trialX, P(trialX), ls='--', label=label)
        return ax

    def plot_saturation(self, ax, P, alpha=1, lw=1.3, **kwargs):
        """ Add saturation lines to a plot. """
        sat, optx, opty = self.threshold(P, **kwargs)
        ax.axhline(y=sat, c='r', alpha=alpha, lw=lw)
        ax.axvline(x=optx, ls='-', alpha=alpha, lw=lw)
        ax.axhline(y=opty, ls='-', alpha=alpha, lw=lw)
        return ax, optx

    def get_unique_list(self, predictors):
        """ Return a list of unique predictors. """
        results = []
        for P in predictors:
            if not P.name in [p.name for p in results] : results.append(P)
        return results
        
