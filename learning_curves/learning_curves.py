# coding:utf-8
import warnings
import gc
import time
import dill
from pathlib import Path

from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, ShuffleSplit, learning_curve
import scipy.optimize as optimize
from scipy.optimize import OptimizeWarning
from scipy.special import lambertw
import numpy as np
import matplotlib.pyplot as plt

from .tools import *
from .predictor import Predictor


class LearningCurve():

    def __init__(self, predictors=[], scoring=r2_score):

        defaults_predictors = [
            Predictor("pow",        lambda x, a, b, c, d    : a - (b*x+d)**c,                  [.9, 1.7, -.5, 1e-3], 
                                    lambda x, a, b, c, d    : (-d + (-x + a)**(1/c))/b),

            Predictor("pow_log",    lambda x, a, b, c, d, m, n : a - (b*x+d)**c + m*np.log(x**n), [.9, 1.7, -.5, 1e-3, 1e-3, 1e-3], diverging=True),

            Predictor("pow_log_2",  lambda x, a, b, c       : a / (1 + (x/np.exp(b))**c),      [.9, 1.7, -.5],
                                    lambda x, a, b, c       : np.exp(b)*(a / x - 1)**(1/c)),

            Predictor("inv_log",    lambda x, a, b          : a - b/np.log(x),                 [.9, 1.6],
                                    lambda x, a, b,         : np.exp(b/(a - x))),

            Predictor("exp",        lambda x, a, b, c       : np.exp((a-1)+b/x +c*np.log(x)),  [.9, -1e3, 1e-3], diverging=True) # c = 0 -> convergence
        ]
        
        self.predictors = get_unique_list(defaults_predictors+predictors)
        self.recorder = {}
        self.scoring = scoring


    def save(self, path="./lc_data.pkl"):
        """ Save the LearningCurve object as a pickle object in disk. Use dill to save because the object contains lambda functions. """
        with open(path, 'wb') as f:
            dill.dump(self, f)


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
            # [0.001, 0.003, 0.005, 0.007, 0.01, 0.05, 0.09, 0.13, 0.17, 0.2, 0.28, 0.36, 0.44, 0.52, 0.6, 0.68, 0.76, 0.84, 0.92, 0.99999] # 1 isn't supported
            train_sizes = [i/1000 for i in range(1, 9, 2)] + [i/100 for i in range(1, 20, 4)] + [i/50 for i in range(10, 47, 4)] + [0.99999]

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


    def fit_all(self, **kwargs):
        """ Fit a curve with all the predictors and retrieve r2 score if y_pred is finite.
            Returns an array of predictors with the updated params and score."""
        return self.fit_all_cust(self.recorder["data"]["train_sizes"], self.recorder["data"]["test_mean_scores"], self.predictors)


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


    def fit(self, P, x, y):
        """ Fit a curve with a predictor and retrieve score (default:R2) if y_pred is finite.
            Returns the predictor with the updated params and score."""

        assert isinstance(P, Predictor), "The given Predictor is not a Predictor object."

        # Enforce parameter a to be in [0,1] if Predictor is converging
        bounds = (-np.inf, np.inf) if P.diverging else ([0]+[-np.inf] * (len(P.params) - 1), [1]+[np.inf] * (len(P.params) - 1))
        try:
            with warnings.catch_warnings():                
                warnings.simplefilter("ignore", RuntimeWarning)
                warnings.simplefilter("ignore", OptimizeWarning)                
                P.params, P.cov = optimize.curve_fit(P, x, y, P.params, bounds=bounds)       
            y_pred = P(x)
            P.score = self.scoring(y, y_pred) if np.isfinite(y_pred).all() else np.nan
        except ValueError:
            P.score = None
        finally:
            return P


    def threshold(self, P="best", **kwargs):
        """ See threshold_cust documentation. This function calls threshold_cust with the LearningCurve data points."""

        if not 'data' in self.recorder: raise RuntimeError("recorder is empty. You must first compute learning curve data points using the train method.")
        if isinstance(P, str) : P = self.get_predictor(P)
        return self.threshold_cust(P, self.recorder["data"]["train_sizes"], **kwargs)


    def threshold_cust(self, P, x, threshold=0.99, max_scaling=2, resolution=1e4, strategies=dict(max_scaling=1), **kwargs):
       
        return self.threshold_cust_inv(P,x,threshold, **kwargs) if callable(P.inv) else self.threshold_cust_approx(P,x,threshold, max_scaling, resolution, strategies, **kwargs)         


    def threshold_cust_inv(self, P, x, threshold=0.99, **kwargs):
        """ Find the training set size providing the highest accuracy up to a desired threshold for a Predictor having an inverse function.
            P(x) = y and for x -> inf, y -> saturation value. Returns x_thresh such as P(x_thresh) = saturation value * threshold.
            Returns (x_thresh, y_thresh, sat_val, threshold). If P is diverging, the saturation value will be approximated using a big number.
        """
        assert callable(P.inv), "P has no inverse function. You have to call threshold_cust_approx instead."
        assert threshold is not None, "No threshold value"

        sat_acc = P.get_saturation()
        desired_acc = sat_acc * threshold            
        opt_trn_size = P.inv(desired_acc)

        if not np.isfinite(opt_trn_size): return np.nan, np.nan, sat_acc, threshold

        return opt_trn_size, round(desired_acc,4), round(sat_acc,4), threshold


    def threshold_cust_approx(self, P, x, threshold, max_scaling, resolution, strategies, **kwargs):
        """ Find the training set size providing the highest accuracy up to a predefined threshold for a Predictor having no inverse function.
            P(x) = y and for x -> inf, y -> saturation value. This method approximates x_thresh such as P(x_thresh) = threshold * saturation value.            
            Returns (x_thresh, y_thresh, sat_val, threshold). If P is diverging, the saturation value will be approximated using a big number.
            max_scaling is used if the Predictor is diverging. It defines the order of magnitude for determining the saturation value.
            max_scaling is added to the order of magnitude of the maximum value of x.
        """
        assert None not in [threshold, max_scaling, resolution], "Parameter has None value"

        sat_acc = P.get_saturation(max_scaling) # if not P.diverging else y[-1]
        desired_acc = sat_acc * threshold

        x_max_scale = get_scale(x[-1])
        max_val = 10 ** (x_max_scale + max_scaling)
        num_splits = min(resolution, max_val-x[0])
        X = np.linspace(x[0], max_val, num_splits, dtype=np.uintc)
        y = P(X)

        if not np.isfinite(y).all(): return np.nan, np.nan, round(sat_acc,4), threshold

        # If Predictor is a decreasing function, stop computing: there is no solution.
        if not is_strictly_increasing(y): return np.nan, np.nan, round(sat_acc,4), threshold

        i = np.argmax(y >= desired_acc)

        # if not enough values in x to find an x_thresh, apply strategies to adjust parameters
        if i == 0:
            if strategies is not None: 
                params = dict(threshold=threshold, max_scaling=max_scaling, resolution=resolution)
                try:
                    return self.threshold_cust_approx(P, x, strategies=strategies, **params)
                except RecursionError:
                    return np.nan, np.nan, sat_acc, threshold
            else:
                return np.nan, np.nan, sat_acc, threshold
        
        else:
            opt_trn_size, opt_acc = X[i], y[i]
            return opt_trn_size, round(opt_acc,4), round(sat_acc,4), threshold


    def best_predictor(self, **kwargs):
        """ Find the best predictor of the LearningCurve data for the test score learning curve."""

        if not 'data' in self.recorder:
            raise RuntimeError("recorder is empty. You must first compute learning curve data points using the train method.")
        return self.best_predictor_cust(self.predictors, self.recorder["data"]["train_sizes"], self.recorder["data"]["test_scores_mean"], **kwargs)


    def best_predictor_cust(self, predictors, x, y, fit=True, prefer_conv_delta=0.002, **kwargs):
        """ Find the best predictor for the test score learning curve."""

        best_p = None
        if fit is True : predictors = self.fit_all_cust(x, y, predictors)
        for P in predictors:
            if P.score is None: continue
            if best_p is None: best_p = P 
            elif P.score > best_p.score: 
                if P.diverging and not best_p.diverging:    # Prefer not diverging Predictors
                    if P.score -  best_p.score > prefer_conv_delta:
                        best_p = P
                else: best_p = P
        return best_p


    def plot(self, predictor=None, **kwargs):
        """ Plot the training and test learning curve of the LearningCurve data, and optionally a fitted function. 
            - predictor: The name of the predictor or predictor to use for fitting the learning curve. Can also be "all" or "best".
        """
        if not 'data' in self.recorder:
            raise RuntimeError("recorder is empty. You must first compute learning curve data points using the train method.")
        return self.plot_cust(predictor=predictor, **self.recorder["data"], **kwargs)

    def plot_cust(self, train_sizes, train_scores_mean, train_scores_std, test_scores_mean, test_scores_std,
                  predictor=None, ylim=(-0.05,1.05), figsize=(12,6), title=None, saturation=None, max_scaling=2, **kwargs):
        """ Plot any training and test learning curve, and optionally a fitted function. """

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        if 'title' is not None: ax.set_title(title)
        ax.set_xlabel("Training examples")
        ax.set_ylabel("Score")
        ax.grid()
        ax.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color='#1f77b4')
        ax.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.15, color="g")
        ax.plot(train_sizes, train_scores_mean, 'o-', color='#1f77b4', label="Training score")
        ax.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

        predictors = []
        if predictor is not None: 
            to_add = self.get_predictor(predictor)
            predictors += to_add if isinstance(to_add, list) else [to_add]

        if saturation is not None:
            saturation = self.get_predictor(saturation)
            predictors += saturation if isinstance(saturation, list) else [saturation]

        predictors = get_unique_list(predictors) # Remove duplicates
        self.fit_all_cust(train_sizes, test_scores_mean, predictors)

        x_scale = get_scale(train_sizes[-1])
        val = 10**(x_scale + max_scaling)
        max_abs = val if val > train_sizes[-1] else train_sizes[-1]
        x_values = np.linspace(train_sizes[0], max_abs, 50)

        # Plot saturation
        if isinstance(saturation, Predictor):
            ax, optx = self.plot_saturation(ax, saturation, max_abs, **kwargs)

        for P in [P for P in predictors if P.score is not None]:
            if P is not None: 
                sat_lbl = saturation.name == P.name if isinstance(saturation, Predictor) else False 
                ax = self.plot_fitted_curve(ax, P, x_values, sat=sat_lbl, **kwargs)

        # Set y limits
        if ylim is not None:
            ymin, ymax = ax.get_ylim()
            if ymin < ylim[0]: ax.set_ylim(bottom=ylim[0])
            if ymax > ylim[1]: ax.set_ylim(top=ylim[1])

        ax.legend(loc="best")
        plt.close(fig)
        return fig


    def plot_fitted_curve(self, ax, P, x, scores=True, sat=False, sat_ls='-.', **kwargs):
        """ Add to figure ax a fitted curve. if {sat} is True, use zorder higher to make the curve more visible. """

        trialX = np.linspace(x[0], x[-1], 500)
        score = round(P.score,4) if P.score is not None else ""
        label = P.name
        if scores : label += f" ({score})"
        z = 3 if sat else 2
        ls =  sat_ls if sat else '--'
        ax.plot(trialX, P(trialX), ls=ls, label=label, zorder=z)
        return ax


    def plot_saturation(self, ax, P, max_abs, alpha=1, lw=1.3, **kwargs):
        """ Add saturation line to a plot. """

        optx, opty, sat, thresh = self.threshold(P, **kwargs)
        if not P.diverging and np.isfinite(sat): 
            ax.axhline(y=sat, c='r', alpha=alpha, lw=lw)
            err = np.sqrt(np.diag(P.cov))[0]
            ax.axhspan(sat - err, min(1,sat + err), alpha=0.05, color='r')        
        if np.isfinite(optx) and optx < max_abs: ax.axvline(x=optx, ls='-', alpha=alpha, lw=lw)
        if np.isfinite(opty): ax.axhline(y=opty, ls='-', alpha=alpha, lw=lw)
        if np.isfinite(thresh): ax.text(1.02, opty, "{:.2e}".format(optx), va='center', ha="left", bbox=dict(facecolor="w",alpha=0.5), transform=ax.get_yaxis_transform())
        return ax, optx
        
