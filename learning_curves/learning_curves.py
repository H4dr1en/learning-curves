# coding:utf-8
import warnings
import gc
import time
import dill
import copy
from pathlib import Path

from matplotlib import cm
from cycler import cycler
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, ShuffleSplit, learning_curve
import scipy.optimize as optimize
from scipy.optimize import OptimizeWarning
import numpy as np

from .tools import *
from .predictor import Predictor
from .monkey_patch import learning_curve_patched


class LearningCurve():

    def __init__(self, predictors=[], scoring=r2_score, name=None):
        """ Provide helper functions to calculate, plot and fit learning curves. 

            Args:
                predictors (list): List of Predictors
                scoring (Callable): Function used to calculate scores of the model. (Default is sklearn r2_score).
                name (str): Name of the model (used for comparison with other LearningCurve objects).
        """

        defaults_predictors = [
            Predictor("pow",
                      lambda x, a, b, c, d: a - (b * x + d)**c,          
                      [.9, 1.7, -.5, 0],
                      lambda x, a, b, c, d: (-d + (-x + a)**(1 / c)) / b),  # bounds=([-np.inf, 1e-10, -np.inf, -np.inf], [1, np.inf, 0, 1])

            Predictor("pow_log",
                      lambda x, a, b, c, d, m, n: a - (b * x + d)**c + m * np.log(x**n),
                      [.9, 1.7, -.5, 1e-3, 1e-3, 1e-3], 
                      diverging=True),

            Predictor("pow_log_2",
                      lambda x, a, b, c: a / (1 + (x / np.exp(b))**c),      
                      [.9, 1.7, -.5],
                      lambda x, a, b, c: np.exp(b) * (a / x - 1)**(1 / c)),

            Predictor("inv_log",    
                      lambda x, a, b: a - b / np.log(x),       
                      [.9, 1.6],
                      lambda x, a, b: np.exp(b / (a - x))),

            Predictor("exp",        
                      lambda x, a, b, c: np.exp((a - 1) + b / x + c * np.log(x)),
                      [.9, -1e3, 1e-3],
                      diverging=True)  # c = 0 -> convergence
        ]

        self.predictors = get_unique_list(defaults_predictors + predictors)
        self.recorder = {}
        self.scoring = scoring
        self.name = name

    def save(self, path=None):
        """ Save the LearningCurve object as a pickle object in disk, or as a string. 

            It uses the dill library to save the instance because the object contains lambda functions, that can not be pickled otherwise.

            Args:
                path (str): Path where to save the object. If None, the string representing the object is returned
        """
        if path is None:
            return dill.dumps(self)

        with open(path, 'wb') as f:
            dill.dump(self, f)

    def get_lc(self, estimator, X, Y, train_kwargs={}, **kwargs):
        """ Compute and plot the learning curve. See :meth:`train` and :meth:`plot` functions for parameters.

            Args:
                estimator (Object): Must implement a `fit(X,Y)` and `predict(Y)` method.
                X (array): Features to use for prediction
                Y (array): Values to be predicted
                train_kwargs (dict): See :meth:`train` parameters
                kwargs (dict): See:meth:`plot` parameters.
        """

        self.train(estimator, X, Y, **train_kwargs)
        return self.plot(**kwargs)

    def train(self, estimator, X, Y, train_sizes=None, test_size=.2, n_splits=5, verbose=1, n_jobs=-1, n_samples=20, **kwargs):
        """ Compute the learning curve of an estimator over a dataset.

            Args:
                estimator (Object): Must implement a `fit(X,Y)` and `predict(Y)` method.
                X (array): Features to use for prediction
                Y (array): Values to be predicted
                train_sizes (list): See sklearn `learning_curve`_ function documentation. If None, np.geomspace will be used with 20 values
                n_split (int): Number of random cross validation calculated for each train size
                verbose (int): The higher, the more verbose.
                n_jobs (int): See sklearn `learning_curve`_ function documentation. 
                n_samples (int): if train_sizes is None, n_samples is the number of samples of to use for the learning curve.
                kwargs (dict): See sklearn `learning_curve`_ function parameters. Invalid parameters raise errors.
            Returns:
                Dict: The resulting object can then be passed to :meth:`plot` function.
        """
        if train_sizes is None:
            min_scale = 10**(-get_scale(len(Y)) + 1) #  Starting with at least 10 elements
            train_sizes = np.geomspace(min_scale, 1, n_samples)

        cv = ShuffleSplit(n_splits=n_splits, test_size=test_size)
        t_start = time.perf_counter()
        train_sizes, train_scores, test_scores, fit_times, score_times = learning_curve_patched(estimator, X, Y, cv=cv, n_jobs=n_jobs,
                                                                                                train_sizes=train_sizes,
                                                                                                verbose=verbose, **kwargs)
        self.recorder = {
            "total_size": len(X),
            "train_sizes": train_sizes,
            "train_scores_mean": np.mean(train_scores, axis=1),
            "train_scores_std": np.std(train_scores, axis=1),
            "test_scores_mean": np.mean(test_scores, axis=1),
            "test_scores_std": np.std(test_scores, axis=1),
            "times": time.perf_counter() - t_start,
            "fit_times_mean": np.mean(fit_times, axis=1),
            "fit_times_std": np.std(fit_times, axis=1),
            "score_times_mean": np.mean(score_times, axis=1),
            "score_times_std": np.std(score_times, axis=1)
        }

        gc.collect()

        return self.recorder

    def get_predictor(self, pred):
        """ Get a :class:`learning_curves.predictor` from the list of the Predictors.

            Args:
                pred (Predictor, str, list): Predictor name, "best" or "all", a Predictor, a list of string (Predictor names), a list of Predictors`
            Returns:
                Predictor, list: The matching Predictor(s)
            Raises:
                ValueError: If no matching Predictor is found
        """
        if isinstance(pred, str):
            if pred == "best":
                return self.best_predictor()
            elif pred == "all":
                return self.predictors
            else:
                matches = [P for P in self.predictors if P.name == pred]
                if len(matches) > 0: 
                    return matches[0]

        elif isinstance(pred, Predictor): 
            return pred

        elif isinstance(pred, list):
            if "best" in pred or "all" in pred: 
                raise ValueError("A list of predictors can not contain 'best' or 'all'.")
            return [self.get_predictor(P) for P in pred]

        raise ValueError(f"Predictor {pred} could not be found.")

    def fit_all(self, **kwargs):
        """ Fit a curve with all the predictors using the recorder data and retrieve score if y_pred is finite.

            Args:        
                kwargs (dict): Parameters that will be forwarded to internal functions.
            Returns:
                list: an array of predictors with the updated params and score.
        """
        return self.fit_all_cust(self.recorder["train_sizes"], self.recorder["test_scores_mean"], self.predictors, sigma=self.recorder["test_scores_std"], 
                                 **kwargs)

    def fit_all_cust(self, x, y, predictors, **kwargs):
        """ Fit a curve with all the predictors and retrieve score if y_pred is finite.

            Args:
                x (list): 1D array (list) representing the training sizes
                y (list): 1D array (list) representing the test scores
            Returns:
                list: an array of predictors with the updated params and score.
        """
        results = []
        for p in predictors:
            try:
                results.append(self.fit(p, x, y, **kwargs))
            except RuntimeError:
                warnings.warn(f"{p.name}: Impossible to fit the learning curve (change initial gess).")
        return results  # [self.fit(p,x,y) for p in self.predictors] # No error handling

    def fit(self, P, x, y, **kwargs):
        """ Fit a curve with a predictor, compute  and save the score of the fit.

            Args:
                x (list): 1D array (list) representing the training sizes
                y (list): 1D array (list) representing the test scores
                kwargs (dict): Parameters that will be forwarded to Scipy curve_fit.
            Returns:
                Predictor: The Predictor with the updated params and score. Score will be None if a ValueError exception occures while computing the score.
        """
        assert isinstance(P, Predictor), "The given Predictor is not a Predictor object."

        try:
            with warnings.catch_warnings():                
                warnings.simplefilter("ignore", RuntimeWarning)
                warnings.simplefilter("ignore", OptimizeWarning)
                P.params, P.cov = optimize.curve_fit(P, x, y, P.params, bounds=P.bounds, **kwargs)
            y_pred = P(x)
            P.score = self.scoring(y, y_pred) if np.isfinite(y_pred).all() else np.nan
        except ValueError as e:
            P.score = None
            print(e)
        finally:
            return P

    def eval_train_sizes(self):
        """ Compute the difference of scale between the first and last gradients of accuracies of the train_sizes.

            If this number is lower than 2, then it indicates that the provided training set sizes don't cover a wide enough range of the accuracies values
            to fit a curve. In that case, you should look at the generated plot to determine if you need more points close to the minimum or the maximum
            training set size.

            Returns:
                tain_size_score (float): The difference of scale between the first and last gradients of accuracies of the train_sizes
            Example:
                get_train_sizes_grads([   2,    8,  ..., 2599, 2824]) > 2.7156
        """
        if len(self.recorder) == 0:
            raise RuntimeError("Recorder is empty. You must first compute learning curve data points using the train method.")

        X = self.recorder["train_sizes"]
        assert len(X) > 4, "train_sizes must have at least 4 values"

        Y = self.best_predictor()(X)
        grad_low = (Y[1] - Y[0]) / (X[1] - X[0])
        grad_high = (Y[-1] - Y[-2]) / (X[-1] - X[-2])

        return get_scale(grad_low, False) - get_scale(grad_high, False)

    def eval_fitted_curve(self, validation, **kwargs):
        """ Split the data points in two sets then fit predictors in the first set and evaluate them using RMSE on the second set. See :meth:`eval_fitted_curve_cust`

            Args:
                validation (float, int): Percentage or number of samples of the validation set (the highest training sizes will be used for validation)
                kwargs (dict): Parameters passed to :meth:`eval_fitted_curve_cust`
            Returns:
                fit_score (float): The Root Mean Squared Error of the validation set against the fitted curve of the Predictor
        """
        if len(self.recorder) == 0: 
            raise RuntimeError("Recorder is empty. You must first compute learning curve data points using the train method.")

        valid_abs = get_absolute_value(validation, len(self.recorder["train_sizes"]))
        train_sizes_val = self.recorder["train_sizes"][-valid_abs:]
        test_scores_mean_val = self.recorder["test_scores_mean"][-valid_abs:]
        train_sizes_fit = self.recorder["train_sizes"][:-valid_abs]
        test_scores_mean_fit = self.recorder["test_scores_mean"][:-valid_abs]
        test_scores_std_fit = self.recorder["test_scores_std"][:-valid_abs]

        return self.eval_fitted_curve_cust(train_sizes_fit, test_scores_mean_fit, test_scores_std_fit, train_sizes_val, test_scores_mean_val, **kwargs)

    def eval_fitted_curve_cust(self, train_sizes_fit, test_scores_mean_fit, test_scores_std_fit, train_sizes_val, test_scores_mean_val, 
                               predictor="best", fit=True, metric=mean_bias_error):
        """ Compute the error of a fitted curve on a validation set.

            Args:
                train_sizes_fit (array): List of train sizes used for the fitting of the curve
                test_scores_mean_fit (array): Means computed by the estimator for the train sizes.
                test_scores_std_fit (array): Standard deviations computed by the estimator for the train sizes.
                train_sizes_val (array): List of train sizes used for vscoring of the fitting of the curve (the computation of the RMSE).
                test_scores_mean_val (array): Values computed by the estimator for the validation train sizes.
                predictor (Predictor, "best"): Predictor to consider
                fit (bool): If True, perform a fit of the Predictors using the test_scores_mean_fit data points.
                metric (function): Function to use for the evaluation of the fit of the validation points.
            Returns:
                fit_score (float): The score of the extrapolation using the validation set
        """
        if predictor == "best": 
            predictors = self.predictors
        elif isinstance(predictor, Predictor):
            predictors = [predictor]
        elif not isinstance(predictor, list):
            raise ValueError(f"predictor parameter must be a list of Predictors or 'best', not {predictor}.")

        if fit: 
            self.fit_all_cust(train_sizes_fit, test_scores_mean_fit, predictors, sigma=test_scores_std_fit)

        P = self.best_predictor(predictors)

        return metric(test_scores_mean_val, P(train_sizes_val))

    def threshold(self, P="best", **kwargs):
        """ See :meth:`threshold_cust` function. This function calls :meth:`threshold_cust` with the recorder data.

            Args:
                P (Predictor, string): Predictor to use.
                kwargs (dict): Parameters that will be forwarded to internal functions.
            Returns:
                Tuple: (x_thresh, y_thresh, sat_val, threshold). If P is diverging, the saturation value will be 1.
            Raises:
                RuntimeError: If the recorder is empty.
        """
        if len(self.recorder) == 0: 
            raise RuntimeError("Recorder is empty. You must first compute learning curve data points using the train method.")
        if isinstance(P, str): 
            P = self.get_predictor(P)
        return self.threshold_cust(P, self.recorder["train_sizes"], **kwargs)

    def threshold_cust(self, P, x, threshold=0.99, max_scaling=1, resolution=1e4, strategies=dict(max_scaling=1, threshold=-0.01), **kwargs):
        """ Find the training set size providing the highest accuracy up to a predefined threshold.

            P(x) = y and for x -> inf, y -> saturation value. This method approximates x_thresh such as P(x_thresh) = threshold * saturation value. 

            Args:
                P (str, Predictor): The predictor to use for the calculation of the saturation value.
                x (array): Training set sizes
                threshold (float): In [0.0, 1.0]. Percentage of the saturation value to use for the calculus of the best training set size.
                max_scaling (float): Order of magnitude added to the order of magnitude of the maximum train set size. Generally, a value of 1-2 is enough.
                resolution (float): Only considered for diverging Predictors without inverse function. The higher it is, the more accurate the value of the 
                    training set size will be.
                strategies (dict): A dictionary of the values to add / substract to the other parameters in case a saturation value can not be found.
                    If an RecursionError raises, (None, None, sat_val, threshold) will be returned.
                kwargs (dict): Parameters that will be forwarded to internal functions.
            Returns:
                Tuple: (x_thresh, y_thresh, saturation_arrucacy, threshold)
        """
        return (self.threshold_cust_inv(P, x, threshold, **kwargs) if callable(P.inv) else
                self.threshold_cust_approx(P, x, threshold, max_scaling, resolution, strategies, **kwargs))

    def threshold_cust_inv(self, P, x, threshold, **kwargs):
        """ Find the training set size providing the highest accuracy up to a desired threshold for a Predictor having an inverse function.
            See :meth:`threshold_cust`. """
        assert callable(P.inv), "P has no inverse function. You have to call threshold_cust_approx instead."
        assert threshold is not None, "No threshold value"

        sat_acc = P.get_saturation()
        desired_acc = sat_acc * threshold
        opt_trn_size = P.inv(desired_acc)

        if not np.isfinite(opt_trn_size): 
            return np.nan, np.nan, sat_acc, threshold

        return round(opt_trn_size, 0), round(desired_acc, 4), round(sat_acc, 4), threshold

    def threshold_cust_approx(self, P, x, threshold, max_scaling, resolution, strategies, **kwargs):
        """ Find the training set size providing the highest accuracy up to a predefined threshold for a Predictor having no inverse function. 
            See :meth:`threshold_cust`. """
        assert None not in [threshold, max_scaling, resolution], "Parameter has None value"

        sat_acc = P.get_saturation()
        desired_acc = sat_acc * threshold

        x_max_scale = get_scale(x[-1])
        max_val = 10 ** (x_max_scale + max_scaling)
        num_splits = min(resolution, max_val - x[0])
        X = np.linspace(x[0], max_val, num_splits, dtype=np.uintc)
        y = P(X)

        if not np.isfinite(y).all():
            return np.nan, np.nan, round(sat_acc, 4), threshold

        # If Predictor is a decreasing function, stop computing: there is no solution.
        if not is_strictly_increasing(y):
            return np.nan, np.nan, round(sat_acc, 4), threshold

        i = np.argmax(y >= desired_acc)

        # if not enough values in x to find an x_thresh, apply strategies to adjust parameters
        if i == 0:
            if strategies is not None: 
                params = dict(threshold=threshold, max_scaling=max_scaling, resolution=resolution)
                params = update_params(params, strategies)
                try:
                    return self.threshold_cust_approx(P, x, strategies=strategies, **params)
                except RecursionError:
                    return np.nan, np.nan, sat_acc, threshold
            else:
                return np.nan, np.nan, sat_acc, threshold

        else:
            opt_trn_size, opt_acc = X[i], y[i]
            return round(opt_trn_size, 0), round(opt_acc, 4), round(sat_acc, 4), threshold

    def best_predictor(self, predictors="all", prefer_conv_delta=2e-3, **kwargs):
        """ Find the Predictor having the best fit of a learning curve.

            Args:
                predictors (list(Predictor), "all"): A list of Predictors to consider.
                prefer_conv_delta (float): If the difference of the two best Predictor fit scores is lower than prefer_conv_delta, then if the converging 
                    Predict will be prefered (if any).
                kwargs (dict): Parameters that will be forwarded to internal functions.
            Returns:
                Predictor: The Predicto having the best fit of the learning curve.
        """
        if predictors == "all":
            predictors = self.predictors

        best_p = None
        for P in predictors:
            if P.score is None:
                continue
            if best_p is None:
                best_p = P
            elif P.score + prefer_conv_delta >= best_p.score: 
                if P.diverging:
                    if best_p.diverging:
                        if P.score > best_p.score:
                            best_p = P
                    else:
                        if P.score - prefer_conv_delta > best_p.score:
                            best_p = P
                else:          
                    if best_p.diverging:
                        best_p = P
                    elif P.score > best_p.score:
                        best_p = P
        return best_p

    def plot(self, predictor=None, figsize=(12, 6), fig=None, **kwargs):
        """ Plot the training and test learning curves of the recorder data, with optionally fitted functions and saturation. See :meth:`plot_cust`:

            Args:
                predictor (str, list(str), Predictor, list(Predictor)): The predictor(s) to use for plotting the fitted curve. Can also be "all" and "best".
                figsize (2uple): Size of the figure (only taken in account if ax is None)
                ax (Matplotlib.axes): A figure on which the learning curve will be drawn. If None, a new one is created.
                kwargs (dict): Parameters that will be forwarded to internal functions.
            Returns:
                A Matplotlib figure of the result.
            Raises:
                RuntimeError: If the recorder is empty.
        """
        if len(self.recorder) == 0:
            raise RuntimeError("recorder is empty. You must first compute learning curve data points using the train method.")

        if fig is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        else:
            ax = fig.axes[0]

        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        colors = [colors[0], colors[2]] + colors  # learning curve of training set in blue, of validation set in green
        ax.set_prop_cycle('color', colors)

        return self.plot_cust(predictor=predictor, fig=fig, **self.recorder, **kwargs)

    def plot_cust(self, train_sizes, train_scores_mean, train_scores_std, test_scores_mean, test_scores_std,
                  predictor=None, what="both", xlim=None, ylim=None, figsize=(12, 6), title=None, saturation=None, 
                  target=None, validation=0, close=True, uncertainty=False, fig=None, alpha=0.1, alpha_fit=1, std=True, **kwargs):
        """ Plot any training and test learning curves, with optionally fitted functions and saturation.

            Args:
                train_sizes (list): Training sizes (x values).
                train_scores_std (list): Train score standard deviations.
                test_scores_mean (list): Test score means(y values).
                test_scores_std (list): Train score means.
                predictor (str, list(str), Predictor, list(Predictor)): The predictor(s) to use for plotting the fitted curve. Can be "all" or "best".
                what ("train", "valid", "both"): learning curves to show
                xlim (2uple): Limits of the x axis of the plot.
                ylim (2uple): Limits of the y axis of the plot.
                figsize (2uple): Size of the figure
                title (str): Title of the figure
                saturation (str, list(str), Predictor, list(Predictor)): Predictor(s) to consider for displaying the saturation on the plot. 
                    Can be "all" or "best".
                target (int): Training size to reach. The training size axis will be extended and the fitted curve extrapolated until reaching this value.
                validation (float): Percentage or number of data points to keep for validation of the curve fitting (they will not be used during the fitting 
                    but displayed afterwards)
                close (bool): If True, close the figure before returning it. This is usefull if a lot of plots are being created because Matplotlib won't close 
                    them, potentially leading to warnings.
                    If False, the plot will not be closed. This can be desired when working on Jupyter notebooks, so that the plot will be rendered in the 
                    output of the cell.
                uncertainty (bool): If True, plot the standard deviation of the best fitted curve for the validation data points.
                fig (Matplotlib.figure): A figure which the learning curve will be drawn. If None, a new one is created.
                alpha (float): Controls transparency of the learning curve
                alpha_fit (float): Controls transparency of the fitted line
                std (bool): Whether to plot standard deviations of points or not.
                kwargs (dict): Parameters that will be forwarded to internal functions.
            Returns:
                fig (Matplotlib.figure)
        """
        assert what in ["both", "train", "valid"]

        if fig is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        else:
            ax = fig.axes[0]

        if 'title' is not None:
            ax.set_title(title)
        ax.set_xlabel("Training size")
        ax.set_ylabel("Estimator score")
        ax.grid()

        max_train_size = train_sizes[-1] * 1.05  # Extend a bit so that the curves don't stop before the last points.

        if validation > 0:
            valid_abs = get_absolute_value(validation, len(train_sizes))
            train_sizes_val, test_scores_mean_val, test_scores_std_val = \
                train_sizes[-valid_abs:], test_scores_mean[-valid_abs:], test_scores_std[-valid_abs:]

            train_sizes_fit, test_scores_mean_fit, test_scores_std_fit = \
                train_sizes[:-valid_abs], test_scores_mean[:-valid_abs], test_scores_std[:-valid_abs]
        else:
            train_sizes_fit, test_scores_mean_fit, test_scores_std_fit = train_sizes, test_scores_mean, test_scores_std

        # Plot the learning curve of training and validation sets
        if what in ["train", "both"]:
            lines = ax.plot(train_sizes, train_scores_mean, 'o-', label=self.get_label("Training score"), alpha=alpha)
            if std is True:
                ax.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, 
                                color=lines[0].get_color(), alpha=0.1)

        if what in ["valid", "both"]:
            # ax.plot(train_sizes, test_scores_mean, 'o-', color="g", label=self.get_label("Cross-validation score"))
            errorbar = ax.errorbar(train_sizes_fit, test_scores_mean_fit, test_scores_std_fit, fmt='o-', label=self.get_label("Cross-validation score"),
                                   elinewidth=1, alpha=alpha)
            if std is True:
                ax.fill_between(train_sizes_fit, test_scores_mean_fit - test_scores_std_fit, test_scores_mean_fit + test_scores_std_fit,
                            color=errorbar.lines[0].get_color(), alpha=0.15)

        # Get the list of Predictors to consider
        predictors_to_fit = []

        if predictor == "best":
            predictors_to_fit = self.predictors  # If "best", wait for fit_all before retrieving the best Predictor

        elif predictor is not None:
            to_add = self.get_predictor(predictor)
            predictors_to_fit += to_add if isinstance(to_add, list) else [to_add]

        if saturation is not None and saturation != "best":  # If "best", wait for fit_all before retrieving the best Predictor
            saturation = self.get_predictor(saturation)
            predictors_to_fit += saturation if isinstance(saturation, list) else [saturation]

        # Fitting Predictors
        predictors_to_fit = get_unique_list(predictors_to_fit)  # Remove duplicates
        self.fit_all_cust(train_sizes_fit, test_scores_mean_fit, predictors_to_fit, sigma=test_scores_std_fit)
        best_p = self.best_predictor(predictors_to_fit)

        if target is not None:
            max_abs = target if target > max_train_size else max_train_size
            ax.axvline(x=target, ls='--', color="#1f77b4", lw=1.3)
        else:
            max_abs = max_train_size

        x_values = np.linspace(train_sizes[0], max_abs, 50)

        # Plot fitted curves
        preds_to_plot = [P for P in predictors_to_fit if P is not None and P.score is not None and not np.isnan(P.score)] if predictor != "best" else [best_p]

        for P in preds_to_plot:
            best_lbl = best_p == P if isinstance(best_p, Predictor) else False
            ax = self.plot_fitted_curve(ax, P, x_values, best=best_lbl, alpha=alpha_fit, **kwargs)

        if saturation == "best":
            saturation = best_p

        # Plot saturation
        if isinstance(saturation, Predictor):
            if not P.diverging:
                sat_val = P.get_saturation()
                err = P.get_error_std()[0]
                ax.axhline(y=sat_val, c='r', alpha=1, lw=1.3)
                ax.axhspan(sat_val - err, min(1, sat_val + err), alpha=0.05, color='r')

        # Plot validation of best predictor
        if validation > 0:
            RMSE = self.eval_fitted_curve(validation=validation, predictor=best_p, fit=False)
            label = f"Fit CV (score:{RMSE:.2e})"
            ax.errorbar(train_sizes_val, test_scores_mean_val, test_scores_std_val, fmt='x', color="r", label=self.get_label(label), elinewidth=1)

        # Set limits
        if ylim is not None:
            ax.set_ylim(ylim)

        if xlim is not None:
            ax.set_xlim(xlim)

        ax.legend(loc=4)  # loc=4 # Lower right

        if close:
            plt.close(fig)

        return fig

    def plot_fitted_curve(self, ax, P, x, scores=True, best=False, best_ls='-.', alpha=1, **kwargs):
        """ Add to figure ax a fitted curve.

            Args:
                ax (Matplotlib.axes): Figure used to print the curve.
                P (Predictor): Predictor to use for the computing of the curve.
                x (array): 1D array (list) representing the training sizes.
                scores (bool): Print the score of each curve fit in the legend if True.
                best (bool): use a higher zorder to make the curve more visible if True.
                best_ls (Matplotlib line-style): line-style of the curve whose Predictor is used for computing saturation accuracy.
                alpha (float): Controls the transparency of the fitted curve
                kwargs (dict): Parameters that will be forwarded to internal functions.
            Returns:
                Matplotlib axes: The updated figure.
        """
        trialX = np.linspace(x[0], x[-1], 500)
        score = round(P.score, 4) if P.score is not None else ""
        label = P.name
        if scores:
            label += f" ({score})"
        z = 3 if best else 2
        ls = best_ls if best else '--'
        lw = 2.5 if best else None
        ax.plot(trialX, P(trialX), ls=ls, label=self.get_label(label), zorder=z, linewidth=lw, alpha=alpha)
        return ax

    @staticmethod
    def compare(lcs, fit=True, figsize=(12, 6), colors=None, what="both", fig=None, **kwargs):
        """ Stack learning curves on a single plot (max 10).

            Args:
                lcs (list(LearningCurve)): List of LearningCurves to stack.
                fit (bool): If True, calls :meth:`LearningCurve.fit_all` on all the learning curve objects.
                figsize (tuple): Dimensions of the figure
                colors (cycle, list): cycle of the learning curves colors. A cycler an be created as follows: cycle = cycle('color', ["color1", "color2", ...])
                what ("train", "valid", "both"): curves to show
                fig (Matplotlib.figure): The resulting figure
                kwargs (dict): Dictionary of values that will be passed to each :meth:`LearningCurve.plot` method
            Returns:
                fig (Matplotlib.figure): The resulting figure
        """
        assert all([isinstance(lc, LearningCurve) for lc in lcs]), "parameters must all be Predictors"
        assert all([len(lc.recorder) > 0 for lc in lcs]), "All Predictors must have been trained."
        assert len(lcs) <= 10, "Maximum 10 learning curve object can be stacked."

        if fig is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        else:
            ax = fig.axes[0]

        if colors is None:
            colormap = cm.get_cmap("tab20" if what == "both" else "tab10")
            colors = [colormap(i + 1) if i % 2 == 0 else colormap(i - 1) for i in range(20 if what == "both" else 10)]

        if not isinstance(colors, list):
            raise TypeError("colors must be as cycle object.")

        cycle = cycler('color', colors)
        ax.set_prop_cycle(cycle)  

        for lc in lcs:
            if fit:
                lc.fit_all()
            fig = lc.plot_cust(fig=fig, what=what, **lc.recorder, **kwargs)

        return fig

    def get_label(self, label):
        """ Prefix the label with the name of the LearningCurve instance.

            Args:
                label (str): label to prefix
            Returns:
                label (str): label prefixed with name, if any.
        """
        return label if self.name is None else f"{self.name} - {label}"

    def plot_time(self, fig=None, what="both", figsize=(12, 6)):
        """ Plot training sizes against fit/score computing times.

            Args:
                fig (Matplotlib.figure): A figure on which the curves will be drawn. If None, a new one will be created.
                what (str): Value in ["both", "fit", "score"]. Select the curve to show.
                figsize (2-uple): Dimensions of the figure (ignored if ax is not None).
            Returns:
                fig (Matplotlib.figure): A Matplotlib figure of the result.
        """
        if len(self.recorder) == 0:
            raise RuntimeError("recorder is empty. You must first compute learning curve data points using the train method.")
        assert what in ["both", "fit", "score"]

        if fig is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        else:
            ax = fig.axes[0]

        if what in ["fit", "both"]:
            ax.scatter(self.recorder["train_sizes"], self.recorder["fit_times_mean"], label=self.get_label("fit"))
            lines = ax.plot(self.recorder["train_sizes"], self.recorder["fit_times_mean"])
            ax.fill_between(self.recorder["train_sizes"],
                            self.recorder["fit_times_mean"] - self.recorder["fit_times_std"],
                            self.recorder["fit_times_mean"] + self.recorder["fit_times_std"],
                            color=lines[0].get_color(), alpha=0.15)

        if what in ["score", "both"]:
            ax.scatter(self.recorder["train_sizes"], self.recorder["score_times_mean"], label=self.get_label("score"))
            lines = ax.plot(self.recorder["train_sizes"], self.recorder["score_times_mean"])
            ax.fill_between(self.recorder["train_sizes"], 
                            self.recorder["score_times_mean"] - self.recorder["score_times_std"],
                            self.recorder["score_times_mean"] + self.recorder["score_times_std"],
                            color=lines[0].get_color(), alpha=0.15)

        ax.set_xlabel("Training size")
        ax.set_ylabel("Training time (s)")
        ax.grid()
        ax.legend()

        return fig

    @staticmethod
    def compare_time(lcs, what="both", figsize=(12, 6), colors=None, **kwargs):
        """ Stack times of the computing of the learning curves on a single plot.

            Args:
                lcs (list(LearningCurve)): List of LearningCurves to stack (max 10).
                what (str): Value in ["both", "fit", "score"]. Select the curve to show.
                figsize (tuple): Dimensions of the figure.
                colors (cycle, list): cycle of the learning curves colors. A cycler can be created as follows: cycle = cycle('color', ["color1", "color2", ...])
                kwargs (dict): Dictionary of values that will be passed to each :meth:`LearningCurve.plot_time` method
            Returns:
                ax (Matplotlib.axes): The resulting figure ax
        """
        assert all([isinstance(lc, LearningCurve) for lc in lcs]), "parameters must all be Predictors"
        assert all([len(lc.recorder) > 0 for lc in lcs]), "All Predictors must have been trained first."
        assert len(lcs) <= 10, "Maximum 10 learning curve object can be stacked."

        fig, ax = plt.subplots(1, 1, figsize=figsize)

        if colors is None:
            colormap = cm.get_cmap("tab20" if what == "both" else "tab10")
            colors = [colormap(i) for i in range(20 if what == "both" else 10)]

        if not isinstance(colors, list):
            raise TypeError("colors must be as cycle object.")

        cycle = cycler('color', colors)
        ax.set_prop_cycle(cycle)  

        for lc in lcs:
            fig = lc.plot_time(fig, what=what, **kwargs)

        return fig
