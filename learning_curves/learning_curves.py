# coding:utf-8
import warnings
import gc
import time
import dill
import copy
from pathlib import Path

from sklearn.metrics import r2_score, mean_squared_error
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
        """ Provide helper functions to calculate, plot and fit learning curves. 
    
        Args:
            predictors (list): List of Predictors
            scoring (Callable): Function used to calculate scores of the model. (Default is sklearn r2_score)."""

        defaults_predictors = [
            Predictor(  
                        "pow",        
                        lambda x, a, b, c, d    : a - (b*x+d)**c,                  
                        [.9, 1.7, -.5, 0], 
                        lambda x, a, b, c, d    : (-d + (-x + a)**(1/c))/b, 
                        bounds=([-np.inf, 0, -np.inf, -np.inf], [1, np.inf, 0, 1])
                    ),
                   # 1.        0.001016 -0.115942  0.641653]

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
        """ Save the LearningCurve object as a pickle object in disk. 
        
            It uses the dill library to save the instance because the object contains lambda functions, that can not be pickled otherwise. 
        """
        with open(path, 'wb') as f:
            dill.dump(self, f)


    def get_lc(self, estimator, X, Y, **kwargs):
        """ Compute and plot the learning curve. See :meth:`train` and :meth:`plot` functions for parameters."""

        self.train(estimator, X, Y, **kwargs)
        return self.plot(**kwargs)


    def train(self, estimator, X, Y, train_sizes=None, test_size=0.15, n_splits=5, verbose=1, n_jobs=-1, **kwargs):
        """ Compute the learning curve of an estimator over a dataset.

            Args:
                estimator (Object): Must implement a `fit(X,Y)` and `predict(Y)` method.
                train_sizes (list): See sklearn `learning_curve`_ function documentation. If None, default value will be used: ``[0.001, 0.003, 0.005, 0.007, 0.01, 0.05, 0.09, 0.13, 0.17, 0.2, 0.28, 0.36, 0.44, 0.52, 0.6, 0.68, 0.76, 0.84, 0.92, 1]``
                n_split (int): Number of random cross validation calculated for each train size
                verbose (int): The higher, the more verbose.
                n_jobs (int): See sklearn `learning_curve`_ function documentation. 
                kwargs (dict): Parameters that will be forwarded to internal functions.
            Returns:
                Dict: The resulting object can then be passed to :meth:`plot` function.
        """
        if train_sizes is None:
            # [0.001, 0.003, 0.005, 0.007, 0.01, 0.05, 0.09, 0.13, 0.17, 0.2, 0.28, 0.36, 0.44, 0.52, 0.6, 0.68, 0.76, 0.84, 0.92, 1]
            train_sizes = [i/1000 for i in range(1, 9, 2)] + [i/100 for i in range(1, 20, 4)] + [i/50 for i in range(10, 51, 4)]

        cv = ShuffleSplit(n_splits=n_splits, test_size=test_size)
        t_start = time.perf_counter()
        train_sizes, train_scores, test_scores = learning_curve(estimator, X, Y,
                                                                cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, verbose=verbose)
        self.recorder = {
            "total_size": len(X),
            "train_sizes": train_sizes,
            "train_scores_mean": np.mean(train_scores, axis=1),
            "train_scores_std": np.std(train_scores, axis=1),
            "test_scores_mean": np.mean(test_scores, axis=1),
            "test_scores_std": np.std(test_scores, axis=1),
            "time": time.perf_counter() - t_start
        }

        gc.collect()

        return self.recorder


    def get_predictor(self, pred):
        """ Get a :class:`learning_curves.predictor` from the list of the Predictors.
            
            Args:
                pred (Predictor, str, list): Predictor name, "best" or "all", a Predictor, a list of string (Predictor names), a list of Predictors`
            Returns:
                Predictor: The matching Predictor
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
                if len(matches) > 0: return matches[0]

        elif isinstance(pred, Predictor): return pred

        elif isinstance(pred, list):
            if "best" in pred or "all" in pred: raise ValueError("A list of predictors can not contain 'best' or 'all'.")
            return [self.get_predictor(P) for P in pred]
            
        raise ValueError(f"Predictor {pred} could not be found.")


    def fit_all(self, **kwargs):
        """ Fit a curve with all the predictors using the recorder data and retrieve score if y_pred is finite.

            Args:                
                kwargs (dict): Parameters that will be forwarded to internal functions.
            Returns:
                list: an array of predictors with the updated params and score.
        """
        return self.fit_all_cust(self.recorder["train_sizes"], self.recorder["test_mean_scores"], self.predictors, sigma=self.recorder["test_scores_std"], **kwargs)


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
        except ValueError:
            P.score = None
        finally:
            return P


    def eval_train_sizes(self):
        """ Compute the difference of scale between the first and last gradients of accuracies of the train_sizes.

            If this number is lower than 2, then it indicates that the provided training set sizes don't cover a wide enough range of the accuracies values
            to fit a curve. In that case, you should look at the generated plot to determine if you need more points close to the minimum or the maximum training set size.

            Returns:
                tain_size_score (float): The difference of scale between the first and last gradients of accuracies of the train_sizes
            Example:
                get_train_sizes_grads([   2,    8,  ..., 2599, 2824]) > 2.7156
        """
        if len(self.recorder) == 0: raise RuntimeError("Recorder is empty. You must first compute learning curve data points using the train method.")
        
        X = self.recorder["train_sizes"]
        assert len(X) > 4, "train_sizes must have at least 4 values"

        Y = self.best_predictor()(X)
        grad_low  = (Y[1] -Y[0])    / (X[1] -X[0])
        grad_high = (Y[-1]-Y[-2])   / (X[-1]-X[-2])

        return get_scale(grad_low, False) - get_scale(grad_high, False)


    def eval_fitted_curve(self, validation, **kwargs):
        """ Split the data points in two sets then fit predictors in the first set and evaluate them using RMSE on the second set. See :meth:`eval_fitted_curve_cust` """
        if len(self.recorder) == 0: raise RuntimeError("Recorder is empty. You must first compute learning curve data points using the train method.")
        return self.eval_fitted_curve_cust(self.recorder["train_sizes"], self.recorder["test_scores_mean"], self.recorder["test_scores_std"], validation=validation, **kwargs)


    def eval_fitted_curve_cust(self, train_sizes, test_scores_mean, test_scores_std, validation, P="best", fit=True):
        """ Split the data points in two sets then fit predictors in the first set and evaluate them using RMSE on the second set.

            Args:
                validation (float, int): Percentage or number of samples of the validation set (the highest training sizes will be used).
                P (Predictor, "best"): Predictor to consider
                fit (bool): If True, fit the curve of the Predictors.
            Returns:
                fit_score (float): The Root Mean Squared Error of the validation set against the fitted curve of the Predictor
        """
        valid_abs = get_absolute_value(validation, len(train_sizes))
        train_sizes_val, test_scores_mean_val = train_sizes[-valid_abs:], test_scores_mean[-valid_abs:]
        train_sizes_fit, test_scores_mean_fit, test_scores_std_fit = train_sizes[0:-valid_abs], test_scores_mean[0:-valid_abs], test_scores_std[0:-valid_abs]

        predictors = self.predictors if P == "best" else P

        if not isinstance(predictors, list):
             raise ValueError("P must be a list of Predictors or 'best'.")

        if fit: self.fit_all_cust(train_sizes_fit, test_scores_mean_fit, predictors, sigma=test_scores_std_fit)
        P = self.best_predictor(predictors)

        return mean_squared_error(test_scores_mean_val, P(train_sizes_val))**0.5


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
        if len(self.recorder) == 0: raise RuntimeError("Recorder is empty. You must first compute learning curve data points using the train method.")
        if isinstance(P, str) : P = self.get_predictor(P)
        return self.threshold_cust(P, self.recorder["train_sizes"], **kwargs)


    def threshold_cust(self, P, x, threshold=0.99, max_scaling=1, resolution=1e4, strategies=dict(max_scaling=1, threshold=-0.01), **kwargs):
        """ Find the training set size providing the highest accuracy up to a predefined threshold. 

            P(x) = y and for x -> inf, y -> saturation value. This method approximates x_thresh such as P(x_thresh) = threshold * saturation value. 

            Args:
                P (str, Predictor): The predictor to use for the calculation of the saturation value.    
                x (array): Training set sizes
                threshold (float): In [0.0, 1.0]. Percentage of the saturation value to use for the calculus of the best training set size.
                max_scaling (float): Order of magnitude added to the order of magnitude of the maximum train set size. Generally, a value of 1-2 is enough.
                resolution (float): Only considered for diverging Predictors without inverse function. The higher it is, the more accurate the value of the training set size will be.
                strategies (dict): A dictionary of the values to add / substract to the other parameters in case a saturation value can not be found.
                    If an RecursionError raises, (None, None, sat_val, threshold) will be returned.
                kwargs (dict): Parameters that will be forwarded to internal functions.
            Returns:
                Tuple: (x_thresh, y_thresh, saturation_arrucacy, threshold)
        """
        return self.threshold_cust_inv(P,x,threshold, **kwargs) if callable(P.inv) else self.threshold_cust_approx(P,x,threshold, max_scaling, resolution, strategies, **kwargs)         


    def threshold_cust_inv(self, P, x, threshold, **kwargs):
        """ Find the training set size providing the highest accuracy up to a desired threshold for a Predictor having an inverse function. See :meth:`threshold_cust`. """
        assert callable(P.inv), "P has no inverse function. You have to call threshold_cust_approx instead."
        assert threshold is not None, "No threshold value"

        sat_acc = P.get_saturation()
        desired_acc = sat_acc * threshold            
        opt_trn_size = P.inv(desired_acc)

        if not np.isfinite(opt_trn_size): return np.nan, np.nan, sat_acc, threshold

        return round(opt_trn_size,0), round(desired_acc,4), round(sat_acc,4), threshold


    def threshold_cust_approx(self, P, x, threshold, max_scaling, resolution, strategies, **kwargs):
        """ Find the training set size providing the highest accuracy up to a predefined threshold for a Predictor having no inverse function. See :meth:`threshold_cust`. """
        assert None not in [threshold, max_scaling, resolution], "Parameter has None value"

        sat_acc = P.get_saturation() # if not P.diverging else y[-1]
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
                params = update_params(params, strategies)
                #print(f"Can not find an x_thresh. Trying with parameters: {params}", flush=True)
                try:
                    return self.threshold_cust_approx(P, x, strategies=strategies, **params)
                except RecursionError:
                    return np.nan, np.nan, sat_acc, threshold
            else:
                return np.nan, np.nan, sat_acc, threshold
        
        else:
            opt_trn_size, opt_acc = X[i], y[i]
            return round(opt_trn_size,0), round(opt_acc,4), round(sat_acc,4), threshold


    def best_predictor(self, predictors="all", prefer_conv_delta=2e-3, **kwargs):
        """ Find the Predictor having the best fit of a learning curve. 
        
            Args:
                predictors (list(Predictor), "all"): A list of Predictors to consider.
                prefer_conv_delta (float): If the difference of the two best Predictor fit scores is lower than prefer_conv_delta, then if the converging Predict will be prefered (if any).
                kwargs (dict): Parameters that will be forwarded to internal functions.
            Returns:
                Predictor: The Predicto having the best fit of the learning curve.
        """
        if predictors == "all": predictors = self.predictors

        best_p = None
        for P in predictors:
            if P.score is None: continue
            if best_p is None: best_p = P 
            elif P.score + prefer_conv_delta >= best_p.score: 
                if P.diverging:
                    if best_p.diverging:
                        if P.score > best_p.score: best_p = P
                    else:
                        if P.score - prefer_conv_delta > best_p.score: best_p = P
                else:                    
                    if best_p.diverging: best_p = P
                    elif P.score > best_p.score: best_p = P
        return best_p


    def plot(self, predictor=None, **kwargs):
        """ Plot the training and test learning curves of the recorder data, with optionally fitted functions and saturation. See :meth:`plot_cust`:

            Args:
                predictor (str, list(str), Predictor, list(Predictor)): The predictor(s) to use for plotting the fitted curve. Can also be "all" and "best".
                kwargs (dict): Parameters that will be forwarded to internal functions.
            Returns:
                A Matplotlib figure of the result.
            Raises:
                RuntimeError: If the recorder is empty.
        """
        if len(self.recorder) == 0: raise RuntimeError("recorder is empty. You must first compute learning curve data points using the train method.")
        return self.plot_cust(predictor=predictor, **self.recorder, **kwargs)


    def plot_cust(self, train_sizes, train_scores_mean, train_scores_std, test_scores_mean, test_scores_std,
                  predictor=None, xlim=None, ylim=None, figsize=(12,6), title=None, saturation=None, 
                  max_scaling=1, validation=0, close=False, uncertainty=False, **kwargs):
        """ Plot any training and test learning curves, with optionally fitted functions and saturation.
        
            Args:
                train_sizes (list): Training sizes (x values).
                train_scores_std (list): Train score standard deviations.
                test_scores_mean (list): Test score means(y values).
                test_scores_std (list): Train score means.
                predictor (str, list(str), Predictor, list(Predictor)): The predictor(s) to use for plotting the fitted curve. Can be "all" and "best".
                xlim (2uple): Limits of the x axis of the plot.
                ylim (2uple): Limits of the y axis of the plot.
                figsize (2uple): Size of the figure
                title (str): Title of the figure
                saturation (str, list(str), Predictor, list(Predictor)): Predictor(s) to consider for displaying the saturation on the plot. Can be "all" and "best".
                max_scaling (float): Order of magnitude added to the order of magnitude of the maximum train set size. Generally, a value of 1-2 is enough. 
                validation (float): Percentage or number of data points to keep for validation of the curve fitting (they will not be used during the fitting but displayed afterwards)
                close (bool): If True, close the figure before returning it. This is usefull if a lot of plots are being created because Matplotlib won't close them, potentially leading to warnings.
                    If False, the plot will not be closed. This can be desired when working on Jupyter notebooks, so that the plot will be rendered in the output of the cell.
                uncertainty (bool): If True, plot the uncertainty of the best fitted curve.
                kwargs (dict): Parameters that will be forwarded to internal functions.
        """
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        if 'title' is not None: ax.set_title(title)
        ax.set_xlabel("Training examples")
        ax.set_ylabel("Score")
        ax.grid()

        max_train_size = train_sizes[-1] * 1.05 # Extend a bit so that the curves don't stop before the last points.

        if validation > 0:
            valid_abs = get_absolute_value(validation, len(train_sizes))
            train_sizes_val, test_scores_mean_val, test_scores_std_val = \
                train_sizes[-valid_abs:], test_scores_mean[-valid_abs:], test_scores_std[-valid_abs:]

            train_sizes_fit, test_scores_mean_fit, test_scores_std_fit = \
                train_sizes[0:-valid_abs], test_scores_mean[0:-valid_abs], test_scores_std[0:-valid_abs]
        else:
            train_sizes_fit, test_scores_mean_fit, test_scores_std_fit = train_sizes, test_scores_mean, test_scores_std

        ax.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color='#1f77b4')
        ax.fill_between(train_sizes_fit, test_scores_mean_fit - test_scores_std_fit, test_scores_mean_fit + test_scores_std_fit, alpha=0.15, color="g")
        ax.plot(train_sizes, train_scores_mean, 'o-', color='#1f77b4', label="Training score")
        #ax.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
        ax.errorbar(train_sizes_fit, test_scores_mean_fit, test_scores_std_fit, fmt='o-', color="g", label="Cross-validation score", elinewidth=1)

        # Get the list of Predictors to consider
        predictors = []

        if predictor == "best": predictors = self.predictors # If "best", wait for fit_all before retrieving the best Predictor

        elif predictor is not None: 
            to_add = self.get_predictor(predictor)
            predictors += to_add if isinstance(to_add, list) else [to_add]

        if saturation is not None and saturation != "best": # If "best", wait for fit_all before retrieving the best Predictor
            saturation = self.get_predictor(saturation)
            predictors += saturation if isinstance(saturation, list) else [saturation]

        predictors = get_unique_list(predictors) # Remove duplicates
        self.fit_all_cust(train_sizes_fit, test_scores_mean_fit, predictors, sigma=test_scores_std_fit)
        best_p = self.best_predictor()

        x_scale = get_scale(max_train_size)
        val = 10**(x_scale + max_scaling)
        max_abs = val if val > max_train_size else max_train_size
        x_values = np.linspace(train_sizes[0], max_abs, 50)

        # Plot fitted curves
        if predictor == "best": predictors = [best_p]

        for P in [P for P in predictors if P is not None and P.score is not None]:
            best_lbl = best_p == P if isinstance(best_p, Predictor) else False 
            ax = self.plot_fitted_curve(ax, P, x_values, best=best_lbl, **kwargs)

        # Plot saturation
        if saturation == "best": saturation = best_p
        if isinstance(saturation, Predictor): ax = self.plot_saturation(ax, saturation, max_abs, **kwargs)

        # Plot validation of best estimator
        if validation > 0:
            #RMSE = mean_squared_error(test_scores_mean_val, best_p(train_sizes_val))**0.5 # Don't call eval_fitted_curve to prevent refit
            RMSE = self.eval_fitted_curve_cust(train_sizes, test_scores_mean, test_scores_std, validation, best_p, False)
            label = f"Fit CV (rmse:{RMSE:.2e})"
            ax.errorbar(train_sizes_val, test_scores_mean_val, test_scores_std_val, fmt='x', color="r", label=label, elinewidth=1)

        # Plot uncertainty of best estimator
        if uncertainty: ax = self.plot_uncertainty(ax, best_p, train_sizes_val[0], max_train_size)

        # Set limits
        if ylim is not None: ax.set_ylim(ylim)
            # ymin, ymax = ax.get_ylim()
            # if ymin < ylim[0]: ax.set_ylim(bottom=ylim[0])
            # if ymax > ylim[1]: ax.set_ylim(top=ylim[1])

        if xlim is not None: ax.set_xlim(xlim)

        ax.legend(loc=4) # Lower right
        if close: plt.close(fig)

        return fig


    def plot_fitted_curve(self, ax, P, x, scores=True, best=False, best_ls='-.', **kwargs):
        """ Add to figure ax a fitted curve. 
            
            Args:
                ax (Matplotlib.axes): Figure used to print the curve.
                P (Predictor): Predictor to use for the computing of the curve.
                x (array): 1D array (list) representing the training sizes.
                scores (bool): Print the score of each curve fit in the legend if True.
                best (bool): use a higher zorder to make the curve more visible if True.
                best_ls (Matplotlib line-style): line-style of the curve whose Predictor is used for computing saturation accuracy.
                kwargs (dict): Parameters that will be forwarded to internal functions.
            Returns:
                Matplotlib axes: The updated figure.
        """
        trialX = np.linspace(x[0], x[-1], 500)
        score = round(P.score,4) if P.score is not None else ""
        label = P.name
        if scores : label += f" ({score})"
        z = 3 if best else 2
        ls =  best_ls if best else '--'
        lw = 2 if best else None
        ax.plot(trialX, P(trialX), ls=ls, label=label, zorder=z, linewidth=lw)
        return ax


    def plot_saturation(self, ax, P, max_abs, alpha=1, lw=1.3, **kwargs):
        """ Add saturation line to a plot. 

            The saturation line is a red horizontal line that shows the maximum accuracy achievable. This saturation might be unreachable in the case of a converging Predictor
            (a infinite number of samples would be required). If the Predictor is diverging, the saturation value will be 1 (because the Predictor function can be higher than this value).
            A threshold must be specified to calculate the optimal training set size.
        
            Args:
                ax (Matplotlib.axes): Figure used to print the curve.
                P (Predictor): Predictor to use for the computing of the curve.
                max_abs (float): The maximum training set size value to display in the plot.
                alpha (float): In [0.0, 1.0]. Controls the opacity of the saturation line.
                lw (float): line-width of the of the threshold lines.
                kwargs (dict): Parameters that will be forwarded to internal functions.
            Returns:
                Matplotlib axes: The updated figure.
        """
        optx, opty, sat, thresh = self.threshold(P=P, **kwargs)
        if not P.diverging and np.isfinite(sat): 
            ax.axhline(y=sat, c='r', alpha=alpha, lw=lw)
            err = np.sqrt(np.diag(P.cov))[0]
            ax.axhspan(sat - err, min(1,sat + err), alpha=0.05, color='r')        
        if np.isfinite(optx) and optx < max_abs: ax.axvline(x=optx, ls='-', alpha=alpha, lw=lw)
        if np.isfinite(opty): ax.axhline(y=opty, ls='-', alpha=alpha, lw=lw)
        if np.isfinite([thresh,opty]).all(): 
            ax.text(1.02, opty, "{:.2e}".format(optx), va='center', ha="left", bbox=dict(facecolor="w",alpha=0.5), transform=ax.get_yaxis_transform())
            ax.text(0.95, opty, f"{round(thresh*100,2)}%", color="#1f77b4", bbox=dict(color="w",alpha=0.8), va='center', ha="center", transform=ax.get_yaxis_transform())
        return ax


    def plot_uncertainty(self, ax, P, start, max_val):
        """ Add uncertainty of Predictor curve to a plot.

            Args:
                ax (Matplotlib.axes): Figure used to print the uncertainty.
                P (Predictor, str): The predictor to use. Uncertainty is based on standard deviation errors on parameters of the Predictor.
                start (int): first value of the x vector
                max_val (int): last value of the x vector
            Returns:
                Matplotlib axes: The updated figure.
        """
        if isinstance(P, str): P = self.get_predictor(P)
        perr = P.get_error_std()

        # Determine upper and lower boundaries based on perr
        p_upper, p_lower = [], []
        for i in range(len(P.params)):
            params = P.params.copy()
            params[i] += perr[i]
            factor = 1 if P(max_val) < P(max_val, *params) else -1
            p_upper.append(P.params[i] + perr[i] * factor)
            p_lower.append(P.params[i] - perr[i] * factor)

        x = np.linspace(start, max_val, 100)
        ax.fill_between(x, P(x, *p_lower), P(x, *p_upper), alpha=0.1, color='#f1c40f')
        return ax