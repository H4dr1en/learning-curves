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

    def __init__(self, name, func, guess):
        """ Create a Predictor with a given {name}, {function} and initial parameters: {guess}."""
        self.name = name        
        self.func = func
        self.guess = guess
        self.params = self.guess
        self.score = None
        self.cov = {}        

    def __call__(self, x, *args):
        return self.func(x,*args) if len(args) > 1 else self.func(x, *self.params)

    def __repr__(self):
        return f"{self.name} - params:{self.params} - score:{self.score}"


class LearningCurve():

    def __init__(self, predictors=[], scoring=r2_score):
        
        self.predictors = [
            Predictor("pow",        lambda x,a,b,c    : a - b*x**c,                  [1, 1.7,-.5]),
            Predictor("pow_2",      lambda x,a,b,c,d  : a - (b*x+d)**c,              [1, 1.7,-.5, 1e-3]),
            Predictor("pow_log",    lambda x,a,b,c,m,n: a - b*x**c + m*np.log(x**n), [1.3, 1.7,-.5,1e-3,2]),
            Predictor("pow_log_2",  lambda x,a,b,c    : a / (1 + (x/np.exp(b))**c),  [1, 1.7,-.5]),
            Predictor("log_lin",    lambda x,a,b      : np.log(a*np.log(x)+b),       [1, 1.7]),
            Predictor("log",        lambda x,a,b      : a - b/np.log(x),             [1.6, 1.1])
        ]
        
        if len(predictors) > 0:
            self.predictors.append(predictors)

        self.recorder = {}
        self.scoring = scoring


    def save(self, path="./lc_data.pkl"):
        """ Save the LearningCurve object as a pickle object in disk. Use dill to save because the object contains lambda functions. """
        with open(path, 'wb') as f: dill.dump(self, f)


    @staticmethod
    def load(path="./lc_data.pkl"):
        """ Load a LearningCurve object from disk. """
        with open(path, 'rb') as f: return dill.load(f)


    def get_lc(self, estimator, X, Y, train_sizes=None, test_size=0.2, n_splits=3, verbose=1, n_jobs=-1, **kwargs):
        """ Compute and plot the learning curve."""

        self.train(estimator, X, Y, train_sizes=train_sizes, test_size=test_size, n_splits=n_splits, verbose=verbose, n_jobs=n_jobs, **kwargs)
        return self.plot(predictor="best")


    def train(self, estimator, X, Y, train_sizes=None, test_size=0.2, n_splits=3, verbose=1, n_jobs=-1):
        """ Compute the learning curve of an estimator over a dataset. Returns an object that can then be passed to plot_lc function.
            Parameters:
                - estimator: object type that implements the “fit” and “predict” methods.
                - train_sizes: See sklearn learning_curve function documentation. Default is 
                - n_splits: int. Number of random cross validation calculated for each train size
                - verbose: int. The higher, the more verbose.
                - n_jobs: See sklearn learning_curve function documentation.
        """         

        if train_sizes is None:
            # [0.001, 0.003, 0.005, 0.007, 0.01, 0.05, 0.09, 0.13, 0.17, 0.2, 0.28, 0.36, 0.44, 0.52, 0.6, 0.68, 0.76]
            train_sizes = [i/1000 for i in range(1,9,2)] + [i/100 for i in range(1,20,4)] + [i/50 for i in range(10,40,4)]

        cv = ShuffleSplit(n_splits=n_splits, test_size=test_size)
        t_start = time.perf_counter()        
        train_sizes, train_scores, test_scores = learning_curve(estimator, X, Y, 
                                                                cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, verbose=verbose)  
        self.recorder["data"] = {
            "total_size"          : len(X),
            "train_sizes"         : train_sizes,
            "train_scores_mean"   : np.mean(train_scores, axis=1),
            "train_scores_std"    : np.std(train_scores, axis=1),
            "test_scores_mean"    : np.mean(test_scores, axis=1),
            "test_scores_std"     : np.std(test_scores, axis=1),
            "time"                : time.perf_counter() - t_start
        }  

        gc.collect()         

        return self.recorder["data"]


    def get_predictor(self, name):
        """ Get the first predictor with matching {name}. Returns None if no predictor matches. """        
        return next((P for P in self.predictors if P.name == name), None)


    def fit_all(self, x, y):
        """ Fit a curve with all the functions and retrieve r2 score if y_pred is finite.
            Returns an array of predictors with the updated params and score."""

        predictors = []
        for p in self.predictors:
            try:
                predictors.append(self.fit(p,x,y))
            except RuntimeError:
                warnings.warn(f"{p.name}: Impossible to fit the learning curve (change initial gess).")
        return predictors # [self.fit(p,x,y) for p in self.predictors] # No error handling


    def fit(self, predictor, x, y):
        """ Fit a curve with a predictor and retrieve score (default:R2) if y_pred is finite.
            Returns the predictor with the updated params and score."""

        assert isinstance(predictor, Predictor), "The given Predictor is not a Predictor object (or could not be found)."

        predictor.params, predictor.cov = optimize.curve_fit(predictor, x, y, predictor.params)
        #try:
        y_pred = predictor(x)
        predictor.score = self.scoring(y,y_pred) if np.isfinite(y_pred).all() else np.nan    
        #except ValueError:
        #    print(x, P(x))
        return predictor


    def best_predictor(self):
        """ Find the best predictor of the LearningCurve data for the test score learning curve."""

        if not 'data' in self.recorder:
            raise RuntimeError("recorder is empty. You must first compute learning curve data points using the train method.")
        return self.best_predictor_cust(self.recorder["data"]["train_sizes"], self.recorder["data"]["test_scores_mean"])


    def best_predictor_cust(self, x, y):
        """ Find the best predictor for the test score learning curve."""

        best_p = self.predictors[0]
        for P in self.fit_all(x,y):
            if P.score is not None and P.score > best_p.score: 
                best_p = P
        return best_p


    def plot(self, predictor=None, **kwargs):
        """ Plot the training and test learning curve of the LearningCurve data, and optionally a fitted function. 
            - predictor: The name of the predictor to use for fitting the learning curve. Can also be "all" or "best".
        """
        if not 'data' in self.recorder:
            raise RuntimeError("recorder is empty. You must first compute learning curve data points using the train method.")
        return self.plot_cust(predictor=predictor, **self.recorder["data"], **kwargs)


    def plot_cust(self, train_sizes, train_scores_mean, train_scores_std, test_scores_mean, test_scores_std, 
                        predictor=None, ylim=None, figsize=None, title=None, scores=True, **kwargs):
        """ Plot any training and test learning curve, and optionally a fitted function. """
    
        fig, ax = plt.subplots(1,1,figsize=figsize)
        if 'title' is not None:
            ax.set_title(title)
        if ylim is not None:
            ax.ylim(*ylim)
        ax.set_xlabel("Training examples")
        ax.set_ylabel("Score")
        ax.grid()
        ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std, alpha=0.1,
                        color="r")
        ax.fill_between(train_sizes, test_scores_mean - test_scores_std,
                        test_scores_mean + test_scores_std, alpha=0.15, color="g")
        ax.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
        ax.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

        predictors_to_print = []

        if predictor == "all":            
            predictors_to_print = self.fit_all(train_sizes, test_scores_mean)
        elif predictor == "best":    
            self.fit_all(train_sizes, test_scores_mean)
            predictors_to_print.append(self.best_predictor_cust(train_sizes, test_scores_mean))  
        elif predictor is not None:
            P = self.get_predictor(predictor)
            if P is not None:
                self.fit(P, train_sizes, test_scores_mean)
                predictors_to_print.append(P) 

        for predictor in predictors_to_print:
            ax = self.plot_fitted_curve(ax, predictor, train_sizes, scores)          
                
        ax.legend(loc="best")
        plt.close(fig)
        return fig


    def plot_fitted_curve(self, ax, P, x, score=True):
        """ Add to figure ax a fitted curve. """
        trialX = np.linspace(x[0], x[-1], 500)
        label = P.name + f" ({round(P.score,4)})" if score is True and P.score is not None else ""
        ax.plot(trialX, P(trialX), ls='--', label=label)
        return ax