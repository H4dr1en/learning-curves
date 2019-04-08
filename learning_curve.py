# coding:utf-8
import warnings
import gc
import time 

from sklearn.metrics import r2_score
import scipy.optimize as optimize
import numpy as np
from sklearn.model_selection import train_test_split, ShuffleSplit, learning_curve

import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText


class Predictor():

    def __init__(self, name, func, guess, scoring=r2_score):
        """ Create a Predictor with a given {name}, {function} and initial parameters: {guess}. A custom scoring function can also be used."""
        self.name = name        
        self.func = func
        self.guess = guess
        self.params = self.guess
        self.scoring = scoring
        self.score = None
        self.cov = {}        

    def __call__(self, x, *args):
        return self.func(x,*args) if len(args) > 1 else self.func(x, *self.params)


class LearningCurve():

    def __init__(self, predictors=[]):
        if len(predictors) > 0:
            self.predictors = predictors
        else:
            def exp_log(x,a,b,c,m,n):    
                return a-b*x**c + m*np.log(x**n)
            def exp(x,a,b,c):    
                return a-b*x**c
            self.predictors = [
                Predictor("exp",     exp,      [1.3, 1.7,-.5]),
                Predictor("exp_log", exp_log,  [1.3, 1.7,-.5,1e-3,2])
            ]
        self.recorder = {}

    def get_lc(self, estimator, X, Y, train_sizes=None, test_size=0.2, n_splits=3, verbose=1, n_jobs=-1, **kwargs):
        """ Compute and plot the learning curve."""
        self.train(estimator, X, Y, train_sizes=None, test_size=0.2, n_splits=3, verbose=verbose, n_jobs=-1, **kwargs)
        return self.plot_lc(predictor="best")

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
        return [self.fit(p,x,y) for p in self.predictors]


    def fit(self, predictor, x, y):
        """ Fit a curve with a predictor and retrieve r2 score if y_pred is finite.
            Returns the predictor with the updated params and score."""

        P = self.get_predictor(predictor) if isinstance(predictor, str) else predictor
        assert isinstance(P, Predictor), "The given Predictor is not a Predictor object (or could not be found)."

        P.params, P.cov = optimize.curve_fit(P, x, y, P.params)
        #try:
        y_pred = P(x)
        P.score = P.scoring(y,y_pred) if np.isfinite(y_pred).all() else np.nan    
        #except ValueError:
        #    print(x, P(x))
        return P


    def best_predictor(self):
        """ Find the best predictor of the LearningCurve data for the test score learning curve."""

        if not 'data' in self.recorder:
            raise RuntimeError("recorder is empty. You must first compute learning curve data points using the train method.")
        return self.best_predictor_cust(self.recorder["data"]["train_sizes"], self.recorder["data"]["test_scores_mean"])

    def best_predictor_cust(self, x, y):
        """ Find the best predictor for the test score learning curve."""
        
        first_p = self.predictors[0]
        if not all([first_p.scoring == P.scoring for P in self.predictors]):
            warnings.warn("Scoring functions should be the same accross Predictors in order to have comparable results.")

        best_p = first_p
        for P in self.fit_all(x,y):
            if P.score > best_p.score: 
                best_p = P
        return P.name, P.score, P


    def plot_lc(self, predictor=None, ylim=None, figsize=None, title=None, **kwargs):
        """ Plot the training and test learning curve of the LearningCurve data, and optionally a fitted function. 
            - predictor: The name of the predictor to use for fitting the learning curve. Can also be "all" or "best".
        """
        if not 'data' in self.recorder:
            raise RuntimeError("recorder is empty. You must first compute learning curve data points using the train method.")
        return self.plot_cust_lc(**self.recorder["data"], **kwargs)


    def plot_cust_lc(self, train_sizes, train_scores_mean, train_scores_std, test_scores_mean, test_scores_std, 
                        predictor=None, ylim=None, figsize=None, title=None, **kwargs):
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
            self.fit_all(train_sizes, test_scores_mean)
            predictors_to_print = self.predictors
        elif predictor == "best":    
            self.fit_all(train_sizes, test_scores_mean)
            predictors_to_print.append(self.best_predictor(train_sizes, test_scores_mean)[2])  
        elif predictor is not None:
            P = self.get_predictor(predictor)
            if P is not None:
                self.fit(P, train_sizes, test_scores_mean)
                predictors_to_print.append(P) 

        for predictor in predictors_to_print:
            ax = self.plot_fitted_curve(ax, predictor, train_sizes)             
                
        ax.legend(loc="best")
        plt.close(fig)
        return fig


    def plot_fitted_curve(self, ax, P, x):
        """ Add to ax figure a fitted curve. """
        # Plot fitted curve
        trialX = np.linspace(x[0], x[-1], 500)
        ax.plot(trialX, P(trialX), ls='--', label=P.name)
        # Print score
        text = f"{P.scoring.__name__}:{round(P.score,4)}"
        text = AnchoredText(text, loc=2)   
        ax.add_artist(text)
        return ax
