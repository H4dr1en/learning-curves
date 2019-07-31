from .learning_curves import LearningCurve
import matplotlib.pyplot as plt
import numpy as np


class LearningCurveCV():
    """ Provide helper functions to plot, fit and extrapolate learning curve using cross validation. """

    def __init__(self, n):
        self.lcs = [LearningCurve(name=f"cv_{i}") for i in range(n)]

    def train(self, *args, **kwargs):
        for i, lc in enumerate(self.lcs):
            if "verbose" in kwargs and kwargs["verbose"] > 0:
                print(f"[LearningCurveCV] {i} / {len(self.lcs)}")
            lc.train(*args, **kwargs)

    def get_accuracies(self, train_sizes, predictor="best"):
        ret = []    
        for lc in self.lcs:
            lc.fit_all()
            ret.append(lc.get_predictor(predictor)(train_sizes))
        return ret

    def get_dist(self, accuracies):
        return np.mean(accuracies, axis=0), np.std(accuracies, axis=0)

    def plot(self, target=None, figsize=(12, 6)):
        plt.figure(figsize=figsize)
        train_sizes = self.lcs[0].recorder["train_sizes"]
        acc = self.get_accuracies(train_sizes)
        acc_mean, acc_std = self.get_dist(acc)
        plt.title(f"Learning Curve CV N={len(self.lcs)}")
        plt.grid()
        plt.xlabel(f"Training sizes")
        plt.ylabel(f"Accuracy")
        plt.errorbar(train_sizes, acc_mean, acc_std, fmt="o-")
        plt.fill_between(train_sizes, acc_mean - acc_std, acc_mean + acc_std, color="r", alpha=0.1)

        if target is not None:
            y_target = self.target(target)
            plt.errorbar(target, y_target[0], y_target[1], fmt="o-", label=f"target:{target}")
            #plt.legend()

    def plot_all(self, **kwargs):
        return LearningCurve.compare(self.lcs, **kwargs)

    def plot_times(self, **kwargs):
        return LearningCurve.compare_time(self.lcs, **kwargs)        

    def target(self, n):
        return self.get_dist(self.get_accuracies([n]))

    def fit_all(self, **kwargs):
        for lc in self.lcs:
            lc.fit_all(**kwargs)
