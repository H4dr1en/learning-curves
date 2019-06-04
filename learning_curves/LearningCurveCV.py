from .learning_curves import LearningCurve
import matplotlib.pyplot as plt
import numpy as np


class LearningCurveCV():

    def __init__(self, n):
        self.lcs = []
        for i in range(n):
            self.lcs.append(LearningCurve(name=f"cv_{i}"))

    def train(self, *args, **kwargs):
        for i, lc in enumerate(self.lcs):
            if "verbose" in kwargs and kwargs["verbose"] > 0:
                print(f"[LearningCurveCV] {i} / {len(self.lcs)}")
            lc.train(*args, **kwargs)

    def get_accuracies(self, train_sizes):
        ret = []    
        for lc in self.lcs:
            lc.fit_all()
            ret.append(lc.best_predictor()(train_sizes))
        return ret

    def get_dist(self, accuracies):
        return np.mean(accuracies, axis=0), np.std(accuracies, axis=0)

    def plot(self):
        plt.figure(figsize=(12, 6))
        train_sizes = self.lcs[0].recorder["train_sizes"]
        acc = self.get_accuracies(train_sizes)
        acc_mean, acc_std = self.get_dist(acc)
        plt.plot(train_sizes, acc_mean, "o-")
        plt.fill_between(train_sizes, acc_mean - acc_std, acc_mean + acc_std, color="r", alpha=0.1)

    def plot_all(self, **kwargs):
        return LearningCurve.compare(self.lcs, **kwargs)

    def target(self, n):
        return self.get_dist(self.get_accuracies([n]))
