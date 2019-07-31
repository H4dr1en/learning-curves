from .learning_curves import LearningCurve
import matplotlib.pyplot as plt
import numpy as np


class LearningCurveCombined():
    """ Provide helper functions to plot, fit and extrapolate learning curve using multiple learning curves."""

    def __init__(self, n):
        """ Instantiante a LearningCurveCombined object.

            Args:
                n (int): Number of learning curves to use. More learning curves will result in a better estimation 
                            of the extrapolation but will be longer to compute.
        """
        self.lcs = [LearningCurve(name=f"cv_{i}") for i in range(n)]

    def train(self, *args, **kwargs):
        """ Train all learning curves using :meth:`LearningCurve.train` method. Parameters are forwarded to this method."""
        for i, lc in enumerate(self.lcs):
            if "verbose" in kwargs and kwargs["verbose"] > 0:
                print(f"[LearningCurveCV] {i} / {len(self.lcs)}")
            lc.train(*args, **kwargs)

    def get_scores(self, train_sizes, predictor="best"):
        """ Fit each learning curve and compute the scores for the given predictor.

            Args:
                train_sizes (list): array of train sizes.
                predictor (string, Predictor): Predictor to use.
            Returns:
                List: list of size n containing arrays with the same length as train_sizes

        """
        ret = []    
        for lc in self.lcs:
            lc.fit_all()
            ret.append(lc.get_predictor(predictor)(train_sizes))
        return ret

    def get_dist(self, scores):
        """ Compute the mean and the standard deviation of the result of :meth:`LearningCurveCombined.get_scores` """
        return np.mean(scores, axis=0), np.std(scores, axis=0)

    def plot(self, target=None, figsize=(12, 6), close=True, legend=True):
        """ Plot the combined learning curve.

            Args:
                target (int): Training size to reach. The training size axis will be extended and the fitted curve extrapolated until reaching this value.
                figsize (2uple): Size of the figure
                close (bool): If True, close the figure before returning it. This is usefull if a lot of plots are being created because Matplotlib won't close 
                    them, potentially leading to warnings. If False, the plot will not be closed. 
                    This can be desired when working on Jupyter notebooks, so that the plot will be rendered in the output of the cell.
                legend (bool): Controls whether to show legend or not.
            Returns:
                fig (Matplotlib.figure): The resulting plot
        """

        if len(self.lcs[0].recorder) == 0:
            raise RuntimeError("recorder is empty. You must first compute learning curve data points using the train method.")

        fig = plt.figure(figsize=figsize)
        train_sizes = self.lcs[0].recorder["train_sizes"]

        scores = self.get_scores(train_sizes)
        scores_mean, scores_std = self.get_dist(scores)

        plt.title(f"Combined learning curve (N={len(self.lcs)})")
        plt.grid()
        plt.xlabel(f"Training sizes")
        plt.ylabel(f"Scores")
        plt.plot(train_sizes, scores_mean, "o-", label="Averaged scores")
        plt.fill_between(train_sizes, scores_mean - scores_std, scores_mean + scores_std, color="r", alpha=0.15, label="Standard deviation")

        if target is not None:
            y_target = self.target(target)
            plt.errorbar(target, y_target[0], y_target[1], fmt="o-", label=f"target:{target}")

            train_sizes = np.linspace(train_sizes[-1], target)
            scores = self.get_scores(train_sizes)
            scores_mean, scores_std = self.get_dist(scores)
            plt.plot(train_sizes, scores_mean, "--", label="Averaged extrapolation")
            plt.fill_between(train_sizes, scores_mean - scores_std, scores_mean + scores_std, color="r", alpha=0.15)

        if legend is True:
            plt.legend(loc=4)

        if close is True:
            plt.close(fig)

        return fig

    def plot_all(self, legend=True, **kwargs):
        """ Calls :meth:`LearningCurve.compare` to plot the different learning curves. 

            Args:
                legend (bool): Controls whether to show legend or not.
        """
        fig = LearningCurve.compare(self.lcs, **kwargs)

        if legend is False:
            fig.axes[0].get_legend().remove()

        return fig

    def plot_times(self, **kwargs):
        """ Calls :meth:`LearningCurve.compare_time` to plot the different learning curves times. """
        return LearningCurve.compare_time(self.lcs, **kwargs)        

    def target(self, n):
        """ Shorcut to get the mean and the standard deviation for one value. """
        return self.get_dist(self.get_scores([n]))

    def fit_all(self, **kwargs):
        """ Fit each learning curve. """
        for lc in self.lcs:
            lc.fit_all(**kwargs)