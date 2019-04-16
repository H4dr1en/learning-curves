import warnings
import numpy as np


class Predictor():
    """ Object representing a function to fit a learning curve (LearningCurve object). """

    def __init__(self, name, func, guess, inv=None, diverging = False):
        """ Create a Predictor.
            Parameters:
                - name: name of the function
                - func: lambda expression, function to fit
                - guess: tuple of initial parameters 
                - inf: lambda expression corresponding to the inverse function of {func}.
                - diverging: Bool, False if the function converge. In this case the first parameter must be the convergence parameter (enforced to be < 1)."""

        self.name = name
        self.func = func
        self.guess = guess
        self.params = self.guess
        self.score = None
        self.cov = {}
        self.diverging = diverging        

        if callable(inv):
            self.inv = lambda x, *args: inv(x, *args) if len(args) > 1 else inv(x, *self.params)
        else:
            self.inv = None


    def __call__(self, x, *args):
        with warnings.catch_warnings():                
            warnings.simplefilter("ignore", RuntimeWarning) 
            return self.func(x, *args) if len(args) > 1 else self.func(x, *self.params)


    def __repr__(self):
        return f"({self.name} [params:{self.params}][score:{self.score}])"


    def get_saturation(self, max_scale=9):
        """ Retrieve the saturation accuracy of the Predictor. Returns 1 if the Predictor is diverging without inverse function."""
    
        if not self.diverging: return self.params[0]

        if callable(self.inv):
            sat_acc = 1     # if predictor is diverging, set saturation accuracy to 1
            sat_val = self.inv(sat_acc)
            while not np.isfinite(sat_val):   # Decrease the saturation accuracy until finding a value that is not inf
                sat_acc -= 0.01
                sat_val = self.inv(sat_acc)

        else: sat_acc = 1 # Default value if diverging Perdictor

        return sat_acc