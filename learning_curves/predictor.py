import warnings
import numpy as np


class Predictor():
    """ Object representing a function to fit a learning curve (See :class:`learning_curves.LearningCurve`). """

    def __init__(self, name, func, guess, inv=None, diverging = False):
        """ Create a Predictor.

            Args:
                name (str): name of the function
                func (Callable): lambda expression, function to fit
                guess (Tuple): Initial parameters 
                inv (Callable): lambda expression corresponding to the inverse function.
                diverging (bool): False if the function converge. In this case the first parameter must be the convergence parameter (enforced to be in [0,1]).
        """
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


    def get_saturation(self):
        """ Retrieve the saturation accuracy of the Predictor. 

            The saturation accuracy is the best accuracy you will get from the model without changing any other parameter than the training set size.
            If the Predictor is diverging, this value should be disregarded, being meaningless.
        
            Returns:
                float: saturation accuracy of the Predictor. 
                    This value is 1 if the Predictor is diverging without inverse function.
                    This valus is the first parameter of the Predictor if it is converging.
                    This value is calculated if the Predictor is diverging with inverse function.
        """
        if not self.diverging: return self.params[0]

        if callable(self.inv):
            sat_acc = 1     # if predictor is diverging, set saturation accuracy to 1
            sat_val = self.inv(sat_acc)
            while not np.isfinite(sat_val):   # Decrease the saturation accuracy until finding a value that is not inf
                sat_acc -= 0.01
                sat_val = self.inv(sat_acc)

        else: sat_acc = 1 # Default value if diverging Perdictor

        return sat_acc