import warnings
import numpy as np


class Predictor():
    """ Object representing a function to fit a learning curve (See :class:`learning_curves.LearningCurve`). """

    def __init__(self, name, func, guess, inv=None, diverging=False, bounds=None):
        """ Create a Predictor.

            Args:
                name (str): name of the function
                func (Callable): lambda expression, function to fit
                guess (Tuple): Initial parameters 
                inv (Callable): lambda expression corresponding to the inverse function.
                diverging (bool): False if the function converge. In this case the first parameter must be the convergence parameter (enforced to be in [-inf,1]).
                bounds (array of tuples): Bounds of the parameters. Default is [-inf, inf] for all parameters, except for the convergence parameter whose bounds are [-inf,1]
                    if diverging is True.

        """
        self.name = name
        self.func = func
        self.guess = guess
        self.params = self.guess
        self.score = None
        self.cov = {}
        self.diverging = diverging
        self.params_up = None     
        self.params_low = None        

        if callable(inv):
            self.inv = lambda x, *args: inv(x, *args) if len(args) > 0 else inv(x, *self.params)
        else:
            self.inv = None

        if bounds:
            self.bounds = bounds
        else:
            self.bounds = (-np.inf, np.inf) if self.diverging else ([-np.inf] * (len(self.params)), [1]+[np.inf] * (len(self.params) - 1))


    def __call__(self, x, *args):
       # with warnings.catch_warnings():                
            #warnings.simplefilter("ignore", RuntimeWarning) 
        x = np.array(x) # Enforce x to be a np array because a list of floats would throw a TypeError
        return self.func(x, *args) if len(args) > 1 else self.func(x, *self.params)


    def __repr__(self):
        return f"Predictor {self.name} with params {self.params} and score {self.score}"


    def get_saturation(self):
        """ Compute the saturation accuracy of the Predictor. 

            The saturation accuracy is the best accuracy you will get from the model without changing any other parameter than the training set size.
            If the Predictor is diverging, this value should be disregarded, being meaningless.
        
            Returns:
                float: saturation accuracy of the Predictor. 
                    This value is 1 if the Predictor is diverging without inverse function.
                    This valus is the first parameter of the Predictor if it is converging.
                    This value is calculated if the Predictor is diverging with inverse function.
        """
        if not self.diverging: sat_acc = self.params[0]

        elif callable(self.inv):
            sat_acc = 1     # if predictor is diverging, set saturation accuracy to 1
            sat_val = self.inv(sat_acc)
            while not np.isfinite(sat_val):   # Decrease the saturation accuracy until finding a value that is not inf
                sat_acc -= 0.01
                sat_val = self.inv(sat_acc)

        else: sat_acc = 1 # Default value if diverging Perdictor

        return sat_acc


    def __eq__(self, other): 
        if not isinstance(other, Predictor): return RuntimeError("Trying to compare Predictor with not Predictor object.")
        return self.name == other.name

    
    def get_error_std(self):
        """ Compute the standard deviation errors on the parameters. """
        return np.sqrt(np.diag(self.cov))


    def get_fit_std_params(self, start, end):
        """ Compute the parameters giving the lowest and the highest fit curve possible with respect to the standard deviation of the parameters. """
        
        perr = self.get_error_std()

        # Determine upper and lower boundaries params based on perr
        params_up, params_low = [], []
        for i in range(len(self.params)):
            params = self.params.copy()
            params[i] += perr[i]
            factor = 1 if self(end) < self(end, *params) else -1
            params_up.append(self.params[i] + perr[i] * factor)
            params_low.append(self.params[i] - perr[i] * factor)

        # Restraining params_up inside bounds
        for i in range(len(self.params)):
            if self.bounds == (-np.inf, np.inf): continue
            if self.bounds[0][i] > params_up[i]:   params_up[i] = self.bounds[0][i]
            elif self.bounds[1][i] < params_up[i]: params_up[i] = self.bounds[1][i]

        # Restraining params_low inside bounds
        for i in range(len(self.params)):
            if self.bounds == (-np.inf, np.inf): continue
            if self.bounds[0][i] > params_low[i]:   params_low[i] = self.bounds[0][i]
            elif self.bounds[1][i] < params_low[i]: params_low[i] = self.bounds[1][i]

        self.params_low = params_low
        self.params_up = params_up
        
        return self.params_low, self.params_up