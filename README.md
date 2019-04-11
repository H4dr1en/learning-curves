# learning-curves

Learning-curves is Python module that extends [sklearn's learning curve feature](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.learning_curve.html). It will help you visualizing the learning curve of your models.

Learning curves give an opportunity to diagnose bias and variance in supervised learning models, but also to visualize how training set size influence the performance of the models (more informations [here](https://www.dataquest.io/blog/learning-curves-machine-learning/)).

Such plots help you answer the following questions:
 - Do I have enough data?
 - What would be the best accuracy I would have if I had more data?
 - Can I train my model with less data?
 - Is my training set biased?
 
 Learning-curves will also help you fitting the learning curve to extrapolate and find the saturation value of the curve.

### Installation

```
$ pip install learning-curves
```

To create learning curve plots, first import the module with `import learning_curves`.

### Usage

It is as simple as:

```
lc = LearningCurve()
lc.get_lc(estimator, X, Y)
```
Where `estimator` implements `fit(X,Y)` and `predict(X,Y)`.

Output:

![alt text](https://github.com/H4dr1en/learning-curves/blob/master/images/learning_curve_no_fit.png)

On this example the green curve suggests that adding more data to the training set is likely to improve a bit the model accuracy. The green curve also shows a saturation near 0.96. We can easily fit a function to this curve:

```
lc.plot(predictor="best")
```
Output:

![alt text](https://github.com/H4dr1en/learning-curves/blob/master/images/learning_curve_simple.png)

Here we used a predefined function, `pow`, to fit the green curve. The R2 score is very close to 1, meaning that the fit is optimal. We can therefore use this curve to extrapolate the evolution of the accuracy with the training set size.

This also tells us how many data we should use to train our model to maximize performances and accuracy.

### Add custom functions to fit the learning curve
Such function are called `Predictor`. You can create a `Predictor` like this:
```
predictor = Predictor("myPredictor", lambda x,a,b : a*x + b, [1,0])
```
Here we created a Predictor called "myPredictor" with the function `y(x) = a*x + b`.
Because internally SciPy `optimize.curve_fit` is called, a first guess of the parameters `a` and `b` are required. Here we gave them respective value 1 and 0.
You can then add the `Predictor` to the `LearningCurve` object in two different ways:
- Pass the `Predictor` to the `LearningCurve` constructor:
```
lc = LearningCurve([predictor])
```
- Register the `Predictor` inside the predictors of the `LearningCurve` object:
```
lc.predictors.append(predictor)
```

By default, 4 `Predictors` are instantiated: 
```
self.predictors = [
    Predictor("pow",        lambda x, a, b, c, d    : a - (b*x+d)**c,                [1, 1.7, -.5, 1e-3]),
    Predictor("pow_log",    lambda x, a, b, c, m, n : a - b*x**c + m*np.log(x**n),   [1, 1.7, -.5, 1e-3, 1e-3], True),
    Predictor("pow_log_2",  lambda x, a, b, c       : a / (1 + (x/np.exp(b))**c),    [1, 1.7, -.5]),
    Predictor("inv_log",    lambda x, a, b          : a - b/np.log(x),               [1, 1.6])
]
```
Some predictors perform better (R2 score is closer to 1) than others, depending on the dataset, the model and the value to be preditected. 

### Find the best Predictor

To find the Predictor that will fit best your learning curve, we can call `get_predictor` function:
```
lc.get_predictor("best")
```
Output:
```
(pow [params:[   0.9588563    11.74747659   -0.36232639 -236.46115903]][score:0.9997458683912492])
```

### Plot the Predictors

You can plot any `Predictor`s fitted function with the `plot` function:
```
lc.plot(predictor="all")
```
Output:

![alt text](https://github.com/H4dr1en/learning-curves/blob/master/images/learning_curve_all.png)

### Save and load LearningCurve instances

Because `Predictor` contains lambda functions, you can not simply save a `LearningCurve` instance. One possibility is to only save the data points of the curve inside `lc.recorder["data"]` and retrieve then later on. But then the custom predictors are not saved. Therefore it is recommended to use the `save` and `load` methods:
```
lc.save("path/to/save.pkl")
lc = LearningCurve.load("path/to/save.pkl")
```
This internally uses the `dill` library to save the `LearningCurve` instance with all the `Predictor`s.

### Find the best training set size

`learning-curves` will help you finding the best training set size by extrapolation of the best fitted curve:
```
lc.plot(predictor="all", saturation="best")
```
Output:

![alt text](https://github.com/H4dr1en/learning-curves/blob/master/images/learning_curve_fit_sat_all.png)

The horizontal red line shows the saturation of the curve. The intersection of the two blue lines shows the best accuracy we can get, given a certain `threshold` (see below).

To retrieve the value of the best training set size:
```
lc.threshold(predictor="best", saturation="best")
```
Output:
```
(0.9589, 31668, 0.9493)
```
This tells us that the saturation value (the maximum accuracy we can get from this model without changing any other parameter) is `0.9589`. This value corresponds to an infinite number of samples in our training set! But with a threshold of `0.99` (this parameter can be changed with `threshold=x`), we can have an accuracy `0.9493` if our training set contains `31668` samples.

Note: The saturation value is always the _second parameter_ of the function. Therefore, if you create your own `Predictor`, place the saturation factor in second position (called a in the predefined `Predictor`s). If the function of your custom `Predictor` is diverging, then no saturation value can be retrieven. In that case, pass `diverging=True` to the constructor of the `Predictor`. The saturation value will then be calculated considering the `max_scaling` parameter of the 
`threshold_cust` function (see documentation for details). You should set this parameter to the maximum number of sample you can add to your training set.

## Documentation

Some functions have their `function_name_cust` equivalent. Calling the function without the `_cust` suffix will internally call the function with the `_cust` suffix with default parameters (such as the data points of the learning curves). Thanks to `kwargs`, you can pass exactly the same parameters to both functions.

| Function/Class         | Parameters        | Type                                       | Default     | Description                                                                                                                                                                                                                                                                                   |
|------------------------|-------------------|--------------------------------------------|-------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Predictor.\_\_init\_\_ |                   |                                            |             | Instantiate a `Predictor` object.                                                                                                                                                                                                                                                             |
|                        | name              | str                                        | _Required_  | Name of the `Predictor`                                                                                                                                                                                                                                                                       |
|                        | func              | Lambda                                     | _Required_  | Lambda function used for fitting of the learning curve                                                                                                                                                                                                                                        |
|                        | guess             | List                                       | _Required_  | Starting parameters used for fitting the curve                                                                                                                                                                                                                                                |
|                        | diverging         | Bool                                       | False       | If the function is diverging, set diverging to True. If the function is converging, then the first parameter of the function has to be the convergence value.                                                                                                                                 |
| LC.\_\_init\_\_        |                   |                                            |             | Instantiate a `LearningCurve` object.                                                                                                                                                                                                                                                         |
|                        | predictors        | List                                       | empty       | Predictors to add to the `LearningCurve` object                                                                                                                                                                                                                                               |
|                        | scoring           | Callable                                   | r2_score    | Scoring function used to evaluate the fits of the learning curve                                                                                                                                                                                                                              |
| LC.get_lc              |                   |                                            |             | Compute and plot the learning curve                                                                                                                                                                                                                                                           |
|                        | estimator         | Object                                     | _Required_  | Model (any object implementing `fit(X,Y)` and `predict(X,Y)` methods)                                                                                                                                                                                                                         |
|                        | X                 | array                                      | _Required_  | X numpy array used for prediction                                                                                                                                                                                                                                                             |
|                        | Y                 | array                                      | _Required_  | Y numpy array used for prediction                                                                                                                                                                                                                                                             |
| LC.train               |                   |                                            |             | Compute the learning curve of an estimator over a dataset. Returns an object that can then be passed to plot_lc function                                                                                                                                                                      |
|                        | X                 | array                                      | _Required_  | X numpy array used for prediction                                                                                                                                                                                                                                                             |
|                        | Y                 | array                                      | _Required_  | Y numpy array used for prediction                                                                                                                                                                                                                                                             |
|                        | train_sizes       | List                                       | Predefined  | List of training size used for calculating the learning curve. Can be a  list of floats between 0 and 1 (assumed to be percentages)  or a list of integers (assumed to be number of values)                                                                                                   |
|                        | test_size         | int/float                                  | 0.2         | percentage / value of the test set size                                                                                                                                                                                                                                                       |
|                        | n_splits          | int                                        | 3           | Number of splits used for cross validation                                                                                                                                                                                                                                                    |
|                        | verbose           | int                                        | 1           | The higher, the more verbose                                                                                                                                                                                                                                                                  |
|                        | n_jobs            | int                                        | -1          | Number of workers. -1 sets to maximum possible. See sklearn.                                                                                                                                                                                                                                  |
| LC.get_predictor       |                   |                                            |             | Get the first predictor with matching {name}. Returns None if no predictor matches.                                                                                                                                                                                                           |
|                        | pred              | str, List(str), Predictor, List(Predictor) | _Required_  | Name of the predictor(s). Can be "all" or "best" or even Predictor(s).                                                                                                                                                                                                                        |
| LC.fit_all             |                   |                                            |             | Fit a curve with all the Predictors and retrieve score if y_pred is finite. Returns an array of predictors with the updated params and score.                                                                                                                                                 |
|                        | x                 | Array                                      | _Required_  | 1D array (list) representing the training sizes                                                                                                                                                                                                                                               |
|                        | y                 | Array                                      | _Required_  | 1D array (list) representing the scores                                                                                                                                                                                                                                                       |
| LC.fit_all_cust        |                   |                                            |             | Same as `fit_all`                                                                                                                                                                                                                                                                             |
|                        | x,y               | Array                                      | _Required_  | See `fit_all`                                                                                                                                                                                                                                                                                 |
|                        | predictors        | List(Predictors)                           | _Required_  | The predictors to use for the fitting.                                                                                                                                                                                                                                                        |
| LC.fit                 |                   |                                            |             | Fit a curve with a predictor and retrieve score (default:R2) if y_pred is finite. Returns the predictor with the updated params and score.                                                                                                                                                    |
|                        | predictor         | Predictor                                  | _Required_  | The predictor to use for fitting the learning curve                                                                                                                                                                                                                                           |
|                        | x                 | Array                                      | _Required_  | 1D array (list) representing the training sizes                                                                                                                                                                                                                                               |
|                        | y                 | Array                                      | _Required_  | 1D array (list) representing the scores                                                                                                                                                                                                                                                       |
| LC.threshold           |                   |                                            |             | Find the training set size providing the highest accuracy up to a predefined threshold.  P(x) = y and for x -> inf, y -> saturation value.   This method approximates x_thresh such as P(x_thresh) = threshold * saturation value. Returns (saturation value, x_thresh, y_thresh)             |
|                        | P                 | str, List(str), Predictor, list(Predictor  | "best"      | The predictor to use for the calculation of the saturation value.                                                                                                                                                                                                                             |
|                        | kwargs            | dict                                       | Emtpy       | See `LC.threshold_cust` for optional parameters.                                                                                                                                                                                                                                              |
| LC.threshold_cust      |                   |                                            |             | See `threshold`                                                                                                                                                                                                                                                                               |
|                        | P                 | str, List(str), Predictor, list(Predictor  | "best"      | The predictor to use for the calculation of the saturation value.                                                                                                                                                                                                                             |
|                        | x                 | array                                      | _Required_  | X values (training set sizes)                                                                                                                                                                                                                                                                 |
|                        | threshold         | float [0.0, 1.0]                           | 0.99        | Percentage of the saturation value to use for the calculus of the best training set size.                                                                                                                                                                                                     |
|                        | max_scaling       | int                                        | 3           | Order of magnitude added to the order of magnitude of the maximum train set size.  If `Predictor` is diverging, the total order of magnitude is used for the calculation of the saturation value. Generally, a value of `3` is enough. A value bigger than `5` may lead to `MemoryException`. |
|                        | force             | Bool                                       | False       | Set to `True` not to raise a `ValueError` if `max_scaling` is > 5                                                                                                                                                                                                                             |
| LC.get_scale           |                   |                                            |             | Returns the order of magnitude of the mean of an array                                                                                                                                                                                                                                        |
|                        | val               | array                                      | _Required_  |                                                                                                                                                                                                                                                                                               |
| LC.best_predictor      |                   |                                            |             | Returns the best predictor of the `LearningCurve` data for the test score learning curve                                                                                                                                                                                                      |
|                        | kwargs            | dict                                       | Empty       | See `LC.best_predictor_cust` for optional parameters.                                                                                                                                                                                                                                         |
| LC.best_predictor_cust |                   |                                            |             | See `best_predictor`                                                                                                                                                                                                                                                                          |
|                        | predictors        | List(Predictors)                           | _Required_  | `Predictor`s to consider                                                                                                                                                                                                                                                                      |
|                        | x                 | Array                                      | _Required_  | 1D array (list) representing the training sizes                                                                                                                                                                                                                                               |
|                        | y                 | Array                                      | _Required_  | 1D array (list) representing the scores                                                                                                                                                                                                                                                       |
| LC.best_predictor_cust |                   |                                            |             | Find the best predictor for a custom learning curve                                                                                                                                                                                                                                           |
|                        | x                 | Array                                      | _Required_  | 1D array (list) representing the training sizes                                                                                                                                                                                                                                               |
|                        | y                 | Array                                      | _Required_  | 1D array (list) representing the scores                                                                                                                                                                                                                                                       |
|                        | fit               | Bool                                       | True        | Perform a fit of the `Predictor`s before classifying                                                                                                                                                                                                                                          |
| LC.plot                |                   |                                            |             | Plot the training and test learning curve of the `LearningCurve` data, and optionally a fitted function                                                                                                                                                                                       |
|                        | predictor         | str, List(str), Predictor, List(Predictor) | None        | `Predictor`s to use for plotting the fitted curve. Can also be "all" and "best".                                                                                                                                                                                                              |
|                        | kwargs            | dict                                       | None        | See `LC.plot_cust` for optional parameters                                                                                                                                                                                                                                                    |
| LC.plot_cust           |                   |                                            |             | Plot any training and test learning curve, and optionally a fitted function.                                                                                                                                                                                                                  |
|                        |  train_sizes      | array                                      | _Required_  | Data points of the learning curve. The output of `LC.train` can be used as parameters of this function                                                                                                                                                                                        |
|                        | train_scores_mean | array                                      | _Required_  | See `train_sizes` parameter                                                                                                                                                                                                                                                                   |
|                        | train_scores_std  | array                                      | _Required_  | See `train_sizes` parameter                                                                                                                                                                                                                                                                   |
|                        | test_scores_mean  | array                                      | _Required_  | See `train_sizes` parameter                                                                                                                                                                                                                                                                   |
|                        | test_scores_std   | array                                      | _Required_  | See `train_sizes` parameter                                                                                                                                                                                                                                                                   |
|                        | predictor         | array                                      | _Required_  | See `LC.plot`                                                                                                                                                                                                                                                                                 |
|                        | ylim              | 2-uple                                     | None        | Limits of the y axis of the plot                                                                                                                                                                                                                                                              |
|                        | figsize           | 2-uple                                     | None        | Size of the figure                                                                                                                                                                                                                                                                            |
|                        | title             | str                                        | None        | Title of the plot                                                                                                                                                                                                                                                                             |
|                        | saturation        | str, List(str), Predictor, List(Predictor) | None        | `Predictor`s to consider for displaying the saturation on the plot.                                                                                                                                                                                                                           |
|                        | kwargs            | dict                                       | Empty       | See `plot_saturation` for optional parameters                                                                                                                                                                                                                                                 |
| LC.plot_fitted_curve   |                   |                                            |             | Add to a matplotlib figure a fitted curve                                                                                                                                                                                                                                                     |
|                        | ax                | axe                                        | _Required_  | Figure where the curve will be printed                                                                                                                                                                                                                                                        |
|                        | P                 | Predictor                                  | _Required_  | `Predictor` to use for the computing of the curve                                                                                                                                                                                                                                             |
|                        | x                 | array                                      | _Required_  | 1D array (list) representing the training sizes                                                                                                                                                                                                                                               |
|                        | scores            | Bool                                       | True        | Show the score of the `Predictor`s                                                                                                                                                                                                                                                            |
| LC.save                |                   |                                            |             | Save the `LearningCurve` object in disk using `dill`                                                                                                                                                                                                                                          |
|                        | path              | Path/str                                   | lc_data.pkl | Path to the file where the save will be done                                                                                                                                                                                                                                                  |
| LC.load                |                   |                                            |             | Load a `LearningCurve` object from disk.                                                                                                                                                                                                                                                      |
|                        | path              | Path/str                                   | lc_data.pkl | Path to the file where the save is located                                                                                                                                                                                                                                                    |
| LC.plot_saturation     |                   |                                            |             | Add saturation lines to a plot.                                                                                                                                                                                                                                                               |
|                        | ax                | matplotlib ax                              | _Required_  | figure to use                                                                                                                                                                                                                                                                                 |
|                        | P                 | Predictor                                  | _Required_  | `Predictor` to consider                                                                                                                                                                                                                                                                       |
|                        | alpha             | float                                      | 1           | alpha applied to the lines                                                                                                                                                                                                                                                                    |
|                        | lw                | float                                      | 1.3         | matplotlib lw parameter applied to the lines.                                                                                                                                                                                                                                                 |
| LC.get_unique_list     |                   |                                            |             | Return a list of unique predictors.                                                                                                                                                                                                                                                           |
|                        | predictors        | List(Predictor)                            | _Required_  | List of `Predictor`s to consider.                                                                                                                                                                                                                                                             |
