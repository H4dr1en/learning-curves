# learning-curves

Learning-curves is a Python module that will help you calculating and ploting the learning curve of a model.

Learning curves give an opportunity to diagnose bias and variance in supervised learning models, but also to visualize how training set size influence the performance of the models.

### Installation

```
$ pip install git+https://github.com/H4dr1en/learning-curves.git
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

On this example the green curve suggests that adding more data to the training set is likely to improve a bit the model accuracy.
The green curve also shows a saturation near 0.95. We can easily fit a function to this curve:

```
lc.plot(**lc.recorder["data"], predictor="best")
```
Output:

![alt text](https://github.com/H4dr1en/learning-curves/blob/master/images/learning_curve_simple.png)

Here we used a predefined function, `exp_log`, to fit the green curve. The R2 score is very close to 1, meaning that the fit is optimal. We can therefore use this curve to extrapolate the evolution of the accuracy with the training set size.

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

By default, two `Predictors` are instantiated: `exp_log` and `exp`. Some predictors perform better (R2 score is closer to 1) than others, depending on the dataset, the model and the value to be preditected.

### Find the best Predictor

To find the Predictor that will fit best your learning curve, we provide a `best_predictor` function:
```
lc.best_predictor()
```
Output:
```
('exp_log', 0.999147437907635, <learning_curve.Predictor at 0x7feb9f2a4ac8>)
```

### Plot the Predictors

You can plot any `Predictor`s fitted function with the `plot` function:
```
lc.plot(predictor="best")
```
Ouput:

![alt text](https://github.com/H4dr1en/learning-curves/blob/master/images/learning_curve_simple.png)

Note that this is the exact same output as calling `get_lc` because internally this function just calls `train` to compute the data points of the learning curve and then call `plot(predictor="best")`.

You can also plot all of the `Predictor` curves:
```
lc.plot(predictor="all")
```
Output:

![alt text](https://github.com/H4dr1en/learning-curves/blob/master/images/learning_curve_all.png)

## Documentation

| Function/Class         | Parameters                                                                          | Type      | Default    | Description                                                                                                                                                                                    |
|---|---|---|---|---|
| Predictor.\_\_init\_\_ |                                                                                     |           |            | Instantiate a `Predictor` object.                                                                                                                                                              |
|                        | name                                                                                | str       | _Required_ | Name of the `Predictor`                                                                                                                                                                        |
|                        | func                                                                                | Lambda    | _Required_ | Lambda function used for fitting of the learning curve                                                                                                                                         |
|                        | guess                                                                               | List      | _Required_ | Starting parameters used for fitting the curve                                                                                                                                                 |
| LC.\_\_init\_\_        |                                                                                     |           |            | Instantiate a `LearningCurve` object.                                                                                                                                                          |
|                        | predictors                                                                          | List      | empty      | Predictors to add to the `LearningCurve` object                                                                                                                                                |
|                        | scoring                                                                             | Callable  | r2_score   | Scoring function used to evaluate the fits of the learning curve                                                                                                                               |
| LC.get_lc              | estimator                                                                           | Object    | _Required_ | Model (any object implementing `fit(X,Y)` and `predict(X,Y)` methods)                                                                                                                          |
|                        | X                                                                                   | array     | _Required_ | X numpy array used for prediction                                                                                                                                                              |
|                        | Y                                                                                   | array     | _Required_ | Y numpy array used for prediction                                                                                                                                                              |
|                        | train_sizes                                                                         | List      | Predefined |  List of training size used for calculating the learning curve.   Can be a list of floats between 0 and 1 (assumed to be percentages)   or a list of integers (assumed to be number of values) |
|                        | test_size                                                                           | int/float | 0.2        | percentage / value of the test set size                                                                                                                                                        |
|                        | n_splits                                                                            | int       | 3          | Number of splits used for cross validation                                                                                                                                                     |
|                        | verbose                                                                             | int       | 1          | The higher, the more verbose                                                                                                                                                                   |
|                        | n_jobs                                                                              | int       | -1         | Number of workers. -1 sets to maximum possible. See sklearn.                                                                                                                                   |
| LC.train               |                                                                                     |           |            | Compute the learning curve of an estimator over a dataset. Returns an object that can then be passed to plot_lc function                                                                       |
|                        |                                                                                     |           |            | Same as get_lc                                                                                                                                                                                 |
| LC.get_predictor       |                                                                                     |           |            | Get the first predictor with matching {name}. Returns None if no predictor matches.                                                                                                            |
|                        | name                                                                                | str       | _Required_ | Name of the predictor                                                                                                                                                                          |
| LC.fit_all             |                                                                                     |           |            | Fit a curve with all the Predictors and retrieve score if y_pred is finite. Returns an array of predictors with the updated params and score.                                                  |
|                        | x                                                                                   | Array     | _Required_ | 1D array (list) representing the training sizes                                                                                                                                                |
|                        | y                                                                                   | Array     | _Required_ | 1D array (list) representing the scores                                                                                                                                                        |
| LC.fit                 |                                                                                     |           |            | Fit a curve with a predictor and retrieve score (default:R2) if y_pred is finite. Returns the predictor with the updated params and score.                                                     |
|                        | predictor                                                                           | Predictor | _Required_ | The predictor to use for fitting the learning curve                                                                                                                                            |
|                        | x                                                                                   | Array     | _Required_ | 1D array (list) representing the training sizes                                                                                                                                                |
|                        | y                                                                                   | Array     | _Required_ | 1D array (list) representing the scores                                                                                                                                                        |
| LC.best_predictor      |                                                                                     |           |            | Returns the best predictor of the `LearningCurve` data for the test score learning curve                                                                                                       |
| LC.best_predictor_cust |                                                                                     |           |            | Find the best predictor for a custom learning curve                                                                                                                                            |
|                        | x                                                                                   | Array     | _Required_ | 1D array (list) representing the training sizes                                                                                                                                                |
|                        | y                                                                                   | Array     | _Required_ | 1D array (list) representing the scores                                                                                                                                                        |
| LC.plot                |                                                                                     |           |            | Plot the training and test learning curve of the `LearningCurve` data, and optionally a fitted function                                                                                        |
|                        | predictor                                                                           | str       | None       | Name of the `Predictor` to use for plotting the fitted curve. Can also be "all" and "best".                                                                                                    |
|                        | kwargs                                                                              | dict      | None       | See `LC.plot_cust` for optional parameters                                                                                                                                                     |
| LC.plot_cust           |                                                                                     |           |            | Plot any training and test learning curve, and optionally a fitted function.                                                                                                                   |
|                        | train_sizes | array     | _Required_ | And: `train_scores_mean`, `train_scores_std`, `test_scores_mean`, `test_scores_std`. Data points of the learning curve. The output of `LC.train` can be used as parameters of this function                                                                                         |
|                        | predictor                                                                           | array     | _Required_ | See `LC.plot`                                                                                                                                                                                  |
|                        | ylim                                                                                | 2-uple    | None       | Limits of the y axis of the plot                                                                                                                                                               |
|                        | figsize                                                                             | 2-uple    | None       | Size of the figure                                                                                                                                                                             |
|                        | title                                                                               | str       | None       | Title of the plot                                                                                                                                                                              |
|                        | scores                                                                              | Bool      | True       | if `predictor` parameter is not `None`, then if `scores` is `True` then the score of the fitted Predictor(s) are shown on the plot.                                                            |
|                        | kwargs                                                                              | dict      | None       | Ignored                                                                                                                                                                                        |
| LC.plot_fitted_curve   |                                                                                     |           |            | Add to a matplotlib figure a fitted curve                                                                                                                                                      |
|                        | ax                                                                                  | axe       | _Required_ | Figure where the curve will be printed                                                                                                                                                         |
|                        | P                                                                                   | Predictor | _Required_ | `Predictor` to use for the computing of the curve                                                                                                                                              |
|                        | x                                                                                   | array     | _Required_ | 1D array (list) representing the training sizes                                                                                                                                                |
