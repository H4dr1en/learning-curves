# learning-curves

Learning-curves is Python module that extends [sklearn's learning curve feature](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.learning_curve.html). It will help you visualizing the learning curve of your models:

![alt text](https://github.com/H4dr1en/learning-curves/blob/master/images/learning_curve_no_fit.png)

Learning curves give an opportunity to diagnose bias and variance in supervised learning models, but also to visualize how training set size influence the performance of the models (more informations [here](https://www.dataquest.io/blog/learning-curves-machine-learning/)).

Such plots help you answer the following questions:
 - Would my model perform better with more data?
 - Can I train my model with less data without reducing accuracy?
 - Is my training/validation set biased?
 - What is the best model for my data?
 - What is the perfect training size for tuning parameters?
 
 Learning-curves will also help you fitting the learning curve to extrapolate and find the saturation value of the curve.

### Installation

This module is still under development. Therefore it is recommended to use:
```
$ pip install git+https://github.com/H4dr1en/learning-curves#egg=learning-curves
```

### Usage

To create learning curve plots, you can start with the following lines:

```
import learning_curves as LC
lc = LC.LearningCurve()
lc.get_lc(estimator, X, Y)
```
Where `estimator` implements `fit(X,Y)` and `predict(X,Y)` (Sklearn interface).

Output:

![alt text](https://github.com/H4dr1en/learning-curves/blob/master/images/learning_curve_no_fit.png)

On this example the green curve suggests that adding more data to the training set is not likely to improve the model accuracy. The green curve also shows a saturation near 0.7. We can easily fit a function to this curve:

```
lc.plot(predictor="best")
```
Output:

![alt text](https://github.com/H4dr1en/learning-curves/blob/master/images/learning_curve_simple.png)

Here we used a predefined function, `pow`, to fit the green curve. The R2 score (0.999) is very close to 1, meaning that the fit is optimal. We can therefore use this curve to extrapolate the evolution of the accuracy with the training set size.

This also tells us how many data we should use to train our model to maximize performances and accuracy.

## And much more!

- Write your [own predictors](https://h4dr1en.github.io/learning-curves/intro.html#custom-predictors)
- Find the [best Predictor](https://h4dr1en.github.io/learning-curves/intro.html#find-the-best-predictor)
- Compare learning curves of [various models](https://h4dr1en.github.io/learning-curves/intro.html#compare-learning-curves-of-various-models)
- Extrapolate learning curve [using multiple instances](https://h4dr1en.github.io/learning-curves/intro.html#average-learning-curves-for-better-extrapolation)
- Evaluate extrapolation using [mse validation](https://h4dr1en.github.io/learning-curves/intro.html#evaluate-extrapolation-using-mse-validation)
- Evaluate and compare your [models scalability](https://h4dr1en.github.io/learning-curves/intro.html#compare-the-models-performances)
- Save and load [LearningCurve instances](https://h4dr1en.github.io/learning-curves/intro.html#save-and-load-learningcurve-instances)

## Documentation

The documentation is available [here](https://h4dr1en.github.io/learning-curves/).

Some functions have their `function_name_cust` equivalent. Calling the function without the `_cust` suffix will internally call the function with the `_cust` suffix with default parameters (such as the data points of the learning curves). Thanks to `kwargs`, you can pass exactly the same parameters to both functions.

## Contributing

PRs, bug reports as well as improvment suggestions are welcomed :)
