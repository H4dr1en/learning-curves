# learning-curves

Learning-curves is a Python module that will help you calculating and ploting the learning curve of a model.

Learning curves give an opportunity to diagnose bias and variance in supervised learning models, but also to visualize how training set size influence the performance of the models.

To create such plots, simply import it with `import learning_curves`

Then it is as simple as:

```
lc = LearningCurve()
lc.get_lc(estimator, X, Y)
```
Output:

![alt text](https://github.com/H4dr1en/learning-curves/blob/dev/images/learning_curve_no_fit.png)

On this example the green curve suggests that adding more data to the training set is likely to improve a bit the model accuracy.
The green curve also shows a saturation near 0.95. We can easily fit a function to this curve:

```
lc.plot_lc(**lc.recorder["data"], predictor="best")
```
Output:

![alt text](https://github.com/H4dr1en/learning-curves/blob/dev/images/learning_curve_simple.png)

Here we used a predefined function, `exp_log`, to fit the green curve. The R2 score is very close to 1, meaning that the fit is optimal. We can therefore use this curve to extrapolate the evolution of the accuracy with the training set size.

This also tells us how many data should we use to train our model to maximize performances and accuracy.
