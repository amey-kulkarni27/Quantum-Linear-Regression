# Quantum-Linear-Regression
Using Quantum Computing Resources to perform Linear Regression.

## Existing
The State of the Art method (to the best of our knowledge) to perform Linear Regression by D-Wave System's Adiabatic Quantum Processors is given by https://onikle.com/articles/319449. This method achieves metrics that are practically as good as the ones obtained by sklearn's Linear Regression module. Additionally, it achieves a 2.8x speedup over the same. One limitation, however, is that the number of attributes the model allows is limited to only 32. This is because the D-Wave 2000Q Quantum Computer allows for only a certain size of the embedding matrix.

## Our Modification
We propose a novel approach that allows for a larger features to be allowed. We achieve this by ranking the features in their decreasing order of importance and allocating resources (quantum bits) in that order.

## Results
We have been able to obtain a comparable MSE to that of sklearn's Linear Regression module, while allowing for a large number of features (~128).
