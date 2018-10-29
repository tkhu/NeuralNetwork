# NeuralNetwork
In this project, I created a neural network with two hidden layers to classify digit handwritings from scratch using R.

### Data
The data is taken from MNIST hand-written digits collection database. The original data can be found [here](http://yann.lecun.com/exdb/mnist/)
..* Due to lack of computing power, I restricted the training set to be 200 rows of data.

### Acknowledgements
I would like to acknowledge David Selby for his excellent introduction to neural network in R [here](https://selbydavid.com/2018/01/09/neural-network/).
I also would like to acknowledge Jonathan Weisberg for his excellent tutorial in neural netowork in Python on MNIST database [here](https://jonathanweisberg.org/post/A%20Neural%20Network%20from%20Scratch%20-%20Part%201/).

### Network
..* The input layer contains 784 variables (28 x 28 pixels)
..* The first hidden layer contains 100 nodes
..* The second hidden layer contains 10 nodes
..* The final output contains 10 classes (i.e. from digit 0 to 9)

### Activation Functions
..* Logit function is used for the hidden layers calculaiton.
```r
sigmoid <- function(x){;
  1/(1 + exp(-x));
};
```
..* Softmax function is used for the final output calculation.
```r
softmax <- function(x){;
  exp(x)/rowSums(exp(x));
};
```
