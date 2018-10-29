# NeuralNetwork
In this project, I created a neural network with two hidden layers to classify digit handwritings from scratch using R.

### Data
The data is taken from MNIST hand-written digits collection database. The original data can be found [here](http://yann.lecun.com/exdb/mnist/)

### Acknowledgements
I would like to acknowledge David Selby for his excellent introduction to neural network in R [here](https://selbydavid.com/2018/01/09/neural-network/).
I also would like to acknowledge Jonathan Weisberg for his excellent tutorial in neural netowork in Python on MNIST database [here](https://jonathanweisberg.org/post/A%20Neural%20Network%20from%20Scratch%20-%20Part%201/).

### Network
* The input layer contains 784 variables (28 x 28 pixels)
* The first hidden layer contains 100 nodes
* The second hidden layer contains 10 nodes
* The final output contains 10 classes (i.e. from digit 0 to 9)

### Activation Functions
* Logit function is used for the hidden layers calculaiton.
```r
sigmoid <- function(x){
  1/(1 + exp(-x))
}
```
* Softmax function is used for the final output calculation.
```r
softmax <- function(x){
  exp(x)/rowSums(exp(x))
}
```

### Feedforward Function
```r
feedforward <- function(x, w1, w2, w3) { 
  z1 <- cbind(1, x) %*% w1     # add intercept to the input layer
  h1 <- sigmoid(z1)            # compute sigmoid function of the first input layer
  z2 <- cbind(1, h1) %*% w2    # add intercept to the second hidden layer
  h2 <- sigmoid(z2)            # compute sigmoid function of the second input layer
  z3 <- cbind(1, h2) %*% w3
  list(yhat=softmax(z3), h2=h2, h1=h1)
}
```

### Backpropagation Function
```r
backpropagation <- function(x, y, yhat, w1, w2, w3, h1, h2, learnrate){
  
  dLdz3  <- as.matrix(yhat - y)
  dz3dw3 <- as.matrix(cbind(1, h2))
  dLdw3  <- t(dz3dw3) %*% dLdz3
  
  dz3dh2 <- w3[-1, ,drop = FALSE]
  dLdh2  <- dLdz3 %*% t(dz3dh2)
  dh2dz2 <- as.matrix(h2 * (1 - h2))
  dLdz2  <- dLdh2 * dh2dz2
  dz2dw2 <- cbind(1,h1)
  dLdw2  <- t(dz2dw2) %*% dLdz2
  
  dz2dh1 <- w2[-1, ,drop = FALSE]
  dLdh1  <- dLdz2 %*% t(dz2dh1)
  dh1dz1 <- as.matrix(h1 * (1 - h1))
  dLdz1  <- dLdh1 * dh1dz1
  dz1dw1 <- cbind(1,x)
  dLdw1  <- t(dz1dw1) %*% dLdz1
  
  w1 <- w1 - learnrate * dLdw1
  w2 <- w2 - learnrate * dLdw2
  w3 <- w3 - learnrate * dLdw3
  
  list(w1 = w1, w2 = w2, w3 = w3)
}
```

### Putting Everything Together
```r
train <- function(x, y, hidden1=100, hidden2=10, learnrate=0.03, iteration=2000){
  
  m  <- ncol(x) + 1
  n  <- ncol(y)
  w1 <- matrix(rnorm(m * hidden1), m, hidden1)
  w2 <- matrix(rnorm((hidden1 + 1) * hidden2), (hidden1 +1), hidden2)
  w3 <- matrix(rnorm((hidden2 + 1) * n), (hidden2 +1), n)
  
  for (i in 1:iteration){
    ff  <- feedforward(x,w1,w2,w3)
    bp  <- backpropagation(x, y, ff$yhat, w1, w2, w3, ff$h1, ff$h2, learnrate)
    w1  <- bp$w1
    w2  <- bp$w2
    w3  <- bp$w3
    err <- L(ff$yhat, y) 
    print(paste0("epoch:", i,", cost:", err))
  }
  
  list(output = ff$yhat, w1 = w1, w2 = w2, w3 = w3)
}
```

### Results
The following assumptions were input in the training process:
* 200 rows of training set (limited due to lack of computing power)
* Learning rate of 0.03
* 2,000 iteration

After the training session, the algorithm is able to correctly guess 69/100 rows of testing data.
I also did another run with 100 rows of training set, and the accuracy was less than 60%.
I suspect that with larger training set, the algorithm will be more accurate.
