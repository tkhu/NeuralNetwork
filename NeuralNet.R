# MNIST

# Read the MNIST dataset
library(readr)
library(dplyr)
library(tinydr)
library(fastDummies)

mnist_raw <- read_csv("https://pjreddie.com/media/files/mnist_train.csv", col_names = FALSE)

# Clean the data
trainx <- as.matrix(mnist_raw[,-c(1)])
trainy <- as.matrix(mnist_raw[,1])
trainy <- dummy_cols(trainy)
trainy <- trainy[,-c(1)]
trainy <- trainy[c("X1_0", "X1_1","X1_2","X1_3","X1_4","X1_5","X1_6","X1_7","X1_8","X1_9")]
trainy <- trainy %>% rename(y0 = X1_0, y1 = X1_1, y2 = X1_2, y3 = X1_3, y4 = X1_4, y5 = X1_5, y6 = X1_6, y7 = X1_7, y8 = X1_8, y9 = X1_9)
### Reduce the training set size to reduce run time
trainx <- trainx[c(1:5000),]
trainy <- trainy[c(1:5000),]

# Define the sigmoid function as an activation function
sigmoid <- function(x){
  1/(1 + exp(-x))
}

# Define the softmax function as an output function
softmax <- function(x){
  exp(x)/rowSums(exp(x))
}

# Define a loss function based on the softmax activation function
L <- function(yhat, y){
  Err <- -1/nrow(trainy) * sum(yhat*y) 
}


# Build a 2 hidden layers feedforward process
# Start with 784 + 1 (785) input variables
# The first hidden layer transforms it to 100 variables
# The second hidden layer transforms it to 10 variables
# Output 1 in 10 digits
# w1 is the weights for the first hidden layer
# w2 is the weights for the second hidden layer
# w3 is the weights for the outpue layer
feedforward <- function(x, w1, w2, w3) {
  
  z1 <- cbind(1, x) %*% w1     # add intercept to the input layer
  h1 <- sigmoid(z1)            # compute sigmoid function of the first input layer
  z2 <- cbind(1, h1) %*% w2    # add intercept to the second hidden layer
  h2 <- sigmoid(z2)            # compute sigmoid function of the second input layer
  z3 <- cbind(1, h2) %*% w3
  list(yhat=softmax(z3), h2=h2, h1=h1)
  
}

# # Test the feedforward function using the initialized random weight matrices
# d  <- ncol(trainx) + 1                        # Dimension is the number of variables plus an intercept
# hidden1 <- 100                                # First hidden layer has 100 neurons
# hidden2 <- 10                                 # Second hidden layer has 10 neurons
# output <- 10                                  # There are 10 digits, so we have 10 outputs
# w1 <- matrix(rnorm(d * hidden1), d, hidden1)  # Random weight matrix as an initialization for the first hidden layer
# w2 <- matrix(rnorm((hidden1 + 1) * hidden2), (hidden1 +1), hidden2) # Random weight matrix as an initialization for the second hidden layer
# w3 <- matrix(rnorm((hidden2 + 1) * output), (hidden2 +1), output)   # Random weight matrix for an output layer         
# 
# z1 <- cbind(1,t(trainx[1,])) %*% w1
# h1 <- sigmoid(z1) 
# z2 <- cbind(1, h1) %*% w2
# h2 <- sigmoid(z2)
# z3 <- cbind(1, h2) %*% w3
# yhat <- softmax(z3)
# 
# ff <- feedforward(trainx,w1,w2,w3) # Tested feedforward with the first row of training set
# 
# # Build a backpropogation mechanism
# # dL/dw3 = dL/dy3 dy3/dz3 dz3/dw3
# 
# ### Output for the first sample (number 5)
# y1 <- trainy[1,]
# 
# ### Compute the gradient for w3 (second hidden layer weights)
# dLdz3  <- as.matrix(yhat - y1)
# dz3dw3 <- as.matrix(cbind(1, h2))
# dLdw3  <- t(dz3dw3) %*% dLdz3
# 
# ### Compute the gradient for w2 (first hidden layer weights)
# # dL/dw2 = (dL/dy3)*(dy3/dz3)*(dz3/dh2)*(dh2/dz2)*(dz2/dw2)
# dz3dh2 <- w3[-1, ,drop = FALSE]
# dLdh2  <- dLdz3 %*% t(dz3dh2)
# dh2dz2 <- as.matrix(h2 * (1 - h2))
# dLdz2  <- dLdh2 * dh2dz2
# dz2dw2 <- cbind(1,h1)
# dLdw2  <- t(dz2dw2) %*% dLdz2
# 
# 
# ### Compute the gradient for w1 (first hidden layer weights)
# # dL/dw2 = (dL/dy3)*(dy3/dz3)*(dz3/dh2)*(dh2/dz2)*(dz2/dh1)*(dh1/dz1)*(dz1/dw1)
# # We already have dldz2 from the prior calculation
# dz2dh1 <- w2[-1, ,drop = FALSE]
# dLdh1  <- dLdz2 %*% t(dz2dh1)
# dh1dz1 <- as.matrix(h1 * (1 - h1))
# dLdz1  <- dLdh1 * dh1dz1
# dz1dw1 <- cbind(1,t(trainx[1,]))
# dLdw1  <- t(dz1dw1) %*% dLdz1

## Write backpropagation function
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

# test backpropagate function
# bp <- backpropgation(trainx, trainy, ff$yhat, w1, w2, w3, ff$h1, ff$h2, learnrate=1)

# Create a training function
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

x <- as.matrix(trainx[c(1:100),])
y <- as.matrix(trainy[c(1:100),])
nnet <- train(x,y)

actualy <- apply(y,1, which.max)-1
actualyhat <- apply(nnet$output,1, which.max)-1