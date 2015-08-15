

source("mnist_loader.R")
load_mnist()

## RUN
source("NN_class.R")
net2 = Network3$new(
  
  
  FullyConnectedLayer(n_in = 784,
                      n_out = 50,
                      activation_fn = SigmoidActivation$new())
  
  ,FullyConnectedLayer(n_in = 50,
                       n_out = 30,
                       activation_fn = SigmoidActivation$new())
  
  ,FullyConnectedLayer(n_in = 30,
                       n_out = 10,
                       activation_fn = SigmoidActivation$new())
  
  , cost = CrossEntropyCost$new()
  
)


net2$SGD(x = train[, 2:ncol(train)], y = train[,1],
         test.x = test[, 2:ncol(test)], test.y = test[,1],
         epochs = 30, mini_batch_size = 20,
         eta = 0.5,
         lambda = 0,
         monitor_evaluation_accuracy = TRUE,
         monitor_training_accuracy = TRUE,
         monitor_training_cost = TRUE,
         monitor_evaluation_cost = TRUE) ## 30 epochs






#########################################################################################################
#########################################################################################################


## DOES NOT WORK
source('my_network_old.R')
net2 = Network3$new(
  
  ConvolutionLayer$new(image_shape = c(1, 28, 28),
                       num_filters = 6,
                       num_input_filters = 1,
                       filter_shape = c(5, 5))
  
  , MeanPoolingLayer(pool_size = 2,
                     activation_fn = SigmoidActivation$new())
  
  , FullyConnectedLayer(n_in = 6*12*12,
                        n_out = 30,
                        activation_fn = SigmoidActivation$new())
  
  , FullyConnectedLayer(n_in = 30,
                        n_out = 10,
                        activation_fn = SigmoidActivation$new())
  
  
  , cost = CrossEntropyCost$new()
  
)


net2$SGD(x = train[1:5000, 2:ncol(train)], y = train[1:5000,1],
         test.x = test[1:5000, 2:ncol(test)], test.y = test[1:5000,1],
         epochs = 20, mini_batch_size = 10,
         eta = 0.2,
         lambda = 0,
         monitor_evaluation_accuracy = TRUE,
         monitor_training_accuracy = TRUE,
         monitor_training_cost = TRUE,
         monitor_evaluation_cost = TRUE) ## 30 epochs









#########################################################################################################################
#########################################################################################################################





require('ggplot2')
require('reshape2')

plot_data <- data.frame("epoch" = 1:length(net2$training_cost), 
                        "evaluation_cost" = net2$evaluation_cost
                        ,"training_cost" = net2$training_cost
)
plot_data <- melt(plot_data, id.vars = "epoch")
ggplot() + geom_line(data = plot_data, aes(x = epoch, y = value, group = variable, colour = variable))


plot_data <- data.frame("epoch" = 1:length(net2$training_cost), 
                        "evaluation_accuracy" = net2$evaluation_accuracy
                        ,"training_accuracy" = net2$training_accuracy
)
plot_data <- melt(plot_data, id.vars = "epoch")
ggplot() + geom_line(data = plot_data, aes(x = epoch, y = value, group = variable, colour = variable))










source('my_network_old.R')
net = Network3$new(c(784, 30, 10), cost = CrossEntropyCost$new())
net$SGD(x = train[, 2:ncol(train)], y = train[,1],
        test.x = test[, 2:ncol(test)], test.y = test[,1],
        epochs = 3, mini_batch_size = 10, eta = 0.1, monitor_evaluation_accuracy = TRUE) ## 30 epochs











net <- Network(
  FullyConnectedLayer(n_in = 784, n_out = 30, activation_fn = sigmoid),
  FullyConnectedLayer(n_in = 30, n_out = 10, activation_fn = sigmoid)
)


net$SGD(x = train[, 2:ncol(train)], y = train[,1],
        test.x = test[, 2:ncol(test)], test.y = test[,1],
        epochs = 3, mini_batch_size = 10, eta = 0.5, monitor_evaluation_accuracy = TRUE) ## 30 epochs





net$weights
net3$weights
net$sizes
net3$sizes




net2 = Network2$new(c(784, 12, 10), cost = QuadraticCost$new())
net2$SGD(training_data = training_data[1:1000],
         epochs = 10,
         mini_batch_size = 10,
         eta = 0.5, evaluation_data = test_data,
         monitor_evaluation_accuracy = TRUE) ## 30 epochs





dim(train)
dim(test)


head(train)
head(test)




require('DEoptim')
require('raster')
obj_f <- function(img){
  
  net3$feedforward(matrix(img, ncol = 1))[3,1] * -1000## 3rd element is corresponds to 2
}

## find an image of a 2

inds <- which(train[,1] == 2)
initialpop <- train[inds,2:ncol(train)]
initialpop <- rbind(initialpop, initialpop)
initialpop <- initialpop[1:(784*10),]

res <- DEoptim(fn = obj_f, lower = rep(0, 784), upper = rep(1, 784), 
               control=list(NP = 784*10, parallelType=1, initialpop = initialpop, parVar = list("net3", "sigmoid"), trace = FALSE, itermax = 500))



res$member$bestvalit


net3$feedforward(matrix(test[2,2:ncol(test)], ncol = 1))[3,1] * -1000
net3$feedforward(matrix(res$optim$bestmem, ncol = 1))[3,1] * -1000

a <- matrix(test[2,2:ncol(test)], ncol = 28, nrow = 28, byrow = TRUE)
plot(raster(a), col = gray(12:1/12))

b <- matrix(res$optim$bestmem, ncol = 28, nrow = 28, byrow = TRUE)
plot(raster(b), col = gray(12:1/12))

all.equal(res$optim$bestmem, test[2,2:ncol(test)], check.attributes = FALSE)

#########################




## check equality




# net2$SGD(training_data = training_data, epochs = 10, mini_batch_size = 1000, eta = 0.5, evaluation_data = test_data, monitor_evaluation_accuracy = TRUE) ## 30 epochs






training_data <- lapply(1:train$n, function(i) {
  
  y <- rep(0, 10)
  y[train$y[i] + 1] <- 1
  y <- matrix(y, ncol = 1)
  
  list(as.matrix(train$x[i,]/255, ncol = 1), y)
})


test_data <- lapply(1:test$n, function(i) {
  
  y <- rep(0, 10)
  y[test$y[i] + 1] <- 1
  y <- matrix(y, ncol = 1)
  
  list(as.matrix(test$x[i,]/255, ncol = 1), y)
})


net = Network$new(c(784, 30, 10))
net2 = Network2$new(c(784, 30, 10), cost = CrossEntropyCost$new())

net2$weights <- net$weights
net2$biases <- net$biases


net$evaluate(test_data = training_data)
net$evaluate(test_data = test_data)

net2$evaluate(test_data = training_data)
net2$evaluate(test_data = test_data)


net$SGD(training_data, epochs = 10, mini_batch_size = 10, eta = 3, test_data = test_data) ## 30 epochs
net2$SGD(training_data = training_data, epochs = 10, mini_batch_size = 1000, eta = 0.5, evaluation_data = test_data, monitor_evaluation_accuracy = TRUE) ## 30 epochs
net2$SGD(training_data = training_data, epochs = 10, mini_batch_size = 100, eta = 0.5, evaluation_data = test_data, monitor_evaluation_accuracy = TRUE) ## 30 epochs
net2$SGD(training_data = training_data, epochs = 5, mini_batch_size = 10000, eta = 0.2, evaluation_data = test_data, monitor_evaluation_accuracy = TRUE) ## 30 epochs


## lets classify my image
show_digit(arr784 = test_data[[1903]][[1]])
show_digit(matrix(t(b), ncol = 1))


a <- matrix(test_data[[1903]][[1]], nrow=28, byrow = TRUE)[,28:1]
require('jpeg')
writeJPEG(image = a, target = "out.bmp")

b <- readJPEG("my_4.jpg")
b <- b[,,1]
res <- net2$feedforward(a = matrix(t(b), ncol = 1))
which.max(res) - 1


b <- readJPEG("my_3.jpg")
b <- b[,,1]
res <- net2$feedforward(a = matrix(t(b), ncol = 1))
which.max(res) - 1





show_digit(arr784 = b[,28:1])

monitor_evaluation_accuracy=True

net2$accuracy(data = test_data)
net2$total_cost(data = test_data, lambda = 0)
net2$total_cost(data = test_data, lambda = 0)



## find an input vector which maximises output a5 (this means the input is a 4)


a <- lapply(test_data, "[[", 2)
a <- sapply(a, which.max)
