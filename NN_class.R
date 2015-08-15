

## Things that must be defined
# A cost function C (quadratic, cross entropy, ...)
# An activation function (sigmoid, tanh, ReLu)

# Steps
# Initialise network weights and biases
# Pick an activation function for each layer
# Feed-forward is calculated as: activation_function(w*x + b)
## SGD
# - get a mini_batch
# - feed-forward through network to get activations at each level
# - calculate output layer delta_L
# - back propagate delta through network

## Usage:
# source('my_network_old.R')
# net2 = Network3$new(
#   
#   
#   FullyConnectedLayer(n_in = 784,
#                       n_out = 50,
#                       activation_fn = SigmoidActivation$new())
#   
#   ,FullyConnectedLayer(n_in = 50,
#                        n_out = 30,
#                        activation_fn = SigmoidActivation$new())
#   
#   ,FullyConnectedLayer(n_in = 30,
#                        n_out = 10,
#                        activation_fn = SigmoidActivation$new())
#   
#   
#   
#   , cost = CrossEntropyCost$new()
#   
# )
# 
# 
# net2$SGD(x = train[1:6000, 2:ncol(train)], y = train[1:6000,1],
#          test.x = test[1:5000,2:ncol(test)], test.y = test[1:5000,1],
#          epochs = 20, mini_batch_size = 20,
#          eta = 0.5,
#          lambda = 0,
#          monitor_evaluation_accuracy = TRUE,
#          monitor_training_accuracy = TRUE,
#          monitor_training_cost = TRUE,
#          monitor_evaluation_cost = TRUE) ## 30 epochs


mean_pooling <- function(x){
  # pooling step
  res <- matrix(0, 12, 12)
  for(i in seq(1, ncol(x), by = 2)){
    for(j in seq(1, ncol(x), by = 2)){
      res[i/2+0.5,j/2+0.5] <- mean(x[i:(i+1), j:(j+1)])
    }
  }
  res
}


mean_pooling_upsample <- function(x){
  res <- matrix(0, 24, 24)
  for(i in seq(1, ncol(res), by = 2)){
    for(j in seq(1, ncol(res), by = 2)){
      res[i:(i+1), j:(j+1)] <- x[i/2+0.5,j/2+0.5] /4
    }
  }
  res
}

FullyConnectedLayer <- setRefClass(
  
  Class = "FullyConnectedLayer",
  
  fields = list(
    
    n_in = "numeric",
    n_out = "numeric",
    activation_fn = "ANY",
    p_dropout = "numeric",
    w = "matrix",
    b = "matrix"
    
  )
  
  ,methods = list(
    
    initialize = function(n_in, n_out, activation_fn = SigmoidActivation$new(), p_dropout = 0){
      
      n_in <<- n_in
      n_out <<- n_out 
      activation_fn <<- activation_fn
      p_dropout <<- p_dropout
      w <<- matrix(rnorm(n_in * n_out) / n_out, nrow = n_out, ncol = n_in)
      b <<- matrix(rnorm(n_out)  / n_out, ncol = 1)
    }
    
    ,feed_forward_raw = function(input){
      
      apply((w %*% input) , MARGIN = 2, FUN = function(x) x + b)
      
    }
    
    ,feed_forward = function(input){
      
      .self$activation_fn$fn( apply((w %*% input) , MARGIN = 2, FUN = function(x) x + b) )
      
    }
    
  )
  
)



SoftmaxLayer <- setRefClass(
  
  Class = "SoftmaxLayer",
  
  fields = list(
    
    n_in = "numeric",
    n_out = "numeric",
    p_dropout = "numeric",
    activation_fn = "ANY",
    w = "matrix",
    b = "matrix"
    
  )
  
  ,methods = list(
    
    initialize = function(n_in, n_out, activation_fn = SoftmaxActivation$new(), p_dropout = 0){
      
      n_in <<- n_in
      n_out <<- n_out 
      p_dropout <<- p_dropout
      w <<- matrix(0, nrow = n_out, ncol = n_in)
      b <<- matrix(0, ncol = 1)
      activation_fn <<- activation_fn
      
    }
    
    ,feed_forward_raw = function(input){
      
      apply((w %*% input) , MARGIN = 2, FUN = function(x) x + b)
      
    }
    
    ,feed_forward = function(input){
      
      .self$activation_fn$fn( .self$feed_forward_raw(input) )
      
    },
    
    accuracy = function(){
      
      1
    }
    
  )
  
)



ConvolutionLayer <- setRefClass(
  
  Class = "ConvolutionLayer",
  
  fields = list(
    
    n_filters = "integer",
    n_input_feature_maps = "integer",
    b = "ANY",
    w = "list",
    output = "ANY",
    output_dropout = "ANY",
    filter_height = 'integer',
    filter_width = 'integer',
    mini_batch_size = "integer",
    image_height = "integer",
    image_width = "integer",
    activation_fn = "ANY"
    
  )
  
  ,methods = list(
    
    initialize = function(image_shape, num_filters, num_input_filters, filter_shape, activation_fn = SigmoidActivation$new()){
      
      ## w is a list of lists. each inner list is a list of matrixs of filter size
      #  `filter_shape` is a tuple of length 2: the filter height and width.
      #  `image_shape` is a tuple of length 3, whose entries are the
      #         mini-batch size, the image height, and the image width.
      
      image_shape <- as.integer(image_shape)
      filter_shape <- as.integer(filter_shape)
      
      n_filters <<- as.integer(num_filters)
      n_input_feature_maps <<- as.integer(num_input_filters)
      
      filter_height <<- filter_shape[1]
      filter_width <<- filter_shape[2]
      
      mini_batch_size <<- image_shape[1]
      image_height <<- image_shape[2]
      image_width <<- image_shape[3]
      
      w <<- lapply(1:n_filters, function(i){
        matrix(rnorm(filter_height * filter_width) / filter_width, nrow = filter_height, ncol = filter_width)
      })
      
      b <<- rnorm(n_filters)
      
      activation_fn <<- activation_fn
    },
    
    
    feed_forward_raw = function(input){
      
      # want x X num_records
      res <- lapply(1:ncol(input), function(j){
        img <- matrix(input[,j], nrow = 28, ncol = 28, byrow = TRUE)
        out <- lapply(1:length(w), function(i) image_convolve(img, w[[i]]) + b[[i]])
      })
      
      return(res)
      
    },
    
    feed_forward = function(input){
      
      ## convolution step
      feed_forward_raw(input)
      
    }
    
    
  )
  
)



MeanPoolingLayer <- setRefClass(
  
  Class = "MeanPoolingLayer",
  
  fields = list(
    
    activation_fn = "ANY"
    
  )
  
  ,methods = list(
    
    initialize = function(pool_size = 2, activation_fn = SigmoidActivation$new()){
      
      activation_fn <<- activation_fn
    },
    
    
    feed_forward_raw = function(input){
      
      ## for each record
      ## -- for each feature map
      ## ---- mean pooling
      
      # want x X num_records
      res <- sapply(input, function(x){
        out <- lapply(x, mean_pooling) ## pooling
        out <- lapply(out, function(x) matrix(x, ncol = 1))
        out <- matrix(unlist(out), ncol = 1)
      })
      
      return(res)
      
    },
    
    feed_forward = function(input){
      
      ## convolution step
      res <- sapply(input, function(x){
        out <- lapply(x, mean_pooling) ## pooling
        out <- lapply(out, function(x) matrix(x, ncol = 1))
        out <- activation_fn$fn(matrix(unlist(out), ncol = 1))
      })
      
    }
    
    
  )
  
)


Network3 <- setRefClass(
  
  Class = "Network3",
  
  fields = list(
    
    layers = "ANY",
    num_layers = "integer",
    cost = "ANY",
    
    ## progress recording
    evaluation_cost = "ANY",
    evaluation_accuracy = "ANY",
    training_cost = "ANY",
    training_accuracy = "ANY"
    
    
  )
  
  ,methods = list(
    
    initialize = function(..., cost = CrossEntropyCost$new()){
      
      layers <<- list(...)
      cost <<- cost
      
      num_layers <<- length(layers) + 1L
      
      evaluation_cost <<- NULL
      evaluation_accuracy <<- NULL
      training_cost <<- NULL
      training_accuracy <<- NULL
      
    }
    
  )
  
)


Network3$methods(feedforward = function(a){
  
  ## apply each layers feed_forward method
  for(i in 1:(num_layers - 1L))
    a <- layers[[i]]$feed_forward(a)
  
  return(a)
  
})






Network3$methods(SGD = function(x, y,
                                test.x = NULL, test.y = NULL,
                                epochs, mini_batch_size, eta,
                                lambda = 0.0, 
                                evaluation_data = NULL, 
                                monitor_evaluation_cost = FALSE,
                                monitor_evaluation_accuracy = FALSE,
                                monitor_training_cost = FALSE, 
                                monitor_training_accuracy = FALSE
){
  
  ## convert y to its matrix form
  yv <- .self$convert_y_to_matrix(y)
  x <- t(x)
  
  if(!is.null(test.x)){
    n_test_data <- nrow(test.x)
    yv.test <- convert_y_to_matrix(test.y)
    test.x <- t(test.x)
  }
  
  
  ## each column is a records
  ## make y where row is a digit and column is the label for records i
  
  n <- ncol(x)
  for(j in 1:epochs){
    
    ## shuffle data - rather than sampling from we shuffle and take in order
    inds <- sample(1:n)
    x <- x[,inds] 
    yv <- yv[, inds]
    y <- y[inds]
    
    ## split into mini-batches
    n_batches <- n / mini_batch_size
    
    for(i in 1:n_batches){
      inds <- (((i-1)*mini_batch_size) + 1):(i*mini_batch_size) 
      .self$update_mini_batch(mini_batch = x[, inds, drop = FALSE],
                              labels = yv[,inds, drop = FALSE],
                              eta = eta,
                              lambda = lambda,
                              n = n)
    }
    
    cat(sprintf("\nEpoch %s complete\n", j))
    
    if(monitor_training_accuracy){
      accuracy <- .self$accuracy(x = x, y = y)
      training_accuracy <<- c(training_accuracy, accuracy/n)
      cat(sprintf("Accuracy on training data: %s / %s (%s%%)\n", accuracy, n, round(accuracy*100/n, 2))) 
    }
    
    if(monitor_evaluation_accuracy){
      accuracy <- .self$accuracy(x = test.x, y = test.y)
      evaluation_accuracy <<- c(evaluation_accuracy, accuracy/n_test_data)
      cat(sprintf("Accuracy on evaluation data: %s / %s (%s%%)\n", accuracy, n_test_data, round(accuracy*100/n_test_data, 2))) 
    }
    
    ## COSTS
    if(monitor_training_cost){
      cost_m <- .self$total_cost(x, yv, lambda)
      training_cost <<- c(training_cost, cost_m)
      cat(sprintf("Cost on training data: %s\n", round(cost_m, 4))) 
    }
    
    if(monitor_evaluation_cost){
      cost_m <- .self$total_cost(x = test.x, y = yv.test, lambda)
      evaluation_cost <<- c(evaluation_cost, cost_m)
      cat(sprintf("Cost on evaluation data: %s\n", round(cost_m, 4))) 
    }
    
    
  }
  
  NULL
  
})




Network3$methods(update_mini_batch = function(mini_batch, labels, eta, lambda, n){
  
  ## replaced loop with matrix op
  nmb <- ncol(mini_batch)
  
  ## BACKPROP --------------------------------------------------------------------------
  # calculate z = w*a + b
  # calculate next activations as activation_function(z)
  # store z and activations at each step
  # -- the last activation is the network output
  
  # calculate delta as the error in the final layer δL
  #                  (net_output - correct_label) * sigmoid_prime(z_L) OR IF CROSS_ENT (net_output - correct_label)
  nabla_b <- nabla_w <- rep(list(NULL), num_layers - 1)
  
  ## feed forward recording activations as we go
  activation <- mini_batch
  activations <- list(mini_batch)
  zs <- list()
  
  for(i in 1:(num_layers-1)){
    
    z <- layers[[i]]$feed_forward_raw(activation)
    zs <- append(zs, list(z))
    activation <- layers[[i]]$feed_forward(activation)
    activations <- append(activations, list(activation))
  }
  
  
  # error at final layer L.
  delta <- .self$cost$delta(z = zs[[(num_layers - 1L)]], a = activations[[num_layers]], y = labels, activation_fn = layers[[num_layers-1]]$activation_fn) ## BP1
  
  nabla_b[[length(nabla_b)]] <- delta ## BP3
  nabla_w[[length(nabla_w)]] <- delta %*% t(activations[[num_layers-1]]) ## BP4
  
  ## back propagate the error 
  # length(weights) == num_layers - 1
  
  for(l in 2:(num_layers - 1L)){
    
    k <- num_layers - l + 1L
    layer <- layers[[k]]  

    if(class(layers[[k - 1]]) == "FullyConnectedLayer"){
      
      delta <- (t(layer$w) %*% delta) * layer$activation_fn$fn_prime(zs[[num_layers - l]]) ## BP2
      nabla_b[[num_layers - l]] <- delta ## BP3
      nabla_w[[num_layers - l]] <- delta %*% t(activations[[k - 1L]]) ## BP4

      
    }else if(class(layers[[k - 1]]) == "MeanPoolingLayer"){
      
      ## turn delta in to n_kernel x pooled_image_size
      delta <- lapply(1:ncol(delta), function(i) {
        
        tmp <- delta[,i]
        inds <- lapply(1:6, function(i) (((i-1)*144) + 1):(i*144) )
        tmp <- lapply(inds, function(ind) matrix(tmp[ind], ncol = 12, nrow = 12, byrow = TRUE))
        lapply(tmp, mean_pooling_upsample) ## up-sample from pool to conv-layer
      })
      
      a = lapply(1:10, function(i) lapply(1:6, function(j) delta[[i]][[j]] * t(activations[[k - 1L]][[i]][[j]])))
      a = lapply(1:6, function(i){
        tmp <- lapply(a, "[[", i)
        s <- matrix(0, ncol = 24, nrow = 24)
        for(x in tmp)
          s <- s + x
        s
      })
      
      nabla_w[[length(nabla_w) - l + 1L]] <- a ## BP4
      nabla_b[[num_layers - l]] <- NA ## BP3
      
    }else{
      
      browser()
      
      
    }
    
  }
  
  ## BACKPROP END ----------------------------------------------------------------------
  
  ## Update weights
  # w <- w - delta*activation
  nabla_b <- lapply(nabla_b, function(x) as.matrix(rowSums(x), ncol = 1))
  
  for(i in 1:(num_layers - 1L)){
    
    layers[[i]]$w <<- (1 - eta * (lambda / n)) * layers[[i]]$w - (nabla_w[[i]] * eta / nmb)
    layers[[i]]$b <<- layers[[i]]$b - (nabla_b[[i]] * eta / nmb)
    
  }
  
})



Network3$methods(accuracy = function(x, y){
  # browser()
  # x[1:10,1:10]
  
  a <- .self$feedforward(x)
  sum(apply(a, MARGIN = 2, FUN = which.max) == (y+1))
  
})



Network3$methods(total_cost = function(x, yv, lambda){
  
  ## Total cost is dependent on the cost funciton we have chosen .self$cost$fn
  
  a <- .self$feedforward(x)
  costs <- .self$cost$fn(a = a, y = yv) / ncol(x)
  weight_costs <- 0.5 * (lambda / ncol(x)) * sum(sapply(layers, function(layer) norm(x = layer$w, type = "F")^2))
  total_cost <- sum(costs) + weight_costs
  
  return(total_cost)
  
})


Network3$methods(convert_y_to_matrix = function(y){
  sapply(y, function(yi) {
    z <- matrix(0, nrow = 10, ncol = 1)
    z[yi+1] <- 1
    z
  })
})







## Cost functions ---------------------------------------------

QuadraticCost <- setRefClass(
  
  Class = "QuadraticCost",
  
  methods = list(
    fn = function(a, y){
      return ((1/2) * norm(x = matrix(a-y), type = "F")^2)
    },
    
    delta = function(z, a, y, activation_fn){
      return( (a - y) * activation_fn$fn_prime(z) )
    }
  )                             
)



CrossEntropyCost <- setRefClass(
  
  Class = "CrossEntropyCost",
  
  methods = list(
    fn = function(a, y){
      suppressWarnings(r <- -y*log(a) - (1 - y) * log(1 - a))  
      r[is.na(r)] <- 0
      sum(r)
    },
    
    delta = function(z, a, y, activation_fn){
      
      # This is (a - y) / (a*(1-a)) * sigmoid_prime(z)
      # as dC / daj = (a - y) / (a*(1-a))
      ## For sigmoid neurons thie simplifies to (a - y)
      # return((a - y))
      
      return(((a - y) / (a*(1 - a))) * activation_fn$fn_prime(z))
      
    }
  )                   
)

LogLikelihoodCost <- setRefClass(
  
  Class = "LogLikelihoodCost",
  
  methods = list(
    fn = function(a, y){
      
      -log(a)
      
    },
    
    delta = function(z, a, y, activation_fn){
      
      # This is (a - y) / (a*(1-a)) * sigmoid_prime(z)
      # as dC / daj = (a - y) / (a*(1-a))
      ## For sigmoid neurons thie simplifies to (a - y)
      # return((a - y))
      
      return(a - y)
      
    }
  )                   
)


## Activation functions



SigmoidActivation <- setRefClass(
  
  Class = "SigmoidActivation",
  
  methods = list(
    
    ## sigmoid activation function
    fn = function(z){
      return (1 / (1 + exp(-z)))
    },
    
    ## derivative of activation function
    fn_prime = function(z){
      return (fn(z)*(1 - fn(z)))
    }
  )                             
)


SoftmaxActivation <- setRefClass(
  
  Class = "SoftmaxActivation",
  
  methods = list(
    
    ## sigmoid activation function
    fn = function(z){
      browser()
      
      
      
    },
    
    ## derivative of activation function
    fn_prime = function(z){
      return (fn(z)*(1 - fn(z)))
    }
  )                             
)



TanhActivation <- setRefClass(
  
  Class = "TanhActivation",
  
  methods = list(
    
    ## tanh activation function
    fn = function(z){
      return ( tanh(z) )
    },
    
    ## derivative of activation function
    fn_prime = function(z){
      return ( 1 - tanh(z)^2 )
    }
  )                             
)


ReLu <- setRefClass(
  
  Class = "ReLu",
  
  methods = list(
    
    ## relu activation function
    fn = function(z){
      z[z < 0 ] <- 0
      return ( z )
    },
    
    ## derivative of relu function
    fn_prime = function(z){
      # browser()
      z[z <= 0] <- 0
      z[z > 0] <- 1
      return(z)
      
    }
  )                             
)






