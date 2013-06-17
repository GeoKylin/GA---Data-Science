# load required library
library(class)
library("ggplot2")

#################################################
# PREPROCESSING
#################################################

## Global declaration
data <- iris                # create copy of iris dataframe
labels <- data$Species      # store labels
data$Species <- NULL        # remove labels from feature set

## this function os responsible for finding all the generalization error 
## default split is set to 10, if nothing provided it will be used
knn.nfold <- function (x, k=10) {

  N = nrow(x)  # getting the size of dataset

  chunkSize = ifelse(N >1, round(N / k), 0)     # assuming k is greater than 1, getting each block/chunk size
    

  err.gnz <- vector()   # to hold all the generalization errors
  
  for ( i in 1:k ) # i is the test number, 1 to k
  {
  
    ## Building the staring and ending index for each block
    ## e.g. 
    ## if the chunk/block size is 15 then
    ## First test it would 1 - 15, startIx: 1 + ((1-1) * 15) => 1, endIx: 15 * 1 => 15
    ## Second test it would be 16 - 30, startIx: 1 + ((2-1) * 15) => 16 endIx: 15 * 2 => 30
    ## ...... 
    startIx = 1 + ((i-1) * chunkSize)
    endIx = chunkSize * i
    
    ## Creating a sequence of indexes based on start and end Ixs
    train.index = seq(startIx, endIx)
    
    test.data = data[train.index,]    # test data set based on indexes
    train.data = data[-train.index,]  # train data set excluding all the test data sets
    
    train.labels <- as.factor(as.matrix(labels)[-train.index, ])     # extract training set labels
    test.labels <- as.factor(as.matrix(labels)[train.index, ])     # extract test set labels
    

    knn.fit <- knn(train = train.data,          # training set
                   test = test.data,           # test set
                   cl = train.labels,          # true labels
                   k = 8                       # number of NN to poll
    )
    
    err.gnz[i] <- sum(train.labels != knn.fit) / length(train.labels)    # store gzn err for each test

  }
  
  return (sum(err.gnz) / k)  # return average of all the test generalization error 
}


err <-knn.nfold(data,15)
paste("Generalization error: ", err)  # display error output



