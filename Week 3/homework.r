# load required library
library(class)
library("ggplot2")

#################################################
# PREPROCESSING
#################################################

data <- iris                # create copy of iris dataframe
labels <- data$Species      # store labels
data$Species <- NULL        # remove labels from feature set (note: could
# alternatively use neg indices on column index in knn call)


knn.nfold <- function (x, k=10){
  
  N = nrow(x)
  # assuming k is not 0
  chunkSize = ifelse(N >0, round(N / k), 0)
  
  err.gnz <- vector()
  
  for ( i in 1:k ) 
  {
    # i is the test number
    startIx = 1 + ((i-1) * chunkSize)
    endIx = chunkSize * i
    
    train.index = seq(startIx, endIx)
    
    test.data = data[train.index,]
    train.data = data[-train.index,]
    
    train.labels <- as.factor(as.matrix(labels)[-train.index, ])     # extract training set labels
    test.labels <- as.factor(as.matrix(labels)[train.index, ])     # extract test set labels
    
    knn.fit <- knn(train = train.data,          # training set
                   test = test.data,           # test set
                   cl = train.labels,          # true labels
                   k = 8                       # number of NN to poll
    )
    
    err <- sum(train.labels != knn.fit) / length(train.labels)    # store gzn err

    err.gnz[i] <- err    
  }
  
  return (sum(err.gnz) / k)
}

knn.nfold(data)



