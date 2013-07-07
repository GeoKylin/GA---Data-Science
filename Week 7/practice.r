setwd("/home/mahbub/Desktop/sourcecode//GA - Data Science/Week 7/")
dt <- read.csv("L7CountryData.csv")

result1 <- kmeans(dt[,2:5], 3)
result1

plot(dt[, 2:5], col=result1$cluster)
#points(result1$centers, col=1:2, pch=8)