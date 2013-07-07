
set.seed(1)
library(ggplot2)
#Generate some dummy data - a parabola
x <- seq(-10, 10, by = 0.01)
y <- 1 - x ^ 2 + rnorm(length(x), 0, 5)
ggplot(data.frame(X = x, Y = y), aes(x = X, y = Y)) +
geom_point() +
geom_smooth(se = FALSE)
#Transform data so we can fit a straight line
x.squared <- x ^ 2
#Now our data looks straight
ggplot(data.frame(XSquared = x.squared, Y=y), aes(x=XSquared, y=Y)) + geom_point() + geom_smooth(method = 'lm', se=FALSE)
#What impact will this have on R Squared of linear model?
summary(lm(y~x))$r.squared # passed the result of lm to SUmmary function and extract the r.squared value
#[1] 1.231097e-05
summary(lm(y~x.squared))$r.squared
#[1] 0.9732342
#The percentage of variance accounted for increases from 0% to 97%. This is a big improvement from such a simple thing as a transformation. This is a common trick in machine learning and we will revisit this when looking at SVMs


#create a dataset that can't be easily modelled with a linear model
set.seed(1)
x<-seq(0,1, by=0.01)
y<-sin(2 * pi * x) + rnorm(length(x), 0, 0.1)
df <- data.frame(X=x, Y=y)
ggplot(df, aes(x = X, y = Y)) + geom_point()
#The data are not liner but let's see how well a linear model fits the data
summary(lm(Y ~ X, data=df))
#We can explain about 60% of the variance which is surprisingly good
ggplot(data.frame(X = x, Y = y), aes(x = X, y = Y)) + geom_point() + geom_smooth(method = 'lm', se = FALSE)
#The chart shows that the model that does that is a downward sloping line, however this model's performance would deteriorate if we considered more data (if the sine wave continues). The linear model overfits and doesn't find the underlying wave structure.
#Now we take the data we have transformed and fit it as well. Will that improve the mode?
df <- transform(df, X2 = X ^ 2)
df <- transform(df, X3 = X ^ 3)
summary(lm(Y ~ X + X2 + X3, data = df))
#adding two more inputs increases the percentage of variance accounted for from 60% to 97%
#what is the problem with adding inputs indefenitely?
df <- transform(df, X4 = X ^ 4)
df <- transform(df, X5 = X ^ 5)
df <- transform(df, X6 = X ^ 6)
df <- transform(df, X7 = X ^ 7)
df <- transform(df, X8 = X ^ 8)
df <- transform(df, X9 = X ^ 9)
df <- transform(df, X10 = X ^ 10)
df <- transform(df, X11 = X ^ 11)
df <- transform(df, X12 = X ^ 12)
df <- transform(df, X13 = X ^ 13)
df <- transform(df, X14 = X ^ 14)
df <- transform(df, X15 = X ^ 15)
summary(lm(Y ~ X + X2 + X3 + X4 + X5 + X6 + X7 + X8 + X9 + X10 + X11 + X12 + X13 + X14, data = df))

# Among other output we'll get this error message: Coefficients: (1 not defined because of singularities)
#This means that some of the terms are so correlated that lm can't run properly. One way around this is to use a function called poly() wich will generate orthogonal polynomials - which means polynomials are not correlated to each other. Let's check that the polynomial function produces the same output as adding the inputs manually
summary(lm(Y ~ poly(X, degree = 14), data = df))
#the output looks like the model we fit earlier. However the coefficients are different as we are generating orthogonal polynomials.
#Mathematically it can be show that this technique allows us to fit complicated shapes (but we won't discuss this in the here).
#Let's see what happens when we increase the degree parameter.
poly.fit <- lm(Y ~ poly(X, degree = 1), data = df)
df <- transform(df, PredictedY = predict(poly.fit))
ggplot(df, aes(x = X, y = PredictedY)) + geom_point() + geom_line()
poly.fit <- lm(Y ~ poly(X, degree = 3), data = df)
df <- transform(df, PredictedY = predict(poly.fit))
ggplot(df, aes(x = X, y = PredictedY)) + geom_point() + geom_line()

poly.fit <- lm(Y ~ poly(X, degree = 5), data = df)
df <- transform(df, PredictedY = predict(poly.fit))
ggplot(df, aes(x = X, y = PredictedY)) + geom_point() + geom_line()
#What happens now?
poly.fit <- lm(Y ~ poly(X, degree = 25), data = df)
df <- transform(df, PredictedY = predict(poly.fit))
ggplot(df, aes(x = X, y = PredictedY)) + geom_point() +geom_line()
#It fails to capture the structure in the data and when this happens we say the model is more powerful than the data

#Preventing overfitting with regularization
#complexity of a model is given by the size of coefficients. We control complexity by enforcing constraints on coefficients sacrificing fit to prevent overfitting.
#
#We'll use the sine wave data
set.seed(1)
x <- seq(0, 1, by = 0.01)
y <- sin(2 * pi * x) + rnorm(length(x), 0, 0.1)
x <- matrix(x)
#We'll use the glmnet package for this
#install.packages('glmnet')
library('glmnet')
glmnet(x, y)
#x needs to be passed as a matrix
#  Df    %Dev   Lambda
#[1,]  0 0.00000 0.542800
#[2,]  1 0.09991 0.494600
#[3,]  1 0.18290 0.450700
#[4,]  1 0.25170 0.410600
#[5,]  1 0.30890 0.374200
#...
#[51,]  1 0.58840 0.005182
#[52,]  1 0.58840 0.004721
#[53,]  1 0.58850 0.004302
#[54,]  1 0.58850 0.003920
#[55,]  1 0.58850 0.003571
#Using glmnet in this way gives us the result of all regularizations glmnet attempted. At the top is the best model and at the bottom the worst
#The meaning of the output is this: Df is the number of parameters in the model, %Dev is how well the modell fits (effectively R squared). Lambda is the regularization parameter used. 
#If a model has few paramters it is said to be sparse. Find ways to get sparse models is an area of active research in machine learning.
#Lambda is a hyperparameter. The intuition is that the large lambda is the more penalize the coefficients for being large if lambda is small we penalize the model less for having large parameters.
#The ideal value of lambda can be found by using cross-validation and fitting a model for many values of lambda. This is not going to be covered here.

set.seed(1)
x <- seq(0, 1, by = 0.01)
y <- sin(2 * pi * x) + rnorm(length(x), 0, 0.1)
n <- length(x)
indices <- sort(sample(1:n, round(0.5 * n)))
training.x <- x[indices]
training.y <- y[indices]
test.x <- x[-indices]
test.y <- y[-indices]
df <- data.frame(X = x, Y = y)
training.df <- data.frame(X = training.x, Y = training.y)
test.df <- data.frame(X = test.x, Y = test.y)
rmse <- function(y, h)
{
    return(sqrt(mean((y - h) ^ 2)))
}

#glmnet fit retains the data for the models it fitted 
glmnet.fit <- with(training.df, glmnet(poly(X, degree = 10), Y))
lambdas <- glmnet.fit$lambda

performance <- data.frame()
for (lambda in lambdas)
{
    performance <- rbind(performance,
    data.frame(Lambda = lambda,
    RMSE = rmse(test.y, with(test.df, predict(glmnet.fit, poly(X,degree = 10), s = lambda)))))
}
#Plot the data
ggplot(performance, aes(x = Lambda, y = RMSE)) +
geom_point() +
geom_line()

best.lambda <- with(performance, Lambda[which(RMSE == min(RMSE))])
glmnet.fit <- with(df, glmnet(poly(X, degree = 10), Y))
coef(glmnet.fit, s = best.lambda)
