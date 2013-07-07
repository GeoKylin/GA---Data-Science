#ref -- http://www.ats.ucla.edu/stat/r/dae/logit.htm

x <- read.csv("http://www.ats.ucla.edu/stat/data/binary.csv")
head(x)
xtabs(~admit + rank, data=x)
#We used summary to get details of a linear regression, we can also used it to get summary statistics of the dataset
summary(x)
#we can find the standard deviation by applying the sd function to each variable
sapply(x, sd)

#let's fit a linear regression
lin.fit <- lm(admit ~ ., data=x)
summary(lin.fit)
#R squared is 0.09% so not great
#Let's fit a linear regression again but this time without an intercept. (if we don't constrain the line to pass thorugh the y axis then maybe we can find a line that fits better. This means that the line is goes through the origin. Sometimes fits better so we try it.)
lin.fit2 <- lm(admit ~ 0 + ., data=x)
summary(lin.fit2)
#R-squared 38.19% this is poor. Worse than tossing a coin to predict admissions.

x$rank <- factor(x$rank)
logit.fit <- glm(admit ~ ., family='binomial', data=x)
summary(logit.fit)

# deviance resids -> measure of model fit (like resids in linear model)
# coeffs -> chg in log-odds of the output variable for unit increase in the input variable
# coeffs of indicator (dummy) vars are slightly different...for example, the coeff of rank2 represents the change in the log-odds of the output variable that comes going to a rank2 school instead of a rank1 school

# odds ratios can be found by exponentiating the log-odds ratios
exp(coef(logit.fit))

# predict oos data

# have a look at mean gre, gpa
summary(x)

# note: important to give columns the same names as in the original df
new.data <- with(x, data.frame(gre=mean(gre), gpa=mean(gpa), rank=factor(1:4)))

# predict probs for new data (varying rank) - what is the effect of rank?
new.data$rank.prob <- predict(logit.fit, newdata=new.data, type='response')
new.data


library("ggplot2")
# predict probs for new data (varying gre)
new.data2 <- with(x, data.frame(gre=rep(seq(from=200, to=800, length.out=100), 4), gpa=mean(gpa), rank=factor(rep(1:4, each=100))))
new.data2$pred <- predict(logit.fit, newdata=new.data2, type='response')
ggplot(new.data2, aes(x=gre, y=pred)) + geom_line(aes(colour=rank), size=1)

# predict probs for new data (varying gpa)
new.data3 <- with(x, data.frame(gpa=rep(seq(from=0, to=4.0, length.out=100), 4), gre=mean(gre), rank=factor(rep(1:4, each=100))))
new.data3$pred <- predict(logit.fit, newdata=new.data3, type='response')
ggplot(new.data3, aes(x=gpa, y=pred)) + geom_line(aes(colour=rank), size=1)

# one thing to keep in mind (no zero/sparse cells)
xtabs(~admit + rank, data = x)
