setwd('/home/mahbub/sourcecode/GA/week 2/')
# data set source - http://data.princeton.edu/wws509/datasets/#births
ds <- read.table('phbirths.dat')
plot(ds)

# check the correlation between varibales
cor(ds)


lm.fitAll <- lm(formula = gestate ~ ., data = ds)
summary(lm.fitAll)

# considering cor and linear model summary, going to fit a model of gestate and grams
lm.fit1 <- lm(formula = gestate ~ grams, data = ds)
summary(lm.fit1)

# very low r squared figure
lm.fit2 <- lm(formula = gestate ~ educ, data = ds)
summary(lm.fit2)

# now poly fit on lm.fit1

# Polynomial of degree 2
lm.pfit1 <- lm(formula=gestate ~ poly(grams, degree=2), data=ds)
summary(lm.pfit1)

Polynomial of degree 3
lm.pfit2 <- lm(formula=gestate ~ poly(grams, degree=3), data=ds)
summary(lm.pfit2)

Polynomial of degree 4
lm.pfit3 <- lm(formula=gestate ~ poly(grams, degree=4), data=ds)
summary(lm.pfit3)

## difference between lm.pfit1, lm.pfit2 & lm.pfit3 is very minimum, so it's better to 
#  stop at second degree which is lm.pfit1in this scenario

library(MASS)
## both the regid and normal linear model showed almost same predictions and coefficient
## used this as command reference http://www.stat.sc.edu/~hitchcock/ridgeRexample704.txt
select (lm.ridge(gestate ~ black + educ + smoke + grams, data = ds, lambda = seq(0, 1, 0.001)))
# using the smallest value of GCV for the labda value
lm.ridge(formula = gestate ~ black + educ + smoke + grams, data = ds, lambda = 1)

lm(formula = gestate ~ black + educ + smoke + grams, data = ds)
## in this scenario both the ridge and normal regression are almost same
