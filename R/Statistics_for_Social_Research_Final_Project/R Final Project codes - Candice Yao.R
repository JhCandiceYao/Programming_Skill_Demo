rm(list = ls())
setwd("C:/Users/lavender/Desktop/Fall 2021/Statistics for Social Research/R")

library(readxl)

rm(list = ls())

data <- read.csv("C:/Users/lavender/Desktop/Fall 2021/Statistics for Social Research/R/GSS2018_soc302.csv", header=T)
attach(data)

##(a)Y histogram
hist((satsoc), 
     prob = TRUE,
     main ="Histogram of social satisfaction",
     xlab="social satisfaction",
     col="lightpink") 
hist(satsoc, breaks = seq(from=0.5, to=5.5, by=1 ),col="lightpink")

mean(data$satsoc)
#[1] 2.465843
sd(data$satsoc)
#[1] 0.9642311
#The sample mean of satsoc is 2.466 and the sample standard deviation is 0.964.

##(b)Constructing a 95% confidence interval of Y
#as calculated, sd= 0.964 and mean = 2.466
sd = 0.964
mean = 2.466
n= nrow(data)
#the sample size is 1376
margin_error = 1.96 *sd/sqrt(n)
margin_error = 0.051

lower_bound = mean - margin_error 
lower_bound = 2.415
upper_bound = mean + margin_error
upper_bound = 2.517

#Therefore, the 95% confidence interval of one's social satisfaction is between 2.415 and 2.517, which means that we are 95% percent sure that the true mean lies within this interval.


##(c). Hypothesis test 
mean_male = mean(satsoc[sex==1])
mean_female= mean(satsoc[sex==2])
t.test(satsoc[sex==1],satsoc[sex==2])


##(d). Binary regression model between X(marital, one's marital status) and Y(satsoc, one's satisfaction with social activities and relationships)
model2 <- lm(satsoc ~ quallife, data=data)
summary(model2)


##(e).
library(ggplot2)

ggplot(data) +aes(x = quallife, y =satsoc ) + geom_point(alpha= .05, color = "blue" )
cor(data$age, data$satsoc,use = "complete.obs" )
ggplot(data) +aes(x = quallife, y =satsoc ) + geom_point(alpha= .05, color = "blue" ) +geom_smooth(method = lm, se = FALSE)


##(f).
model3 <- lm(satsoc ~  hlthphys + as.factor(race) + hlthmntl,data = data)
summary(model3)

