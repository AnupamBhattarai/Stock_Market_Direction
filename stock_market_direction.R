# Stock Market Direction: Logistic Regression vs Decision Tree
# Dataset: ISLR2::Smarket
# Submitted by: Anupam Bhattarai, Darrel Primandaru, Si June Jun

# loading necessary libraries
library(caret)
library(rpart)
library(rpart.plot)
library(pROC)


# loading the data set from ISLR2 library

Dat <- ISLR2::Smarket

dim(Dat)
summary(Dat)


# checking for missing values

table(Dat$Direction, useNA = "always")

Dat <- na.omit(Dat)

Dat$Direction <- factor(Dat$Direction,
                        levels = c("Down", "Up"))


# class distribution

table(Dat$Direction)
prop.table(table(Dat$Direction)) * 100

barplot(table(Dat$Direction),
        xlab = "Market direction",
        ylab = "Frequency",
        main = "Class distribution")


# 4. Distribution of the predictors
# (these are continuous, so histograms instead of bar plots)

par(mfrow = c(3, 2))

hist(Dat$Lag1, xlab = "Lag1 (% return 1 day before)", main = "")
hist(Dat$Lag2, xlab = "Lag2 (% return 2 days before)", main = "")
hist(Dat$Lag3, xlab = "Lag3 (% return 3 days before)", main = "")
hist(Dat$Lag4, xlab = "Lag4 (% return 4 days before)", main = "")
hist(Dat$Lag5, xlab = "Lag5 (% return 5 days before)", main = "")
hist(Dat$Volume, xlab = "Volume (billions of shares)", main = "")

par(mfrow = c(1, 1))


# 5. Correlation between the numeric variables
# High correlation between predictors = multicollinearity,
# which affects logistic regression

cor(Dat[, 1:8], use = "pairwise.complete.obs")


# 6. Split into training (80%) and validation (20%)

set.seed(1040)

id <- createDataPartition(Dat$Direction,
                          p = 0.80,
                          list = FALSE)

Train <- Dat[id, ]
Val <- Dat[-id, ]


# 7. Fit the models
# Today is left out: Direction is just whether Today > 0 (it would give the answer away)
# Year is left out: it is a time label, not market information

ModGLM <- glm(Direction ~ Lag1 + Lag2 + Lag3 + Lag4 + Lag5 + Volume,
              data = Train,
              family = binomial)

ModTree <- rpart(Direction ~ Lag1 + Lag2 + Lag3 + Lag4 + Lag5 + Volume,
                 data = Train,
                 method = "class")

summary(ModGLM)

rpart.plot(ModTree,
           type = 2,
           extra = 104,
           fallen.leaves = TRUE)


# 8. Predict on the validation data (0.5 cutoff)

pGLM <- predict(ModGLM, newdata = Val, type = "response")
pTree <- predict(ModTree, newdata = Val, type = "prob")[, "Up"]

predGLM <- factor(ifelse(pGLM >= 0.5, "Up", "Down"),
                  levels = c("Down", "Up"))

predTree <- factor(ifelse(pTree >= 0.5, "Up", "Down"),
                   levels = c("Down", "Up"))


# 9. Confusion matrices (accuracy, sensitivity, specificity)

confusionMatrix(predGLM, Val$Direction, positive = "Up")
confusionMatrix(predTree, Val$Direction, positive = "Up")

# Baseline: accuracy if we always guessed the most common class
max(prop.table(table(Val$Direction)))


# 10. ROC curves and AUC

rocGLM <- roc(Val$Direction, pGLM, levels = c("Down", "Up"),
              direction = "<", quiet = TRUE)

rocTree <- roc(Val$Direction, pTree, levels = c("Down", "Up"),
               direction = "<", quiet = TRUE)

auc(rocGLM)
auc(rocTree)

plot(rocGLM, col = "blue", lwd = 2,
     main = "ROC Curves - Validation Data")

plot(rocTree, col = "darkgreen", lwd = 2, add = TRUE)

abline(0, 1, lty = 2, col = "gray")

legend("bottomright",
       legend = c("Logistic regression", "Decision tree"),
       col = c("blue", "darkgreen"),
       lwd = 2,
       bty = "n")
