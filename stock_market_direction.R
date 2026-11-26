
# Stock Market Direction: Logistic Regression vs Decision Tree
# Dataset: ISLR2::Smarket

# Loading necessary libraries
library(caret)
library(rpart)
library(rpart.plot)
library(pROC)


# Loading the dataset from ISLR2 library

Dat <- ISLR2::Smarket

dim(Dat)

summary(Dat)


# Checking for missing values

table(Dat$Direction, useNA = "always")

Dat <- na.omit(Dat)

Dat$Direction <- factor(Dat$Direction, levels = c("Down", "Up"))


# Class distribution

table(Dat$Direction)

# Convert frequencies into percentages
prop.table(table(Dat$Direction)) * 100


# Visualising the class distribution

barplot(table(Dat$Direction),
        xlab = "Market direction",
        ylab = "Frequency",
        main = "Class distribution")


# Distribution of the predictors
# Histograms to visualise the data

par(mfrow = c(3, 2))

hist(Dat$Lag1, xlab = "Lag1 (% return for previous day)", main = "")
hist(Dat$Lag2, xlab = "Lag2 (% return for 2 days previous)", main = "")
hist(Dat$Lag3, xlab = "Lag3 (% return for 3 days previous)", main = "")
hist(Dat$Lag4, xlab = "Lag4 (% return for 4 days previous)", main = "")
hist(Dat$Lag5, xlab = "Lag5 (% return for 5 days previous)", main = "")
hist(Dat$Volume, xlab = "Volume (daily shares traded in billions)", main = "")

par(mfrow = c(1, 1))


# Correlation between numeric variables
# High correlation between predictors may indicate multicollinearity

cor(Dat[, 1:8], use = "pairwise.complete.obs")


# Splitting dataset into training (80%) and validation (20%)

set.seed(1040)

id <- createDataPartition(Dat$Direction, p = 0.80, list = FALSE)

Train <- Dat[id, ]
Val <- Dat[-id, ]


# Fitting the models
# Today is excluded because it directly determines the current market direction.
# Year is excluded because it is a time label.

# Logistic Regression

ModGLM <- glm(Direction ~ Lag1 + Lag2 + Lag3 + Lag4 + Lag5 + Volume,
              data = Train,
              family = binomial)


# Decision Tree

ModTree <- rpart(Direction ~ Lag1 + Lag2 + Lag3 + Lag4 + Lag5 + Volume,
                 data = Train,
                 method = "class")


# Model summary

summary(ModGLM)


# Visualising the Decision Tree

rpart.plot(ModTree,
           type = 2,
           extra = 104,
           fallen.leaves = TRUE)


# Testing on validation data (0.5 probability cutoff)

pGLM <- predict(ModGLM, newdata = Val, type = "response")

pTree <- predict(ModTree, newdata = Val, type = "prob")[, "Up"]


# Converting probabilities into predicted classes

predGLM <- factor(ifelse(pGLM >= 0.5, "Up", "Down"),
                  levels = c("Down", "Up"))

predTree <- factor(ifelse(pTree >= 0.5, "Up", "Down"),
                   levels = c("Down", "Up"))


# Summarize model performance

evaluate <- function(pred, prob, actual, cutoff) {
  
  TN <- sum(pred == "Down" & actual == "Down")
  
  FP <- sum(pred == "Up" & actual == "Down")
  
  FN <- sum(pred == "Down" & actual == "Up")
  
  TP <- sum(pred == "Up" & actual == "Up")
  
  roc.obj <- roc(actual, prob,
                 levels = c("Down", "Up"),
                 direction = "<",
                 quiet = TRUE)
  
  data.frame(
    Cutoff = cutoff,
    TN = TN,
    FP = FP,
    FN = FN,
    TP = TP,
    Accuracy = (TP + TN) / length(actual),
    Sensitivity = TP / (TP + FN),
    Specificity = TN / (TN + FP),
    AUC = as.numeric(auc(roc.obj))
  )
}


# Evaluating Logistic Regression and Decision Tree

ResultsGLM <- evaluate(predGLM, pGLM, Val$Direction, 0.5)

ResultsTree <- evaluate(predTree, pTree, Val$Direction, 0.5)


# Combining results into one comparison table

Results <- rbind(
  Logistic_Regression = ResultsGLM,
  Decision_Tree = ResultsTree
)

print(Results)


# ROC curves and AUC

rocGLM <- roc(Val$Direction, pGLM,
              levels = c("Down", "Up"),
              direction = "<",
              quiet = TRUE)

rocTree <- roc(Val$Direction, pTree,
               levels = c("Down", "Up"),
               direction = "<",
               quiet = TRUE)


# Plotting ROC curves

plot(rocGLM,
     col = "blue",
     lwd = 2,
     main = "ROC Curves - Validation Data")

plot(rocTree,
     col = "darkgreen",
     lwd = 2,
     add = TRUE)

abline(0, 1, lty = 2, col = "gray")

legend("bottomright",
       legend = c("Logistic Regression", "Decision Tree"),
       col = c("blue", "darkgreen"),
       lwd = 2,
       bty = "n")
