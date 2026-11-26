Overview:

This code predicts whether the S&P 500 goes Up or Down on a given day using the 
previous five days' returns and trading volume. It compares a statistical model 
(logistic regression) with a machine learning model (decision tree) to see 
whether the more flexible ML approach does any better.

Data: 

Smarket from the ISLR2 package

Variables:

Variable	  Meaning
Year	      The year that the observation was recorded
Lag1–Lag5	  % return of the S&P 500 on each of the previous 5 days
Volume	    Shares traded the previous day (billions)
Today	      % return today
Direction	  A factor with levels Down and Up indicating whether the market had 
            a positive or negative return on a given day
            
Outputs and plots:
![Barplot of Classification](barplot.png)
![Histograms of Lag variables and Volume](histograms.png)
