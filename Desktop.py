#!/usr/bin/env python
# coding: utf-8

# # GRIP : The Sparks Foundation
# 
# # Data Science and Buiseness Analytics Intern
# 
# # Author: ASHU RANA
# 
# # Task 1: Prediction using Supervised ML
# ***
# In this task, We are going to predict the percentage score of a student based on the number of hours studied. The task has two variables where the feature is the no. of hours studied and the target value is the percentage score. This can be solved using simple linear regression.
# 
# Data :
# 
# Hours,Scores : 
# 2.5,21 ;
# 5.1,47 ;
# 3.2,27 ;
# 8.5,75 ;
# 3.5,30 ;
# 1.5,20 ;
# 9.2,88 ;
# 5.5,60 ;
# 8.3,81 ;
# 2.7,25 ;
# 7.7,85 ;
# 5.9,62 ;
# 4.5,41 ;
# 3.3,42 ;
# 1.1,17 ;
# 8.9,95 ;
# 2.5,30 ;
# 1.9,24 ;
# 6.1,67 ;
# 7.4,69 ;
# 2.7,30 ;
# 4.8,54 ;
# 3.8,35 ;
# 6.9,76 ;
# 7.8,86 .
# 

# In[43]:


# Importing the required libraries

import pandas as pd                          # Pandas stands for panel data, library for data manipulation and data analysis
import numpy as np                           # Numpy stands for numerical pythn, library for numeric and scientific computing
import matplotlib.pyplot as plt              # library for Data Visualisation and draw various plots and charts
import seaborn as sns                        # library for visualisation and built over matplotlib


# In[44]:


# Reading data from remote URL and printing that data

url = 'http://bit.ly/w-data'
data = pd.read_csv(url)
print(data.shape)


# In[45]:


# showing first 5 Rows of dataset
data.head()             


# In[46]:


# describing about data like count, mean, min etc.
data.describe()         


# In[47]:


# showing info about data like how many null values, no. of columns, memory, type of variable etc.
data.info()            


# In[48]:


# plotting the scatterplot of 2 variables where the no. of hours studied on x-axis & scores on y-axis
data.plot(kind='scatter', x='Hours', y='Scores')
plt.show()


# In[49]:


# from above,we can see there is a linear relationship between two variables which can be validated from correlation coefficient
data.corr(method='pearson')


# In[50]:


# coefficient is 0.976 approximately equal to 1 and is positive which means there is a positive linear relationship 
# implies Hours is directly proportional to scores which also makes sense


# In[51]:


# Assigning hours and scores columns of dataset in hours and scores as lists type So that we can use it directly
hours = data['Hours']
scores = data['Scores']


# In[52]:


# plotting the distribution plot of the two variable, getting variables are in particular range & there are no outliers
sns.distplot(hours)


# In[53]:


sns.distplot(scores)


# # Linear Regression
# 
# Here, we are using Linear Regression model. What linear regression does is that, it finds the slope and intercept(denoted by m) of the line where all the points fall. But, according to scatter plot, there can't be any such line in which all the points fall so, we find a line in which difference of the predicted values of the line and the actual values of the point is minimum i.e. sum of the difference of the predicted values and the actual values of the score has to be minimum.

# In[54]:


# allocating data into x and y
x = data.iloc[:, :1].values
y = data.iloc[:, 1:].values


# In[55]:


x


# In[56]:


y


# In[57]:


# In this, we are first dividing our data into Train dataset(80% data) and Test dataset(20% data).
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=50)


# In[58]:


from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x_train, y_train)


# In[59]:


# Two outputs of the line: Slope of the line & Intercepts
m = reg.coef_
c = reg.intercept_
line = m*x + c
plt.scatter(x,y)
plt.plot(x,line)
plt.show()


# In[61]:


# Target values which we got in the data and the predicted values 
y_pred=reg.predict(x_test)


# In[71]:


# This is the Predicted score
y_pred


# In[72]:


y_test


# In[78]:


# Comparison Actual vs Predicted
actual_predicted = pd.DataFrame({'Target':[y_test], 'Predicted':[y_pred]})
actual_predicted


# In[82]:


# plotting a distribution plot for the difference between the targeted value and predicted value 
# This difference is very close to zero and range is -5 to 5 which tells our model is fitting the data well
sns.set_style('whitegrid')
sns.distplot(np.array(y_test-y_pred))
plt.show()


# # What will be the predicted score if a student studies for 9.25 hours/day ? 

# In[83]:


h=9.25
s=reg.predict([[h]])
print("If a student studies for {} hours per day he/she will score {} % in exam.".format(h,s))


# # Model Evaluation

# In[86]:


from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))


# In[ ]:




