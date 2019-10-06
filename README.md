
# Predicting cold call success for car insurance
_Our objective in this notebook is to predict whether or not a person will sign up for car insurance over the phone, given information such as marital status, education, etc._

# Table of contents
1. [Background](#Background)
2. [Data ingestion](#Data-ingestion)
3. [Data preprocessing](#Data-preprocessing)
4. [Exploratory Data Analysis](#EDA)
5. [Model Training](#Model-Training)
6. [Testing the Model](#Testing-the-Model)
7. [Conclusion](#Conclusion)
8. [Credits](#Credits)

# Background

## Purpose
Cold calling has been a controversial business route for both companies and customers alike, and the growth of mobile phones across the globe means that reaching people over the phone has never been easier. [92% of all customer interactions happen over the phone](https://blog.thebrevetgroup.com/21-mind-blowing-sales-stats), meaning that cold calling is still very relevant today.  

Both companies and consumers dislike cold calling for two main reasons: cold calling takes a long amount of time and cold calling can have detrimental effects when done improperly.  

This article will detail one possible route to improving cold call efficiency by utilizing public callee data to pre-emptively predict if a call will be successful. 

## Introduction

In this notebook, I used a public dataset from Kaggle located [HERE](https://www.kaggle.com/kondla/carinsurance). Thanks to the Technical University of Munich and Kaggle for providing the data.  

The first step will be observing our data and understanding what we're dealing with. Second, we will "clean" the dataset, dropping empty values and scaling where needed. Third, we will balance our dataset to get reliable predictions for fraudulent and non-fraudulent transactions. Fourth, we will build and train a model to help us predict outcomes. 

This notebook contains various columns with categorical data. Let's do a quick rundown of the contents of the dataset. 

**Id** is a unique number given to each person called.  

**Age**, **Job**, **Marital Status**, and **Education**, etc. are all self-explanatory. 

There are some columns that are not detailed in the Kaggle page for this dataset. I've tried to use my best reasoning behind their meaning, but there may be some errors in my understanding. 

**CallStart** and **CallEnd** are used to calculate the duration of the call.  

**Outcome** is whether the call resulted in a sale or not. 

# Data ingestion

## Initial thoughts

This dataset requires a lot of modification in order to be able to accurately predict successes. This is because our prediction models use numerical data to train themselves. If these numbers are very different from one another, or they appear to be randomly occurring, the algorithms we train will have a tough time trying to gain any "insight" on how to predict.  

Keep in mind that for people, predicting the next number after 2, 4, 8, 16 is simple *because we can understand the pattern*. We see that the subsequent number is double the previous, and we use that *rule* to predict the following number.  

In machine learning, the computer uses algorithms to help it find the same rules we use. This is where the learning occurs. The machine would go through the data points and find a function, in this case y = 2^x, that it will use to predict the following number. Of course, the actual math and theory behind this is infinitely more complicated, but for our use case, this simple understanding is enough. 

Like the credit card model, there are a lot of null values for the 'Outcome' column, but not very many successes or failures. We will most likely have to restructure our dataset by dropping these null values to improve the accuracy of our models. This has the negative effect of also reducing the size of our dataset, but the accuracy of the model won't change too much compared to if we had left the null values alone. 

CallStart and CallEnd aren't very useful attributes on their own, but we can generate a new column, 'CallLength', to better reflect what we want. It seems really important to know the length of a sales call, especially the differences in length between a successful call and a failed one. This process is called feature engineering, which uses existing data to create new data that is more relevant to our use case.  

We will also have to numerically map a lot of the categorical data present in this file. Categorical data is anything that isn't a number. Since our machine learning models use numbers to find the prediction rules, we map the non-numerical data to numbers and use that to train the model. 


# Data preprocessing

The purpose of the data preprocessing stage is to minimize potential error in the model as much as possible. Generally, a model is only as good as the data passed into it, and the data preprocessing we do ensures that the model has as accurate a dataset as possible. While we cannot perfectly clean the dataset, we can at least follow some basics steps to ensure that our dataset has the best possible chance of generating a good model. 

First, let's check for null values in our dataset. Null values are empty, useless entries within our dataset that we don't need. If we skip removing null values, our model will be inaccurate as we create "connections" for useless values, rather than focusing all resources onto creating connections for useful values. 


```python
import pandas as pd
import numpy as np

df = pd.read_csv('carInsurance_train.csv')
print("Presence of any null values: " + str(df.isnull().values.any()))
```

    Presence of any null values: True
    

We will be clearing out these null values before we can use this dataset. Let's take a peek at what our dataset looks like.


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>Age</th>
      <th>Job</th>
      <th>Marital</th>
      <th>Education</th>
      <th>Default</th>
      <th>Balance</th>
      <th>HHInsurance</th>
      <th>CarLoan</th>
      <th>Communication</th>
      <th>LastContactDay</th>
      <th>LastContactMonth</th>
      <th>NoOfContacts</th>
      <th>DaysPassed</th>
      <th>PrevAttempts</th>
      <th>Outcome</th>
      <th>CallStart</th>
      <th>CallEnd</th>
      <th>CarInsurance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>32</td>
      <td>management</td>
      <td>single</td>
      <td>tertiary</td>
      <td>0</td>
      <td>1218</td>
      <td>1</td>
      <td>0</td>
      <td>telephone</td>
      <td>28</td>
      <td>jan</td>
      <td>2</td>
      <td>-1</td>
      <td>0</td>
      <td>NaN</td>
      <td>13:45:20</td>
      <td>13:46:30</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>32</td>
      <td>blue-collar</td>
      <td>married</td>
      <td>primary</td>
      <td>0</td>
      <td>1156</td>
      <td>1</td>
      <td>0</td>
      <td>NaN</td>
      <td>26</td>
      <td>may</td>
      <td>5</td>
      <td>-1</td>
      <td>0</td>
      <td>NaN</td>
      <td>14:49:03</td>
      <td>14:52:08</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>29</td>
      <td>management</td>
      <td>single</td>
      <td>tertiary</td>
      <td>0</td>
      <td>637</td>
      <td>1</td>
      <td>0</td>
      <td>cellular</td>
      <td>3</td>
      <td>jun</td>
      <td>1</td>
      <td>119</td>
      <td>1</td>
      <td>failure</td>
      <td>16:30:24</td>
      <td>16:36:04</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>25</td>
      <td>student</td>
      <td>single</td>
      <td>primary</td>
      <td>0</td>
      <td>373</td>
      <td>1</td>
      <td>0</td>
      <td>cellular</td>
      <td>11</td>
      <td>may</td>
      <td>2</td>
      <td>-1</td>
      <td>0</td>
      <td>NaN</td>
      <td>12:06:43</td>
      <td>12:20:22</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>30</td>
      <td>management</td>
      <td>married</td>
      <td>tertiary</td>
      <td>0</td>
      <td>2694</td>
      <td>0</td>
      <td>0</td>
      <td>cellular</td>
      <td>3</td>
      <td>jun</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>NaN</td>
      <td>14:35:44</td>
      <td>14:38:56</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



We've been given a testing and training dataset, but I like to merge the test and training set, and split them later for my own use. 


```python
test_df = pd.read_csv("carInsurance_test.csv")
test_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>Age</th>
      <th>Job</th>
      <th>Marital</th>
      <th>Education</th>
      <th>Default</th>
      <th>Balance</th>
      <th>HHInsurance</th>
      <th>CarLoan</th>
      <th>Communication</th>
      <th>LastContactDay</th>
      <th>LastContactMonth</th>
      <th>NoOfContacts</th>
      <th>DaysPassed</th>
      <th>PrevAttempts</th>
      <th>Outcome</th>
      <th>CallStart</th>
      <th>CallEnd</th>
      <th>CarInsurance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4001</td>
      <td>25</td>
      <td>admin.</td>
      <td>single</td>
      <td>secondary</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>NaN</td>
      <td>12</td>
      <td>may</td>
      <td>12</td>
      <td>-1</td>
      <td>0</td>
      <td>NaN</td>
      <td>17:17:42</td>
      <td>17:18:06</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4002</td>
      <td>40</td>
      <td>management</td>
      <td>married</td>
      <td>tertiary</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>cellular</td>
      <td>24</td>
      <td>jul</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>NaN</td>
      <td>09:13:44</td>
      <td>09:14:37</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4003</td>
      <td>44</td>
      <td>management</td>
      <td>single</td>
      <td>tertiary</td>
      <td>0</td>
      <td>-1313</td>
      <td>1</td>
      <td>1</td>
      <td>cellular</td>
      <td>15</td>
      <td>may</td>
      <td>10</td>
      <td>-1</td>
      <td>0</td>
      <td>NaN</td>
      <td>15:24:07</td>
      <td>15:25:51</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4004</td>
      <td>27</td>
      <td>services</td>
      <td>single</td>
      <td>secondary</td>
      <td>0</td>
      <td>6279</td>
      <td>1</td>
      <td>0</td>
      <td>cellular</td>
      <td>9</td>
      <td>nov</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>NaN</td>
      <td>09:43:44</td>
      <td>09:48:01</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4005</td>
      <td>53</td>
      <td>technician</td>
      <td>married</td>
      <td>secondary</td>
      <td>0</td>
      <td>7984</td>
      <td>1</td>
      <td>0</td>
      <td>cellular</td>
      <td>2</td>
      <td>feb</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>NaN</td>
      <td>16:31:51</td>
      <td>16:34:22</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
print("Presence of any null values: " + str(test_df.isnull().values.any()))
```

    Presence of any null values: True
    

We will have to clear out these null values before we can do any operations with this data. First, let's merge the datasets together, and then we can clean out the null values from our data. 


```python
df = pd.concat([df, test_df], keys=('train','test'))
np.where(pd.isnull(df))
```




    (array([   0,    1,    1, ..., 4998, 4999, 4999], dtype=int64),
     array([15,  9, 15, ..., 18, 15, 18], dtype=int64))



Judging from the dataset previews above, most of the NaNs appear to be coming from Communication, Outcome, and CarInsurance. Let's drop Communication and CarInsurance, and fill NaN Outcome values with 0. I would drop these rows entirely normally, but since we have very few data points already, it's better to fill them with 'unsuccessful' values rather than have 100 valid data points. 


```python
df.pop('Communication')
df.pop('CarInsurance')
df.fillna('failure')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>Id</th>
      <th>Age</th>
      <th>Job</th>
      <th>Marital</th>
      <th>Education</th>
      <th>Default</th>
      <th>Balance</th>
      <th>HHInsurance</th>
      <th>CarLoan</th>
      <th>LastContactDay</th>
      <th>LastContactMonth</th>
      <th>NoOfContacts</th>
      <th>DaysPassed</th>
      <th>PrevAttempts</th>
      <th>Outcome</th>
      <th>CallStart</th>
      <th>CallEnd</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="30" valign="top">train</th>
      <th>0</th>
      <td>1</td>
      <td>32</td>
      <td>management</td>
      <td>single</td>
      <td>tertiary</td>
      <td>0</td>
      <td>1218</td>
      <td>1</td>
      <td>0</td>
      <td>28</td>
      <td>jan</td>
      <td>2</td>
      <td>-1</td>
      <td>0</td>
      <td>failure</td>
      <td>13:45:20</td>
      <td>13:46:30</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>32</td>
      <td>blue-collar</td>
      <td>married</td>
      <td>primary</td>
      <td>0</td>
      <td>1156</td>
      <td>1</td>
      <td>0</td>
      <td>26</td>
      <td>may</td>
      <td>5</td>
      <td>-1</td>
      <td>0</td>
      <td>failure</td>
      <td>14:49:03</td>
      <td>14:52:08</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>29</td>
      <td>management</td>
      <td>single</td>
      <td>tertiary</td>
      <td>0</td>
      <td>637</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>jun</td>
      <td>1</td>
      <td>119</td>
      <td>1</td>
      <td>failure</td>
      <td>16:30:24</td>
      <td>16:36:04</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>25</td>
      <td>student</td>
      <td>single</td>
      <td>primary</td>
      <td>0</td>
      <td>373</td>
      <td>1</td>
      <td>0</td>
      <td>11</td>
      <td>may</td>
      <td>2</td>
      <td>-1</td>
      <td>0</td>
      <td>failure</td>
      <td>12:06:43</td>
      <td>12:20:22</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>30</td>
      <td>management</td>
      <td>married</td>
      <td>tertiary</td>
      <td>0</td>
      <td>2694</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>jun</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>failure</td>
      <td>14:35:44</td>
      <td>14:38:56</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>32</td>
      <td>technician</td>
      <td>single</td>
      <td>tertiary</td>
      <td>0</td>
      <td>1625</td>
      <td>0</td>
      <td>0</td>
      <td>22</td>
      <td>may</td>
      <td>1</td>
      <td>109</td>
      <td>1</td>
      <td>failure</td>
      <td>14:58:08</td>
      <td>15:11:24</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>37</td>
      <td>admin.</td>
      <td>single</td>
      <td>tertiary</td>
      <td>0</td>
      <td>1000</td>
      <td>1</td>
      <td>0</td>
      <td>17</td>
      <td>mar</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>failure</td>
      <td>13:00:02</td>
      <td>13:03:17</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>35</td>
      <td>management</td>
      <td>divorced</td>
      <td>tertiary</td>
      <td>0</td>
      <td>538</td>
      <td>1</td>
      <td>0</td>
      <td>12</td>
      <td>may</td>
      <td>4</td>
      <td>-1</td>
      <td>0</td>
      <td>failure</td>
      <td>15:39:43</td>
      <td>15:40:49</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>30</td>
      <td>admin.</td>
      <td>married</td>
      <td>secondary</td>
      <td>0</td>
      <td>187</td>
      <td>1</td>
      <td>0</td>
      <td>18</td>
      <td>nov</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>failure</td>
      <td>12:20:56</td>
      <td>12:22:42</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>30</td>
      <td>blue-collar</td>
      <td>married</td>
      <td>secondary</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>12</td>
      <td>may</td>
      <td>2</td>
      <td>-1</td>
      <td>0</td>
      <td>failure</td>
      <td>09:22:20</td>
      <td>09:27:46</td>
    </tr>
    <tr>
      <th>10</th>
      <td>11</td>
      <td>27</td>
      <td>services</td>
      <td>single</td>
      <td>secondary</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>jun</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>failure</td>
      <td>13:04:28</td>
      <td>13:07:29</td>
    </tr>
    <tr>
      <th>11</th>
      <td>12</td>
      <td>53</td>
      <td>blue-collar</td>
      <td>married</td>
      <td>primary</td>
      <td>1</td>
      <td>-462</td>
      <td>0</td>
      <td>0</td>
      <td>29</td>
      <td>jan</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>failure</td>
      <td>16:45:50</td>
      <td>16:53:40</td>
    </tr>
    <tr>
      <th>12</th>
      <td>13</td>
      <td>44</td>
      <td>technician</td>
      <td>married</td>
      <td>secondary</td>
      <td>0</td>
      <td>2776</td>
      <td>1</td>
      <td>0</td>
      <td>27</td>
      <td>jan</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>failure</td>
      <td>16:19:41</td>
      <td>16:31:22</td>
    </tr>
    <tr>
      <th>13</th>
      <td>14</td>
      <td>38</td>
      <td>self-employed</td>
      <td>divorced</td>
      <td>secondary</td>
      <td>0</td>
      <td>2674</td>
      <td>1</td>
      <td>0</td>
      <td>19</td>
      <td>jun</td>
      <td>2</td>
      <td>-1</td>
      <td>0</td>
      <td>failure</td>
      <td>10:29:56</td>
      <td>10:32:39</td>
    </tr>
    <tr>
      <th>14</th>
      <td>15</td>
      <td>25</td>
      <td>services</td>
      <td>single</td>
      <td>failure</td>
      <td>0</td>
      <td>2022</td>
      <td>0</td>
      <td>0</td>
      <td>29</td>
      <td>jul</td>
      <td>8</td>
      <td>97</td>
      <td>12</td>
      <td>other</td>
      <td>14:15:09</td>
      <td>14:19:45</td>
    </tr>
    <tr>
      <th>15</th>
      <td>16</td>
      <td>61</td>
      <td>management</td>
      <td>single</td>
      <td>tertiary</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>12</td>
      <td>aug</td>
      <td>1</td>
      <td>114</td>
      <td>3</td>
      <td>failure</td>
      <td>16:18:48</td>
      <td>16:20:59</td>
    </tr>
    <tr>
      <th>16</th>
      <td>17</td>
      <td>34</td>
      <td>admin.</td>
      <td>single</td>
      <td>secondary</td>
      <td>0</td>
      <td>69</td>
      <td>1</td>
      <td>0</td>
      <td>6</td>
      <td>may</td>
      <td>3</td>
      <td>362</td>
      <td>4</td>
      <td>other</td>
      <td>11:48:45</td>
      <td>11:50:17</td>
    </tr>
    <tr>
      <th>17</th>
      <td>18</td>
      <td>46</td>
      <td>management</td>
      <td>married</td>
      <td>tertiary</td>
      <td>0</td>
      <td>7331</td>
      <td>0</td>
      <td>0</td>
      <td>11</td>
      <td>sep</td>
      <td>4</td>
      <td>95</td>
      <td>2</td>
      <td>other</td>
      <td>11:23:26</td>
      <td>11:34:24</td>
    </tr>
    <tr>
      <th>18</th>
      <td>19</td>
      <td>49</td>
      <td>blue-collar</td>
      <td>married</td>
      <td>secondary</td>
      <td>0</td>
      <td>2039</td>
      <td>1</td>
      <td>0</td>
      <td>6</td>
      <td>may</td>
      <td>1</td>
      <td>169</td>
      <td>2</td>
      <td>failure</td>
      <td>12:42:54</td>
      <td>12:50:25</td>
    </tr>
    <tr>
      <th>19</th>
      <td>20</td>
      <td>50</td>
      <td>blue-collar</td>
      <td>married</td>
      <td>primary</td>
      <td>0</td>
      <td>82</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>apr</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>failure</td>
      <td>10:34:08</td>
      <td>10:41:44</td>
    </tr>
    <tr>
      <th>20</th>
      <td>21</td>
      <td>57</td>
      <td>blue-collar</td>
      <td>married</td>
      <td>secondary</td>
      <td>0</td>
      <td>773</td>
      <td>1</td>
      <td>1</td>
      <td>18</td>
      <td>jun</td>
      <td>8</td>
      <td>-1</td>
      <td>0</td>
      <td>failure</td>
      <td>09:05:50</td>
      <td>09:07:27</td>
    </tr>
    <tr>
      <th>21</th>
      <td>22</td>
      <td>80</td>
      <td>retired</td>
      <td>married</td>
      <td>secondary</td>
      <td>0</td>
      <td>8304</td>
      <td>0</td>
      <td>0</td>
      <td>8</td>
      <td>sep</td>
      <td>3</td>
      <td>91</td>
      <td>13</td>
      <td>success</td>
      <td>17:37:41</td>
      <td>17:47:47</td>
    </tr>
    <tr>
      <th>22</th>
      <td>23</td>
      <td>31</td>
      <td>management</td>
      <td>single</td>
      <td>tertiary</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>21</td>
      <td>aug</td>
      <td>2</td>
      <td>-1</td>
      <td>0</td>
      <td>failure</td>
      <td>14:28:54</td>
      <td>14:32:32</td>
    </tr>
    <tr>
      <th>23</th>
      <td>24</td>
      <td>63</td>
      <td>retired</td>
      <td>married</td>
      <td>secondary</td>
      <td>0</td>
      <td>2896</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>jul</td>
      <td>1</td>
      <td>119</td>
      <td>2</td>
      <td>success</td>
      <td>15:37:19</td>
      <td>15:42:23</td>
    </tr>
    <tr>
      <th>24</th>
      <td>25</td>
      <td>60</td>
      <td>technician</td>
      <td>married</td>
      <td>secondary</td>
      <td>0</td>
      <td>824</td>
      <td>1</td>
      <td>0</td>
      <td>9</td>
      <td>feb</td>
      <td>1</td>
      <td>558</td>
      <td>7</td>
      <td>other</td>
      <td>16:30:52</td>
      <td>16:32:59</td>
    </tr>
    <tr>
      <th>25</th>
      <td>26</td>
      <td>29</td>
      <td>management</td>
      <td>married</td>
      <td>tertiary</td>
      <td>0</td>
      <td>900</td>
      <td>0</td>
      <td>0</td>
      <td>17</td>
      <td>jul</td>
      <td>2</td>
      <td>-1</td>
      <td>0</td>
      <td>failure</td>
      <td>17:06:47</td>
      <td>17:11:22</td>
    </tr>
    <tr>
      <th>26</th>
      <td>27</td>
      <td>54</td>
      <td>management</td>
      <td>married</td>
      <td>primary</td>
      <td>0</td>
      <td>3859</td>
      <td>0</td>
      <td>1</td>
      <td>20</td>
      <td>nov</td>
      <td>3</td>
      <td>-1</td>
      <td>0</td>
      <td>failure</td>
      <td>10:09:02</td>
      <td>10:10:46</td>
    </tr>
    <tr>
      <th>27</th>
      <td>28</td>
      <td>45</td>
      <td>failure</td>
      <td>divorced</td>
      <td>failure</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>21</td>
      <td>apr</td>
      <td>3</td>
      <td>-1</td>
      <td>0</td>
      <td>failure</td>
      <td>12:33:22</td>
      <td>12:38:00</td>
    </tr>
    <tr>
      <th>28</th>
      <td>29</td>
      <td>34</td>
      <td>services</td>
      <td>married</td>
      <td>secondary</td>
      <td>0</td>
      <td>-62</td>
      <td>1</td>
      <td>0</td>
      <td>16</td>
      <td>jun</td>
      <td>3</td>
      <td>-1</td>
      <td>0</td>
      <td>failure</td>
      <td>13:50:40</td>
      <td>13:52:29</td>
    </tr>
    <tr>
      <th>29</th>
      <td>30</td>
      <td>42</td>
      <td>blue-collar</td>
      <td>married</td>
      <td>primary</td>
      <td>0</td>
      <td>832</td>
      <td>0</td>
      <td>0</td>
      <td>20</td>
      <td>jun</td>
      <td>14</td>
      <td>-1</td>
      <td>0</td>
      <td>failure</td>
      <td>10:36:52</td>
      <td>10:37:05</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th rowspan="30" valign="top">test</th>
      <th>970</th>
      <td>4971</td>
      <td>56</td>
      <td>retired</td>
      <td>married</td>
      <td>primary</td>
      <td>0</td>
      <td>340</td>
      <td>1</td>
      <td>0</td>
      <td>21</td>
      <td>jul</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>failure</td>
      <td>09:14:13</td>
      <td>09:34:33</td>
    </tr>
    <tr>
      <th>971</th>
      <td>4972</td>
      <td>38</td>
      <td>unemployed</td>
      <td>married</td>
      <td>primary</td>
      <td>0</td>
      <td>890</td>
      <td>1</td>
      <td>0</td>
      <td>5</td>
      <td>feb</td>
      <td>2</td>
      <td>-1</td>
      <td>0</td>
      <td>failure</td>
      <td>16:33:22</td>
      <td>16:34:05</td>
    </tr>
    <tr>
      <th>972</th>
      <td>4973</td>
      <td>62</td>
      <td>housemaid</td>
      <td>married</td>
      <td>failure</td>
      <td>0</td>
      <td>2021</td>
      <td>0</td>
      <td>0</td>
      <td>26</td>
      <td>feb</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>failure</td>
      <td>11:36:31</td>
      <td>11:42:32</td>
    </tr>
    <tr>
      <th>973</th>
      <td>4974</td>
      <td>83</td>
      <td>housemaid</td>
      <td>divorced</td>
      <td>primary</td>
      <td>0</td>
      <td>5944</td>
      <td>0</td>
      <td>0</td>
      <td>30</td>
      <td>jul</td>
      <td>3</td>
      <td>-1</td>
      <td>0</td>
      <td>failure</td>
      <td>12:40:31</td>
      <td>12:43:51</td>
    </tr>
    <tr>
      <th>974</th>
      <td>4975</td>
      <td>33</td>
      <td>student</td>
      <td>single</td>
      <td>tertiary</td>
      <td>0</td>
      <td>882</td>
      <td>1</td>
      <td>0</td>
      <td>6</td>
      <td>feb</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>failure</td>
      <td>11:20:01</td>
      <td>11:25:12</td>
    </tr>
    <tr>
      <th>975</th>
      <td>4976</td>
      <td>50</td>
      <td>entrepreneur</td>
      <td>married</td>
      <td>secondary</td>
      <td>0</td>
      <td>78</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>jun</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>failure</td>
      <td>16:42:22</td>
      <td>16:44:45</td>
    </tr>
    <tr>
      <th>976</th>
      <td>4977</td>
      <td>53</td>
      <td>entrepreneur</td>
      <td>married</td>
      <td>secondary</td>
      <td>0</td>
      <td>230</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
      <td>feb</td>
      <td>2</td>
      <td>250</td>
      <td>1</td>
      <td>other</td>
      <td>09:17:21</td>
      <td>09:18:17</td>
    </tr>
    <tr>
      <th>977</th>
      <td>4978</td>
      <td>28</td>
      <td>management</td>
      <td>single</td>
      <td>tertiary</td>
      <td>0</td>
      <td>3285</td>
      <td>1</td>
      <td>0</td>
      <td>9</td>
      <td>may</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>failure</td>
      <td>09:27:45</td>
      <td>09:29:23</td>
    </tr>
    <tr>
      <th>978</th>
      <td>4979</td>
      <td>30</td>
      <td>admin.</td>
      <td>divorced</td>
      <td>secondary</td>
      <td>0</td>
      <td>377</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
      <td>jun</td>
      <td>15</td>
      <td>-1</td>
      <td>0</td>
      <td>failure</td>
      <td>14:04:32</td>
      <td>14:13:16</td>
    </tr>
    <tr>
      <th>979</th>
      <td>4980</td>
      <td>51</td>
      <td>admin.</td>
      <td>single</td>
      <td>secondary</td>
      <td>0</td>
      <td>726</td>
      <td>1</td>
      <td>0</td>
      <td>27</td>
      <td>apr</td>
      <td>8</td>
      <td>-1</td>
      <td>0</td>
      <td>failure</td>
      <td>12:12:01</td>
      <td>12:13:30</td>
    </tr>
    <tr>
      <th>980</th>
      <td>4981</td>
      <td>55</td>
      <td>technician</td>
      <td>divorced</td>
      <td>secondary</td>
      <td>0</td>
      <td>184</td>
      <td>0</td>
      <td>0</td>
      <td>13</td>
      <td>may</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>failure</td>
      <td>15:04:11</td>
      <td>15:04:29</td>
    </tr>
    <tr>
      <th>981</th>
      <td>4982</td>
      <td>53</td>
      <td>technician</td>
      <td>married</td>
      <td>secondary</td>
      <td>0</td>
      <td>9146</td>
      <td>0</td>
      <td>0</td>
      <td>13</td>
      <td>aug</td>
      <td>4</td>
      <td>-1</td>
      <td>0</td>
      <td>failure</td>
      <td>10:52:10</td>
      <td>10:54:26</td>
    </tr>
    <tr>
      <th>982</th>
      <td>4983</td>
      <td>31</td>
      <td>services</td>
      <td>single</td>
      <td>secondary</td>
      <td>0</td>
      <td>-475</td>
      <td>1</td>
      <td>0</td>
      <td>12</td>
      <td>may</td>
      <td>3</td>
      <td>-1</td>
      <td>0</td>
      <td>failure</td>
      <td>16:36:34</td>
      <td>16:40:26</td>
    </tr>
    <tr>
      <th>983</th>
      <td>4984</td>
      <td>40</td>
      <td>blue-collar</td>
      <td>married</td>
      <td>primary</td>
      <td>0</td>
      <td>254</td>
      <td>1</td>
      <td>0</td>
      <td>7</td>
      <td>jul</td>
      <td>3</td>
      <td>-1</td>
      <td>0</td>
      <td>failure</td>
      <td>09:07:34</td>
      <td>09:09:21</td>
    </tr>
    <tr>
      <th>984</th>
      <td>4985</td>
      <td>35</td>
      <td>blue-collar</td>
      <td>married</td>
      <td>failure</td>
      <td>0</td>
      <td>568</td>
      <td>1</td>
      <td>0</td>
      <td>14</td>
      <td>may</td>
      <td>2</td>
      <td>293</td>
      <td>1</td>
      <td>failure</td>
      <td>17:27:57</td>
      <td>17:29:56</td>
    </tr>
    <tr>
      <th>985</th>
      <td>4986</td>
      <td>25</td>
      <td>student</td>
      <td>single</td>
      <td>secondary</td>
      <td>0</td>
      <td>2975</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>sep</td>
      <td>2</td>
      <td>-1</td>
      <td>0</td>
      <td>failure</td>
      <td>14:49:12</td>
      <td>15:02:31</td>
    </tr>
    <tr>
      <th>986</th>
      <td>4987</td>
      <td>53</td>
      <td>retired</td>
      <td>married</td>
      <td>secondary</td>
      <td>0</td>
      <td>357</td>
      <td>0</td>
      <td>0</td>
      <td>18</td>
      <td>jun</td>
      <td>2</td>
      <td>-1</td>
      <td>0</td>
      <td>failure</td>
      <td>11:41:30</td>
      <td>11:44:04</td>
    </tr>
    <tr>
      <th>987</th>
      <td>4988</td>
      <td>33</td>
      <td>technician</td>
      <td>married</td>
      <td>tertiary</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>30</td>
      <td>apr</td>
      <td>2</td>
      <td>-1</td>
      <td>0</td>
      <td>failure</td>
      <td>10:51:58</td>
      <td>10:55:45</td>
    </tr>
    <tr>
      <th>988</th>
      <td>4989</td>
      <td>41</td>
      <td>blue-collar</td>
      <td>single</td>
      <td>primary</td>
      <td>0</td>
      <td>-206</td>
      <td>0</td>
      <td>0</td>
      <td>20</td>
      <td>jun</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>failure</td>
      <td>16:02:44</td>
      <td>16:08:26</td>
    </tr>
    <tr>
      <th>989</th>
      <td>4990</td>
      <td>37</td>
      <td>services</td>
      <td>married</td>
      <td>secondary</td>
      <td>0</td>
      <td>1543</td>
      <td>1</td>
      <td>0</td>
      <td>22</td>
      <td>may</td>
      <td>1</td>
      <td>10</td>
      <td>6</td>
      <td>failure</td>
      <td>14:41:32</td>
      <td>14:46:30</td>
    </tr>
    <tr>
      <th>990</th>
      <td>4991</td>
      <td>35</td>
      <td>admin.</td>
      <td>married</td>
      <td>secondary</td>
      <td>0</td>
      <td>1254</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>jul</td>
      <td>2</td>
      <td>-1</td>
      <td>0</td>
      <td>failure</td>
      <td>11:36:14</td>
      <td>11:38:26</td>
    </tr>
    <tr>
      <th>991</th>
      <td>4992</td>
      <td>32</td>
      <td>admin.</td>
      <td>married</td>
      <td>secondary</td>
      <td>0</td>
      <td>366</td>
      <td>0</td>
      <td>0</td>
      <td>14</td>
      <td>jan</td>
      <td>2</td>
      <td>-1</td>
      <td>0</td>
      <td>failure</td>
      <td>17:50:14</td>
      <td>17:56:20</td>
    </tr>
    <tr>
      <th>992</th>
      <td>4993</td>
      <td>45</td>
      <td>admin.</td>
      <td>married</td>
      <td>secondary</td>
      <td>1</td>
      <td>-106</td>
      <td>1</td>
      <td>1</td>
      <td>7</td>
      <td>may</td>
      <td>3</td>
      <td>288</td>
      <td>1</td>
      <td>other</td>
      <td>14:35:31</td>
      <td>14:44:44</td>
    </tr>
    <tr>
      <th>993</th>
      <td>4994</td>
      <td>41</td>
      <td>services</td>
      <td>married</td>
      <td>secondary</td>
      <td>0</td>
      <td>138</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>jul</td>
      <td>3</td>
      <td>-1</td>
      <td>0</td>
      <td>failure</td>
      <td>14:33:09</td>
      <td>14:35:25</td>
    </tr>
    <tr>
      <th>994</th>
      <td>4995</td>
      <td>35</td>
      <td>blue-collar</td>
      <td>married</td>
      <td>secondary</td>
      <td>0</td>
      <td>293</td>
      <td>1</td>
      <td>0</td>
      <td>17</td>
      <td>apr</td>
      <td>1</td>
      <td>325</td>
      <td>1</td>
      <td>other</td>
      <td>10:47:43</td>
      <td>11:04:40</td>
    </tr>
    <tr>
      <th>995</th>
      <td>4996</td>
      <td>31</td>
      <td>admin.</td>
      <td>single</td>
      <td>secondary</td>
      <td>0</td>
      <td>131</td>
      <td>1</td>
      <td>0</td>
      <td>15</td>
      <td>jun</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>failure</td>
      <td>13:54:24</td>
      <td>13:56:55</td>
    </tr>
    <tr>
      <th>996</th>
      <td>4997</td>
      <td>52</td>
      <td>management</td>
      <td>married</td>
      <td>tertiary</td>
      <td>0</td>
      <td>2635</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>sep</td>
      <td>1</td>
      <td>108</td>
      <td>8</td>
      <td>success</td>
      <td>13:13:38</td>
      <td>13:18:51</td>
    </tr>
    <tr>
      <th>997</th>
      <td>4998</td>
      <td>46</td>
      <td>technician</td>
      <td>married</td>
      <td>tertiary</td>
      <td>0</td>
      <td>3009</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>aug</td>
      <td>2</td>
      <td>-1</td>
      <td>0</td>
      <td>failure</td>
      <td>10:23:35</td>
      <td>10:24:33</td>
    </tr>
    <tr>
      <th>998</th>
      <td>4999</td>
      <td>60</td>
      <td>retired</td>
      <td>married</td>
      <td>secondary</td>
      <td>0</td>
      <td>7038</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>oct</td>
      <td>4</td>
      <td>-1</td>
      <td>0</td>
      <td>failure</td>
      <td>17:01:49</td>
      <td>17:04:07</td>
    </tr>
    <tr>
      <th>999</th>
      <td>5000</td>
      <td>28</td>
      <td>management</td>
      <td>single</td>
      <td>tertiary</td>
      <td>0</td>
      <td>957</td>
      <td>0</td>
      <td>0</td>
      <td>25</td>
      <td>may</td>
      <td>2</td>
      <td>-1</td>
      <td>0</td>
      <td>failure</td>
      <td>12:15:19</td>
      <td>12:30:34</td>
    </tr>
  </tbody>
</table>
<p>5000 rows × 17 columns</p>
</div>



Let's also pop Id as well. We can convert the CallStart and CallEnd data columns to DateTime format and create a new column called CallLength to better represent our data. Then, we can begin encoding our categorical data to numerical values. 


```python
df.pop("Id")

col_list = df.columns.values.tolist()
for column in col_list:
    df[column] = df[column].fillna(value = 'failure')
    if column == 'CallStart' or column == 'CallEnd':
        df[column] = pd.to_datetime(df[column])
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>Age</th>
      <th>Job</th>
      <th>Marital</th>
      <th>Education</th>
      <th>Default</th>
      <th>Balance</th>
      <th>HHInsurance</th>
      <th>CarLoan</th>
      <th>LastContactDay</th>
      <th>LastContactMonth</th>
      <th>NoOfContacts</th>
      <th>DaysPassed</th>
      <th>PrevAttempts</th>
      <th>Outcome</th>
      <th>CallStart</th>
      <th>CallEnd</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">train</th>
      <th>0</th>
      <td>32</td>
      <td>management</td>
      <td>single</td>
      <td>tertiary</td>
      <td>0</td>
      <td>1218</td>
      <td>1</td>
      <td>0</td>
      <td>28</td>
      <td>jan</td>
      <td>2</td>
      <td>-1</td>
      <td>0</td>
      <td>failure</td>
      <td>2019-07-12 13:45:20</td>
      <td>2019-07-12 13:46:30</td>
    </tr>
    <tr>
      <th>1</th>
      <td>32</td>
      <td>blue-collar</td>
      <td>married</td>
      <td>primary</td>
      <td>0</td>
      <td>1156</td>
      <td>1</td>
      <td>0</td>
      <td>26</td>
      <td>may</td>
      <td>5</td>
      <td>-1</td>
      <td>0</td>
      <td>failure</td>
      <td>2019-07-12 14:49:03</td>
      <td>2019-07-12 14:52:08</td>
    </tr>
    <tr>
      <th>2</th>
      <td>29</td>
      <td>management</td>
      <td>single</td>
      <td>tertiary</td>
      <td>0</td>
      <td>637</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>jun</td>
      <td>1</td>
      <td>119</td>
      <td>1</td>
      <td>failure</td>
      <td>2019-07-12 16:30:24</td>
      <td>2019-07-12 16:36:04</td>
    </tr>
    <tr>
      <th>3</th>
      <td>25</td>
      <td>student</td>
      <td>single</td>
      <td>primary</td>
      <td>0</td>
      <td>373</td>
      <td>1</td>
      <td>0</td>
      <td>11</td>
      <td>may</td>
      <td>2</td>
      <td>-1</td>
      <td>0</td>
      <td>failure</td>
      <td>2019-07-12 12:06:43</td>
      <td>2019-07-12 12:20:22</td>
    </tr>
    <tr>
      <th>4</th>
      <td>30</td>
      <td>management</td>
      <td>married</td>
      <td>tertiary</td>
      <td>0</td>
      <td>2694</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>jun</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>failure</td>
      <td>2019-07-12 14:35:44</td>
      <td>2019-07-12 14:38:56</td>
    </tr>
  </tbody>
</table>
</div>



Now, let's begin encoding our categorical data by using LabelEncoder.


```python
from sklearn.preprocessing import LabelEncoder

col_list.remove('CallStart')
col_list.remove('CallEnd')

for column in col_list:
    encoder = LabelEncoder()
    encoder.fit(df[column])
    df[column] = encoder.transform(df[column])
    
df['CallLength'] = (df['CallEnd'] - df['CallStart'])/np.timedelta64(1, 'm').astype(float)
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>Age</th>
      <th>Job</th>
      <th>Marital</th>
      <th>Education</th>
      <th>Default</th>
      <th>Balance</th>
      <th>HHInsurance</th>
      <th>CarLoan</th>
      <th>LastContactDay</th>
      <th>LastContactMonth</th>
      <th>NoOfContacts</th>
      <th>DaysPassed</th>
      <th>PrevAttempts</th>
      <th>Outcome</th>
      <th>CallStart</th>
      <th>CallEnd</th>
      <th>CallLength</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">train</th>
      <th>0</th>
      <td>14</td>
      <td>5</td>
      <td>2</td>
      <td>3</td>
      <td>0</td>
      <td>1250</td>
      <td>1</td>
      <td>0</td>
      <td>27</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2019-07-12 13:45:20</td>
      <td>2019-07-12 13:46:30</td>
      <td>00:01:10</td>
    </tr>
    <tr>
      <th>1</th>
      <td>14</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1211</td>
      <td>1</td>
      <td>0</td>
      <td>25</td>
      <td>8</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2019-07-12 14:49:03</td>
      <td>2019-07-12 14:52:08</td>
      <td>00:03:05</td>
    </tr>
    <tr>
      <th>2</th>
      <td>11</td>
      <td>5</td>
      <td>2</td>
      <td>3</td>
      <td>0</td>
      <td>846</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>6</td>
      <td>0</td>
      <td>86</td>
      <td>1</td>
      <td>0</td>
      <td>2019-07-12 16:30:24</td>
      <td>2019-07-12 16:36:04</td>
      <td>00:05:40</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7</td>
      <td>9</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>614</td>
      <td>1</td>
      <td>0</td>
      <td>10</td>
      <td>8</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2019-07-12 12:06:43</td>
      <td>2019-07-12 12:20:22</td>
      <td>00:13:39</td>
    </tr>
    <tr>
      <th>4</th>
      <td>12</td>
      <td>5</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>1791</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2019-07-12 14:35:44</td>
      <td>2019-07-12 14:38:56</td>
      <td>00:03:12</td>
    </tr>
  </tbody>
</table>
</div>



We now pop columns that aren't useful to us anymore, and we prepare to scale the data using sklearn's MinMaxScaler(). We scale values to be between 0 and 1 due to our model using numbers to again find the rules for predicting outcomes. Let's say we had one column called Outcome that was either 0 or 1, and another column called Income that went from 20,000 to 500,000.  

If we had to use both the Income and Outcome columns, our model would be training on values between 0 and 1 and another set of values that are significantly larger. This difference in number size causes the importance of the Outcome column to be dramatically reduced, even if we wanted both columns to have the same importance in our model. Always keep in mind that the computer only sees numbers. If we don't tell the computer that all of the columns are equally important by scaling numbers, the model will have skewed importances between the different features. 

To avoid this problem, we scale all numbers to be between 0 and 1, so that all values will have the same importance during the model training phase. 


```python
df.pop('CallStart')
df.pop('CallEnd')
df.pop('LastContactMonth')
df.pop('LastContactDay')
```




    train  0      27
           1      25
           2       2
           3      10
           4       2
           5      21
           6      16
           7      11
           8      17
           9      11
           10      0
           11     28
           12     26
           13     18
           14     28
           15     11
           16      5
           17     10
           18      5
           19      4
           20     17
           21      7
           22     20
           23      0
           24      8
           25     16
           26     19
           27     20
           28     15
           29     19
                  ..
    test   970    20
           971     4
           972    25
           973    29
           974     5
           975     3
           976     3
           977     8
           978     3
           979    26
           980    12
           981    12
           982    11
           983     6
           984    13
           985     1
           986    17
           987    29
           988    19
           989    21
           990     0
           991    13
           992     6
           993     0
           994    16
           995    14
           996     5
           997     3
           998     6
           999    24
    Name: LastContactDay, Length: 5000, dtype: int64




```python
from sklearn.preprocessing import MinMaxScaler

col_list = df.columns.values.tolist()
scaler = MinMaxScaler(feature_range=(0,1))

for col in col_list:
    scaler = MinMaxScaler(feature_range=(0,1))
    df[col] = (scaler.fit_transform(df[col].values.reshape(-1, 1))).astype(float)
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>Age</th>
      <th>Job</th>
      <th>Marital</th>
      <th>Education</th>
      <th>Default</th>
      <th>Balance</th>
      <th>HHInsurance</th>
      <th>CarLoan</th>
      <th>NoOfContacts</th>
      <th>DaysPassed</th>
      <th>PrevAttempts</th>
      <th>Outcome</th>
      <th>CallLength</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">train</th>
      <th>0</th>
      <td>0.202899</td>
      <td>0.454545</td>
      <td>1.0</td>
      <td>1.000000</td>
      <td>0.0</td>
      <td>0.501002</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.029412</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.020012</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.202899</td>
      <td>0.090909</td>
      <td>0.5</td>
      <td>0.333333</td>
      <td>0.0</td>
      <td>0.485371</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.117647</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.055419</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.159420</td>
      <td>0.454545</td>
      <td>1.0</td>
      <td>1.000000</td>
      <td>0.0</td>
      <td>0.339078</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.238227</td>
      <td>0.047619</td>
      <td>0.0</td>
      <td>0.103140</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.101449</td>
      <td>0.818182</td>
      <td>1.0</td>
      <td>0.333333</td>
      <td>0.0</td>
      <td>0.246092</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.029412</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.250616</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.173913</td>
      <td>0.454545</td>
      <td>0.5</td>
      <td>1.000000</td>
      <td>0.0</td>
      <td>0.717836</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.057574</td>
    </tr>
  </tbody>
</table>
</div>




```python
print("Presence of any null values: " + str(df.isnull().values.any()))
```

    Presence of any null values: False
    

Now that we've cleaned out our dataset, we can begin analyzing some components to observe which features we can exploit to suit our model needs. 

# EDA

The purpose of Exploratory Data Analysis is to enhance our understanding of trends in the dataset without involving complicated machine learning models. Oftentimes, we can see obvious traits using graphs and charts just from plotting columns of the dataset against each other.

We've completed the necessary preprocessing steps, so let's create a correlation map to see the relations between different features.  

A correlation map (or correlation matrix) is a visual tool that illustrates the relationship between different columns of the dataset. The matrix will be lighter when the columns represented move in the same direction together, and it will be darker when one column decreases while the other increases. Strong spots of light and dark spots in our correlation matrix tell us about the future reliabilty of the model. 

Let's plot the correlation heatmap of this dataset.


```python
import seaborn as sns

corr = df.corr()
sns.heatmap(corr)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2abe1791c18>



From this map, we can see that there are some features we can use to get better classification results. DaysPassed has positive correlation with cold call success, and PrevAttempts is also positively correlated with successes. However, beyond these few columns there is very little else we can use. Regardless, let's make a bar graph of the amount of failures and successes. 


```python
pd.value_counts(df['Outcome']).plot.bar()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2abe3a07c88>




![png](output_39_1.png)


Like the credit card model, our distributions for the outcome are very skewed toward failure. We will most likely need to trim this dataset to get better prediction results. Let's apply what we learned from the credit card model to see what is going on in this dataset. 


```python
fail_df = df[df['Outcome'] == 0]
success_df = df[df['Outcome'] >= 0.5]

print("Number of failures: " + str(len(fail_df.index)))
print("Number of successes: " + str(len(success_df.index)))
```

    Number of failures: 4347
    Number of successes: 653
    


```python
from sklearn.utils import shuffle 

fail_df = shuffle(df.loc[df['Outcome'] == 0])[:492]
df = pd.concat([fail_df, shuffle(success_df)])

encoder = LabelEncoder()
df['Outcome'] = encoder.fit_transform(df['Outcome'])
pd.value_counts(df['Outcome']).plot.bar()

```




    <matplotlib.axes._subplots.AxesSubplot at 0x2abe3a962e8>




![png](output_42_1.png)


This distribution is much better for our model, at the cost of sacrificing much of our data. Hopefully, this model should be better. If not, we can always try with the original dataset as well. Since we're finished with our analysis, let's begin training the model. 

# Model Training

In this section, we will be creating and training our model for predicting whether a cold call is successful, unsuccessful, or other. Since there are multiple algorithms we can use to build our model, we will compare the accuracy scores after testing and pick the most accurate algorithm.

Begin by creating testing, training, and validation sets. 


```python
from sklearn.model_selection import train_test_split

training,test = train_test_split(df, train_size = 0.7, test_size = 0.3, shuffle=True)
training, valid = train_test_split(training, train_size = 0.7, test_size =0.3, shuffle=True) 

training_label = training.pop('Outcome')
test_label = test.pop('Outcome')
valid_label = valid.pop('Outcome')
```

Now, we instantiate the different algorithms we will use for classification. Then, we train them and check what the accuracy is.


```python
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model imporbt LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

xgb = XGBClassifier()
dtc = DecisionTreeClassifier()
rfc = RandomForestClassifier()
nbc = GaussianNB()
LR = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial')
SVM = SVC(kernel='rbf', C=1,gamma='auto')
knn = KNeighborsClassifier(n_neighbors=3)
```

From this list, we are using XGBoost, DecisionTree, RandomForest, Naive Bayes, LogisticRegressoin, SVC, and KNeighborsClassifier to perform our predictions. We will then see which algorithm produces the highest accuracy and select it as our algorithm of choice for future use. We also want to partition our dataset into training, testing, and validation, so let's add a method for that ability. 

Let's perform the splitting of our data into test, train, validation using train_test_split.  

Our testing will take three phases: testing, training, and validation. Training is first, and it's where our model generates "intuition" about how to approach fraudulent and not fraudulent transactions. It is similar to a student studying and developing knowledge about a topic before an exam.  

The testing phase is where we see how the model performs against data where we know the outcome. This would be the exam if we continue the analogy from before. The algorithms will perform differently, similar to how students will score differently on the exam. From this phase, we generate an accuracy score to compare the different algorithms.  

The validation testing is how we check that the model isn't overfitting to our specific dataset. Overfitting is when the model starts to develop an intuition that is too specific to the training set. Overfitting is a problem because our model is no longer flexible. It may work on the initial set, but subsequent uses will cause our model to fail. Continuing the exam analogy, the validation testing phase is like another version of the exam with different questions. If a student happened to cheat on the first exam by knowing the questions, the second exam will give a better representation of performance.  

Note that verification doesn't completely disprove or prove overfitting, but the testing does give insight about it. 


```python
# train the models
xgb.fit(training, training_label)
dtc.fit(training, training_label)
rfc.fit(training, training_label)
nbc.fit(training, training_label)
LR.fit(training, training_label)
SVM.fit(training, training_label)
knn.fit(training, training_label)
```

    c:\users\tgmat\appdata\local\programs\python\python37\lib\site-packages\sklearn\ensemble\forest.py:245: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
      "10 in version 0.20 to 100 in 0.22.", FutureWarning)
    




    KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                         metric_params=None, n_jobs=None, n_neighbors=3, p=2,
                         weights='uniform')



The accuracies for the different algorithms are shown below, sorted by algorithm name and then its decimal accuracy:


```python
# try and predict an outcome from the test set
xgb_predict = xgb.predict(test)
dtc_predict = dtc.predict(test)
rfc_predict = rfc.predict(test)
nbc_predict = nbc.predict(test)
LR_predict = LR.predict(test)
SVM_predict = SVM.predict(test)
knn_predict = knn.predict(test)

#get accuracy score from models
from sklearn.metrics import accuracy_score

accuracy = dict()
accuracy['XGBoost'] = accuracy_score(test_label, xgb_predict)
accuracy['Naive_bayes'] = accuracy_score(test_label, nbc_predict)
accuracy['DecisionTree'] = accuracy_score(test_label, dtc_predict)
accuracy['RandomForest'] = accuracy_score(test_label,rfc_predict)
accuracy['support_vector_Machines'] = accuracy_score(test_label,SVM_predict)
accuracy['Linear Regression'] = accuracy_score(test_label,LR_predict)
accuracy['KNN'] = accuracy_score(test_label,knn_predict)

accuracy
```




    {'XGBoost': 0.8197674418604651,
     'Naive_bayes': 0.4563953488372093,
     'DecisionTree': 0.7558139534883721,
     'RandomForest': 0.7761627906976745,
     'support_vector_Machines': 0.6947674418604651,
     'Linear Regression': 0.7267441860465116,
     'KNN': 0.6656976744186046}



These are okay prediction values, perhaps we can do better with more tuning. Let's try the validation testing to see what our predictions look like. Keep in mind that the Outcome variable we are trying to predict was composed of mostly null values, meaning that we would need more relevant data in order to create a better model. 

# Testing the Model


```python
# perform validation testing for dataset
xgb_predict = xgb.predict(valid)
dtc_predict = dtc.predict(valid)
rfc_predict = rfc.predict(valid)
nbc_predict = nbc.predict(valid)
LR_predict = LR.predict(valid)
SVM_predict = SVM.predict(valid)
knn_predict = knn.predict(valid)

accuracy['XGBoost'] = accuracy_score(valid_label, xgb_predict) 
accuracy['Naive_bayes'] = accuracy_score(valid_label, nbc_predict)
accuracy['DecisionTree'] = accuracy_score(valid_label, dtc_predict)
accuracy['RandomForest'] = accuracy_score(valid_label,rfc_predict)
accuracy['support_vector_Machines'] = accuracy_score(valid_label,SVM_predict)
accuracy['Linear Regression'] = accuracy_score(valid_label,LR_predict)
accuracy['KNN'] = accuracy_score(valid_label,knn_predict)

accuracy
```




    {'XGBoost': 0.7717842323651453,
     'Naive_bayes': 0.42738589211618255,
     'DecisionTree': 0.7302904564315352,
     'RandomForest': 0.7261410788381742,
     'support_vector_Machines': 0.6473029045643154,
     'Linear Regression': 0.7012448132780082,
     'KNN': 0.6182572614107884}



For the validation set, it seems XGBoost is performing the best, with an 77% accuracy score. Let's select it as the best performing model for now, and we can tune our model better to see if we can improve the accuracy score.  


```python
max_accuracy = max(accuracy,key=accuracy.get)
max_accuracy
```




    'XGBoost'



# Conclusion

During this notebook, we built a model that could accurately predict whether or not a cold call was successful, given data about the callee such as age, marital status, and education.  

Even though the accuracy of our model wasn't superb, keep in mind that we were given a very empty dataset. While there were many data points, the tiny amount of relevant information in the Outcome column meant that our model had very little actual data to train with. Even with this poor quality data, our model still achieved 77% accuacy with the XGBoost algorithm.  

For real-world use cases, a company could save a large amount of money avoiding calls that have a low chance of success simply by inputting some public information about the callee into our model. The consumer, on the other hand, will also deal with fewer irrelavant calls and will receive calls more related to what they actually need. In both cases, using machine learning to predict cold call success leads to beneficial outcomes for everyone involved.  

If you want to make the most of your company's data and time, Cocolevio's own 5411 product can help you build models that can make your business process as efficient as possible. 

# Credits

Thanks to the Kaggle community for helping me understand many of the concepts that I used during this analysis. 
