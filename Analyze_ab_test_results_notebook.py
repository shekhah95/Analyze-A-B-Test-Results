#!/usr/bin/env python
# coding: utf-8

# ## Analyze A/B Test Results
# 
# You may either submit your notebook through the workspace here, or you may work from your local machine and submit through the next page.  Either way assure that your code passes the project [RUBRIC](https://review.udacity.com/#!/projects/37e27304-ad47-4eb0-a1ab-8c12f60e43d0/rubric).  **Please save regularly.**
# 
# This project will assure you have mastered the subjects covered in the statistics lessons.  The hope is to have this project be as comprehensive of these topics as possible.  Good luck!
# 
# ## Table of Contents
# - [Introduction](#intro)
# - [Part I - Probability](#probability)
# - [Part II - A/B Test](#ab_test)
# - [Part III - Regression](#regression)
# 
# 
# <a id='intro'></a>
# ### Introduction
# 
# A/B tests are very commonly performed by data analysts and data scientists.  It is important that you get some practice working with the difficulties of these 
# 
# For this project, you will be working to understand the results of an A/B test run by an e-commerce website.  Your goal is to work through this notebook to help the company understand if they should implement the new page, keep the old page, or perhaps run the experiment longer to make their decision.
# 
# **As you work through this notebook, follow along in the classroom and answer the corresponding quiz questions associated with each question.** The labels for each classroom concept are provided for each question.  This will assure you are on the right track as you work through the project, and you can feel more confident in your final submission meeting the criteria.  As a final check, assure you meet all the criteria on the [RUBRIC](https://review.udacity.com/#!/projects/37e27304-ad47-4eb0-a1ab-8c12f60e43d0/rubric).
# 
# <a id='probability'></a>
# #### Part I - Probability
# 
# To get started, let's import our libraries.

# In[1]:


import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
#We are setting the seed to assure you get the same answers on quizzes as we set up
random.seed(42)


# `1.` Now, read in the `ab_data.csv` data. Store it in `df`.  **Use your dataframe to answer the questions in Quiz 1 of the classroom.**
# 
# a. Read in the dataset and take a look at the top few rows here:

# In[2]:


df = pd.read_csv('ab_data.csv')
df.head()


# b. Use the cell below to find the number of rows in the dataset.

# In[3]:


df.shape


# c. The number of unique users in the dataset.

# In[4]:


df.user_id.nunique() 


# d. The proportion of users converted.

# In[5]:


df.converted.mean() 


# e. The number of times the `new_page` and `treatment` don't match.

# In[6]:


the_oldtreat = df.query("group == 'treatment' and landing_page == 'old_page'").shape[0]
the_newcontrol = df.query("group == 'control' and landing_page == 'new_page'").shape[0]


# In[7]:


the_oldtreat,the_newcontrol


# Treatment, old = 1965
# Control, new = 1928
# Number of times new_page and treatment don't line up: 1965+1928=3893

# In[8]:


the_oldtreat + the_newcontrol 


# f. Do any of the rows have missing values?

# In[9]:


df.isnull() 


# In[10]:


df.info() 


# `2.` For the rows where **treatment** does not match with **new_page** or **control** does not match with **old_page**, we cannot be sure if this row truly received the new or old page.  Use **Quiz 2** in the classroom to figure out how we should handle these rows.  
# 
# a. Now use the answer to the quiz to create a new dataset that meets the specifications from the quiz.  Store your new dataframe in **df2**.

# In[11]:


df2 = df.query("group == 'control' and landing_page == 'old_page'")
df2 = df2.append(df.query("group == 'treatment' and landing_page == 'new_page'"))


# In[12]:


#Check all of that rows removed 
df2[((df2['group'] == 'treatment') == (df2['landing_page'] == 'new_page')) == False].shape[0]
0


# `3.` Use **df2** and the cells below to answer questions for **Quiz3** in the classroom.

# a. How many unique **user_id**s are in **df2**?

# In[ ]:


# the unique user_ids in df2
df2.user_id.nunique() 


# b. There is one **user_id** repeated in **df2**.  What is it?

# In[208]:


df2['is_duplicated'] = df2.duplicated('user_id')
df2['is_duplicated'].sum()


# In[209]:


df2[df2.duplicated(['user_id'], keep=False)]


# c. What is the row information for the repeat **user_id**? 

# In[210]:


#the row information for the repeatd user_id
df2[df2['user_id'].duplicated()]


# d. Remove **one** of the rows with a duplicate **user_id**, but keep your dataframe as **df2**.

# In[211]:


df2.drop_duplicates() 


# In[212]:



df2.shape[0]


# In[213]:


df2.shape


# `4.` Use **df2** in the cells below to answer the quiz questions related to **Quiz 4** in the classroom.
# 
# a. What is the probability of an individual converting regardless of the page they receive?

# In[214]:


df['converted'].mean() 


# b. Given that an individual was in the `control` group, what is the probability they converted?

# In[215]:


converted_probability = df2.query('group =="control"').converted.mean()


# In[216]:


converted_probability 


# c. Given that an individual was in the `treatment` group, what is the probability they converted?

# In[217]:


treatment_probability  = df2.query('group == "treatment"')['converted'].mean()


# In[218]:


treatment_probability


# d. What is the probability that an individual received the new page?

# In[219]:


df2.query("landing_page == 'new_page'").count()[0]/df2.shape[0]


# e. Consider your results from parts (a) through (d) above, and explain below whether you think there is sufficient evidence to conclude that the new treatment page leads to more conversions.

# # the answer 

# ##### 
# No, it does not seem as though one page leads to more conversions. 
# from the above values, the difference is small the treatment group has a smaller conversion rate.

# <a id='ab_test'></a>
# ### Part II - A/B Test
# 
# Notice that because of the time stamp associated with each event, you could technically run a hypothesis test continuously as each observation was observed.  
# 
# However, then the hard question is do you stop as soon as one page is considered significantly better than another or does it need to happen consistently for a certain amount of time?  How long do you run to render a decision that neither page is better than another?  
# 
# These questions are the difficult parts associated with A/B tests in general.  
# 
# 
# `1.` For now, consider you need to make the decision just based on all the data provided.  If you want to assume that the old page is better unless the new page proves to be definitely better at a Type I error rate of 5%, what should your null and alternative hypotheses be?  You can state your hypothesis in terms of words or in terms of **$p_{old}$** and **$p_{new}$**, which are the converted rates for the old and new pages.

# ###### H0 :Pold = Pnew 
# 
# H1:Pnew > Pold 
# 
# |
# 
# H0: Pold − Pnew = 0
# 
# H1: Pnew − Pold > 0

# **Put your answer here.**

# `2.` Assume under the null hypothesis, $p_{new}$ and $p_{old}$ both have "true" success rates equal to the **converted** success rate regardless of page - that is $p_{new}$ and $p_{old}$ are equal. Furthermore, assume they are equal to the **converted** rate in **ab_data.csv** regardless of the page. <br><br>
# 
# Use a sample size for each page equal to the ones in **ab_data.csv**.  <br><br>
# 
# Perform the sampling distribution for the difference in **converted** between the two pages over 10,000 iterations of calculating an estimate from the null.  <br><br>
# 
# Use the cells below to provide the necessary parts of this simulation.  If this doesn't make complete sense right now, don't worry - you are going to work through the problems below to complete this problem.  You can use **Quiz 5** in the classroom to make sure you are on the right track.<br><br>

# a. What is the **conversion rate** for $p_{new}$ under the null? 

# In[27]:


p_undernull = df2.converted.mean()
p_undernull


# b. What is the **conversion rate** for $p_{old}$ under the null? <br><br>

# In[28]:


old_undernull = df2.converted.mean()
old_undernull


# c. What is $n_{new}$, the number of individuals in the treatment group?

# In[29]:


new_treatment_group_df = df2.query('landing_page == "new_page"')
new_treatment = new_treatment_group_df.shape[0]


# In[30]:


new_treatment 


# d. What is $n_{old}$, the number of individuals in the control group?

# In[31]:


old_control_group_df = df2.query('landing_page == "new_page"')
old_control = old_control_group_df.shape[0]


# In[32]:


old_control


# e. Simulate $n_{new}$ transactions with a conversion rate of $p_{new}$ under the null.  Store these $n_{new}$ 1's and 0's in **new_page_converted**.

# In[33]:


new_page_converted = np.random.binomial(1 ,p_undernull,new_treatment) 
print(new_page_converted) 


# f. Simulate $n_{old}$ transactions with a conversion rate of $p_{old}$ under the null.  Store these $n_{old}$ 1's and 0's in **old_page_converted**.

# In[34]:


old_page_converted = np.random.binomial(1 ,old_undernull ,old_control) 
print(old_page_converted) 


# g. Find $p_{new}$ - $p_{old}$ for your simulated values from part (e) and (f).

# In[35]:


new_page_converted.mean() - old_page_converted.mean()


# h. Create 10,000 $p_{new}$ - $p_{old}$ values using the same simulation process you used in parts (a) through (g) above. Store all 10,000 values in a NumPy array called **p_diffs**.

# In[36]:


p_diffs = []
for _ in range(10000):
    simulation_process_new = np.random.binomial(new_treatment ,p_undernull)/new_treatment 
    simulation_process_old = np.random.binomial(old_control  ,old_undernull,)/old_control
    diff = simulation_process_new - simulation_process_old 
    
    p_diffs.append(diff)


# i. Plot a histogram of the **p_diffs**.  Does this plot look like what you expected?  Use the matching problem in the classroom to assure you fully understand what was computed here.

# In[37]:


plt.hist(p_diffs); 


# j. What proportion of the **p_diffs** are greater than the actual difference observed in **ab_data.csv**?

# In[38]:


treatment_proportion =df2[df2['group']=='treatment']['converted'].mean()
treatment_proportion


# In[39]:


control_proportion =df2[df2['group']=='control']['converted'].mean()
control_proportion


# In[40]:


#calculte Actual diffrence 

diff_observed =  treatment_proportion - control_proportion
p_diffs=np.array(p_diffs)
p_vaules = (p_diffs>diff_observed).mean()
p_vaules


# In[41]:


plt.hist(p_diffs);
plt.axvline(df_diff,c='y',linewidth = 3);


# k. Please explain using the vocabulary you've learned in this course what you just computed in part **j.**  What is this value called in scientific studies?  What does this value mean in terms of whether or not there is a difference between the new and old pages?

# **Put your answer here.**

# 
# The probability of observing with statistic or one more extreme in favor of the alternative , assuming that the null hypothesis is correct. 
# The p-value is a proportion:
# if your p-value is 0.05, that means that 5% of the time you would see a test statistic at least as extreme as the one you found if the null hypothesis was true, smaller p-value means that there is stronger evidence in favor of the alternative hypothesis. 
# Since the p-value is large enough, so will fail to reject the Null hypothesis and hold the old page.

# l. We could also use a built-in to achieve similar results.  Though using the built-in might be easier to code, the above portions are a walkthrough of the ideas that are critical to correctly thinking about statistical significance. Fill in the below to calculate the number of conversions for each page, as well as the number of individuals who received each page. Let `n_old` and `n_new` refer the the number of rows associated with the old page and new pages, respectively.

# In[42]:


import statsmodels.api as sm


# In[43]:


convert_old = len(df2.query('converted==1 and landing_page=="old_page"')) 


# In[44]:


convert_new = len(df2.query('converted==1 and landing_page=="new_page"')) 


# In[45]:


n_old = len(df2.query('landing_page=="old_page"')) 
n_new = len(df2.query('landing_page=="new_page"'))


# m. Now use `stats.proportions_ztest` to compute your test statistic and p-value.  [Here](https://docs.w3cub.com/statsmodels/generated/statsmodels.stats.proportion.proportions_ztest/) is a helpful link on using the built in.

# In[46]:


z_score, p_value = sm.stats.proportions_ztest([convert_old, convert_new], [n_old, n_new], alternative='larger')


# In[47]:


# display the z_score
print(z_score) 


# In[48]:


p_value


# n. What do the z-score and p-value you computed in the previous question mean for the conversion rates of the old and new pages?  Do they agree with the findings in parts **j.** and **k.**?

# **Put your answer here.**

# Z-score is a numerical measurement that describes a value's relationship to the mean of a group of values. Z-score is measured in terms of standard deviations from the mean. If a Z-score is 0, it indicates that the data point's score is identical to the mean score.
# 
# The probability of randomly selecting a score between -1.96 and +1.96 standard deviations from the mean is 95%.
# If there is less than a 5% chance of a raw score being selected randomly, then this is a statistically significant result.
# 
# 
# 
# The null hypothesis is (1.31) standard deviations above the mean.
# This is less than the critical 1.96 we would need to reject the null hypothesis. So the z-test seems to support our result.

# <a id='regression'></a>
# ### Part III - A regression approach
# 
# `1.` In this final part, you will see that the result you achieved in the A/B test in Part II above can also be achieved by performing regression.<br><br> 
# 
# a. Since each row is either a conversion or no conversion, what type of regression should you be performing in this case?

# # the  answer here.**

# ### Since we want to see if there is a significant difference in conversion, the appropriate approach is Logistic Regression.
# 

# b. The goal is to use **statsmodels** to fit the regression model you specified in part **a.** to see if there is a significant difference in conversion based on which page a customer receives. However, you first need to create in df2 a column for the intercept, and create a dummy variable column for which page each user received.  Add an **intercept** column, as well as an **ab_page** column, which is 1 when an individual receives the **treatment** and 0 if **control**.

# In[71]:


import statsmodels.api as sm


# In[72]:


## Adding an intercept column
df2['intercept'] = 1


# In[73]:


#create a dummy variables
df2[['control','treatment']]= pd.get_dummies(df2['group']) 
df2.rename(columns={'treatment':'ab_page'},inplace=True) 


# In[74]:


df2.head()


# c. Use **statsmodels** to instantiate your regression model on the two columns you created in part b., then fit the model using the two columns you created in part **b.** to predict whether or not an individual converts. 

# In[75]:


log_mod = sm.Logit(df2['converted'], df2[['intercept', 'ab_page']])
results = log_mod.fit()


# d. Provide the summary of your model below, and use it as necessary to answer the following questions.

# In[70]:


results.summary2() 


# e. What is the p-value associated with **ab_page**? Why does it differ from the value you found in **Part II**?<br><br>  **Hint**: What are the null and alternative hypotheses associated with your regression model, and how do they compare to the null and alternative hypotheses in **Part II**?

# ## Logistic Regression.
# 
# H0:  pnew − pold = 0
# 
# H1 : pnew − pold!= 0

# ## part2 
# 
# H0: pnew − pold <= 0
# 
# H1:pnew − pold >0

# # the answer here.**

# The p-value associated with ab_page is 0.190 here.
# the model is attempting to predict whether a user will convert depending on their page.
# 
# It's different because of Part II because in the A/B test our null hypothesis states that the old page is better than, or equal to, the new and Regression it is two sided test.

# f. Now, you are considering other things that might influence whether or not an individual converts.  Discuss why it is a good idea to consider other factors to add into your regression model.  Are there any disadvantages to adding additional terms into your regression model?

# # the answer here.**

# ### Considering the outcume that we have, The pages do not appear to have A big effect on converting users. Therefore, it is an excellent idea to consider other factors might predict conversion. and It is important to be when selecting factors, cheking that the factors are not in and of themselves colinear, this is the disadvantege.
# Other considerations:
# Change aversion: gives an unfair advantage to control group/ old page; users might be unhappy with
# change.
# Novelty effect: gives an unfair advantage to treatment group/ new page; users might be drawn to change

# g. Now along with testing if the conversion rate changes for different pages, also add an effect based on which country a user lives in. You will need to read in the **countries.csv** dataset and merge together your datasets on the appropriate rows.  [Here](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.join.html) are the docs for joining tables. 
# 
# Does it appear that country had an impact on conversion?  Don't forget to create dummy variables for these country columns - **Hint: You will need two columns for the three dummy variables.** Provide the statistical output as well as a written response to answer this question.

# In[64]:


countries_df = pd.read_csv('./countries.csv')
countries_df.head(20) 


# In[65]:


df_new = countries_df.set_index('user_id').join(df2.set_index('user_id'), how='inner') 
df_new.head()


# In[66]:


# review country column data, how many unique entries are there?
df_new['country'].unique()


# In[67]:


# Create the necessary dummy variables
df_new[['CA', 'UK', 'US']] = pd.get_dummies(df_new['country'])
df_new.head()


# In[78]:


df_new ['UK_ab_page'] = df_new['UK']* df_new['ab_page']
df_new.head()


# In[79]:


df_new ['CA_ab_page'] = df_new['CA']* df_new['ab_page']
df_new.head() 


# h. Though you have now looked at the individual factors of country and page on conversion, we would now like to look at an interaction between page and country to see if there significant effects on conversion.  Create the necessary additional columns, and fit the new model.  
# 
# Provide the summary results, and your conclusions based on the results.

# In[80]:


logit_mod = sm.Logit(df_new['converted'], df_new[['intercept', 'CA', 'UK']]) 
results = logit_mod.fit()
results.summary2()


# # Based on the results, it also does not appear that the country has a notable influence on conversion.

# # conclusions 

# ##  Considering the available information, we will fail to reject the null hypothesis and, there is no sufficient evidence to suggest that we switch to the new page.
# As a professional data analyst, I recommend there is no need to change the old page, While it is showing fine good results. 

# <a id='conclusions'></a>
# ## Finishing Up
# 
# > Congratulations!  You have reached the end of the A/B Test Results project!  You should be very proud of all you have accomplished!
# 
# > **Tip**: Once you are satisfied with your work here, check over your report to make sure that it is satisfies all the areas of the rubric (found on the project submission page at the end of the lesson). You should also probably remove all of the "Tips" like this one so that the presentation is as polished as possible.
# 
# 
# ## Directions to Submit
# 
# > Before you submit your project, you need to create a .html or .pdf version of this notebook in the workspace here. To do that, run the code cell below. If it worked correctly, you should get a return code of 0, and you should see the generated .html file in the workspace directory (click on the orange Jupyter icon in the upper left).
# 
# > Alternatively, you can download this report as .html via the **File** > **Download as** submenu, and then manually upload it into the workspace directory by clicking on the orange Jupyter icon in the upper left, then using the Upload button.
# 
# > Once you've done this, you can submit your project by clicking on the "Submit Project" button in the lower right here. This will create and submit a zip file with this .ipynb doc and the .html or .pdf version you created. Congratulations!

# In[ ]:


from subprocess import call
call(['python', '-m', 'nbconvert', 'Analyze_ab_test_results_notebook.ipynb'])


# In[ ]:




