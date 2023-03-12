import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
data_set = pd.read_csv('insurance_c.csv')
data_set.head()
data_set.head()
data_set.describe()


# In[7]:


data_set.corr()


# In[8]:


data_set.isnull().sum()


# In[9]:


data_set.columns


# In[13]:


plt.figure(figsize = (14, 10))
sns.barplot(x = 'sex', y = 'charges', data = data_set)

plt.title("AGE__vs__CHARGES")


# In[15]:


plt.figure(figsize = (16, 12))
sns.barplot(x = 'age', y = 'charges', data = data_set)

plt.title("AGE__vs__CHARGES")


# In[17]:


plt.figure(figsize = (10, 6))
sns.barplot(x = 'bmi', y = 'charges', data = data_set)

plt.title("AGE__vs__CHARGES")


# In[18]:


plt.figure(figsize = (13, 7))
sns.barplot(x = 'children', y = 'charges', data = data_set)

plt.title("AGE__vs__CHARGES")


# In[19]:


plt.figure(figsize = (12, 8))
sns.barplot(x = 'smoker', y = 'charges', data = data_set)

plt.title("AGE__vs__CHARGES")


# In[21]:


plt.figure(figsize = (16, 13))
sns.barplot(x = 'region', y = 'charges', data = data_set)

plt.title("AGE__vs__CHARGES")


# In[22]:


data_set = data_set.drop('region', axis = 1)


# In[23]:


data_set.head()


# In[29]:


from sklearn.preprocessing import LabelEncoder

leab = LabelEncoder()
data_set['sex'] = leab.fit_transform(data_set['sex'])
data_set['smoker'] = leab.fit_transform(data_set['smoker'])


# In[30]:


data_set.head()


# In[33]:


x = data_set.iloc[:,:5]
Y = data_set.iloc[:,5]

print(x.shape)
print(Y.shape)


# In[35]:


from sklearn.model_selection import train_test_split

x_train, x_test, Y_train, Y_test = train_test_split(x, Y, test_size = 0.4, random_state = 32)

print(x_train.shape)
print(x_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# In[36]:


from sklearn.preprocessing import StandardScaler
stdsca = StandardScaler()
x_train = stdsca.fit_transform(x_train)
x_test = stdsca.fit_transform(x_test)


# In[37]:


from sklearn.linear_model import LinearRegression
linereg = LinearRegression()
linereg.fit(x_train, Y_train)
Y_pred_linereg = linereg.predict(x_test)

linereg.score(x_train,Y_train)


# In[38]:


from sklearn.svm import SVR
s_v_r = SVR()
s_v_r.fit(x_train, Y_train)
Y_pred_s_v_r = s_v_r.predict(x_test)
s_v_r.score(x_train,Y_train)


# In[41]:


from sklearn.ensemble import RandomForestRegressor
reg_ressor = RandomForestRegressor(n_estimators = 12, random_state = 0)
reg_ressor.fit(x_train, Y_train)
Y_pred = reg_ressor.predict(x_test)
reg_ressor.score(x_train,Y_train)


# In[ ]:




