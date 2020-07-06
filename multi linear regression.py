#!/usr/bin/env python
# coding: utf-8

# # Multi linear regression 

# In[1]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt


# In[53]:


#--- importing data 
df = pd.read_excel('D:\\1\\co2.xlsx')
df


# In[55]:


mdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION','CO2EMISSIONS']]
mdf


# In[56]:


print(mdf.describe()) 
print('====================================================================')
mdf.max()


# In[57]:


mdf.hist()

plt.show()


# In[58]:


plt.figure(figsize=(6,6),dpi=500)
mdf.boxplot()
plt.xticks(rotation='vertical')
plt.show()


# In[59]:



Rmdf = mdf[['CYLINDERS','FUELCONSUMPTION','CO2EMISSIONS']]

fig = plt.figure(figsize=(8,10))
ax = fig.add_subplot(111, projection='3d')

x_axis =  mdf['CYLINDERS']
y_axis = mdf['FUELCONSUMPTION']
z_axis = mdf['CO2EMISSIONS']
ax.scatter(x_axis, y_axis, z_axis, s=50, alpha=0.5, edgecolors='#c443b1')

ax.set_xlabel('CYLINDERS')
ax.set_ylabel('FUELCONSUMPTION')
ax.set_zlabel('CO2EMISSIONS')
plt.savefig('D:\\1\\3Dplot.png', dpi=500)

plt.show()

spliting out data 
# In[65]:


msk = np.random.rand(len(df)) < 0.8
train = mdf[msk]
test = mdf[~msk]


# In[66]:


from sklearn import linear_model
M_regression = linear_model.LinearRegression()
x = np.asanyarray(train[['CYLINDERS','FUELCONSUMPTION']])
y = np.asanyarray(train[['CO2EMISSIONS']])
M_regression.fit(x,y)
print ('Coefficients: ', M_regression.coef_)


# In[67]:


## another way without sklearn
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
model = smf.ols(formula='CO2EMISSIONS ~ CYLINDERS + FUELCONSUMPTION', data=df2)
results_formula = model.fit()
results_formula.params


# In[75]:


x_surf, y_surf = np.meshgrid(np.linspace(mdf.CYLINDERS.min(), mdf.CYLINDERS.max(), 100),np.linspace(mdf.FUELCONSUMPTION.min(), mdf.FUELCONSUMPTION.max(), 100))
onlyX = pd.DataFrame({'CYLINDERS': x_surf.ravel(), 'FUELCONSUMPTION': y_surf.ravel()})
fittedY=results_formula.predict(exog=onlyX)

fittedY=np.array(fittedY)


fig = plt.figure(figsize=(8,10))

ax = fig.add_subplot(111, projection='3d')
ax.scatter(mdf['CYLINDERS'],mdf['FUELCONSUMPTION'],mdf['CO2EMISSIONS'],c='#3fcc54', marker='.', alpha=0.5)
ax.plot_surface(x_surf,y_surf,fittedY.reshape(x_surf.shape), color='#d97ce6', alpha=0.3)
ax.set_xlabel('CYLINDERS')
ax.set_ylabel('FUELCONSUMPTION')
ax.set_zlabel('CO2EMISSIONS')
plt.savefig('D:\\1\\3Dplot2.png', dpi=500)

plt.show()

