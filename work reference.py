#!/usr/bin/env python
# coding: utf-8

# In[2]:



import pandas as pd
import numpy as np
sravan = pd.read_csv('/Users/sravankumarreddypeddamavireddy/Downloads/niftydata.csv')
sravan


# In[3]:


sravan = pd.read_csv('/Users/sravankumarreddypeddamavireddy/Downloads/niftydata.csv', nrows = 20)
sravan


# In[4]:


sravan = pd.read_csv('/Users/sravankumarreddypeddamavireddy/Downloads/niftydata.csv', names = ['trading date','closing price'],nrows = 20)
sravan


# In[ ]:





# In[5]:


import pandas as pd
sravan = pd.read_excel('/Users/sravankumarreddypeddamavireddy/Downloads/Sample - Superstore Sales.xls')
sravan.head()


# In[6]:


sravan.describe()
sravan.describe(include=['object','float'])
sravan.head()


# In[7]:


sravan[sravan['Order Priority'] == 'High']


# In[8]:


sravan[sravan['Order Priority'] == 'High'].describe()


# In[9]:


sravan[(sravan['Order Priority'] == 'High') & (sravan.Profit > 2000)].describe()


# In[10]:


sravan[(sravan['Order Priority'] == 'High') & (sravan.Profit > 2000)].describe().sort_values(['Profit'],ascending=False)


# In[11]:


sravan


# In[12]:


sravan.drop('Row ID',axis=1,inplace=True)


# In[13]:


sravan


# In[14]:


sravan[(sravan['Order Priority'] == 'High') & (sravan.Profit > 2000)].describe().sort_values(['Profit'],ascending=False)


# In[15]:


sravan


# In[ ]:





# In[16]:


sravan.head(20)


# In[17]:


sravan[(sravan['Order Priority'] == 'High') & (sravan.Profit > 2000)].describe().sort_values(['Profit'],ascending=False)


# In[18]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np


# In[19]:


sravan.describe()


# In[20]:


x = sravan['Order Quantity']
y = sravan['Profit']


# In[21]:


plt.scatter(x,y,c='g')
plt.xlabel('no.of orders')
plt.ylabel('profit')
plt.title('orders vs profit')
plt.savefig('img.png')


# In[22]:


sravan.isnull()


# In[23]:


import seaborn as sns
sns.heatmap(sravan.isnull(),yticklabels=False,cbar=False,cmap="viridis")


# In[24]:


sns.countplot(x='Product Container',data=sravan)


# In[25]:


sns.boxplot(x='Product Container',y='Product Base Margin',data=sravan)


# In[26]:


sravan['Product Base Margin'].fillna(sravan.groupby('Product Container')['Product Base Margin'].transform("mean"),inplace=True)


# In[27]:


sns.heatmap(sravan.isnull(),yticklabels=False,cbar=False,cmap="viridis")


# In[28]:


sravan.head(3)


# In[29]:


pd.get_dummies(sravan['Order Priority'])



# In[30]:


pd.get_dummies(sravan['Ship Mode'])


# In[31]:


pd.get_dummies(sravan['Customer Segment'])


# In[32]:


pd.get_dummies(sravan['Product Category'])


# In[33]:


priority=pd.get_dummies(sravan['Order Priority'],drop_first=True)
mode=pd.get_dummies(sravan['Ship Mode'],drop_first=True)
segment=pd.get_dummies(sravan['Customer Segment'],drop_first=True)
category=pd.get_dummies(sravan['Product Category'],drop_first=True)



# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[34]:


sravan.head(3)


# In[35]:


sravan.drop(['Region','Province','Customer Name','Product Sub-Category','Product Name','Product Container','Order ID','Order Date','Ship Date'],axis=1,inplace=True)


# In[36]:


sravan


# In[37]:


priority=pd.get_dummies(sravan['Order Priority'],drop_first=True)
mode=pd.get_dummies(sravan['Ship Mode'],drop_first=True)
segment=pd.get_dummies(sravan['Customer Segment'],drop_first=True)
category=pd.get_dummies(sravan['Product Category'],drop_first=True)



# In[38]:


sravan=pd.concat([priority,mode,segment,category,sravan],axis=1)


# In[39]:


sravan.head(3)


# In[40]:


sravan.drop(['Ship Mode','Customer Segment','Product Category','Order Priority'],axis=1,inplace=True)


# In[41]:


sravan


# In[42]:


titles = list(sravan.columns)
titles


# In[43]:


titles[14],titles[17] = titles[17],titles[14]
titles


# In[44]:


sravan = sravan[titles]
sravan


# In[95]:


x=sravan.iloc[:,:-1]
y=sravan.iloc[:,[17]]


# In[88]:


y


# In[96]:


sravan.head()


# In[97]:


from sklearn.model_selection import train_test_split


# In[98]:



from sklearn.linear_model import LogisticRegression


# In[ ]:





# In[99]:


x_sravan, x_test, y_sravan, y_test = train_test_split(x,y,test_size = 0.30, random_state=1)


# In[ ]:





# In[82]:


model = LogisticRegression()
model.fit(x_sravan, y_sravan)


# In[64]:


predictions = model.predict(x_test)


# In[65]:


from sklearn.metrics import confusion_matrix



# In[ ]:





# In[66]:


accuracy=confusion_matrix(y_test,predictions)


# In[157]:


accuracy


# In[427]:


from sklearn.metrices import accuracy_score


# In[428]:


accuracy=accuracy_score(y_test,predictions)


# In[ ]:




