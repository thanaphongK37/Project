#!/usr/bin/env python
# coding: utf-8

# # RFM Analysis  
# - reference : https://www.blastanalytics.com/blog/rfm-analysis-boosts-sales
# - Analysis and Developed K-Means Clustering  for Segmentation Membership

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import iqr
import seaborn as sns
import os
from pandas_profiling import ProfileReport


# In[2]:


df = pd.read_pickle('raw_data_rfm.data')


# ## STEP 1 : Data Cleansing And EDA

# In[3]:


df.info()


# ## Use Pandas Profileling For EDA data

# In[4]:


profile = ProfileReport(df, title='Pandas Profiling Report')


# In[5]:


profile


# In[6]:


df


# In[7]:


df.shape


# In[8]:


len(df['CUSTOMER_ID'].drop_duplicates())


# In[9]:


currentDate = pd.to_datetime(df['DATE']).max() # Defind Current Date From Data


# In[15]:


def recency(x):
        return (currentDate - pd.to_datetime(x).max()).days


# ## STEP 2 : DATA Preparation

# ### Aggregrate RFM  Values

# In[17]:


df_RFM = df.groupby('CUSTOMER_ID').agg({'DATE':[recency, # Recency
                                                                'count'], # Frequency
                                                                'SALES':'sum'}) #Monetory


# In[19]:


## rename columns
df_RFM.columns = ['Recency','Frequency','Monetary']
df_RFM.head()


# In[20]:


df_RFM2 = df_RFM.copy()


# In[21]:


df_RFM.describe()


# ### Fill Missing Values and Solve Problem Zero Values

# In[22]:


negative_zero = df_RFM2['Monetary'].astype('int') > 0
df_RFM2 = df_RFM2[negative_zero]


# ### *Take Log Scale Because Monetary Values is Positive Skewed mean Values 0 - 20 Milion 
# - Reference : http://bigdataexperience.org/feature-engineer-with-log-transformation/

# In[23]:


np.log10(df_RFM2[df_RFM2['Monetary'] >=0]['Monetary']+1).hist(bins=100)
plt.xticks(np.arange(9),['1','10','100','1k','10k','100k','1M','10M','50M'])
plt.xlabel('Number_of_Sales')
plt.ylabel('Number of Member')
plt.show()


# ## Handle Outlier
# - Use IQR(Interquartile Range) for Cut Outlier  == > Reference :  https://datarockie.com/2019/10/31/excel-detect-outliers/

# In[24]:


np.round(df_RFM.quantile([0.25,0.5,0.75,0.8,0.9,0.95,0.98,0.99,0.995,0.996,0.997,0.998,1]))


# In[25]:


Q1 =  df_RFM2['Monetary'].quantile(0.25)
Q3 = df_RFM2['Monetary'].quantile(0.75)
IQR = Q3 - Q1


# In[26]:


sns.boxplot(x=np.log10(df_RFM2).Monetary)


# In[27]:


final_data = df_RFM2.loc[~((df_RFM2['Monetary'] < (Q1 -3*IQR)) | (df_RFM['Monetary']> (Q3+3*IQR)))]


# In[28]:


final_data['Monetary'].max()


# In[29]:


fig = plt.figure()
ax = fig.add_subplot(111)
ax.boxplot(final_data.Monetary)
ax.set_ylabel('Monetary')
plt.show()


# In[30]:


final_data.describe()


# ## Normalization Data 
# - Normallize Values Sales Because reduce Scale Values to have a range between 0 - 1 
# - reference : https://medium.com/@rrfd/standardize-or-normalize-examples-in-python-e3f174b65dfc
# 

# In[31]:


from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
rfmData_scaled = pd.DataFrame(mms.fit_transform(final_data),
                              columns=final_data.columns,
                              index=final_data.index)


# In[32]:


rfmData_scaled


# ## Use K-Mean Clustering

# In[33]:


from sklearn.cluster import KMeans
cls = KMeans(n_clusters=7, n_jobs=-1, random_state=99)


# In[34]:


cls.fit(rfmData_scaled)


# In[35]:


cluster_centroid = pd.DataFrame(cls.cluster_centers_,
                                columns=rfmData_scaled.columns)
cluster_label = cls.labels_


# In[36]:


plt.figure(figsize=(10,5))
sns.heatmap(cluster_centroid, xticklabels = True ,cmap="YlGnBu",annot=True,vmin=0,vmax=1)
#plt.savefig('D:/RFM Python/RFM MEMBER ALL BRAND/cluster_last_vmin_vmax.jpg', dpi = 1000)
plt.show()


# In[37]:


desc = ['Other',
       'Lost Cheap',
       'Normal Customer Almost',
       'Best Customer',
       'Normal Customer',
       'Big Spender',
       'Lost Cheap',
       'Recent Customer',
       'Lost Cheap',
       'Almost lost']


# In[38]:


## สร้าง colums rfmData_clustered[k(x)]  =  โดยจะเก็บ กลุ่ม Cluster
##  k10_desc = เป็นการ assign ชื่อ label
rfmData_clustered = final_data.copy()
rfmData_clustered['k10'] = cls.labels_
rfmData_clustered['k10_desc'] = [desc[i] for i in rfmData_clustered['k10']]


# In[39]:


cluster_data = pd.DataFrame({'Cluster_Index':rfmData_clustered['k10'],
                             'Cluster':rfmData_clustered['k10_desc']})


# In[40]:


cluster_data = cluster_data.reset_index()


# In[54]:


cluster_data


# In[56]:


result = cluster_data.merge(final_data,on='CUSTOMER_ID')


# ## Plot See Cluster Group Member

# In[58]:


import seaborn as sns
sns.boxplot(data = result, x = 'Cluster_Index', y='Monetary')


# ## Evaluation Performance K-Mean And Tunning
# - Use Sillhouette Score  meaning Instance instance between group prepare other group Values Range Between [-1, 1] (Approach 1 good)
# - Reference 1 :  https://www.facebook.com/datascienceandteach/posts/2608914275837142/
# - Reference 2 :https://medium.com/@lengyi/%E0%B8%AB%E0%B8%B2%E0%B8%88%E0%B8%B3%E0%B8%99%E0%B8%A7%E0%B8%99-clusters-%E0%B8%97%E0%B8%B5%E0%B9%88%E0%B9%80%E0%B8%AB%E0%B8%A1%E0%B8%B2%E0%B8%B0%E0%B8%AA%E0%B8%A1%E0%B8%AA%E0%B8%B3%E0%B8%AB%E0%B8%A3%E0%B8%B1%E0%B8%9A-kmeans-clustering-%E0%B8%94%E0%B9%89%E0%B8%A7%E0%B8%A2-silhouette-analysis-e08220477c50
# 

# In[77]:


from sklearn.metrics import silhouette_score


# In[ ]:


silhouette_score(rfmData_scaled, cls.labels_)


# In[ ]:


import numpy as np
result = []
for c in np.arange(2,10,2):
    print('Clustering: n_cluster=%d'%c)
    cluster = KMeans(n_clusters=c, n_jobs=-1)
    cluster.fit(rfmData_scaled)
    result += [cluster.inertia_]


# In[ ]:


cluster_perf = pd.DataFrame({'n_cluster':np.arange(2,10,2),
                             'inertia': result})
import seaborn as sns
sns.lineplot(x='n_cluster',y='inertia',data=cluster_perf)


# In[ ]:


import seaborn as sns
sns.lineplot(x='n_cluster',y='inertia',data=cluster_perf)


# ## Tuning with silhouette score
# 

# In[ ]:


import numpy as np
sil_score = []
for c in np.arange(10,51,10):
    print('Clustering: n_cluster=%d'%c)
    cluster = KMeans(n_clusters=c, n_jobs=30)
    cluster.fit(rfmData_scaled)
    sil_score += [silhouette_score(rfmData_scaled, cluster.labels_)]


# In[ ]:


import matplotlib.pyplot as plt
cluster_perf_sil = pd.DataFrame({'n_cluster':np.arange(10,51,10),
                             's_score': sil_score})
plt.plot('n_cluster','s_score',data=cluster_perf_sil)


# In[ ]:




