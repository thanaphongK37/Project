#!/usr/bin/env python
# coding: utf-8

# # Market Basket Analysis
#  - Use Market Basket Analysis For looking find the relationship of the products saling in each bill that there is there an opportunity for each bill to buy together?

# In[4]:


import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import os


# ## 1. Load Raw Data

# In[5]:


data_raw = pd.read_pickle('Raw_data_file.data')


# In[32]:


data_raw.head()


# # 2. Checking Data type insight data
# - *<B>NOTE</B>* If there are many feature and Can Use library Pandas Profiling For Check EDA Report HTML Data  <H3>Reference<H3/> : https://github.com/pandas-profiling/pandas-profiling

# In[7]:


data_raw.info()


# In[10]:


data_raw['BILLNO'] = data_raw['BILLNO'].str.strip()
data_raw['PRODUCT_CODE'] = data_raw['PRODUCT_CODE'].str.strip()
data_raw = data_raw[~data_raw['BILLNO'].str.contains('C')] 


# In[11]:


df_pivot = pd.pivot_table(data_raw,index='BILLNO' ,
                          columns='PRODUCT_CODE',
                          values='PRICE_SALES',
                          fill_value=0).sort_values(by='IC3171')


# ## Defind Function For Generate  Mapping Sales 
# - 0 = meaning Not Sales in Product in Billno
# - 1 = meaning Have Sales Product in Billno

# In[14]:

def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1


# In[15]:


data_basket = df_pivot.applymap(encode_units)


# In[35]:


data_basket.head()


# ## Call Library mlxtend Apriori (frequent)  : 
# - Reference http://rasbt.github.io/mlxtend/api_subpackages/mlxtend.frequent_patterns/
# - Output Meaning = Relationship between product_item and product_item  that are common to occur  together.

# In[30]:

frequent_itemsets = apriori(data_basket, min_support=0.01,
                                        use_colnames=True , max_len=None )\
                                        .sort_values(by = 'support', ascending=False) 


# In[34]:


frequent_itemsets.head()


# ## Call Library mlxtend Apriori(association_rules) : 
# - Output Meaning = Find association rules that item the item_product is  to happen with what products in Confidence Values

# In[33]:


rules = association_rules(frequent_itemsets,metric="confidence", min_threshold=0.01)
rules.sort_values(by='confidence', ascending=False).head()

