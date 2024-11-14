#!/usr/bin/env python
# coding: utf-8

# In[6]:


# utilities
import re
import numpy as np
import pandas as pd
#plotting
import seaborn as sns
from wordcloud import WordCloud
import matplotlib.pyplot as plt 
# nltk
from nltk.stem import WordNetLemmatizer
#sklearn
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix,classification_report


# In[7]:


get_ipython().system('pip3 install WordCloud')


# In[9]:


# utilities
import re
import numpy as np
import pandas as pd
#plotting
import seaborn as sns
from wordcloud import WordCloud
import matplotlib.pyplot as plt 
# nltk
from nltk.stem import WordNetLemmatizer
#sklearn
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix,classification_report


# In[12]:


#importing the dataset
DATASET_COLUMNS=['target','ids','date','flag','user','text']
DATASET_ENCODING="ISO-8859-1"
df=pd.read_csv('C:\Users\HP\OneDrive\Desktop\sentiment\training.1600000.processed.noemoticon.csv',encoding=DATASET_ENCODING,names=DATASET_COLUMNS)
df.sample(5)


# In[13]:


# Importing necessary libraries
import pandas as pd

# Defining dataset columns and encoding
DATASET_COLUMNS = ['target', 'ids', 'date', 'flag', 'user', 'text']
DATASET_ENCODING = "utf-8"

# Reading the CSV file into a DataFrame
file_path = r'C:\Users\HP\OneDrive\Desktop\sentiment\training.1600000.processed.noemoticon.csv'
df = pd.read_csv(file_path, encoding=DATASET_ENCODING, names=DATASET_COLUMNS)

# Displaying a sample of the DataFrame
df.sample(5)


# In[14]:


# Importing necessary libraries
import pandas as pd

# Defining dataset columns and encoding
DATASET_COLUMNS = ['target', 'ids', 'date', 'flag', 'user', 'text']
DATASET_ENCODING = "ISO-8859-1"

# Reading the CSV file into a DataFrame
file_path = r'C:\Users\HP\OneDrive\Desktop\sentiment\training.1600000.processed.noemoticon.csv'
df = pd.read_csv(file_path, encoding=DATASET_ENCODING, names=DATASET_COLUMNS)

# Displaying a sample of the DataFrame
df.sample(5)


# In[15]:


df.head() #5 top records


# In[16]:


df.columns


# In[17]:


print('length of data is',len(df))


# In[19]:


df.shape


# In[10]:


import pandas as pd
import numpy as np

# Defining dataset columns and encoding
DATASET_COLUMNS = ['target', 'ids', 'date', 'flag', 'user', 'text']
DATASET_ENCODING = "ISO-8859-1"

# Reading the CSV file into a DataFrame
file_path = r'C:\Users\HP\OneDrive\Desktop\sentiment\training.1600000.processed.noemoticon.csv'
df = pd.read_csv(file_path, encoding=DATASET_ENCODING, names=DATASET_COLUMNS)

# Displaying a sample of the DataFrame
df.sample(5)
df.info()


# In[8]:


df.dtypes


# In[11]:


np.sum(df.isnull().any(axis=1))


# In[12]:


np.sum(df.isnull().any(axis=1))


# In[13]:


print('Count of columns in the data is:  ', len(df.columns))
print('Count of rows in the data is:  ', len(df))


# In[16]:


print('Count of columns in data is:',len(df.columns))


# In[17]:


print('Count the no of rows in the data:',len(df))


# In[18]:


df['target'].unique()


# In[19]:


df['target'].nunique()


# In[20]:


ax = df.groupby('target').count().plot(kind='bar', title='Distribution of data',legend=False)
ax.set_xticklabels(['Negative','Positive'], rotation=0)


# In[26]:


text,sentiment=list(df['text']),list(df['target'])


# In[27]:


import seaborn as sns
sns.countplot(x='target',data=df)


# In[28]:


data=df[['text','target']]


# In[ ]:




