#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd


# In[ ]:





# In[1]:


import pandas as pd

# List of possible encodings to try
encodings = ['utf-8', 'latin1', 'ISO-8859-1', 'cp1252']

file_path = 'spam.csv' # Change this to the path of your CSV file

# Attempt to read the CSV file with different encodings 
for encoding in encodings:
    try:
        df = pd.read_csv(file_path, encoding=encoding) 
        print(f"File successfully read with encoding:{encoding}")
        break # Stop the loop if successful
    except UnicodeDecodeError:
        print(f"Failed to read with encoding: {encoding}")
        continue 

# If the loop completes without success, df will not be defined
if 'df' in locals():
    print("CSV file has been successfully loaded.")
else:
    print("All encoding attempts failed. Unable to read the CSV file.")


# In[2]:


df.sample(5)


# In[ ]:


Data Cleaning


# In[4]:


df.info()


# In[5]:


df.sample(5)


# In[6]:


df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)


# In[7]:


df.sample(5)


# In[8]:


df.rename(columns={'v1' : 'target','v2' : 'text'},inplace=True)


# In[9]:


df.sample(5)


# In[10]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()


# In[11]:


df['target'] = encoder.fit_transform(df['target'])


# In[12]:


df.head()


# In[14]:


df.isnull().sum()


# In[15]:


df.duplicated().sum()


# In[16]:


df = df.drop_duplicates(keep='first')


# In[17]:


df.duplicated().sum()


# In[18]:


df.shape


# In[20]:


eda data exploration


# In[21]:


df.head()


# In[23]:


df['target'].value_counts()


# In[24]:


import matplotlib.pyplot as plt
plt.pie(df['target'].value_counts(), labels=['ham', 'spam'], autopct="%0.2f")
plt.show()


# In[25]:


import nltk


# In[26]:


get_ipython().system('pip install nltk')


# In[27]:


nltk.download('punkt')


# In[28]:


nltk.download('punkt')


# In[29]:


nltk.download('punkt')


# In[30]:


df['num_characters'] = df['text'].apply(len)


# In[31]:


df.head()


# In[33]:


df['num_words'] = df['text'].apply(lambda x:len(nltk.word_tokenize(x)))


# In[34]:


df['num_words'] = df['text'].apply(lambda x:len(nltk.word_tokenize(x)))


# In[35]:


df['num_words'] = df['text'].apply(lambda x:len(nltk.word_tokenize(x)))


# In[36]:


df.head()


# In[37]:


df['num_sentences'] = df['text'].apply(lambda x:len(nltk.word_tokenize(x)))


# In[38]:


nltk.download('punkt')


# In[39]:


get_ipython().system('pip install nltk')


# In[40]:


nltk.download('punkt')


# In[41]:


import urllib
import urllib2
g = "http://www.google.com/"
read = urllib2.urlopen(g, timeout=20)


# In[42]:


import nltk
nltk.download()


# In[44]:


import nltk
nltk.download('punkt')
from nltk import sent_tokenize


# In[ ]:


nltk.download('punkt')


# In[46]:


df.head()


# In[47]:


df['num_words'] = df['text'].apply(lambda x:len(nltk.word_tokenize(x)))


# In[48]:


import nltk


# In[49]:


get_ipython().system('pip install nltk')


# In[50]:


nltk.download('punkt')


# In[51]:


import nltk


# In[52]:


get_ipython().system('pip install nltk')


# In[53]:


nltk.download('punkt')


# In[54]:


df.head()


# In[55]:


df['num_words'] = df['text'].apply(lambda x:len(nltk.word_tokenize(x)))


# In[56]:


df.head()


# In[57]:


df['num_sentences'] = df['text'].apply(lambda x:len(nltk.sent_tokenize(x)))


# In[58]:


df.head()


# In[59]:


df[['num_characters','num_words','num_sentences']].describe()


# In[60]:


df[df['target'] == 0][['num_characters','num_words','num_sentences']].describe()


# In[61]:


df[df['target'] == 1][['num_characters','num_words','num_sentences']].describe()


# In[62]:


import seaborn as sns


# In[63]:


plt.figure(figsize=(12,6))
sns.histplot(df[df['target'] == 0]['num_characters'])
sns.histplot(df[df['target'] == 1]['num_characters'],color='red')


# In[64]:


plt.figure(figsize=(12,6))
sns.histplot(df[df['target'] == 0]['num_words'])
sns.histplot(df[df['target'] == 1]['num_words'],color='red')


# In[65]:


sns.pairplot(df,hue='target')


# In[66]:


sns.heatmap(df.corr(),annot=True)


# In[67]:


sns.heatmap(df.corr(),annot=True)


# In[1]:


sns.heatmap(df.corr(),annot=True)


# In[3]:


sns.pairplot(df,hue='target')


# In[7]:


def transform_text(text):
    text = text.lower()
   
    
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
    
            
    return " ".join(y)

transform_text = transform_text("I'm gonna be home soon and i don't want to talk about this stuff anymore tonight, k? I've cried eno")
print(transform_text)


# In[5]:


transform_text("I'm gonna be home soon and i don't want to talk about this stuff anymore tonight, k? I've cried eno")


# In[ ]:




