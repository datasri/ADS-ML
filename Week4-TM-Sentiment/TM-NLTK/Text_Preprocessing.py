#!/usr/bin/env python
# coding: utf-8

# <img style="float: left;" src="pic2.png">

# ### Sridhar Palle, Ph.D, spalle@emory.edu (Applied ML & DS with Python Program)

# # Text Preprocessing

# **Import the libraries and dependencies**

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from bs4 import BeautifulSoup
import nltk
import contractions
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#nltk.download('all', halt_on_error=False) # do this only once if never done before


# ## 1. Regex operations

# * re.match() - matches pattern at the begnining of the string
# * re.search() - match patterns occuring at any position
# * re.findall() - returns all non-verlapping matches of a specifief regex pattern
# * re.sub() - replaces a pattern with another string

# In[3]:


sample_text = "Learning is a repetitive process. Best way of learning anything in life is to actualllly do it, @@data science wisdom <br> <br>.but what about difficulty in understanding???. Again $ when we take baby steps**,everything becomes easier"
sample_text


# In[4]:


re.match('Learning', sample_text)


# In[5]:


re.match('Learning', sample_text).span()


# In[6]:


re.match('Best', sample_text) #match only works for matching a pattern at the begining


# **re.search()**

# In[7]:


sample_text


# In[8]:


re.search('Best', sample_text) #search works to match pattern at any position


# **re.findall()**

# In[9]:


sample_text


# In[10]:


re.findall('Learning', sample_text, re.I)


# In[11]:


re.findall('is', sample_text)


# In[12]:


re.findall('[^A-Za-z0-9., ]', sample_text) # returns all characters other than A-Za-z0-9.


# **re.sub()**

# In[13]:


sample_text


# In[14]:


re.sub('Best', 'Super', sample_text) # substitutes a regex pattern in a string with another


# In[15]:


re.sub('in', '500', sample_text)


# In[16]:


sample_text.replace('Best', 'Super') #.replace on strings achieves the same and is faster.


# ### 1.1.1 Regex rules

# <img style="float: left;" src="reg.png">

# **. Period**

# In[17]:


sample_text


# In[18]:


re.findall('.ea.', sample_text) # for matching any character before or after period


# In[19]:


re.findall('l..', sample_text)


# **^**

# In[20]:


sample_text


# In[21]:


re.findall('^L', sample_text, re.I) # ^ for matching the start of the string


# **$**

# In[22]:


re.findall('..r$', sample_text) # ^ for matching the end of the string


# **[...]**

# In[23]:


re.findall('[@]', sample_text) # for matching set of characters inside []


# **[^...]**

# In[24]:


re.findall('[^A-Za-z., ]', sample_text) # for matching any character which is not there after ^ in the [^ ]


# In[25]:


sample_text2 = "Learning is a repetitive process. Number 1 way of learning anything in life is to actualllly do it 1000 times, @@data science wisdom <br> <br>.but what about difficulty in understanding???. Again $ when we take baby steps**,everything becomes easier"
sample_text2


# **\d**

# In[26]:


re.findall('\d', sample_text2) # \d for matching decimal digits depicted by [0-9]


# **\D**

# In[27]:


re.findall('\D', sample_text2)[0:5] # \D for matching non-digits


# **\s**

# In[28]:


re.findall('\s', sample_text2)[0:5] # \s for matching whitespaces 


# **\S**

# In[29]:


''.join(re.findall('\S', sample_text2)) # \S for matching non-whitespaces 


# **\w**

# In[30]:


re.findall('\w', sample_text2)[0:5] # \w for matching alphanumeric characters [a-zA-Z0-9_]


# **\W**

# In[31]:


re.findall('\W', sample_text2)[0:9] # \W for matching non alphanumeric characters. Same as  [^a-zA-Z0-9_]


# **For more info on regular expressions please see https://docs.python.org/3.4/library/re.html**

# # 2. Text Preprocessing

# **Lets Load a  bigger imdb reviews dataset**

# In[32]:


imdb_big = pd.read_csv('movie_reviews.csv')


# In[33]:


imdb_big.head(3)


# In[34]:


imdb_big.shape


# In[35]:


imdb_big['review'].describe()


# In[36]:


imdb_big['sentiment'].value_counts()


# ### 2.1 Some basic preprocessing methodologies

# **Lets take a sample review and demonstrate different preprocessing metholodies**

# In[37]:


# reviews with lot of special characters, 20867, 26791, 37153, 42947, 48952    


# In[38]:


sample_review = imdb_big['review'][42947]
sample_review


# **Text Normalization or preprocessing steps**
#     - Converting to lowercase
#     - Remove html tags
#     - Removing punctuation
#     - Removing stop words
#     - Stemming or lemmatization
#     - Expanding contractions
#     - Correcting words, spellings
#     - ngrams

# **Converting to lowercase**

# In[40]:


sample_review


# In[41]:


def lower_case(text):
    return text.lower()


# In[42]:


sample_review = lower_case(sample_review)
sample_review


# **Removing html tags**

# In[43]:


def html_parser(text):
    return BeautifulSoup(text, "html.parser").get_text()


# In[44]:


sample_review = html_parser(sample_review)
sample_review


# **Expanding contractions**

# In[45]:


def replace_contractions(text):
    """Replace contractions in string of text"""
    return contractions.fix(text)


# In[46]:


sample_review = replace_contractions(sample_review)
sample_review


# **Removing punctuation and special characters**

# In[47]:


def remove_special(text):
    return re.sub('[^a-zA-Z0-9]', ' ', text)


# In[48]:


sample_review = remove_special(sample_review)
sample_review


# **Removing stop words**

# In[49]:


def remove_stopwords(text):
    stopword_list = nltk.corpus.stopwords.words('english')
    words = nltk.word_tokenize(text)
    words = [word.strip() for word in words]
    filtered_words = [word for word in words if word not in stopword_list]
    return ' '.join(filtered_words)


# In[50]:


sample_review = remove_stopwords(sample_review)
sample_review


# **Stemming or Lemmatization**

# In[51]:


def word_stem(text, kind='stemming'):
        from nltk.stem import WordNetLemmatizer
        from nltk.stem import PorterStemmer
        wnl = WordNetLemmatizer()
        ps = PorterStemmer()

        words = nltk.word_tokenize(text)
        words = [word.strip() for word in words]
        filtered_words = [wnl.lemmatize(word) if (kind == 'lemmatize') else ps.stem(word) for word in words]
        return ' '.join(filtered_words)


# In[52]:


word_stem(sample_review)


# In[53]:


word_stem(sample_review, 'lemmatize')

