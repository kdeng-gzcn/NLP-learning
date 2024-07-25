#!/usr/bin/env python
# coding: utf-8

# # Sentiment Analysis
# - dataset: Amazon fine food reviews
# 
# baselines:
# 
# 1. VADER - bag of words method
# 2. Roberta pretrained Model from 🤗 (Hugging Face)
# 3. Huggingface Pipeline

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


plt.style.use('ggplot') # R style


# In[3]:


import nltk


# In[4]:


path = 'D:/Users/pc/Documents/VscodeFiles/dataset/NLP projects/Reviews.csv'
df = pd.read_csv(path)


# In[5]:


df.head()
# id product_id user_id profile_name ...


# In[6]:


# we are focus on text
df.Text.values[0]


# In[7]:


df.shape


# In[8]:


# OPTION!!!
# we select 500 rows for initial
# df = df.head(500)
# df = df.sample(n=140000, replace=False)
# df.reset_index(drop=True, inplace=True)


# # EDA

# In[9]:


df.Score.value_counts()


# In[10]:


df.Score.value_counts().sort_index() # star from 1 to 5 asc


# In[11]:


ax = df.Score.value_counts().sort_index().plot( # series.plot(params)
    kind='bar',
    title='distribution of Stars',
    figsize=(10, 5)
)
ax.set_xlabel('Stars')
plt.show()


# # Basic NLTK

# In[12]:


# take example text
example = df.Text[49]
example


# In[13]:


# change text into tokens using NLTK
tokens = nltk.word_tokenize(example)
tokens[:10]


# In[14]:


# get pos of these tokens 
# pos is part of speech
tagged = nltk.pos_tag(tokens)
tagged[:5]


# In[15]:


# nltk.download('maxent_ne_chunker')
# nltk.download('words')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('punkt')

# this is a tree structure

entities = nltk.chunk.ne_chunk(tagged) # NER???
entities.pprint()
entities


# # Baselien1: VADER(Valance Aware Dictionary and Entiment Reasoner)
# 
# - bag of words method
# 
# 1. stop words are removed
# 2. each word is scored and combined to a total score

# In[16]:


from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm


# In[17]:


# nltk.download('vader_lexicon')


# In[18]:


sen_int_analyzer = SentimentIntensityAnalyzer()


# In[19]:


# demo
sen_int_analyzer.polarity_scores('Fuck!')
# compound: aggregation of neg, pos, neu
# from -1 to 1


# In[20]:


sen_int_analyzer.polarity_scores('I don\'t like it')


# In[21]:


# recall our example
print(example)
sen_int_analyzer.polarity_scores(example)


# entire dataset with si_analyzer

# In[22]:


df.head()


# In[23]:


result = {} # empty dict

for i, row in tqdm(df.iterrows(), total=len(df)):
    text = row.Text
    myid = row.Id
    result[myid] = sen_int_analyzer.polarity_scores(text)


# In[24]:


vaders = pd.DataFrame(result).T
vaders = vaders.reset_index().rename(columns={'index': 'Id'})
vaders.head()


# In[25]:


vaders = vaders.merge(df, how='left') # left join with original data


# In[26]:


# left join and we get final dataset
vaders.head()


# plot the result

# In[27]:


ax = sns.barplot(
    data = vaders,
    x='Score',
    y='compound'
)
ax.set_title('compound by score')
plt.show()


# In[28]:


fig, axes = plt.subplots(1, 3, figsize=(15, 3))
# ??? .subplot <=> .subplots
sns.barplot(
    data=vaders,
    x='Score',
    y='neg',
    ax=axes[0]
)
sns.barplot(
    data=vaders,
    x='Score',
    y='neu',
    ax=axes[1]
)
sns.barplot(
    data=vaders,
    x='Score',
    y='pos',
    ax=axes[2]
)
axes[0].set_title('neg')
axes[1].set_title('neu')
axes[2].set_title('pos')
plt.tight_layout() # for better visualization
plt.show()


# # baseline2: hugging face
# - pre-train model is using a large corpus of data
# - transformer not only works on words but also focus on context related to words

# In[29]:


from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax # for classification
import torch


# In[30]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[31]:


MODEL = f'cardiffnlp/twitter-roberta-base-sentiment'
tokenizer = AutoTokenizer.from_pretrained(MODEL) # no need on GPU
model = AutoModelForSequenceClassification.from_pretrained(MODEL).to(device)


# In[32]:


# recall example from vader
# we want to make comparison
print(example)
sen_int_analyzer.polarity_scores(example)


# In[33]:


# example for roberta model
encoded_text = tokenizer(example, return_tensors='pt').to(device)
encoded_text


# In[34]:


with torch.no_grad():
    output = model(**encoded_text) # on GPU
type(output), output


# In[35]:


scores = output[0][0].detach().cpu().numpy()
scores


# In[36]:


scores = softmax(scores)
scores


# In[37]:


scores_dict = {
    'roberta_neg': scores[0],
    'roberta_neu': scores[1],
    'roberta_pos': scores[2]
}
scores_dict


# In[38]:


# create a function of above
def polarity_scores_roberta(text):
    encoded_text = tokenizer(text, return_tensors='pt').to(device=device)
    output = model(**encoded_text) # decoder
    scores = output[0][0].detach().cpu().numpy() # get ndarray
    scores = softmax(scores) # make it a prob vector
    scores_dict = {
    'roberta_neg': scores[0],
    'roberta_neu': scores[1],
    'roberta_pos': scores[2]
    }
    return scores_dict 


# In[39]:


result = {} # empty dict

for i, row in tqdm(df.iterrows(), total=len(df)):
    try:
        text = row.Text
        myid = row.Id
        vader_result = sen_int_analyzer.polarity_scores(text)  
        vader_result_rename = {}
        for key, value in vader_result.items():
            vader_result_rename[f'vader_{key}'] = value 
        roberta_result = polarity_scores_roberta(text) # roberta prefer using GPU

        both = {**vader_result_rename, **roberta_result} # old version
        result[myid] = both
    except RuntimeError:
        print(f'break for id {myid}')


# In[40]:


result_df = pd.DataFrame(result).T
result_df = result_df.reset_index().rename(columns={'index': 'Id'})
result_df.head()


# In[41]:


result_df = result_df.merge(df, how='left') # left join with original data


# In[42]:


result_df.head() # final df


# # Visualization and Comparison

# In[43]:


# before input list, we copy from .columns
result_df.columns


# In[44]:


sns.pairplot(
    data=result_df,
    vars=['vader_neg', 'vader_neu', 'vader_pos', 'vader_compound',
       'roberta_neg', 'roberta_neu', 'roberta_pos'], # paste
    hue='Score', # color
    palette='tab10'
)
plt.show()


# from the fig:
# 1. roberta model is more confident

# # Review Examples (especially outliers)

# In[45]:


result_df.query('Score == 1').sort_values(by='roberta_pos', ascending=False).Text.values[0] # find the most roberta positive but with 1 star


# In[46]:


result_df.query('Score == 1').sort_values(by='vader_pos', ascending=False).Text.values[0] # find the most vader positive but with 1 star


# In[47]:


# on the opposite
result_df.query('Score == 5').sort_values(by='roberta_neg', ascending=False).Text.values[0] # find the most roberta negative but with 1 star


# In[48]:


result_df.query('Score == 5').sort_values(by='vader_neg', ascending=False).Text.values[0] # find the most vader negative but with 1 star


# # Extra: Transformer Pipeline

# In[49]:


from transformers import pipeline


# In[50]:


sentimen_pipeline = pipeline("sentiment-analysis") # download pre-train model


# In[51]:


sentimen_pipeline('FUCK!')


# In[52]:


sentimen_pipeline('No way')


# In[53]:


sentimen_pipeline('I like sentiment analysis!')


# In[ ]:




