import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('wordnet')
import string
from textblob import Word
import re

stop = stopwords.words('english')
spam_msg = ['bot', 'action', 'performed', 'automatically','please', 'contact', 'moderator'] #specify the spam words that you want to move
df['patterns'] = df['patterns'].apply(lambda x:' '.join(x.lower() for x in x.split()))
df['patterns'] = df['patterns'].apply(lambda x:' '.join(x for x in x.split() if x not in string.punctuation)) #remove punctuations
df['patterns'] = df['patterns'].str.replace('https*\S+','')  #remove url
df['patterns'] = df['patterns'].str.replace('\'\w+','')      #remove ticks
df['patterns'] = df['patterns'].str.replace('[^\w\s]','')    
df['patterns'] = df['patterns'].str.replace('@\S+','')       #remove email
df['patterns'] = df['patterns'].str.encode('ascii', 'ignore').str.decode("utf-8")  #remove unicode
df['patterns'] = df['patterns'].str.replace('\w*\d+\w*','')  #remove digits
df['patterns'] = df['patterns'].str.replace('#\S+','')  #remove hashtag
df['patterns'] = df['patterns'].str.replace('_','')  #remove underscore
df['patterns'] = df['patterns'].apply(lambda x: ' '.join(x for x in x.split() if  not x.isdigit()))
df['patterns'] = df['patterns'].apply(lambda x:' '.join(x for x in x.split() if not x in stop)) #remove stop words
df['patterns'] = df['patterns'].apply(lambda x:' '.join(x for x in x.split() if not x in spam_msg)) #remove stop words
df['patterns'] = df['patterns'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
