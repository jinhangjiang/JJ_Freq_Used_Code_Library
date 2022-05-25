# Python program to generate WordCloud
  
# importing all necessery modules
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import pandas as pd
import re
  
# Reads text file here 
text = str([""]) ############ add text here
text = re.sub(r'==.*?==+', '', text)
text = text.replace('\n', '')
  
# Define a function to plot word cloud
def plot_cloud(wordcloud):
    # Set figure size
    plt.figure(figsize=(40, 30))
    # Display image
    plt.imshow(wordcloud) 
    # No axis details
    plt.axis("off");

# Generate word cloud
wordcloud = WordCloud(width = 1200, height = 628, random_state=333, 
                      colormap='rainbow', collocations=False, background_color ='salmon',
                      stopwords = STOPWORDS).generate(text) # background_color = "white"
# Plot
plot_cloud(wordcloud)
