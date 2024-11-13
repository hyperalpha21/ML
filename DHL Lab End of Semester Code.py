#!/usr/bin/env python
# coding: utf-8

# In[8]:


from nltk import pos_tag, word_tokenize, FreqDist
from nltk.corpus import stopwords
import nltk
import re
import matplotlib.pyplot as plt

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('vader_lexicon')

from nltk.sentiment import SentimentIntensityAnalyzer

def analyze_stylometry(text):
    # Basic analyses
    sentences = re.split(r'[.!?]+', text)
    words = re.findall(r'\b\w+\b', text.lower())
    unique_words = set(words)
    
    #Filter out stopwords like "the", "is", "at", "which", and "on".
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words]
    
    #Part-of-Speech Tagging
    tokens = word_tokenize(text)
    tagged_tokens = pos_tag(tokens)
    pos_counts = FreqDist(tag for (word, tag) in tagged_tokens)
    
    #Frequency Distribution of filtered words
    freq_dist = FreqDist(filtered_words)
    most_common_words = freq_dist.most_common(10) #Adjust the number as needed
    
    #Flesch Reading Ease
    syllable_count = sum([text.count(vowel) for vowel in "aeiouAEIOU"])  # Simplified approach
    words_per_sentence = [len(sentence.split()) for sentence in sentences if sentence]
    avg_syllables_per_word = syllable_count / len(words) if words else 0
    avg_sentence_length = sum(words_per_sentence) / len(sentences) if sentences else 0
    flesch_reading_ease = 206.835 - 1.015 * avg_sentence_length - 84.6 * avg_syllables_per_word
    
    #Sentiment Analysis
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = sia.polarity_scores(text)
    ##positive, negative, or netural sounding text
    
    analysis_results = {
        "Average Sentence Length": avg_sentence_length,
        "Lexical Diversity": len(unique_words) / len(words) if words else 0,
        "POS Counts": pos_counts,
        "Flesch Reading Ease": flesch_reading_ease,
        "Sentiment Scores": sentiment_scores,
        "Most Common Words": most_common_words
    }
    
    return analysis_results

#Visualization
def present_results(results):
    for key, value in results.items():
        if key == "POS Counts":
            print("Part-of-Speech Distribution:")
            plt.figure(figsize=(20, 7))
            plt.bar(value.keys(), value.values())
            plt.xlabel('Part of Speech')
            plt.ylabel('Frequency')
            plt.title('Part-of-Speech Distribution')
            plt.show()
        elif key == "Most Common Words":
            print(f"{key}:")
            for word, count in value:
                print(f"  {word}: {count}")
        else:
            print(f"{key}: {value}")


content = read_docx('/Users/neeldavuluri/analects.docx')
stylometry_results = analyze_stylometry(content)
present_results(stylometry_results)


# In[ ]:




