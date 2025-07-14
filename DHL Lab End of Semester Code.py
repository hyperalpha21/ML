import nltk
import re
import matplotlib.pyplot as plt
from nltk import pos_tag, word_tokenize, FreqDist
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('vader_lexicon')

#different writing analysis tests
def analyze_text_style(text):
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    words = re.findall(r'\b\w+\b', text.lower())
    unique_words = set(words)
    
    stop_words = set(stopwords.words('english'))
    content_words = [w for w in words if w not in stop_words]
    
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    pos_frequency = FreqDist(tag for word, tag in pos_tags)
    
    word_frequency = FreqDist(content_words)
    top_words = word_frequency.most_common(15)
    
    vowels = "aeiouAEIOU"
    syllable_estimate = sum(text.count(v) for v in vowels)
    
    sentence_lengths = [len(sentence.split()) for sentence in sentences if sentence]
    avg_syllables_per_word = syllable_estimate / len(words) if words else 0
    avg_sentence_length = sum(sentence_lengths) / len(sentences) if sentences else 0
    
    flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
    
    analyzer = SentimentIntensityAnalyzer()
    sentiment_data = analyzer.polarity_scores(text)
    
    metrics = {
        "avg_sentence_length": avg_sentence_length,
        "lexical_diversity": len(unique_words) / len(words) if words else 0,
        "pos_distribution": pos_frequency,
        "flesch_readability": flesch_score,
        "sentiment_analysis": sentiment_data,
        "frequent_words": top_words
    }
    
    return metrics

def display_analysis(results):
    for metric, data in results.items():
        if metric == "pos_distribution":
            print("Part-of-Speech Distribution:")
            plt.figure(figsize=(18, 6))
            pos_tags = list(data.keys())
            frequencies = list(data.values())
            plt.bar(pos_tags, frequencies, color='steelblue', alpha=0.7)
            plt.xlabel('POS Tags')
            plt.ylabel('Count')
            plt.title('Part-of-Speech Tag Distribution')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
        elif metric == "frequent_words":
            print(f"\nMost Frequent Words:")
            for word, count in data:
                print(f"  {word}: {count}")
        else:
            print(f"\n{metric.replace('_', ' ').title()}: {data}")

content = read_docx('/Users/neeldavuluri/analects.docx')
results = analyze_text_style(content)
display_analysis(results)
