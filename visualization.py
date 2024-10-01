import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import numpy as np

def visualize_word_pairs(word_pairs):
    words, _ = zip(*word_pairs)
    unique_words = list(set(words))
    
    # Create a matrix of similarities
    similarity_matrix = np.zeros((len(unique_words), len(unique_words)))
    for w1, w2, sim in word_pairs:
        i, j = unique_words.index(w1), unique_words.index(w2)
        similarity_matrix[i, j] = similarity_matrix[j, i] = sim
    
    # Create heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(similarity_matrix, xticklabels=unique_words, yticklabels=unique_words, cmap='YlOrRd')
    plt.title('Top 100 Word Pairs Similarity')
    plt.tight_layout()
    plt.savefig('word_pairs_heatmap.png')
    plt.close()

def interactive_visualization(word_pairs):
    words, _ = zip(*word_pairs)
    unique_words = list(set(words))
    
    # Create a matrix of similarities
    similarity_matrix = np.zeros((len(unique_words), len(unique_words)))
    for w1, w2, sim in word_pairs:
        i, j = unique_words.index(w1), unique_words.index(w2)
        similarity_matrix[i, j] = similarity_matrix[j, i] = sim
    
    # Create interactive heatmap
    fig = go.Figure(data=go.Heatmap(
        z=similarity_matrix,
        x=unique_words,
        y=unique_words,
        colorscale='YlOrRd'
    ))
    
    fig.update_layout(
        title='Interactive Top 100 Word Pairs Similarity',
        xaxis_title='Words',
        yaxis_title='Words'
    )
    
    fig.write_html("interactive_word_pairs_heatmap.html")