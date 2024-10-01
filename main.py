import os
from data_preparation import load_and_prepare_data
from model_fine_tuning import fine_tune_model
from model_evaluation import evaluate_model
from text_analysis import preprocess_text, generate_embeddings, find_top_word_pairs
from visualization import visualize_word_pairs, interactive_visualization

def main():
    # Load and prepare data
    train_df, eval_df = load_and_prepare_data('src/impression_300_llm.csv')
    
    # Fine-tune model
    model_name = "google/gemma-2b-it"  # or "google/gemma-7b-it" based on your hardware
    model, tokenizer = fine_tune_model(model_name, train_df, eval_df)
    
    # Evaluate model
    perplexity, rouge_scores = evaluate_model(model, tokenizer, eval_df)
    print(f"Perplexity: {perplexity}")
    print(f"ROUGE Scores: {rouge_scores}")
    
    # Text Analysis
    all_text = ' '.join(train_df['input_text'].tolist() + eval_df['input_text'].tolist())
    processed_texts = [preprocess_text(text) for text in all_text.split('.')]
    embeddings = generate_embeddings(processed_texts)
    top_pairs = find_top_word_pairs(embeddings)
    
    # Visualization
    visualize_word_pairs(top_pairs)
    interactive_visualization(top_pairs)

if __name__ == "__main__":
    main()