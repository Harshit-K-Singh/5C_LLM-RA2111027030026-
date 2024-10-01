import torch
from torch.nn import CrossEntropyLoss
from rouge_score import rouge_scorer

def compute_perplexity(model, tokenizer, eval_dataset):
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    for _, row in eval_dataset.iterrows():
        input_text = row['input_text']
        target_text = row['target_text']
        
        inputs = tokenizer(input_text, return_tensors='pt', truncation=True, max_length=512)
        targets = tokenizer(target_text, return_tensors='pt', truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = model(**inputs, labels=targets['input_ids'])
            loss = outputs.loss
            
        total_loss += loss.item() * targets['input_ids'].numel()
        total_tokens += targets['input_ids'].numel()
    
    perplexity = torch.exp(torch.tensor(total_loss / total_tokens))
    return perplexity.item()

def compute_rouge_scores(generated_impressions, reference_impressions):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = []
    
    for gen, ref in zip(generated_impressions, reference_impressions):
        score = scorer.score(gen, ref)
        scores.append(score)
    
    # Aggregate scores
    avg_scores = {
        'rouge1': sum(s['rouge1'].fmeasure for s in scores) / len(scores),
        'rouge2': sum(s['rouge2'].fmeasure for s in scores) / len(scores),
        'rougeL': sum(s['rougeL'].fmeasure for s in scores) / len(scores),
    }
    
    return avg_scores

def evaluate_model(model, tokenizer, eval_dataset):
    # Generate impressions
    generated_impressions = []
    for _, row in eval_dataset.iterrows():
        input_text = row['input_text']
        input_ids = tokenizer.encode(input_text, return_tensors='pt')
        output = model.generate(input_ids, max_length=150, num_return_sequences=1)
        generated_impression = tokenizer.decode(output[0], skip_special_tokens=True)
        generated_impressions.append(generated_impression)
    
    # Compute metrics
    perplexity = compute_perplexity(model, tokenizer, eval_dataset)
    rouge_scores = compute_rouge_scores(generated_impressions, eval_dataset['target_text'].tolist())
    
    return perplexity, rouge_scores