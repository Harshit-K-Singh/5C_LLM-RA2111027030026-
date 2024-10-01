from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import Dataset

def fine_tune_model(model_name, train_dataset, eval_dataset):
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Tokenize datasets
    def tokenize_function(examples):
        return tokenizer(examples['input_text'], examples['target_text'], truncation=True, padding='max_length', max_length=512)
    
    train_dataset = Dataset.from_pandas(train_dataset)
    eval_dataset = Dataset.from_pandas(eval_dataset)
    
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_eval = eval_dataset.map(tokenize_function, batched=True)
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
    )
    
    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval
    )
    
    # Fine-tune the model
    trainer.train()
    
    return model, tokenizer