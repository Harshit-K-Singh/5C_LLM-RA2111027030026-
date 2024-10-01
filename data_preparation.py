import pandas as pd

def load_and_prepare_data(file_path, num_train=300, num_eval=30):
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Prepare the data for fine-tuning
    df['input_text'] = df['Report Name'] + ' [SEP] ' + df['History'] + ' [SEP] ' + df['Observation']
    df['target_text'] = df['Impression']
    
    # Split the data
    train_df = df.iloc[:num_train]
    eval_df = df.iloc[num_train:num_train+num_eval]
    
    return train_df, eval_df