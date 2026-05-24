import json
import os
import torch
from FlagEmbedding import BGEM3FlagModel
import numpy as np

# Check for CUDA availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load BGE model
model = BGEM3FlagModel('BAAI/bge-large-en-v1.5', use_fp16=True, device=device)

def generate_embeddings(descript_file, output_file):
    embeddings = []
    names = []
    texts = []
    
    with open(descript_file, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            # from ipdb import set_trace; set_trace()
            name = data['name']
            # response = json.loads(data['response'])
            text = data['response']  # Use summarization as the text to embed
            texts.append(text)
            names.append(name)
    
    # Encode in batches
    embeddings = model.encode(texts, batch_size=32, max_length=512)['dense_vecs']
    
    # Save embeddings
    np.save(output_file, embeddings)
    
    # Save name map
    map_file = output_file.replace('.npy', '_map.tsv')
    with open(map_file, 'w') as f:
        for name in names:
            f.write(name + '\n')
    
    print(f"Saved embeddings to {output_file} and map to {map_file}")

# Process for different models and types
# models = ['gpt3.5', 'gpt4o']
models = ['gpt-5']
types = ['cell', 'drug']

for model_name in models:
    for typ in types:
        descript_file = f'datasets/descriptions/{typ}_descripts_all_{model_name}.jsonl'
        if model_name == 'gpt3.5':
            folder = 'gpt-3.5-turbo'
        elif model_name == 'gpt4o':
            folder = 'gpt-4o-mini'  # Assuming this is the folder
        else:
            folder = model_name
        
        output_file = f'ckpts/llms/{folder}/{typ}_embed_bge.npy'
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        generate_embeddings(descript_file, output_file)
