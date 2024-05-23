import torch
import argparse
import pandas as pd
import numpy as np

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import BitsAndBytesConfig
import os
from tqdm import tqdm


def create_model_and_tokenizer(model_name):
    if model_name == 'Gemma':
        model_name = 'google/gemma-2b-it'
        # model_name = "Columbia-NLP/gemma-2b-zephyr-sft"
        bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_compute_dtype=torch.bfloat16
                    )

        model = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    num_labels = 1,
                    use_safetensors=True,
                    quantization_config=bnb_config,
                    trust_remote_code=True,
                    device_map="auto",
                    low_cpu_mem_usage=True
                )
    
    elif model_name == 'GPT2':
        model_name = 'lvwerra/gpt2-imdb'
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        model.config.pad_token_id = model.config.eos_token_id

    return model, tokenizer
    

def get_dataset_for_rlhf(dataset_name, small=True, train_or_test='train'):
    if dataset_name == 'UltraFeedback':
        if small:
            df = pd.read_csv(f'datasets/ultrafeedback_{train_or_test}_small.csv.gz', compression='gzip')
        else:
            df = pd.read_csv(f'datasets/ultrafeedback_{train_or_test}.csv.gz', compression='gzip')

        texts_chosen = [r for r in df["chosen"]]
        texts_reject = [r for r in df["rejected"]]
    
    elif dataset_name == 'Anthropic':
        if small:
            df = pd.read_csv(f'datasets/anthropic_hard_8k_{train_or_test}.csv.gz', compression='gzip')
        else:
            df = pd.read_csv(f'datasets/anthropic_{train_or_test}.csv.gz', compression='gzip')

        texts_chosen = [r for r in df["chosen"]]
        texts_reject = [r for r in df["rejected"]]
    
    elif dataset_name == 'IMDB':
        df = pd.read_csv(f'datasets/path_to_IMDB_{train_or_test}.csv')
    
        texts_chosen = [q + r for q, r in zip(df["query"], df["chosen"])]
        texts_reject = [q + r for q, r in zip(df["query"], df["rejected"])]

    return texts_chosen, texts_reject, len(df)

def save_embeddings(model, tokenizer, dataset_name, model_name, texts_chosen, texts_reject, n, output_path, small=True, train_or_test='train'):

    embed_chosen = []
    embed_reject = []
    bs = 256

    for i in tqdm(range((len(texts_chosen) // bs) + 1)):
        tc = texts_chosen[i*bs : min((i+1) * bs, len(texts_chosen))]
        tr = texts_reject[i*bs : min((i+1) * bs, len(texts_reject))]
        n = len(tc)
        
        #get the tokens
        texts_tokens_chosen = tokenizer(tr, return_tensors='pt', padding=True)
        texts_tokens_reject = tokenizer(tc, return_tensors='pt', padding=True)

        #convert the tokens to ids
        input_ids_chosen = texts_tokens_chosen['input_ids']
        input_ids_reject = texts_tokens_reject['input_ids']


        seq_len_chosen = (torch.eq(input_ids_chosen, model.config.pad_token_id).int().argmax(-1) - 1)
        seq_len_chosen = seq_len_chosen.numpy()

        seq_len_reject = (torch.eq(input_ids_reject, model.config.pad_token_id).int().argmax(-1) - 1)
        seq_len_reject = seq_len_reject.numpy()


        #generate the outputs
        with torch.no_grad():
            if model_name == 'GPT2':
                embed_chosen.append(model.transformer(**texts_tokens_chosen)[0][np.arange(n),seq_len_chosen].detach().cpu().numpy())
                embed_reject.append(model.transformer(**texts_tokens_reject)[0][np.arange(n),seq_len_reject].detach().cpu().numpy())
            elif model_name == 'Gemma':
                embed_chosen.append(model.model(**texts_tokens_chosen)[0][np.arange(n),seq_len_chosen].detach().cpu().numpy())
                embed_reject.append(model.model(**texts_tokens_reject)[0][np.arange(n),seq_len_reject].detach().cpu().numpy())


    embed_chosen = np.vstack(embed_chosen)
    embed_reject = np.vstack(embed_reject)
    if small:
        np.savetxt(os.path.join(output_path, f'{dataset_name}_small_{model_name}_embed_chosen_{train_or_test}.npy'), embed_chosen)
        np.savetxt(os.path.join(output_path, f'{dataset_name}_small_{model_name}_embed_reject_{train_or_test}.npy'), embed_reject)
    else:
        np.savetxt(os.path.join(output_path, f'{dataset_name}_{model_name}_embed_chosen_{train_or_test}.npy'), embed_chosen)
        np.savetxt(os.path.join(output_path, f'{dataset_name}_{model_name}_embed_reject_{train_or_test}.npy'), embed_reject)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', default='IMDB', type=str, help='Name of the dataset. Options: [\'IMDB\', \'UltraFeedback\', \'Anthropic\']')
    parser.add_argument('-m', '--model', default='GPT2', type=str, help='Name of the model. Options: [\'GPT2\', \'Gemma\']')
    parser.add_argument('-o', '--output', default='.', type=str, help='Output folder')
    parser.add_argument('-z', '--size', type=bool, default=True, help='Identifier whether use small or full dataset')

    args = parser.parse_args()

    model, tokenizer = create_model_and_tokenizer(args.model)
    text_chosen, text_rejected, n = get_dataset_for_rlhf(args.dataset, args.size)

    save_embeddings(model, tokenizer, args.dataset, args.model, text_chosen, text_rejected, n, args.output, args.size)