import os
import argparse
import torch
from tqdm import tqdm
import pandas as pd

tqdm.pandas()

from transformers import pipeline, AutoTokenizer, BitsAndBytesConfig

from transformers import AutoModelForSequenceClassification, AutoModelForCausalLM
from trl import AutoModelForCausalLMWithValueHead
#from scripts.ppo_model_training import get_dataset_iterator


def load_model_and_tokenizer(model_name, data_name, id, output_path, small=True, info=''):
    if model_name == 'GPT2':
        true_model_name = 'lvwerra/gpt2-imdb'
        model = AutoModelForCausalLMWithValueHead.from_pretrained(os.path.join(output_path, f'{data_name}_{model_name}_final_model_{id}_{info}'))
    elif model_name == 'Gemma':
        true_model_name = "google/gemma-2b-it"
        bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_compute_dtype=torch.bfloat16
                    )

        if small:
            if id == 'sft':
                model = AutoModelForCausalLM.from_pretrained('google/gemma-2b-it',
                            use_safetensors=True,
                            quantization_config=bnb_config,
                            device_map="auto",
                            low_cpu_mem_usage=True
                        )
            else:
                model = AutoModelForCausalLMWithValueHead.from_pretrained(
                            os.path.join(output_path, f'{data_name}_small_{model_name}_final_model_{id}_{info}_v4'),
                            use_safetensors=True,
                            quantization_config=bnb_config,
                            device_map="auto",
                            low_cpu_mem_usage=True
                        )
        else:
            model = AutoModelForCausalLMWithValueHead.from_pretrained(
                            os.path.join(output_path, f'{data_name}_{model_name}_final_model_{id}_{info}_v4'),
                            use_safetensors=True,
                            quantization_config=bnb_config,
                            device_map="auto",
                            low_cpu_mem_usage=True
                        )
    
    tokenizer = AutoTokenizer.from_pretrained(true_model_name, add_bos_token=False)
    # if tokenizer.pad_token is None:
    #     tokenizer.pad_token = tokenizer.eos_token
    #     tokenizer.padding_side = "right"
    #     model.config.pad_token_id = model.config.eos_token_id
    

    return model, tokenizer


def load_true_reward_model(model_name, data_name):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if model_name == 'GPT2' and data_name == 'IMDB':
        reward_pipeline = pipeline("sentiment-analysis", model="lvwerra/distilbert-imdb", device=device)
        def reward_processor(query, response, kwargs):
            texts = [q + r for q, r in zip(query, response)]
            return [output[1]["score"] for output in reward_pipeline(texts, **kwargs)]
        
        return reward_processor

    elif model_name == 'Gemma' and (data_name in ['UltraFeedback', 'Anthropic']):
        tokenizer = AutoTokenizer.from_pretrained('Ray2333/reward-model-Mistral-7B-instruct-Unified-Feedback')
        bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_compute_dtype=torch.bfloat16
                    )
        reward_model = AutoModelForSequenceClassification.from_pretrained(
                        'Ray2333/reward-model-Mistral-7B-instruct-Unified-Feedback',
                        num_labels=1, quantization_config=bnb_config, device_map="auto", low_cpu_mem_usage=True
                        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "right"
            reward_model.config.pad_token_id = reward_model.config.eos_token_id
        def reward_processor(query, response, kwargs):
            #tokens = [tokenizer.encode_plus(message, **kwargs) for message in response]
            chats = []
            print('Processing coversations...')
            for conv in tqdm(response):
                chats.append(tokenizer.apply_chat_template(conv, tokenize=False))
            print('Done!')
            tokens = tokenizer(chats, **kwargs)
            n = len(tokens['input_ids'])
            reward_arr = []
            for i in tqdm(range((n // 64) + 1)):
                batch_inp_ids = tokens['input_ids'][i*64 : min(n, (i+1)*64)]
                batch_att_msk = tokens['attention_mask'][i*64 : min(n, (i+1)*64)]
                with torch.no_grad():
                    reward_arr.extend(list(reward_model(batch_inp_ids.to(device), attention_mask=batch_att_msk.to(device)).logits.reshape(-1).cpu().detach().numpy()))
            return chats, reward_arr
        
        return reward_processor

def generate_response_from_model(model, tokenizer, dataset_name, model_name, model_id, output_path, small=True, test_size=10, info=''):
    #ds = get_dataset_iterator(dataset_name, model_name, id='test', small=small)
    generation_kwargs = {
                            "min_length": -1,
                            "top_k": 0.0,
                            "top_p": 1.0,
                            "do_sample": True,
                            "pad_token_id": tokenizer.eos_token_id,
                            "max_new_tokens" : 256
                        }
    responses = []
    result = {}
    #ds.set_format("pandas")
    #df = ds[:]
    df = pd.read_csv('./datasets/anthropic_hard_8k_test.csv.gz', compression='gzip')
    tokenizer = AutoTokenizer.from_pretrained('google/gemma-2b-it', add_bos_token=False)
    query_tokens = tokenizer(list(df['query']))
    mask = [len(q) <= 250 for q in query_tokens['input_ids']]
    df = df[mask]
    query_tensors = [q for q, flag in zip(query_tokens["input_ids"], mask) if flag]
    result["prompt"] = df["query"].tolist()
    n = len(query_tensors) if test_size is None else min(test_size, len(query_tensors))
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    for i in tqdm(range(n)):

        output = model.generate(
                                torch.tensor(query_tensors[i]).unsqueeze(dim=0).to(device), **generation_kwargs
                            ).squeeze().cpu()
        responses.append(output)
    
    result["response"] = [tokenizer.decode(responses[i]) for i in range(n)]
    result["prompt"] = [r for r in result["prompt"][:n]]
    df_results = pd.DataFrame(result)
    if small:
        df_results.to_csv(os.path.join(output_path, f'{dataset_name}_small_{model_name}_{model_id}_{info}_result_2k_v4.csv'), index=False)
    else:
        df_results.to_csv(os.path.join(output_path, f'{dataset_name}_{model_name}_{model_id}_{info}_result_2k_v4.csv'), index=False)


def evaluate_responses(reward_processor, dataset_name, model_name, model_id, output_path, small=True, info=''):
    if small:
        df = pd.read_csv(os.path.join(output_path, f'{dataset_name}_small_{model_name}_{model_id}_{info}_result_2k_v4.csv'))
    else:
        df = pd.read_csv(os.path.join(output_path, f'{dataset_name}_{model_name}_{model_id}_{info}_result_2k_v4.csv'))

    if model_name == 'GPT2':
        sent_kwargs = {"return_all_scores": True, "function_to_apply": "none", "batch_size": 16}
        #### reward obtained for query/response pairs for model
        df["reward"] = reward_processor(df["prompt"], df["response"], sent_kwargs)

    elif model_name == 'Gemma':
        arr = []
        for res in tqdm(df['response']):
            idx_user_start = res.find('<start_of_turn>user\n') + len('<start_of_turn>user\n')
            idx_user_end = res.find('<end_of_turn>\n')
            idx_model_start = res.find('<start_of_turn>model\n') + len('<start_of_turn>model\n')
            idx_model_end = res.rfind('<eos>')
            arr.append([{'role' : 'user', 'content' : res[idx_user_start : idx_user_end]},
                        {'role' : 'assistant', 'content' : res[idx_model_start : idx_model_end]}])
        sent_kwargs = {"padding": 'max_length', "truncation": True, "return_tensors": "pt", 'max_length': 600}
        #### reward obtained for query/response pairs for model
        df['chat_format'], df["reward"] = reward_processor(df["prompt"], arr, sent_kwargs)
        df.drop(columns=['response'], inplace=True)
    

    # store results in a dataframe
    if small:
        df.to_csv(os.path.join(output_path, f'{dataset_name}_small_{model_name}_{model_id}_{info}_reward_2k_v4.csv'), index=False)
    else:
        df.to_csv(os.path.join(output_path, f'{dataset_name}_{model_name}_{model_id}_{info}_reward_2k_v4.csv'), index=False)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', default='IMDB', type=str, help='Name of the dataset. Options: [\'IMDB\', \'UltraFeedback\', \'Anthropic\']')
    parser.add_argument('-m', '--model', default='GPT2', type=str, help='Name of the model. Options: [\'GPT2\', \'Gemma\']')
    parser.add_argument('-o', '--output', default='.', type=str, help='Output folder')
    parser.add_argument('-z', '--size', type=bool, default=True, help='Identifier whether use small or full dataset')
    parser.add_argument('-n', '--testsize', type=int, default=10, help='Number of prompts to test on')
    parser.add_argument('-i', '--id', type=str, default='apo', help='Active Learning method. Options: [\'apo\', \'random\', \'aedpo\']')
    parser.add_argument('-f', '--info', type=str, default='', help='Additional text to be added at the end of filenames outputted.')
    
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model, args.dataset, args.id, args.output, args.size, args.info)

    generate_response_from_model(model, tokenizer, args.dataset, args.model, args.id, args.output, args.size, args.testsize, args.info)

    del model
    del tokenizer
    reward_processor = load_true_reward_model(args.model, args.dataset)
    evaluate_responses(reward_processor, args.dataset, args.model, args.id, args.output, args.size, args.info)
    
