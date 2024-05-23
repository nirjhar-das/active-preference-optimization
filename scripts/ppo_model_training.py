import os
import torch
from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse

tqdm.pandas()

from transformers import AutoTokenizer, BitsAndBytesConfig
from transformers import AutoModelForSequenceClassification
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from peft.tuners.lora import LoraLayer

from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from datasets import Dataset


def get_dataset_iterator(dataset_name, model_name, id='train', small=True):
    if dataset_name == 'UltraFeedback':
        if small:
            df = pd.read_csv(f'datasets/ultrafeedback_{id}_small.csv.gz', compression='gzip')
        else:
            df = pd.read_csv(f'datasets/ultrafeedback_{id}.csv.gz', compression='gzip')
    elif dataset_name == 'Anthropic':
        if small:
            df = pd.read_csv(f'datasets/anthropic_hard_8k_{id}.csv.gz', compression='gzip')
        else:
            df = pd.read_csv(f'datasets/anthropic_hard_8k_{id}.csv.gz', compression='gzip')
    elif dataset_name == 'IMDB':
        df = pd.read_csv(f'datasets/path_to_IMDB_{id}.csv')
    
    if model_name == 'GPT2':
        true_model_name = 'lvwerra/gpt2-imdb'

    elif model_name == 'Gemma':
        true_model_name = "google/gemma-2b-it"
    
    df.rename(columns={"query": "prompt"}, inplace=True)
    df = df.sample(frac=1, random_state=9432).reset_index(drop=True)
    tokenizer = AutoTokenizer.from_pretrained(true_model_name, add_bos_token=False)
    # tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.padding_side = "right"

    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(sample["prompt"])
        sample["prompt"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = Dataset.from_pandas(df)
    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")
    return ds


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


def prepare_model_for_PPO(model_name, id=None):
    if model_name == 'GPT2':
        true_model_name = 'lvwerra/gpt2-imdb'
        model = AutoModelForCausalLMWithValueHead.from_pretrained(true_model_name)

    elif model_name == 'Gemma':
        true_model_name = "google/gemma-2b-it"
        # true_model_name = f"./output/Anthropic_small_Gemma_final_model_{id}_le_v4"
        bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_compute_dtype=torch.bfloat16
                    )

        loraconfig = LoraConfig(
                        r=8,
                        lora_alpha=32,
                        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
                        lora_dropout=0.05, # Set Lora drop out properly
                        bias="none",
                        task_type="CAUSAL_LM"
                    )

        model = AutoModelForCausalLMWithValueHead.from_pretrained(
                    true_model_name,
                    use_safetensors=True,
                    quantization_config=bnb_config,
                    trust_remote_code=True,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                    peft_config=loraconfig
                )
        
        
        #model = prepare_model_for_kbit_training(model)

        #model = get_peft_model(model, loraconfig)
        # for name, module in model.named_modules():
        #     if isinstance(module, LoraLayer):
        #         module = module.to(torch.float16)
        #     if 'norm' in name:
        #         module = module.to(torch.float16)
        #     if hasattr(module, 'weight'):
        #         if module.weight.dtype == torch.float32:
        #             module = module.to(torch.float16)
        #     if 'lm_head' in name or 'embed_tokens' in name:
        #         if hasattr(module, 'weight'):
        #             if module.weight.dtype == torch.float32:
        #                 module = module.to(torch.float16)

    tokenizer = AutoTokenizer.from_pretrained(true_model_name, add_bos_token=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        model.config.pad_token_id = model.config.eos_token_id

    return model, tokenizer

def prepare_reward_model_for_PPO(model_name, dataset_name, output_path, id, info=''):
    if model_name == 'GPT2':
        true_model_name = 'lvwerra/gpt2-imdb'
        reward_model = AutoModelForSequenceClassification.from_pretrained(true_model_name)
    elif model_name == 'Gemma':
        true_model_name = "google/gemma-2b-it"
        bnb_config = BitsAndBytesConfig(
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_compute_dtype=torch.bfloat16
                    )

        reward_model = AutoModelForSequenceClassification.from_pretrained(
                            true_model_name,
                            num_labels = 1,
                            use_safetensors=True,
                            quantization_config=bnb_config,
                            trust_remote_code=True,
                            device_map="auto",
                        )

    reward_model_tokenizer = AutoTokenizer.from_pretrained(true_model_name, add_bos_token=False)
    reward_model_tokenizer.pad_token = reward_model_tokenizer.eos_token
    reward_model.config.pad_token_id = reward_model.config.eos_token_id
    reward_model_tokenizer.padding_side = "right"

    #update reward model
    theta = np.loadtxt(os.path.join(output_path, f'{dataset_name}_small_{model_name}_theta_{id}_{info}.npy'))[-1, :].reshape(reward_model.score.weight.shape)
    with torch.no_grad():
        reward_model.score.weight.copy_(torch.from_numpy(theta).float())

    return reward_model, reward_model_tokenizer

def prepare_PPO_config(model_name):
    if model_name == 'GPT2':
        config = PPOConfig(
                    model_name="lvwerra/gpt2-imdb",
                    mini_batch_size = 32,
                    learning_rate=1.41e-5,
                    gradient_accumulation_steps=4
                )
    elif model_name == 'Gemma':
        config = PPOConfig(
                    model_name="google/gemma-2b-it",
                    batch_size = 16,
                    mini_batch_size = 8,
                    learning_rate=1e-4
                )
    
    return config

def prepare_PPO_trainer(config, model, tokenizer, dataset, collator_fn=collator):
    ppo_trainer = PPOTrainer(config=config, model=model, ref_model=None, tokenizer=tokenizer, dataset=dataset, data_collator=collator_fn)
    # if ppo_trainer.accelerator.num_processes == 1:
    #     device = 0 if torch.cuda.is_available() else "cpu"
    
    generation_kwargs = {
                            "min_length": -1,
                            "top_k": 80,
                            "top_p": 1,
                            "do_sample": True,
                            "pad_token_id": tokenizer.eos_token_id,
                            "temperature": 1.1,
                            "max_new_tokens": 256
                        }
    # gen_len = 400
    # generation_kwargs["max_new_tokens"] = gen_len
    
    return ppo_trainer, generation_kwargs


def train_model(ppo_trainer, generation_kwargs, tokenizer, reward_model, reward_model_tokenizer, output_path, model_name, data_name, id, max_iter=-1, small=True, info=''):
    for i, batch in tqdm(enumerate(ppo_trainer.dataloader), total=max_iter):
        if i < 100:
            continue
        if (max_iter != -1) and (i + 1 - 100 > max_iter):
            break
        # print(f'Batch {i} running...')
        query_tensors = batch["input_ids"]
        m = max([q.shape[0] for q in query_tensors])
        if m > 250:
            print(i, m)
            max_iter += 1
            continue
        #### Get response from LLM
        response_tensors = []
        ppo_trainer.model.gradient_checkpointing_disable()
        for query in query_tensors:
            response = ppo_trainer.generate(query, **generation_kwargs)
            response_tensors.append(response.squeeze())
        batch["response"] = [tokenizer.decode(r) for r in response_tensors]

        #### Compute reward
        if model_name == 'GPT2':
            texts = [q + r for q, r in zip(batch["query"], batch["response"])]
            texts_tokens = reward_model_tokenizer(texts, return_tensors='pt')
            with torch.no_grad():
                outputs = reward_model(**texts_tokens)
            rewards = list(outputs.logits)
        elif model_name == 'Gemma':
            texts_tokens = reward_model_tokenizer(batch['response'], return_tensors='pt', padding='max_length', max_length=500, truncation=True)
            with torch.no_grad():
                rewards = reward_model(texts_tokens['input_ids'], attention_mask=texts_tokens['attention_mask']).logits
                rewards = [r for r in rewards]

        #### Run PPO step
        ppo_trainer.model.gradient_checkpointing_enable()
        ppo_trainer.step(query_tensors, response_tensors, rewards)
        #stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        # ppo_trainer.log_stats(stats, batch, rewards)
        del query_tensors
        del response_tensors
        del rewards
        del texts_tokens

        if ((i + 1) % 5) == 0:
            # print('Saving Checkpoint...')
            if not small:
                if not os.path.exists(os.path.join(output_path, 'checkpoints', f'{data_name}_{model_name}_ckpt_{i+1-100}_model_{id}_{info}_v4')):
                    os.mkdir(os.path.join(output_path, 'checkpoints', f'{data_name}_{model_name}_ckpt_{i+1-100}_model_{id}_{info}_v4'))
                ppo_trainer.model.save_pretrained(os.path.join(output_path, 'checkpoints', f'{data_name}_{model_name}_ckpt_{i+1-100}_model_{id}_{info}_v4'))
            else:
                if not os.path.exists(os.path.join(output_path, 'checkpoints', f'{data_name}_{model_name}_ckpt_{i+1-100}_model_{id}_{info}_v4')):
                    os.mkdir(os.path.join(output_path, 'checkpoints', f'{data_name}_small_{model_name}_ckpt_{i+1-100}_model_{id}_{info}_v4'))
                ppo_trainer.model.save_pretrained(os.path.join(output_path, 'checkpoints', f'{data_name}_small_{model_name}_ckpt_{i+1-100}_model_{id}_{info}_v4'))
            # print('Checkpoint saved!')

        # print(f'Batch {i} completed!')
    print('Done!!!')

    #save
    if not small:
        if not os.path.exists(os.path.join(output_path, f'{data_name}_{model_name}_final_model_{id}_{info}_v4')):
            os.mkdir(os.path.join(output_path, f'{data_name}_{model_name}_final_model_{id}_{info}_v4'))
        ppo_trainer.model.save_pretrained(os.path.join(output_path, f'{data_name}_{model_name}_final_model_{id}_{info}_v4'))
    else:
        if not os.path.exists(os.path.join(output_path, f'{data_name}_{model_name}_final_model_{id}_{info}_v4')):
            os.mkdir(os.path.join(output_path, f'{data_name}_small_{model_name}_final_model_{id}_{info}_v4'))
        ppo_trainer.model.save_pretrained(os.path.join(output_path, f'{data_name}_small_{model_name}_final_model_{id}_{info}_v4'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', default='IMDB', type=str, help='Name of the dataset. Options: [\'IMDB\', \'UltraFeedback\', \'Anthropic\']')
    parser.add_argument('-m', '--model', default='GPT2', type=str, help='Name of the model. Options: [\'GPT2\', \'Gemma\']')
    parser.add_argument('-o', '--output', default='.', type=str, help='Output folder')
    parser.add_argument('-i', '--id', type=str, default='apo', help='Identifier whether APO or Random')
    parser.add_argument('-z', '--size', type=bool, default=True, help='Identifier whether use small or full dataset')
    parser.add_argument('-f', '--info', type=str, default='', help='Additional text to be added at the end of filenames outputted.')
    parser.add_argument('-n', '--maxiter', type=int, default=-1, help='Maximum PPO Iteration')

    args = parser.parse_args()

    ds = get_dataset_iterator(args.dataset, args.model, small=args.size, id='train')
    model, tokenizer = prepare_model_for_PPO(args.model, args.id)
    reward_model, rm_tokenizer = prepare_reward_model_for_PPO(args.model, args.dataset, args.output, args.id, args.info)
    ppo_config = prepare_PPO_config(args.model)
    ppo_trainer, gen_kwargs = prepare_PPO_trainer(ppo_config, model, tokenizer, ds)
    train_model(ppo_trainer, gen_kwargs, tokenizer, reward_model, rm_tokenizer, args.output, args.model, args.dataset, args.id, args.maxiter, args.size, args.info)