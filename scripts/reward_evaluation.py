import os
import argparse
import numpy as np


def get_accuracy(model_name, dataset_name, output_path, small=True, id='apo', info=''):

    if small:
        theta = np.loadtxt(os.path.join(output_path, f'{dataset_name}_small_{model_name}_theta_{id}_{info}.npy'))[-1, :]

        embed_chosen = np.loadtxt(os.path.join(output_path, f'{dataset_name}_small_{model_name}_embed_chosen_test.npy'))
        embed_rejected = np.loadtxt(os.path.join(output_path, f'{dataset_name}_small_{model_name}_embed_reject_test.npy'))

    else:
        theta = np.loadtxt(os.path.join(output_path, f'{dataset_name}_{model_name}_theta_{id}_{info}.npy'))[-1, :]

        embed_chosen = np.loadtxt(os.path.join(output_path, f'{dataset_name}_{model_name}_embed_chosen_test.npy'))
        embed_rejected = np.loadtxt(os.path.join(output_path, f'{dataset_name}_{model_name}_embed_reject_test.npy'))
    
    reward_diff = np.dot(embed_chosen - embed_rejected, theta)
    accuracy = np.mean(reward_diff > 0) * 100

    print(f'{id} accuracy: {accuracy}%')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', default='IMDB', type=str, help='Name of the dataset. Options: [\'IMDB\', \'UltraFeedback\', \'Anthropic\']')
    parser.add_argument('-m', '--model', default='GPT2', type=str, help='Name of the model. Options: [\'GPT2\', \'LLAMA2\']')
    parser.add_argument('-o', '--output', default='.', type=str, help='Output folder')
    parser.add_argument('-i', '--id', type=str, default='apo', help='Identifier whether APO or Random')
    parser.add_argument('-z', '--size', type=bool, default=True, help='Identifier whether use small or full dataset')
    parser.add_argument('-f', '--info', type=str, default='', help='Additional text to be added at the end of filenames outputted.')

    args = parser.parse_args()

    get_accuracy(args.model, args.dataset, args.output, args.size, args.id, args.info)
