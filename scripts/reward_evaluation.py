import os
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




