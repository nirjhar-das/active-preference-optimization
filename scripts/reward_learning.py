import os
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

gen = torch.Generator()
gen.manual_seed(167868171)

class BradleyTerryNN(nn.Module):
    def __init__(self, embedding_dim):
        super(BradleyTerryNN, self).__init__()
        self.fc = nn.Linear(embedding_dim, 1, bias=False)  # A single linear layer

    def forward(self, chosen_embedding, rejected_embedding):
        # Compute the logits for chosen and rejected items
        chosen_logits = self.fc(chosen_embedding)
        rejected_logits = self.fc(rejected_embedding)

        # Calculate the probability of chosen over rejected using the Bradley-Terry model
        prob = torch.sigmoid(chosen_logits - rejected_logits)
        return prob


def train_model(model, chosen_embeddings, rejected_embeddings, criterion, optimizer, num_samples, epochs=10):
    tr_loss = []
    
    for epoch in range(epochs):
        total_loss = 0
        
        prob = model(chosen_embeddings, rejected_embeddings)
        loss = criterion(prob, torch.ones(prob.shape, device=prob.device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        
        # for i in range(num_samples):
        #     # Forward pass to compute probability for each pair
        #     prob = model(chosen_embeddings[i], rejected_embeddings[i])

        #     # Compute the loss (assuming chosen is the positive class)
        #     loss = criterion(prob, torch.tensor([1.0], device=prob.device))  # Label for chosen is 1

        #     # Backward pass and optimization
        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()

        #     total_loss += loss.item()

        # Calculate average loss for the epoch
        avg_loss = total_loss / num_samples
        tr_loss.append(avg_loss)
        if ((epoch+1)%50) == 0:
            print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
    
    return tr_loss


def get_V(chosen_embeddings, rejected_embeddings, V_prev, theta) : 
    
    # Compute the difference between embeddings
    diffs = chosen_embeddings - rejected_embeddings

    V = V_prev.detach().clone()

    # Compute the sum of the outer product of diffs
    for _, diff in enumerate(diffs):
        # Compute the outer product of the diff vector with itself
        # mu = torch.sigmoid(torch.dot(theta, diff))
        # outer_product = mu * (1 - mu) * torch.ger(diff, diff)

        outer_product = torch.ger(diff, diff)

        # Update V matrix
        V += outer_product

    return V


def apo_selection(diff_embeddings, V, topK):
    norm_mat =  torch.matmul(torch.matmul(diff_embeddings, torch.inverse(V)), diff_embeddings.T)
    norm = torch.diag(norm_mat)
    idx = torch.argsort(norm, descending=True)

    return idx[0:topK]

def random_selection(n, topK):
    return torch.randint(0, n, (topK,), generator=gen)


def reward_learning(dataset_name, model_name, n_samples_per_epoch, num_epochs, output_path, id, small, info):
    if small:
        chosen_embeddings_np = np.loadtxt(os.path.join(output_path, f'{dataset_name}_small_{model_name}_embed_chosen_test.npy'))
        rejected_embeddings_np = np.loadtxt(os.path.join(output_path, f'{dataset_name}_small_{model_name}_embed_reject_test.npy'))
    else:
        chosen_embeddings_np = np.loadtxt(os.path.join(output_path, f'{dataset_name}_{model_name}_embed_chosen_test.npy'))
        rejected_embeddings_np = np.loadtxt(os.path.join(output_path, f'{dataset_name}_{model_name}_embed_reject_test.npy'))

    chosen_embeddings = torch.from_numpy(chosen_embeddings_np).float()
    rejected_embeddings = torch.from_numpy(rejected_embeddings_np).float()
    diff_embeddings = chosen_embeddings - rejected_embeddings

    del chosen_embeddings_np
    del rejected_embeddings_np

    embedding_dim = int(diff_embeddings.shape[1])
    n = int(diff_embeddings.shape[0])
    model = BradleyTerryNN(embedding_dim)
    optimizer = optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-5)
    criterion = nn.BCELoss()

    tr_loss = []
    chosen_embeddings_sub, rejected_embeddings_sub = [], []
    V = 1e-5 * torch.eye(chosen_embeddings.size(1))
    theta_ls = []

    for _ in tqdm(range(num_epochs)):
        if id == 'apo':
            selected_idx = apo_selection(diff_embeddings, V, n_samples_per_epoch)
        elif id == 'random':
            selected_idx = random_selection(n, n_samples_per_epoch)

        chosen_embeddings_sub.extend(chosen_embeddings[selected_idx])
        rejected_embeddings_sub.extend(rejected_embeddings[selected_idx])

        loss_history = train_model(model, torch.stack(chosen_embeddings_sub), 
                                   torch.stack(rejected_embeddings_sub), criterion, optimizer, num_samples = len(chosen_embeddings_sub))
        
        # loss_history = train_model(model, chosen_embeddings[selected_idx], 
        #                            rejected_embeddings[selected_idx], criterion, optimizer, num_samples = n_samples_per_epoch)
        V = get_V(chosen_embeddings[selected_idx], rejected_embeddings[selected_idx], V, model.fc.weight.detach().reshape(-1))

        theta_ls.append(model.fc.weight.detach().cpu().numpy().reshape(-1))
        tr_loss.append(loss_history)
    
    if small:
        torch.save(model.state_dict(), os.path.join(output_path, f'{dataset_name}_small_{model_name}_reward_{id}_{info}.pt'))
        np.savetxt(os.path.join(output_path, f'{dataset_name}_small_{model_name}_theta_{id}_{info}.npy'), np.array(theta_ls))
    
    else:
        torch.save(model.state_dict(), os.path.join(output_path, f'{dataset_name}_{model_name}_reward_{id}_{info}.pt'))
        np.savetxt(os.path.join(output_path, f'{dataset_name}_{model_name}_theta_{id}_{info}.npy'), np.array(theta_ls))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', default='IMDB', type=str, help='Name of the dataset. Options: [\'IMDB\', \'UltraFeedback\', \'Anthropic\']')
    parser.add_argument('-m', '--model', default='GPT2', type=str, help='Name of the model. Options: [\'GPT2\', \'LLAMA2\']')
    parser.add_argument('-o', '--output', default='.', type=str, help='Output folder')
    parser.add_argument('-e', '--epochs', default=10, type=int, help='Number of epochs')
    parser.add_argument('-s', '--samples_per_epoch', default=10, type=int, help='Number of samples per epoch')
    parser.add_argument('-i', '--id', type=str, default='apo', help='Identifier whether APO or Random')
    parser.add_argument('-z', '--size', type=bool, default=True, help='Identifier whether use small or full dataset')
    parser.add_argument('-f', '--info', type=str, default='', help='Additional text to be added at the end of filenames outputted.')

    args = parser.parse_args()

    reward_learning(args.dataset, args.model, args.samples_per_epoch, args.epochs, args.output, args.id, args.size, args.info)

    