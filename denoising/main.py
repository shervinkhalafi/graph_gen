import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import eigsh
from tqdm import tqdm
from accelerate import Accelerator
import os
import argparse
from utils import *

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    #NEW#
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed for reproducibility",
    )

    parser.add_argument(
        "--noise_type",
        type=str,
        default='Gaussian',
        help="noise type, can be 'Gaussian', 'Rotation', or 'Digress'",
    )

    parser.add_argument(
        "--eps",
        type=float,
        default=0.5,
        help="noise level",
    )

    parser.add_argument(
        "--num_heads",
        type=int,
        default=4,
        help="number of attention heads",
    )

    parser.add_argument(
        "--num_layers",
        type=int,
        default=1,
        help="number of transformer layers",
    )

    parser.add_argument(
        "--block_sizes",
        type=str,
        default='[10, 5, 3, 2]',
        help="block sizes",
    )
    
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=400,
        help="number of epochs",
    )
     
    parser.add_argument(
        "--num_samples_per_epoch",
        type=int,
        default=128,
        help="number of samples per epoch",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="batch size",
    )
    
    
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


def main(args):
    # Define different block sizes
    block_sizes = eval(args.block_sizes)
    n = np.sum(np.array(block_sizes))

    # Set probabilities for intra-block and inter-block connections
    p_intra = 1.0  # High probability for connections within blocks
    q_inter = 0.0  # Low probability for connections between blocks

    # Set random seed for reproducibility
    rng = np.random.default_rng(args.seed)

    # Generate the adjacency matrix
    A = generate_sbm_adjacency(block_sizes, p_intra, q_inter)

    # Create dataset and dataloader
    dataset = AdjacencyMatrixDataset(A, num_samples_per_epoch=args.num_samples_per_epoch)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # h is the hidden dimension of the attention layer
    # the top k eigenvectors are used as denoising.
    h = 20
    k = 20


    skew = random_skew_symmetric_matrix(k)


    noise_type = args.noise_type
    eps = args.eps
    num_heads = args.num_heads
    num_layers = args.num_layers


    model = MultiLayerAttention(k, num_heads = num_heads, d_k = k, d_v = k, num_layers = num_layers, bias = True)


    criterion = nn.MSELoss()  # Mean Squared Error as the loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop


    n_epochs = args.num_epochs


    loss_hist = np.zeros(n_epochs)
    for epoch in range(n_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        for i, batch in enumerate(dataloader):
            # inputs = batch.unsqueeze(dim = 1)
            inputs = batch

            
        
            
            #add noise by rotating the eigenvectors
            if noise_type == 'Rotation':
                noisy_inputs, Vs, l_noisy = add_rotation_noise(inputs, eps, skew)
                    
            elif noise_type == 'Gaussian': 
                noisy_inputs, Vs, l_noisy = add_gaussian_noise(inputs, eps)
                    
            elif noise_type == 'Digress':
                noisy_inputs, Vs, l_noisy = add_digress_noise(inputs, eps)

                
            optimizer.zero_grad()  # Zero the gradients
            
            # Forward pass
            #the last layer's attention scores are used as the denoised adjacency matrix
            outputs = model(Vs)[1][-1]
            loss = criterion(outputs, inputs)
            
            # Backward pass
            loss.backward()
            
            # Update model parameters
            optimizer.step()
            
            running_loss += loss.item()
        
        loss_hist[epoch] = running_loss / len(dataloader)

        

    # Optionally, evaluate the model on a test set
    model.eval()  # Set model to evaluation mode

    # print(loss_hist)
    # plt.plot(loss_hist)
    l,V = eigsh(A, k = k, which = 'LM', maxiter = 10000)

    if noise_type == 'Rotation':
        A_noisy, V_noisy, l_noisy = add_rotation_noise(torch.tensor(A, dtype=torch.float32).unsqueeze(0), eps, skew)
        A_noisy = A_noisy.squeeze(0)    
        V_noisy = V_noisy.squeeze(0)
        l_noisy = l_noisy.squeeze(0)


    elif noise_type == 'Gaussian':
        A_noisy, V_noisy, l_noisy = add_gaussian_noise(A, eps)

        
    elif noise_type == 'Digress':
        A_noisy, V_noisy, l_noisy = add_digress_noise(A, eps)


    A_denoised = model(torch.unsqueeze(torch.FloatTensor(V_noisy), 0))[1][-1]
    A_denoised = np.array(A_denoised.detach())


    l_denoised, V_denoised = eigsh(A_denoised[0], k = k, which = 'LM', maxiter = 10000)


    A_denoised_eigvalsonly = V_noisy @ np.diag(l_denoised) @ V_noisy.T

    fig, axs = plt.subplots(3, 4, figsize=(10, 5))
    axs

    axs[0,0].imshow(A)
    axs[0,0].set_title('Original')

    axs[1,0].imshow(V)
    axs[1,0].set_title('Original Eigenvectors')

    axs[2,0].imshow(np.diag(l))
    axs[2,0].set_title('Original Eigenvalues')

    axs[0,1].imshow(A_noisy)
    axs[0,1].set_title('Noisy')

    axs[1,1].imshow(V_noisy)
    axs[1,1].set_title('Noisy Eigenvectors')   

    axs[2,1].imshow(np.diag(l_noisy))
    axs[2,1].set_title('Noisy Eigenvalues')

    axs[0,2].imshow(A_denoised[0])
    axs[0,2].set_title('Denoised')

    axs[1,2].imshow(V_denoised)
    axs[1,2].set_title('Denoised Eigenvectors')

    axs[2,2].imshow(np.diag(l_denoised)) 
    axs[2,2].set_title('Denoised Eigenvalues')

    axs[0,3].imshow(A_denoised_eigvalsonly)
    axs[0,3].set_title('Denoised using only denoised eigvals')

    axs[1,3].imshow(V_noisy)
    axs[1,3].set_title('Noisy Eigenvectors')

    axs[2,3].imshow(np.diag(l_denoised))
    axs[2,3].set_title('Denoised Eigenvalues')  


    axs.flatten()

    for i in range(12):
        axs.flatten()[i].axis('off')

    title = noise_type + ' Noise, eps = ' + str(eps) + ', Transformer Denoiser, num_layers = ' + str(num_layers) + ', num_heads = ' + str(num_heads)
    fig.suptitle(title, fontsize=16)

    fig.tight_layout()
    # Create results directory if it doesn't exist
    
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/' + title + '.png')
    plt.show()



if __name__ == "__main__":
    args = parse_args()
    main(args)