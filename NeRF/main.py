import torch
from model import NeRFModel
from utils import ray_renderer
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader

def train(
        NeRFModel,
        optimizer,
        scheduler,
        device,
        data_loader,
        hn=0,
        hf=0.5,
        num_bins=192,
        H=400,
        W = 400,
        num_epochs=1000,
):
    # Supervised Training
    loss_history = []
    running_loss = 0.0
    total_batches = len(data_loader) * num_epochs
    progress_bar = tqdm(total=total_batches, desc="Training")
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        batch_count = 0
        
        for batch in data_loader:
            ray_origins, ray_directions, pixels = batch[:, :3].to(device), batch[:, 3:6].to(device), batch[:, 6:].to(device)

            # print(f"Pixels: {pixels.shape} ray_origins: {ray_origins.shape} ray_directions: {ray_directions.shape}")  # [B, 3]

            optimizer.zero_grad()

            regenerated_pixels = ray_renderer(
                NeRFModel,
                ray_origins,
                ray_directions,
                hn=hn,
                hf=hf,
                num_bins=num_bins,
            )

            loss = torch.nn.functional.mse_loss(regenerated_pixels, pixels)
            loss.backward()
            optimizer.step()
            
            current_loss = loss.item()
            loss_history.append(current_loss)
            epoch_loss += current_loss
            batch_count += 1
            
            # running loss for progress bar
            running_loss = epoch_loss / batch_count
            progress_bar.set_postfix({
                'epoch': f'{epoch+1}/{num_epochs}',
                'loss': f'{running_loss:.6f}',
                'lr': f'{scheduler.get_last_lr()[0]:.6f}'
            })
            progress_bar.update(1)

        scheduler.step()
        
        for idx in range(100):
            test(test_dataset, idx, hn, hf, num_bins, H, W) # test while we're training for monitoring

    progress_bar.close()
    return loss_history

@torch.no_grad()
def test(
    test_dataset,
    idx, 
    hn, 
    hf, 
    num_bins, 
    H, 
    W,
):
    """
    Test the model on subset of dataset using same hyperparams as training
    """
    ray_origins = test_dataset[idx * H * W: (idx + 1) * H * W, :3]
    ray_directions = test_dataset[idx*H*W:(idx + 1) * H * W, 3:6]

    test_data = []
    chunk_size = 10
    h = int(np.ceil(H / chunk_size))
    for i in range(h):
        ray_origins_chunked = ray_origins[i * W * chunk_size:(i + 1) * W * chunk_size].to(device)
        ray_directions_chunked = ray_directions[i * W * chunk_size:(i + 1) * W * chunk_size].to(device)
        reconstructed_pixels = ray_renderer(
            model,
            ray_origins_chunked,
            ray_directions_chunked,
            hn=hn,
            hf=hf,
            num_bins=num_bins,
        )
        test_data.append(reconstructed_pixels)

    test_image = torch.cat(test_data).data.cpu().numpy().reshape(H, W, 3)
    plt.figure()
    plt.imshow(test_image)
    plt.savefig(f"reconstructed_views/test_image_{idx}.png", bbox_inches='tight')
    plt.close()

def plot_loss_curve(loss_history):
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history)
    plt.title('Training Loss Over Time')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.grid(True)
    plt.savefig('loss_curve.png', bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_dataset = torch.from_numpy(np.load("training_data.pkl", allow_pickle=True))
    test_dataset = torch.from_numpy(np.load("testing_data.pkl", allow_pickle=True))

    model = NeRFModel(hidden_dim=256).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)   # karpathy constant
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2, 4, 8], gamma=0.5)

    data_loader = DataLoader(
        train_dataset,
        batch_size=1024,
        shuffle=True,
    )
    loss_history = train(
        model,
        optimizer,
        scheduler,
        device,
        data_loader,
        hn=0,
        hf=0.5,
        num_bins=192,
        H=400,
        W=400,
        num_epochs=25,
    )

    # Plot the loss curve
    plot_loss_curve(loss_history)
