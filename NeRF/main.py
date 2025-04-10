import torch
from model import NeRFModel
from utils import ray_renderer
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
import os
from pathlib import Path
import logging

from model import Cache

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def save_checkpoint(state, filename="checkpoint.pth"):
    torch.save(state, filename)

def load_checkpoint(model, optimizer, scheduler, filename="checkpoint.pth"):
    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        loss_history = checkpoint['loss_history']
        print(f"Loaded checkpoint '{filename}' (epoch {checkpoint['epoch']})")
        return start_epoch, loss_history
    else:
        print(f"No checkpoint found at '{filename}'")
        return 0, []

def ensure_directory_exists(*paths):
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)

def train(
        model,
        optimizer,
        scheduler,
        device,
        data_loader,
        hn=0,
        hf=0.5,
        num_bins=192,
        num_epochs=1000,
        checkpoint_dir="checkpoints",
        output_dir="output",
        checkpoint_frequency=5,
):
    ensure_directory_exists(
        checkpoint_dir,
        output_dir,
        os.path.join(output_dir, "reconstructed_views_truck"),
        os.path.join(output_dir, "loss_curves")
    )

    checkpoint_path = os.path.join(checkpoint_dir, "latest_checkpoint.pth")
    start_epoch, loss_history = load_checkpoint(model, optimizer, scheduler, checkpoint_path)
    
    # Supervised Training
    total_batches = len(data_loader) * (num_epochs - start_epoch)
    progress_bar = tqdm(total=total_batches, desc="Training")
    running_loss = 0.0
    
    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0.0
        batch_count = 0
        
        for batch in data_loader:
            ray_origins, ray_directions, pixels = batch[:, :3].to(device), batch[:, 3:6].to(device), batch[:, 6:].to(device)

            # Log data samples
            # print(f"Sample ray_origins: {ray_origins[:5]}")
            # print(f"Sample ray_directions: {ray_directions[:5]}")
            # print(f"Sample pixels: {pixels[:5]}")

            optimizer.zero_grad()

            regenerated_pixels = ray_renderer(
                model,
                ray_origins,
                ray_directions,
                hn=hn,
                hf=hf,
                num_bins=num_bins,
            )

            loss = ((regenerated_pixels - pixels) ** 2).sum()
            loss.backward()
            optimizer.step()
            
            # Check gradients
            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         logging.debug(f"Gradient for {name}: {param.grad.abs().mean()}")
            #     else:
            #         logging.debug(f"No gradient for {name}")

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
        
        # Save checkpoint
        if (epoch + 1) % checkpoint_frequency == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'loss_history': loss_history
            }
            save_checkpoint(checkpoint, os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth"))
            save_checkpoint(checkpoint, os.path.join(checkpoint_dir, "latest_checkpoint.pth"))

    progress_bar.close()
    return loss_history

@torch.no_grad()
def test(
    model,
    test_dataset,
    idx, 
    hn, 
    hf, 
    num_bins, 
    H, 
    W,
    output_dir="output"
):
    """
    Test the model to generate continuous novel views
    """
    ray_origins = test_dataset[idx * H * W: (idx + 1) * H * W, :3].to(device)
    ray_directions = test_dataset[idx*H*W:(idx + 1) * H * W, 3:6].to(device)
    reconstructed_pixels = ray_renderer(
                model,
                ray_origins,
                ray_directions,
                hn=hn,
                hf=hf,
                num_bins=num_bins,
            )

    test_image = reconstructed_pixels.data.cpu().numpy().reshape(H, W, 3).clip(0, 1)
    plt.figure()
    plt.imshow(test_image)
    plt.axis("off")
    plt.savefig(os.path.join(output_dir, "reconstructed_views_truck", f"test_image_{idx}.png"), bbox_inches='tight')
    plt.close()

def plot_loss_curve(loss_history, output_dir="output"):
    """Plot and save the loss curve"""
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history)
    plt.title('Training Loss Over Time')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "loss_curves", 'loss_curve.png'), bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    train_dataset = torch.from_numpy(np.load("training_data.pkl", allow_pickle=True))
    test_dataset = torch.from_numpy(np.load("testing_data.pkl", allow_pickle=True))

    model = NeRFModel(hidden_dim_pos=384, hidden_dim_dir=128).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)   # karpathy constant
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2, 4, 8], gamma=0.5)

    data_loader = DataLoader(
        train_dataset,
        batch_size=1024,
        shuffle=True,
    )
    
    checkpoint_dir = "checkpoints"
    output_dir = "output"
    
    loss_history = train(
        model,
        optimizer,
        scheduler,
        device,
        data_loader,
        hn=2.,
        hf=6.,
        num_bins=192,
        num_epochs=20,
        checkpoint_dir=checkpoint_dir,
        output_dir=output_dir,
        checkpoint_frequency=5
    )

    cache = Cache(model, 2.2, device, 192, 128)
    for i in range(200):
        test(
            model=cache,
            test_dataset=test_dataset,
            idx=i,
            hn=2.,
            hf=6.,
            num_bins=192,
            H=400,
            W=400,
            output_dir=output_dir,
        )

    # plot the loss curve
    plot_loss_curve(loss_history, output_dir='output2') 
