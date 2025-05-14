
import wandb


def log_images(images, epoch, name):
    wandb.log({
        f"{name}Epoch {epoch + 1}": [wandb.Image(img) for img in images[:100].cpu()],
    })

def log_metrics(loss, epoch, logdet_loss, gaussian_loss):
    wandb.log({
        "Loss": loss,
        "Epoch": epoch,
        "Logdet loss": logdet_loss,
        "Gaussian loss": gaussian_loss,
    })
wandb.finish()
wandb.init(project = "TAR_NSF")
import torch
import torchvision as tv
import os
import utils
import pathlib
import nflows
from time import *

from tarnsfinv import Model

dataset = 'mnist'
num_classes = 10
img_size = 28
channel_size = 1
num_bins = 2

# we use a small model for fast demonstration, increase the model size for better results
patch_size = 4
channels = 256
blocks = 8
layers_per_block = 4
# try different noise levels to see its effect

noise_std = 0.1

batch_size = 256
lr = 5e-4
# increase epochs for better results
epochs = 100
sample_freq = 1
notebook_output_path = pathlib.Path('runs/notebook')
if torch.cuda.is_available():
    device = 'cuda' 
elif torch.backends.mps.is_available():
    device = 'mps' # if on mac
else:
    device = 'cpu' # if mps not available
print(f'using device {device}')

fixed_noise = torch.randn(num_classes * 10, (img_size // patch_size)**2, channel_size * patch_size ** 2, device=device)
fixed_y = torch.arange(num_classes, device=device).view(-1, 1).repeat(1, 10).flatten()
transform = tv.transforms.Compose([
    tv.transforms.Resize((img_size, img_size)),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize((0.5,), (0.5,))
])
data = tv.datasets.MNIST('.', transform=transform, train=True, download=True)
data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True)

model = Model(in_channels=channel_size, img_size=img_size, patch_size=patch_size, 
              channels=channels, num_blocks=blocks, layers_per_block=layers_per_block, num_bins = num_bins,
              num_classes=num_classes).to(device)

optimizer = torch.optim.AdamW(model.parameters(), betas=(0.9, 0.95), lr=lr, weight_decay=1e-4)
lr_schedule = utils.CosineLRSchedule(optimizer, len(data_loader), epochs * len(data_loader), 1e-6, lr)

model_name = f'{patch_size}_{channels}_{blocks}_{layers_per_block}_{noise_std:.2f}'
sample_dir = notebook_output_path / f'{dataset}_samples_{model_name}'
ckpt_file = notebook_output_path / f'{dataset}_model_{model_name}.pth'
sample_dir.mkdir(exist_ok=True, parents=True)

log_images(model.unpatchify(fixed_noise[:100]), epoch = -1, name = "noise") #saving the noisy images

start_time = time()
for epoch in range(epochs):
    losses = 0
    counter = 0
    for x, y in data_loader:
        x = x.to(device) #256 1 28 28
        
        eps = noise_std * torch.randn_like(x)
        x = x + eps
        y = y.to(device)
        optimizer.zero_grad()
        z, outputs, logdets = model(x, y)
        gaussian_loss, logdet_loss, loss = model.get_loss(z, logdets)
        loss.backward()
        optimizer.step()
        lr_schedule.step()
        losses += loss.item()
        if counter %10 == 0:
            print("loss", loss.item(), "logdet_loss", logdet_loss.item(), "gaussian_loss", gaussian_loss.item())
        log_metrics(loss, epoch, logdet_loss, gaussian_loss)
        counter += 1

    print(f"epoch {epoch} lr {optimizer.param_groups[0]['lr']:.6f} loss {losses / len(data_loader):.4f}")
    print('layer norms', ' '.join([f'{z.pow(2).mean():.4f}' for z in outputs]))

    if (epoch + 1) % sample_freq == 0:
        with torch.no_grad():
            samples = model.reverse(fixed_noise, fixed_y)
            #print(samples.size())
        tv.utils.save_image(samples, sample_dir / f'samples_{epoch:03d}.png', normalize=True, nrow=10)
        tv.utils.save_image(model.unpatchify(z[:100]), sample_dir / f'latent_{epoch:03d}.png', normalize=True, nrow=10)
        log_images(samples, epoch, "sample_images")
        log_images(model.unpatchify(z[:100]), epoch, "latent_images")
        print('sampling complete')
    
    end_time = time()
    print(f"Time taken: {(end_time - start_time)/(epoch+1):.2f} seconds")
torch.save(model.state_dict(), ckpt_file)
