import os
from PIL import Image
from collections import OrderedDict
import imageio.v2 as imageio
import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
import OpenEXR
import Imath
import torch.optim as optim
import matplotlib.pyplot as plt
import tqdm


def load_exr_image(exr_path):
    file = OpenEXR.InputFile(exr_path)
    
    
    dw = file.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

   
    float_type = Imath.PixelType(Imath.PixelType.FLOAT)
    depth_data = file.channel('Y', float_type) 

    
    depth_image = np.frombuffer(depth_data, dtype=np.float32).reshape(size[1], size[0])

    return depth_image

class RGBDImageDataset(Dataset):
    def __init__(self, rgb_dir, depth_dir, transform=None):
        self.rgb_dir = rgb_dir
        self.depth_dir = depth_dir
        self.transform = transform
        
        # Listar apenas arquivos de imagem
        self.rgb_images = sorted([f for f in os.listdir(rgb_dir) if f.endswith('.jpg')])
        self.depth_images = sorted([f for f in os.listdir(depth_dir) if f.endswith('.exr')])
        
        # Verificar se o número de arquivos nas duas pastas é igual
        assert len(self.rgb_images) == len(self.depth_images), "Número de imagens RGB e Depth não é o mesmo!"

    def __len__(self):
        return len(self.rgb_images)

    def __getitem__(self, idx):

        rgb_path = os.path.join(self.rgb_dir, self.rgb_images[idx])
        rgb_image = Image.open(rgb_path).convert('RGB')

        depth_path = os.path.join(self.depth_dir, self.depth_images[idx])
        depth_image = load_exr_image(depth_path) 

        depth_image = (depth_image - depth_image.min()) / (depth_image.max() - depth_image.min())

        depth_image = torch.from_numpy(depth_image).unsqueeze(0).float()

        if self.transform:
            rgb_image = self.transform(rgb_image)

        # Combinar RGB e profundidade em um único tensor com 4 canais
        rgbd_image = torch.cat((rgb_image, depth_image), dim=0)

        return rgbd_image

# # Definir transformações para as imagens RGB
# rgb_transform = transforms.Compose([
#     transforms.Resize((256, 256)),  
#     transforms.ToTensor(),          
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],  
#                          std=[0.229, 0.224, 0.225])
# ])

# dataset = RGBDImageDataset(
#     rgb_dir='/home/cesar/RCareWorld/template/Scene_dataset_final/Scene_dataset/run/rgb', 
#     depth_dir='/home/cesar/RCareWorld/template/Scene_dataset_final/Scene_dataset/run/depth', 
#     transform=rgb_transform
# )

# dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

# Função para calcular o tamanho da saída do recurso após convolução
def calc_out_size(w, h, kernel_size, padding=0, stride=1):
    width = (w - kernel_size + 2 * padding) // stride + 1
    height = (h - kernel_size + 2 * padding) // stride + 1
    return width, height

    
def get_encoder_network(observation_space):
    if len(observation_space) != 3:
        raise ValueError(f"Esperado que observation_space tenha 3 elementos (C, H, W), mas recebeu: {observation_space}")

    h, w = observation_space[1], observation_space[2]
    print(f"Altura: {h}, Largura: {w}")

    encoder = torch.nn.Sequential(
        torch.nn.Conv2d(observation_space[0], 64, kernel_size=4, stride=2, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
        torch.nn.ReLU(),
        torch.nn.Flatten(),
        torch.nn.Linear(256 * (h // 8) * (w // 8), 512),
        torch.nn.ReLU()
    )
    
    return encoder

class ImageAutoencoder(nn.Module):
    def __init__(self, encoder_network, output_dim):
        super(ImageAutoencoder, self).__init__()
        self.encoder_network = encoder_network
        self.fc = nn.Identity()
        # self.fc = nn.Linear(512, output_dim)  
       
        self.decoder = DecoderNetwork(4, 256, 256, latent_dim=512)  

    def forward(self, x, detach_encoder=False):
        if x.ndim == 3:
            x = x.unsqueeze(0)
        
        if x.shape[1] == 256:
            x = x.permute(0, 3, 1, 2)

        encoded = self.encoder_network(x)
        
        if detach_encoder:
            encoded = encoded.detach()

        # latent space
        latent_space = self.fc(encoded)

        decoded = self.decoder(latent_space)
        
        return decoded, latent_space 


class DecoderNetwork(nn.Module):
    def __init__(self, channels, height, width, latent_dim):
        super(DecoderNetwork, self).__init__()
        self.i_h, self.i_w = height, width  
        h, w = self.i_h, self.i_w
        h, w = calc_out_size(h, w, 8, stride=4)
        self.h, self.w = calc_out_size(h, w, 4, stride=2)
        self.fc = nn.Linear(latent_dim, 128)
        self.fc2 = nn.Linear(128, self.h * self.w * 32)
        self.decoder = nn.Sequential(OrderedDict([
            ('dec_cnn_trans_1', nn.ConvTranspose2d(32, 16, 4, stride=2)),
            ('dec_cnn_trans_elu_1', nn.ELU()),
            ('dec_cnn_trans_2', nn.ConvTranspose2d(16, channels, 8, stride=4))  # Ajustado para ter 4 canais
        ]))

    def forward(self, x):
        batch_size = x.shape[0]
        x = F.elu(self.fc(x))
        s = F.elu(self.fc2(x))
        s = s.view(batch_size, 32, self.h, self.w)
        s = self.decoder(s)
        output = F.interpolate(s, size=(self.i_h, self.i_w))
        return output




# for batch in dataloader:
#     print(batch.shape)  
#     break

# observation_space = batch.shape[1:]  


# encoder = get_encoder_network(observation_space)
# autoencoder = ImageAutoencoder(encoder_network=encoder, output_dim=64).to('cuda' if torch.cuda.is_available() else 'cpu')
        

# # optimizer and loss
# optimizer = optim.Adam(autoencoder.parameters(), lr=1e-3)
# criterion = nn.MSELoss()  

# train_losses = []

# # Train
# num_epochs = 5


# for epoch in range(num_epochs):
#     autoencoder.train()
#     running_loss = 0.0
#     pbar = tqdm.tqdm(total = 314, desc="Processando", unit="dados", ncols=100) 
#     for batch in dataloader:

#         batch = batch.to('cuda' if torch.cuda.is_available() else 'cpu')
        
#         optimizer.zero_grad()
#         outputs, latent_space = autoencoder(batch)  
        
#         loss = criterion(outputs, batch)  
#         loss.backward()
#         optimizer.step()
        
#         running_loss += loss.item()
#         pbar.update(1)
#     train_losses.append(running_loss / len(dataloader))

    
#     print(f"\nEpoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(dataloader)}\n")
    

# plt.figure(figsize=(10,5))
# plt.plot(range(1, num_epochs+1), train_losses, marker='o', label='Train Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Reconstruction Loss Over Epochs')
# plt.legend()
# plt.show()


# autoencoder.eval()
# latent_spaces = [] 

    # with torch.no_grad():
    #     for batch in dataloader:
    #         batch = batch.to('cuda' if torch.cuda.is_available() else 'cpu')

    #         reconstructed, latent_space = autoencoder(batch)  

    #         latent_spaces.append(latent_space.cpu())  

    #         # Plot image
    #         original_image = batch[0].cpu().numpy()  
    #         reconstructed_image = reconstructed[0].cpu().numpy()  

    #         original_image = original_image.transpose(1, 2, 0)
    #         reconstructed_image = reconstructed_image.transpose(1, 2, 0)

    #         original_image = (original_image - original_image.min()) / (original_image.max() - original_image.min())
    #         reconstructed_image = (reconstructed_image - reconstructed_image.min()) / (reconstructed_image.max() - reconstructed_image.min())

    #         plt.figure(figsize=(10, 5))
    #         plt.subplot(1, 2, 1)
    #         plt.imshow(original_image)
    #         plt.title("Original Image")
    #         plt.axis('off')

    #         plt.subplot(1, 2, 2)
    #         plt.imshow(reconstructed_image)
    #         plt.title("Reconstructed Image")
    #         plt.axis('off')

    #         plt.show()
            
    #         break  
    
# torch.save(encoder.state_dict(), 'encoder_trained.pth')
# print("Encoder salvo em 'encoder_trained.pth'")
