import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


from torchvision import transforms
from sklearn.decomposition import PCA
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from cosmos_tokenizer.image_lib import ImageTokenizer
import time
from PIL import Image
import OpenEXR
from torch.utils.data import Dataset
import Imath
import tqdm


# Configuração do IncrementalPCA
batch_size = 256  # Tamanho do batch
n_components = 1024  # Número desejado de componentes principais

pca = PCA(n_components=n_components)

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
        rgb_image = cv.imread(rgb_path)

        depth_path = os.path.join(self.depth_dir, self.depth_images[idx])
        depth_image = load_exr_image(depth_path) 

        depth_image = (depth_image - depth_image.min()) / (depth_image.max() - depth_image.min())

        depth_image = torch.from_numpy(depth_image).unsqueeze(0).float()

        rgb_image = torch.tensor(rgb_image, dtype=torch.float32)

        rgb_image /= 255

        rgb_image = rgb_image.permute(2, 0, 1)
        rgb_image = rgb_image.unsqueeze(0)

        # Combinar RGB e profundidade em um único tensor com 4 canais
        #rgbd_image = torch.cat((rgb_image,), dim=0)

        return rgb_image

# Definição do Autoencoder
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 4096),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, latent_dim)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 4096),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, input_dim),
            nn.Sigmoid()  # Para normalização (se os dados estiverem normalizados entre 0 e 1)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed
    

dataset = RGBDImageDataset(
    rgb_dir='/home/adrolab/Cesar/RCareWorld/template/Data/dataset/Scene_dataset/run/rgb', 
    depth_dir='/home/adrolab/Cesar/RCareWorld/template/Data/dataset/Scene_dataset/run/depth', 
    transform=True
)

print("Carregando tokenizer")
model_name8 = "Cosmos-Tokenizer-CI8x8"
encoder8 = ImageTokenizer(checkpoint_enc=f'pretrained_ckpts/{model_name8}/encoder.jit').to('cuda')
decoder8 = ImageTokenizer(checkpoint_dec=f'pretrained_ckpts/{model_name8}/decoder.jit').to('cuda')

# Definir transformações para as imagens RGB
rgb_transform = transforms.Compose([
    transforms.Resize((256, 256)),  
    transforms.ToTensor(),          
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  
                         std=[0.229, 0.224, 0.225])
])

# Configuração
input_dim = 16384 
latent_dim = 1024  # Tamanho da representação latente

dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)


# Train
num_epochs = 1
train_losses = []

# Perform PCA fitting on the entire dataset (usually done outside the epoch loop)
# Assuming dataloader provides batches of `batch_size` samples
all_data = []
print("calculating PCA")
pbar = tqdm.tqdm(total=len(dataloader), ncols=100)
for batch in dataloader:
    all_data.append(batch.view(batch.shape[0], -1))  # Flatten to 2D
    pbar.update(1)
all_data = torch.cat(all_data, dim=0).numpy()  # Convert to numpy for PCA
pca.fit(all_data)


running_loss = 0.0
pbar = tqdm.tqdm(total=len(dataloader), unit="batch", ncols=100)

for batch in dataloader:
    batch = batch.to('cpu')  # PCA works with CPU data; ensure the data is on CPU
    batch_flat = batch.view(batch.shape[0], -1).numpy()  # Flatten images for PCA

    # Encode with PCA
    latents_batch = pca.transform(batch_flat)

    # Optional: Decode to reconstruct and calculate reconstruction loss
    reconstructed_batch = pca.inverse_transform(latents_batch)
    reconstructed_batch = torch.tensor(reconstructed_batch).view_as(batch)  # Reshape back to original dims

    # Calculate reconstruction loss (e.g., Mean Squared Error)
    loss = ((batch - reconstructed_batch) ** 2).mean().item()
    running_loss += loss


    pbar.close()
    train_losses.append(running_loss / len(dataloader))
    print(f"Reconstruction Loss: {running_loss / len(dataloader):.6f}\n")
