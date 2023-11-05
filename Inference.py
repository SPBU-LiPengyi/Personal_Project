import dataset_process
import unet_model
from config import cfgs
import torch
import diffusion_model
import matplotlib.pyplot as plt
import diffusion_model
from tqdm import tqdm

unet = unet_model.UNet(labels = True)
unet.load_state_dict(torch.load('./weight/epoch_200.pt'))
print("Load Weight file")
diffusion_model = diffusion_model.DiffusionModel()

classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
NUM_CLASSES = len(classes)
NUM_DISPLAY_IMAGES = 5
device = cfgs.device
image_shape = tuple(cfgs.image_shape)

torch.manual_seed(16)

plt.figure(figsize = (15, 15))
f, ax = plt.subplots(NUM_CLASSES, NUM_DISPLAY_IMAGES, figsize = (100, 100))

for c in tqdm(range(NUM_CLASSES), desc = "Generative..."):
    imgs = torch.randn((NUM_DISPLAY_IMAGES, 3) + image_shape).to(device)

    for i in reversed(range (diffusion_model.timesteps)):
        t = torch.full((1,), i, dtype = torch.long, device = device)
        labels = torch.tensor([c] * NUM_DISPLAY_IMAGES).resize_(NUM_DISPLAY_IMAGES, 1).float().to(device)
        imgs = diffusion_model.backward(x=imgs, t=t, model = unet.eval().to(device), labels = labels)
        
    for idx, img in enumerate(imgs):
        ax[c][idx].imshow(dataset_process.reverse_transform(img))
        ax[c][idx].set_title(f"Class:{classes[c]}", fontsize = 10)
    
plt.savefig('inference_image/Inference_2.png')
plt.show()