import numpy as np
from torchvision import transforms
from config import cfgs

transform = transforms.Compose([
    transforms.Resize(cfgs.image_shape),  # Resize the input image
    transforms.ToTensor(),  # Convert to torch tensor (scales data into [0,1])
    transforms.Lambda(lambda t: (t * 2) - 1),  # Scale data between [-1, 1]
])

reverse_transform = transforms.Compose([
    transforms.Lambda(lambda t: (t + 1) / 2),  # Scale data between [0,1]
    transforms.Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
    transforms.Lambda(lambda t: t * 255.),  # Scale data between [0.,255.]
    transforms.Lambda(lambda t: t.cpu().detach().numpy().astype(np.uint8)),  # Convert into an uint8 numpy array
    transforms.ToPILImage(),  # Convert to PIL image
])