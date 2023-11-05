import dataset_process
import unet_model
from config import cfgs
import torch
from tqdm import tqdm
import time
import diffusion_model
import matplotlib.pyplot as plt
import numpy as np
import method
import torchvision
import os

# 使用GPU运算
device = cfgs.device

# 初始化模型
diffusion_model = diffusion_model.DiffusionModel()

# load model：UNet
unet = unet_model.UNet(labels=True)
unet.to(device)

def main():
    trainset = torchvision.datasets.CIFAR10(root = '.\data', train = True, download = True, transform = dataset_process.transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=cfgs.batch_size, shuffle = True, num_workers = 0, drop_last = True)

    testset = torchvision.datasets.CIFAR10(root = '.\data', train = False, download = True, transform = dataset_process.transform)
    testloader = torch.utils.data.DataLoader(testset, cfgs.batch_size, shuffle = False, num_workers = 0, drop_last = True)

    # set optimizer type
    optimizer = torch.optim.Adam(unet.parameters(), lr = cfgs.learning_rate)

    # 每个epoch的loss存储
    epoch_loss = []
    epoch_loss_val = []
    
    for epoch in range(cfgs.num_epochs):
        print("正在进行第" + str(epoch+1) + "个epoch，总共" + str(cfgs.num_epochs) + "个epoch.")
        time.sleep(0.0001)
        mean_epoch_loss = []
        mean_epoch_loss_val = []
    
        for batch, label in  tqdm(trainloader, desc="Training"):
            t = torch.randint(0, diffusion_model.timesteps, (cfgs.batch_size,)).long().to(device)
            batch = batch.to(device)
            batch_noisy, noise = diffusion_model.forward(batch, t, device)
            predicted_noise = unet(batch_noisy, t, labels = label.reshape(-1, 1).float().to(device))
            
            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(noise, predicted_noise)
            mean_epoch_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            
        for batch, label in tqdm(testloader, desc="Testing"):
            t = torch.randint(0, diffusion_model.timesteps, (cfgs.batch_size,)).long().to(device)
            batch = batch.to(device)
            
            batch_noisy, noise = diffusion_model.forward(batch, t, device)
            predicted_noise = unet(batch_noisy, t, labels = label.reshape(-1, 1).float().to(device))
            
            loss = torch.nn.functional.mse_loss(noise, predicted_noise)
            mean_epoch_loss_val.append(loss.item())
        
        epoch_loss.append(np.mean(mean_epoch_loss))
        epoch_loss_val.append(np.mean(mean_epoch_loss_val))
        
        if (epoch+1) % cfgs.PRINT_FREQUENCY == 0:
            print("-----------保存权重文件----------")
            print(f"Epoch: {epoch + 1} | Train Loss {np.mean(mean_epoch_loss)} | Val Loss {np.mean(mean_epoch_loss_val)}")
            
            # 每个Epoch中的loss平均相加来，但是第二个epoch也平均了第一个epoch中的loss，但是 也能代表Loss下降。
            if cfgs.VERBOSE:
                with torch.no_grad():
                    method.plot_noise_prediction(noise[0], predicted_noise[0])
                    method.plot_noise_distribution(noise, predicted_noise)
            
            torch.save(unet.state_dict(), f"./weight/epoch_{epoch+1}.pt")

    folder_path = './new_list'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
    file_path_loss = os.path.join(folder_path, "loss.txt")
    file_path_loss_val = os.path.join(folder_path, "loss_val.txt")
    with open(file_path_loss, "w") as file:
        for item in epoch_loss:
            file.write(str(item) + "\n")

        print("数组已写入文件:", file_path_loss)
    
    with open(file_path_loss_val, "w") as file:
        for item in epoch_loss_val:
            file.write(str(item) + "\n")

        print("数组已写入文件:", file_path_loss_val)
    
    # 输出损失函数 并画图
    print("Train loss of diffusion model")
    # print(mean_epoch_loss)
    # 绘图
    plt.plot(list(range(1, len(epoch_loss) + 1)), epoch_loss, label='Train Loss', marker='o', linestyle='-')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig('./trian_image/loss.png')
    plt.show()

    print("Test loss of diffusion model")
    # print(mean_epoch_loss_val)

    plt.plot(list(range(1, len(epoch_loss_val) + 1)), epoch_loss_val, label='Test Loss', marker='o', linestyle='-')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Testing Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig('./trian_image/loss_val.png')
    plt.show()

if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    training_time = (end_time - start_time) / 3600
    print(f"训练经过的时间: {training_time} 小时")