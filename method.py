import dataset_process
import matplotlib.pyplot as plt

def plot_noise_prediction(noise, predicted_noise):
    plt.figure(figsize=(15,15))
    f, ax = plt.subplots(1, 2, figsize=(5, 5))
    ax[0].imshow(dataset_process.reverse_transform(noise))
    ax[0].set_title(f"ground truth noise", fontsize=10)
    ax[1].imshow(dataset_process.reverse_transform(predicted_noise))
    ax[1].set_title(f"predicted noise", fontsize=10)
    plt.show()

def plot_noise_distribution(noise, predicted_noise):
    plt.hist(noise.cpu().detach().numpy().flatten(), density=True, alpha=0.8, label="ground truth noise")
    plt.hist(predicted_noise.cpu().detach().numpy().flatten(), density=True, alpha=0.8, label="predicted noise")
    plt.legend()
    plt.show()