#%%
## Load the mnist_samples.npy and mnist_examples.npy files
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
## Set whitegrid
sns.set_style("whitegrid")

# Load the mnist_samples.npy and mnist_examples.npy files
mnist_pred = np.load("mnist_samples.npy").astype(np.int32)
mnist_true = np.load("mnist_examples.npy").astype(np.int32)

# Print min and max values of the loaded arrays
print("mnist_pred min:", mnist_pred.min(), "max:", mnist_pred.max())
print("mnist_true min:", mnist_true.min(), "max:", mnist_true.max())

print("mnist_pred shape:", mnist_pred.shape)
print("mnist_true shape:", mnist_true.shape)

#%%





# ## Plot the first 16 samples, along with their true images, using imshow
# def plot_mnist_samples(mnist_pred, mnist_true):
#     # Create a figure with 4 rows and 8 columns
#     fig, axs = plt.subplots(16, 2, figsize=(2*2, 2*16))
#     fig.suptitle("MNIST Samples and True Images", fontsize=16)

#     ## Plot the true, then the predicted image next to each other
#     for i in range(16):
#         # Plot the true image
#         # axs[i, 0].imshow(mnist_true[i, ...].reshape(28, 28, 3), cmap='gray')
#         axs[i, 0].imshow(mnist_true[i, ...,0].reshape(28, 28, 1), cmap='gray')
#         axs[i, 0].axis('off')
#         axs[i, 0].set_title("True Image", fontsize=12)

#         # Plot the predicted image
#         # axs[i, 1].imshow(mnist_pred[i, ...].reshape(28, 28, 3), cmap='gray')
#         axs[i, 1].imshow(mnist_pred[i, ..., 1].reshape(28, 28, 1), cmap='gray')
#         axs[i, 1].axis('off')
#         axs[i, 1].set_title("Predicted Image", fontsize=12)

#     # Adjust layout
#     plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
#     plt.show()

# good in dices = [12, 49, 43, 35]
# Corrected function to plot MNIST samples
def plot_mnist_samples(mnist_pred, mnist_true, nb_images_to_plot=50):
    # Create a figure with 16 rows and 2 columns
    fig, axs = plt.subplots(nb_images_to_plot, 2, figsize=(4, 2*nb_images_to_plot))
    fig.suptitle("MNIST Samples and True Images", fontsize=16)

    ## Plot the true, then the predicted image next to each other
    for a, i in enumerate(range(nb_images_to_plot)):
    # for a, i in enumerate(np.random.randint(0, min(mnist_pred.shape[0], mnist_true.shape[0]), size=16)):
    # for a, i in enumerate([12, 49, 43, 35]):
        # print("Plotting sample index:", i)
        # Plot the true image (from the green channel)
        axs[a, 0].imshow(mnist_true[i, :, :].reshape(28,28, -1), cmap='gray')  # Green channel
        axs[a, 0].axis('off')

        axs[a, 0].set_title(f"True Image: {i}", fontsize=12)

        # Plot the predicted image (from the red channel)
        axs[a, 1].imshow(mnist_pred[i, :, :].reshape(28,28, -1), cmap='gray')  # Red channel
        axs[a, 1].axis('off')
        axs[a, 1].set_title("Predicted Image", fontsize=12)

    # Adjust layout
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.tight_layout()
    plt.show()

    ## Save the figure
    fig.savefig("mnist_samples_and_true_images.png", dpi=300, bbox_inches='tight')

# Call the corrected function to plot the samples
plot_mnist_samples(mnist_pred/256, mnist_true/256)


print("First predicted image data:", mnist_pred[0, :, :].shape)


#%%
60000-128*468


## Devise a dataloader and a visualisation to show the last 50 images from the MNIST test dataset

#%%
# Load the MNIST test dataset
from torchvision import datasets, transforms
mnist_test = datasets.MNIST(root='../data', train=False, download=True, transform=transforms.ToTensor())
# Print the length of the test dataset
print("Length of MNIST test dataset:", len(mnist_test)) 
# Print the first image and its label
print("First image shape:", mnist_test[0][0].shape)
print("First image label:", mnist_test[0][1])   

# Visualize the last 50 images from the MNIST test dataset
def visualize_last_50_mnist_images(mnist_test, labels=None):
    # Create a figure with 10 rows and 5 columns
    fig, axs = plt.subplots(10, 5, figsize=(5, 10))
    fig.suptitle("MNIST Test Images", fontsize=16)

    # Loop through the last 50 images
    for i in range(50):
        # img, label = mnist_test[-(i + 1)]  # Get the last 50 images
        img = mnist_test[-(i + 1)]  # Get the last 50 images
        if labels is not None:
            label = labels[-(i + 1)]

        if label == 9:
            axs[i // 5, i % 5].imshow(img.squeeze(), cmap='gray')
        axs[i // 5, i % 5].axis('off')
        # axs[i // 5, i % 5].set_title(f"Label: {label}", fontsize=12)

    # Adjust layout
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.show()


## Setup a daloader with a batch size of 50 and plot the 78th batch
from torch.utils.data import DataLoader
mnist_loader = DataLoader(mnist_test, batch_size=50, shuffle=False)
# Get the 78th batch from the dataloader
for i, (images, labels) in enumerate(mnist_loader):
    if i == 77:  # 78th batch (0-indexed)
        print("Batch index:", i)
        print("Batch images shape:", images.shape)
        print("Batch labels shape:", labels.shape)

        mnist_test = images.numpy()  # Convert images to numpy array for visualization
        visualize_last_50_mnist_images(mnist_test, labels.numpy())
        # break

# Call the function to visualize the last 50 images
# visualize_last_50_mnist_images(mnist_test)