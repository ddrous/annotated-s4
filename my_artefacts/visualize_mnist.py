#%%
## Load the mnist_samples.npy and mnist_examples.npy files
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
## Set whitegrid
sns.set_style("whitegrid")

# Load the mnist_samples.npy and mnist_examples.npy files
mnist_pred = np.load("mnist_samples.npy")
mnist_true = np.load("mnist_examples.npy")

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


# Corrected function to plot MNIST samples
def plot_mnist_samples(mnist_pred, mnist_true):
    # Create a figure with 16 rows and 2 columns
    fig, axs = plt.subplots(16, 2, figsize=(4, 32))
    fig.suptitle("MNIST Samples and True Images", fontsize=16)

    ## Plot the true, then the predicted image next to each other
    for i in range(16):
        # Plot the true image (from the green channel)
        axs[i, 0].imshow(mnist_true[i, :, :].reshape(28,28), cmap='gray')  # Green channel
        axs[i, 0].axis('off')

        axs[i, 0].set_title("True Image", fontsize=12)

        # Plot the predicted image (from the red channel)
        axs[i, 1].imshow(mnist_pred[i, :, :].reshape(28,28), cmap='gray')  # Red channel
        axs[i, 1].axis('off')
        axs[i, 1].set_title("Predicted Image", fontsize=12)

    # Adjust layout
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.show()


# Call the corrected function to plot the samples
plot_mnist_samples(mnist_pred, mnist_true)


print("First predicted image data:", mnist_pred[0, :, :].shape)


#%%