#%%
## Load the celeba_samples.npy and celeba_examples.npy files
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
## Set whitegrid
sns.set_style("whitegrid")

# Load the celeba_samples.npy and celeba_examples.npy files
celeba_pred = np.load("celeba_samples.npy").astype(np.int32)
celeba_true = np.load("celeba_examples.npy").astype(np.int32)

# Print min and max values of the loaded arrays
print("celeba_pred min:", celeba_pred.min(), "max:", celeba_pred.max())
print("celebatrue min:", celeba_true.min(), "max:", celeba_true.max())

print("celeba_pred shape:", celeba_pred.shape)
print("celeba_true shape:", celeba_true.shape)

#%%





# ## Plot the first 16 samples, along with their true images, using imshow
# def plot_celeba_samples(celeba_pred, celeba_true):
#     # Create a figure with 4 rows and 8 columns
#     fig, axs = plt.subplots(16, 2, figsize=(2*2, 2*16))
#     fig.suptitle("MNIST Samples and True Images", fontsize=16)

#     ## Plot the true, then the predicted image next to each other
#     for i in range(16):
#         # Plot the true image
#         # axs[i, 0].imshow(celeba_true[i, ...].reshape(28, 28, 3), cmap='gray')
#         axs[i, 0].imshow(celeba_true[i, ...,0].reshape(28, 28, 1), cmap='gray')
#         axs[i, 0].axis('off')
#         axs[i, 0].set_title("True Image", fontsize=12)

#         # Plot the predicted image
#         # axs[i, 1].imshow(celeba_pred[i, ...].reshape(28, 28, 3), cmap='gray')
#         axs[i, 1].imshow(celeba_pred[i, ..., 1].reshape(28, 28, 1), cmap='gray')
#         axs[i, 1].axis('off')
#         axs[i, 1].set_title("Predicted Image", fontsize=12)

#     # Adjust layout
#     plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
#     plt.show()

# good in dices = [12, 49, 43, 35]
# Corrected function to plot MNIST samples
def plot_celeba_samples(celeba_pred, celeba_true, nb_images_to_plot=50):
    # Create a figure with 16 rows and 2 columns
    fig, axs = plt.subplots(nb_images_to_plot, 2, figsize=(4, 2*nb_images_to_plot))
    fig.suptitle("CelebA Samples and True Images", fontsize=16)

    ## Plot the true, then the predicted image next to each other
    for a, i in enumerate(range(nb_images_to_plot)):
    # for a, i in enumerate(np.random.randint(0, min(celeba_pred.shape[0], celeba_true.shape[0]), size=16)):
    # for a, i in enumerate([12, 49, 43, 35]):
        # print("Plotting sample index:", i)
        # Plot the true image (from the green channel)
        axs[a, 0].imshow(celeba_true[i, :, :].reshape(32,32, -1), cmap='gray')  # Green channel
        axs[a, 0].axis('off')

        axs[a, 0].set_title(f"True Image: {i}", fontsize=12)

        # Plot the predicted image (from the red channel)
        axs[a, 1].imshow(celeba_pred[i, :, :].reshape(32,32, -1), cmap='gray')  # Red channel
        axs[a, 1].axis('off')
        axs[a, 1].set_title("Predicted Image", fontsize=12)

    # Adjust layout
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.tight_layout()
    plt.show()

    ## Save the figure
    fig.savefig("celeba_samples_and_true_images.png", dpi=300, bbox_inches='tight')

# Call the corrected function to plot the samples
plot_celeba_samples(celeba_pred/255, celeba_true/255)


print("First predicted image data:", celeba_pred[0, :, :].shape)


#%%


