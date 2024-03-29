import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from torchvision.utils import make_grid


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Replace 'your_dataset_directory' with the path to your dataset
dataset = datasets.ImageFolder(root='/nfs/hpc/share/joshishu/DL_Project/BirdClass_Robustness/Adversarial_Wide/Mixed_version_Adv_Images', transform=transform)


def show_and_save_images_grid_with_labels(dataset, n_images=16, nrows=4, padding=2, save_path='nachani_sample_grid_with_labels.png'):
    """
    Displays and saves a grid of sample images from the dataset, with padding between images and class labels at the bottom.
    
    Parameters:
    - dataset: The dataset from which to draw the images.
    - n_images: Total number of images to display in the grid.
    - nrows: Number of rows in the image grid.
    - padding: Amount of padding between images in the grid.
    - save_path: Path to save the image grid.
    """
    # Randomly select images from the dataset
    indices = torch.randperm(len(dataset))[:n_images]
    images = [dataset[i][0] for i in indices]
    labels = [dataset.classes[dataset[i][1]] for i in indices]  
    
    # Create a grid of images with specified padding
    grid_img = make_grid(torch.stack(images), nrow=nrows, padding=padding, pad_value=1) 
    
    # Convert the grid to a NumPy array for plotting with matplotlib
    plt.figure(figsize=(12, 12))
    np_grid = grid_img.numpy().transpose((1, 2, 0))
    plt.imshow(np_grid)
    plt.axis('off')

    # Calculate dimensions for annotations
    cols = n_images // nrows if n_images % nrows == 0 else n_images // nrows + 1
    step_x = np_grid.shape[1] / cols
    step_y = np_grid.shape[0] / nrows

    # Adjust margin_y to place labels at the bottom of the images
    margin_y = step_y - (padding * 40)  # Increase or decrease as needed

    for idx, label in enumerate(labels):
        x = (idx % cols) * step_x + (step_x / 2)  
        y = (idx // cols + 1) * step_y - margin_y
        plt.text(x, y, label, color='black', backgroundcolor='white', fontsize=12, ha='center', va='top')

    plt.show()
    
    plt.savefig(save_path, bbox_inches='tight')

show_and_save_images_grid_with_labels(dataset, padding=5, save_path='sample_grid_with_labels.png')
