import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_patches_on_image(image_path, patch_size=256, output_path='patch_visualization.png'):
    image = Image.open(image_path)
    width, height = image.size

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image, cmap='gray' if image.mode == 'L' else None)

    # Draw rectangles to represent patches
    for top in range(0, height, patch_size):
        for left in range(0, width, patch_size):
            if top + patch_size <= height and left + patch_size <= width:
                rect = patches.Rectangle((left, top), patch_size, patch_size, linewidth=2, edgecolor='r', facecolor='none')
                ax.add_patch(rect)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()

# Example usage
image_path = '/Users/pierregabrielbibalsobeaux/Documents/python/VUB_git/VUB_Image_denoising/DIV2K_train_HR.nosync/0001.png'
output_path = 'patch_visualization.png'
draw_patches_on_image(image_path, patch_size=256, output_path=output_path)
