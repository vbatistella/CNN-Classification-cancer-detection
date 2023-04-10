import os
from PIL import Image
import torchvision.transforms as transforms

def pre_process():
    # Set the directory path where the images are stored
    directory = 'raw/'
    processed = 'data/'

    # Set the new size of the image
    new_size = (32, 32)

    # Loop through each image file in the directory
    count = 0
    classes = ["yes", "no"]
    for c in classes:
        # Define subdirectiories
        sub_dir = os.path.join(directory, c)
        proc_sub_dir = os.path.join(processed, c)

        for filename in os.listdir(sub_dir):
            # Open the image and convert it to grayscale
            image = Image.open(os.path.join(sub_dir, filename)).convert('L')
            
            # Resize the image
            image = image.resize(new_size)
            
            # Save the image with a new filename
            new_filename = str(count) + '.png'
            image.save(proc_sub_dir+"/"+new_filename)
            count += 1