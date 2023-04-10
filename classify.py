import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

# Set the path to the image you want to predict
image_path_no = 'images/example_no.png'
image_path_yes = 'images/example_yes.png'

# Define the transformation
transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])

# Load the image and apply the transformation
image = Image.open(image_path_no)
image_tensor_no = transform(image).unsqueeze(0)

# Load the image and apply the transformation
image = Image.open(image_path_yes)
image_tensor_yes = transform(image).unsqueeze(0)

# Load the saved model
model = torch.load('cancer_classifier.pt')

# Set the model to evaluation mode
model.eval()

# Make the prediction
with torch.no_grad():
    outputs_no = model(image_tensor_no)
    outputs_yes = model(image_tensor_yes)

label1 = "No cancer \npredicted yes: %.2f\n predicted no: %.2f" % (outputs_no[0][1].item(), outputs_no[0][0].item())
label2 = "With cancer \npredicted yes: %.2f\n predicted no: %.2f" % (outputs_yes[0][1].item(), outputs_yes[0][0].item())

# Load the images
image1 = Image.open(image_path_no)
image2 = Image.open(image_path_yes)

# Create a figure with two subplots
fig, axs = plt.subplots(1, 2)

# Set the first subplot to show the first image and label
axs[0].imshow(image1)
axs[0].set_title(label1)

# Set the second subplot to show the second image and label
axs[1].imshow(image2)
axs[1].set_title(label2)

# Show the figure
plt.show()