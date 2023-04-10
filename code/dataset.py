from torch.utils.data import Dataset
import os
from PIL import Image
import torchvision.transforms as transforms

class CancerDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.classes = os.listdir(root_dir)
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        self.samples = []
        
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            for filename in os.listdir(class_dir):
                filepath = os.path.join(class_dir, filename)
                self.samples.append((filepath, self.class_to_idx[class_name]))
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        filepath, label = self.samples[idx]
        image = Image.open(filepath).convert('L')
        tensor = transforms.ToTensor()(image)
        return tensor, label