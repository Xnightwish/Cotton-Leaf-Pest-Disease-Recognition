import os
import json
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class PestDataset(Dataset):
    def __init__(self, txt_file, root_dir, transform=None):
        self.root_dir = root_dir
        with open(txt_file, 'r') as file:
            self.image_labels = [line.strip().split() for line in file.readlines()]
        self.transform = transform
        

    def __len__(self):
        return len(self.image_labels)

    def __getitem__(self, idx):
        img_name, label = self.image_labels[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        label = int(label)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label



# Example usage
if __name__ == "__main__":
    data_dir = './datasets/cotton_insect/'
    batch_size = 32
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_dataset = PestDataset(txt_file=os.path.join(data_dir, 'train.txt'), root_dir=os.path.join(data_dir, 'train'), transform=transform)
    val_dataset = PestDataset(txt_file=os.path.join(data_dir, 'val.txt'), root_dir=os.path.join(data_dir, 'val'), transform=transform)
    test_dataset = PestDataset(txt_file=os.path.join(data_dir, 'test.txt'), root_dir=os.path.join(data_dir, 'test'), transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Iterate through the data
    for images, labels in train_loader:
        print(images.size(), labels.size())
        break