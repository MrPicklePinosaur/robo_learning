import os
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image

class AnimeFaces(Dataset):
    def __init__(self, img_dir, preprocess):
        super().__init__()

        self.img_dir = img_dir
        self.preprocess = preprocess

    def __len__(self):
        png_files = [f for f in os.listdir(self.img_dir) if f.endswith('.png') and os.path.isfile(os.path.join(self.img_dir, f))]
        return len(png_files)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, str(index+1)+'.png')
        img_file = Image.open(img_path)

        if self.preprocess != None:
            img_file = self.preprocess(img_file)

        return img_file


