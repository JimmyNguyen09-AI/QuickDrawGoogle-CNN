# author : Trung Thanh Nguyen(Jimmy) | 09/12/2004  | ng.trungthanh04@gmail.com
from torch.utils.data import Dataset
import torch
from torchvision.transforms import Compose,ToTensor

import numpy as np
import os
class QuickDrawDataset(Dataset):
    def __init__(self,root,is_train,nums_images_per_class = 10000,ratio = 0.8,transform = None):
        self.categories =  ["airplane","angel","apple","axe","bat","book","boomerang","camera","cup","fish","flower","mushroom","radio","sun","sword"]
        self.images = []
        self.labels = []
        self.transform = transform
        for label, cate in enumerate(self.categories):
            file_path = os.path.join(root,"full_numpy_bitmap_{}.npy".format(cate))
            images = np.load(file_path)
            idx_image = np.arange(nums_images_per_class)
            percent = int(nums_images_per_class*ratio)
            if is_train:
                data_idx = idx_image[:percent]
            else:
                data_idx = idx_image[percent:]
            self.images.append(images[data_idx])
            self.labels.extend([label]*len(data_idx))
        self.images = np.concatenate(self.images, axis=0)
        self.labels = np.array(self.labels)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        image = self.images[idx].astype(np.float32)/255.0
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        image = torch.tensor(image.reshape((1,28,28)),dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        return image,label
if __name__ == '__main__':
    dataset = QuickDrawDataset(root = "./dataset_Quick_Draw",is_train = False,nums_images_per_class=500)
    image,label = dataset[100]
    print(type(image))
    print(image.shape)
