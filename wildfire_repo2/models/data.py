import torch, pandas as pd, numpy as np, cv2
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms as T
import json

class ImageDS(Dataset):
    def __init__(self, csv_path, split, mean, std, size=224):
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df.split==split].reset_index(drop=True)
        self.size = size
        self.mean, self.std = mean, std
        self.train = (split=="train")
        self.tf = self._build_tf()

    def _build_tf(self):
        if self.train:
            return T.Compose([
                T.ToPILImage(),
                T.RandomResizedCrop(self.size, scale=(0.8,1.0)),
                T.RandomHorizontalFlip(0.5),
                T.ColorJitter(0.2,0.2,0.1,0.05),
                T.ToTensor(),
                T.Normalize(mean=self.mean, std=self.std)
            ])
        else:
            return T.Compose([
                T.ToPILImage(),
                T.Resize(int(self.size*1.15)),
                T.CenterCrop(self.size),
                T.ToTensor(),
                T.Normalize(mean=self.mean, std=self.std)
            ])

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        p = self.df.loc[i,'path']
        img = cv2.imread(p)
        if img is None:
            raise FileNotFoundError(p)
        img = img[:,:,::-1]  # BGR->RGB
        y = int(self.df.loc[i,'label'])
        img = self.tf(img)
        return img, y

def make_loaders(csv_path, batch_size=64, num_workers=4):
    with open("out/norm_and_class.json") as f:
        st = json.load(f)
    mean, std = st["mean"], st["std"]

    ds_tr = ImageDS(csv_path, "train", mean, std)
    ds_va = ImageDS(csv_path, "val",   mean, std)
    ds_te = ImageDS(csv_path, "test",  mean, std)

    # Weighted sampler (train only)
    labels = ds_tr.df['label'].values
    class_count = np.bincount(labels)
    weights = (1.0 / class_count)[labels]
    sampler = WeightedRandomSampler(weights, num_samples=len(labels), replacement=True)

    tr = DataLoader(ds_tr, batch_size=batch_size, sampler=sampler, pin_memory=True, num_workers=num_workers)
    va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
    te = DataLoader(ds_te, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
    return tr, va, te

