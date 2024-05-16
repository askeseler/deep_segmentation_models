import os
import glob

from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.functional import one_hot

def preprocess_img(img, scale):
    w, h = img.size
    _h = int(h * scale)
    _w = int(w * scale)
    assert _w > 0
    assert _h > 0

    _img = img.resize((_w, _h))
    _img = np.array(_img)

    if len(_img.shape) == 2:  ## gray/mask images
        _img = np.expand_dims(_img, axis=-1)

    _img = _img.transpose((2, 0, 1))
    return _img

class DirDataset(Dataset):
    def __init__(self, img_dir, mask_dir, scale=4, grayscale = False, n_channels = 3, n_classes = 1):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.scale = scale
        self.grayscale = grayscale

        try:
            self.ids = [s.split('.')[0] for s in os.listdir(self.img_dir)]
        except FileNotFoundError:
            self.ids = []
        self._h = None
        self._w = None
        self.n_classes = n_classes
        self.n_channels = n_channels
        
    def __len__(self):
        return len(self.ids)

    def preprocess(self, img):
        if type(self._w) == type(None):#Determine target size with first image; All subsequent images will have same size (resize)
            w, h = img.size
            _h = int(h * self.scale)
            _w = int(w * self.scale)
            assert _w > 0
            assert _h > 0
            self._h = _h
            self._w = _w

        _img = img.resize((self._w, self._h))
        _img = np.array(_img)

        if len(_img.shape) == 2:  ## gray/mask images
            _img = np.expand_dims(_img, axis=-1)

        # hwc to chw
        _img = _img.transpose((2, 0, 1))
        return _img

    def __getitem__(self, i):
        idx = self.ids[i]#filename
        img_files = glob.glob(os.path.join(self.img_dir, idx+'.*'))
        mask_files = glob.glob(os.path.join(self.mask_dir, idx+'.*'))

        assert len(img_files) == 1, f'{idx}: {img_files}'
        assert len(mask_files) == 1, f'{idx}: {mask_files}'

        img = Image.open(img_files[0])
        mask = Image.open(mask_files[0])
        mask = mask.convert('L')
        assert img.size == mask.size, f'{img.shape} # {mask.shape}'

        if self.grayscale:
            img = img.convert("L")

        img = self.preprocess(img)

        if not self.grayscale:
            img = img[:self.n_channels]

        if img.max() > 1:
            img = img / 255

        mask = self.preprocess(mask)
        mask[mask>=self.n_classes] = self.n_classes -1
        if self.n_classes == 1:# binarize
            mask = (mask > 0).astype(int)
            mask = torch.from_numpy(mask).float()
        else:
            #binmask = (mask > 0).astype(int)#debug TODO remove
            #binmask = torch.from_numpy(binmask).float()#TODO remove
 
            mask = one_hot(torch.tensor(mask, dtype=torch.int64), self.n_classes)
            mask = mask.squeeze().permute(2,0,1).float()
            #mask = mask[:,:,:]#conceptualize: background (int=0) is not a class but the absence of all other classes
            #mask[0] = mask[0]#only flower
            #mask[1] = mask[0]
            #mask[2] = mask[0]
            #mask[3] = mask[0]

        ret = torch.from_numpy(img).float(), mask
        return ret
