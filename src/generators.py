# -*- coding: utf-8 -*-
# Generator classes to dynamically load the data.
# COPYRIGHT: TU Delft, Netherlands. 2022.
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

class Volumes(Dataset):
    """
    Loads input planning and repeat images.
    Assumes that the data is stored in a folder (e.g. /data) 
    inside a single .h5 file.
    """

    def __init__(self, list_IDs, file_handle, masks=False, xkey='pCT',
        ykey='rCT', xmask='pCST', ymask='rCST', maxv=1, minv=0):
        """
        Parameters:
            list_IDs: a list with the file identifiers.
            ikey, okey: input and output h5 dataset names.
            scale: dict with max and min values.
        """
        self.list_IDs = list_IDs
        self.fh = file_handle
        self.masks = masks
        self.xkey  = xkey
        self.ykey  = ykey
        self.xmask = xmask
        self.ymask = ymask
        self.maxv  = maxv
        self.minv  = minv

    def __len__(self):
        # upper bound of sample index
        return len(self.list_IDs)

    def __getitem__(self, index):
        # select sample
        ID = self.list_IDs[index]

        # load data and normalize to [0,1]
        planning = np.expand_dims(np.transpose(self.fh[self.xkey][:,:,:,ID]), -1)
        planning = (planning - self.minv) / (self.maxv - self.minv)
        repeat   = np.expand_dims(np.transpose(self.fh[self.ykey][:,:,:,ID]), -1)
        repeat   = (repeat - self.minv) / (self.maxv - self.minv)

        if not self.masks:
            return planning, repeat

        else:    
            planning_mask = np.expand_dims(np.transpose(self.fh[self.xmask][:,:,:,ID]), -1)
            repeat_mask   = np.expand_dims(np.transpose(self.fh[self.ymask][:,:,:,ID]), -1)
            # planning[planning_mask==4] = 0.25
            # repeat[repeat_mask==4] = 0.25
            return planning, repeat, planning_mask, repeat_mask
