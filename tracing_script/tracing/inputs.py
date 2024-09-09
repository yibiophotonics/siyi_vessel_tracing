import numpy as np
from skimage import io
import torch
import torch.nn.functional
import torchvision.transforms as transforms

np.random.seed(0)

class Image():
    def __init__(self, name, mask_name = None, transforms = None):
        self.image = io.imread(name)[:,:,:3]
        self.transforms = transforms

        if mask_name != None:
            self.mask = io.imread(mask_name)
            if len(self.mask.shape) == 3:
                self.mask = self.mask[:,:,0]
            self.mask[self.mask > 0] = 1
        else:
            self.mask = np.ones((self.image.shape[0], self.image.shape[1]))

    def __sizeof__(self):
        return self.image.shape[0], self.image.shape[1]

    def GenerateRegion(self, sp, size):
        region_row = slice(sp[0] - int(size / 2), sp[0] +int(size / 2), 1)
        region_column = slice(sp[1] - int(size / 2), sp[1]+int(size / 2), 1)
        return region_row, region_column

    def ExtractMask(self, slicer_row, slicer_column):

        size = (slicer_row.stop - slicer_row.start, slicer_column.stop - slicer_column.start)
        mask_enlarge = np.pad(self.mask, size, 'constant', constant_values = (0, 0))
        slicer_row = slice(slicer_row.start + size[0], slicer_row.stop + size[0], 1)
        slicer_column = slice(slicer_column.start + size[1], slicer_column.stop + size[1], 1)
        mask_piece = mask_enlarge[slicer_row, slicer_column]

        return mask_piece.astype(bool)

    def ExtractPiece(self, slicer_row, slicer_column):
        # Extract Piece
        size = slicer_row.stop - slicer_row.start
        image_enlarge = np.pad(self.image, ((size, size), (size, size), (0, 0)), 'constant', constant_values = 0 )
        slicer_row = slice(slicer_row.start + size, slicer_row.stop + size, 1)
        slicer_column = slice(slicer_column.start + size, slicer_column.stop + size, 1)

        image_piece = image_enlarge[slicer_row, slicer_column, :]
        trans = transforms.Compose([transforms.ToTensor()])
        image_piece = trans(image_piece)

        # Transform
        image_piece = self.transforms(image_piece)
        image_piece = image_piece.to(torch.float32)

        return image_piece
    def ExtractBase(self, slicer_row, slicer_column):
        image_piece = torch.zeros((3, slicer_row.stop-slicer_row.start,slicer_column.stop-slicer_column.start))
        if slicer_row.start < 0:
            slicer_row = slice(0, slicer_row.stop, 1)
        elif slicer_row.stop > self.image.shape[0]:
            slicer_row = slice(slicer_row.start, self.image.shape[0], 1)

        if slicer_column.start < 0:
            slicer_column = slice(0, slicer_column.stop, 1)
        elif slicer_column.stop > self.image.shape[1]:
            slicer_column = slice(slicer_column.start, self.image.shape[1], 1)
        return slicer_row, slicer_column

    def LoadSequence(self, point_list, size):
        time = len(point_list)
        for i in range(time): # time is updated in last part
            coordinate = point_list[i,:]
            region_row, region_column = self.GenerateRegion(coordinate, size)
            current_img = self.ExtractPiece(region_row,region_column)
            current_img_t = torch.unsqueeze(current_img, axis=0)
            if i == 0:
                img = current_img_t
            else:
                img = torch.cat([img, current_img_t], axis=0)

        return img, time # time is updated one