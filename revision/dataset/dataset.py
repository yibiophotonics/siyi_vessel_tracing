import numpy as np
import torch
import torchvision.transforms as transforms
import torch.utils.data
from skimage import io
from pathlib import Path
import os

class DriveDataset():

    def __init__(self, root_dir, start, end, time, max_instance_number,
                 remove_overlay = False, usebackground = False,
                 transform_image = None, transform_all = None):
        """
        Args:
            root_dir (string): Directory with all the images
            start (int): start number of the image pair
            end (int): end number of the image pair
            time (int): length of the temporal sequence
            max_instance_number (int): maximum number of instances in the image, to avoid out of index
            remove_overlay (bool): whether to remove the overlay of vessels
            usebackground (bool): whether to use the background as one instance
            transform_image (transform): transform to be applied only on images but not masks
            transform_all (transform): transform to be applied on both images and masks
        """
        self.root_dir = root_dir
        self.start = start
        self.end = end
        self.time = time
        self.max_instance_number = max_instance_number
        self.remove_overlay = remove_overlay
        self.usebackground = usebackground
        self.transform_image = transform_image
        self.transform_all = transform_all

        # train.txt and test.txt are the files that contain the information of the dataset
        # the order is [subject id, sequence id, subject number, sequence number]
        path1 = Path(self.root_dir + '/' + 'train.txt')
        path2 = Path(self.root_dir + '/' + 'test.txt')

        if os.path.exists(path1):
            self.instance_information = np.loadtxt(path1)
        elif os.path.exists(path2):
            self.instance_information = np.loadtxt(path2)

        # if the instance_information is 1D, expand it to 2D
        if self.instance_information.ndim == 1:
            self.instance_information = np.expand_dims(self.instance_information, axis=0)

        # Generate the list of image pairs
        self.list = []
        for i in range(start, end+1):
            # find the row of the image pair
            row = np.argwhere(self.instance_information[:, 0] == i)
            subject_id = self.instance_information[row, 2].astype(int)
            sequence_id = self.instance_information[row, 3].astype(int)

            # find the row of first image patch (the most previous patch) in the temporal sequence
            start_row = row - (self.time - 1)
            start_subject_id = self.instance_information[start_row, 2].astype(int)
            start_sequence_id = self.instance_information[start_row, 3].astype(int)

            # make sure the subject and subsequences are consistent of the temporal sequence
            if subject_id == start_subject_id and sequence_id == start_sequence_id and start_row >= 0:
                self.list.append(i)
        print('The number of image patches is: ',len(self.list))

    def __len__(self):
        return len(self.list)

    def __getitem__(self, idx):
        # prepare the idx
        if torch.is_tensor(idx):
            idx = idx.tolist()

        idx = self.list[idx]
        row = np.argwhere(self.instance_information[:, 0] == idx)
        start = ((row - (self.time - 1)).astype(int)).item(0)
        end = ((row + 1).astype(int)).item(0)
        instance_number = self.instance_information[start: end, 1].astype(int)

        # the transformation from numpy to tensor
        Trans = transforms.Compose([
                transforms.ToTensor()
            ])

        # read the image sequence (T * 3 * H * W)
        # T order: [previous, latest]
        for t in range(self.time-1, -1, -1):
            img_name = self.root_dir + '/' +str(idx - t) + '_I.tif'
            image_t = io.imread(img_name)
            image_t = Trans(image_t)
            image_t = torch.unsqueeze(image_t, axis = 0)
            if self.transform_image:
                image_t = self.transform_image(image_t)
            if t == self.time-1:
                image = image_t
            else:
                image = torch.cat([image, image_t], axis=0)

        # read the semantic GT sequence (T * 2 * H * W)
        for t in range(self.time-1, -1, -1):
            semantic_name = self.root_dir + '/' + str(idx - t) + '_Semantic.tif'
            semantic_t = io.imread(semantic_name)
            semantic_t = Trans(semantic_t).bool()
            semantic_s = torch.cat([~semantic_t, semantic_t], axis=0)
            semantic_s = torch.unsqueeze(semantic_s, axis=0)
            if t == self.time-1:
                semantic = semantic_s
            else:
                semantic = torch.cat([semantic, semantic_s], axis=0)

        # read the instance GT sequence (T * max_instance_number * H * W)
        for t in range(self.time - 1, -1, -1):
            instance_number_t = instance_number[self.time - 1 - t]
            for i in range(instance_number_t):
                instance_name = self.root_dir + '/' + str(idx - t) + '_Instance_' + str(i + 1) + '.tif'
                single_instance = io.imread(instance_name)
                single_instance = np.expand_dims(single_instance, axis=2)
                if i == 0:
                    instance = single_instance
                else:
                    instance = np.concatenate([instance, single_instance],axis=2)

            if self.remove_overlay:
                ins_color_img = np.zeros((instance.shape[0], instance.shape[1]))
                for i in range(instance_number_t):
                    ins_color_img[instance[:, :, i] != 0] = i + 1
                instance = np.zeros((instance.shape[0], instance.shape[1], instance_number_t))
                for i in range(instance_number_t):
                    instance[ins_color_img== i + 1 , i] = 1

            # make instance same layer (add zeros layer) help for the stack
            for i in range(self.max_instance_number - instance_number_t):
                zero = np.zeros((instance.shape[0], instance.shape[1], 1))
                instance = np.concatenate([instance, zero],axis=2)

            if self.usebackground:
                instance_number[self.time - 1 - t] = instance_number[self.time - 1 - t] + 1
                background = np.expand_dims(semantic[:,:,0], axis=2)
                instance = np.concatenate([background,instance],axis=2)

            instance = Trans(instance)

            instance_t = torch.unsqueeze(instance, axis=0)
            if t == self.time-1:
                instance_f = instance_t
            else:
                instance_f = torch.cat([instance_f, instance_t], axis=0)

        instance = instance_f

        # torch channel * height * width
        # image T * 3 * H * W
        # semantic T * 2 * H * W
        # instance T * max_instance_number * H * W

        if self.transform_all:
            hieght = image.shape[2]
            width = image.shape[3]
            image = image.view(self.time * 3, hieght, width)
            semantic = semantic.view(self.time * 2, hieght, width)

            if self.usebackground:
                instance = instance.view(self.time * (self.max_instance_number + 1), hieght, width)
            else:
                instance = instance.view(self.time * self.max_instance_number, hieght, width)

            combine = torch.cat((image, semantic, instance), 0)
            combine = self.transform_all(combine)
            combine = combine.to(torch.float32)

            if self.usebackground:
                image, semantic, instance = torch.split(combine,
                                                        [3 * self.time, 2 * self.time, (self.max_instance_number + 1) * self.time])
            else:
                image, semantic, instance = torch.split(combine,
                                                        [3 * self.time, 2 * self.time, (self.max_instance_number) * self.time])

            image = image.view(self.time, 3, hieght, width)
            semantic = semantic.view(self.time, 2, hieght, width)
            if self.usebackground:
                instance = instance.view(self.time, (self.max_instance_number + 1), hieght, width)
            else:
                instance = instance.view(self.time, (self.max_instance_number), hieght, width)

        return image, semantic, instance, instance_number