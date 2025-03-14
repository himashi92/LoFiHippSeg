import monai
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from monai.transforms import MapTransform
from torch.utils.data import Dataset, DataLoader
import os
from glob import glob
import nibabel as nib
import pickle
from tqdm import tqdm
import random
from torchvision.transforms import transforms
from scipy import ndimage
import matplotlib.pyplot as plt
import SimpleITK as sitk
"""functions to correctly pad or crop non uniform sized MRI (before batching in the dataloader).
"""
import random
import numpy as np


device = 'cuda' if torch.cuda.is_available() else 'cpu'
def pkload(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)


def pad_or_crop_image(image, image2, seg=None, target_size=(128, 144, 144)):
    c, z, y, x = image.shape
    z_slice, y_slice, x_slice = [get_crop_slice(target, dim) for target, dim in zip(target_size, (z, y, x))]
    image = image[:, z_slice, y_slice, x_slice]
    image2 = image2[:, z_slice, y_slice, x_slice]
    if seg is not None:
        seg = seg[:, z_slice, y_slice, x_slice]

    todos = [get_left_right_idx_should_pad(size, dim) for size, dim in zip(target_size, [z, y, x])]
    padlist = [(0, 0)]  # channel dim
    for to_pad in todos:
        if to_pad[0]:
            padlist.append((to_pad[1], to_pad[2]))
        else:
            padlist.append((0, 0))
    image = np.pad(image, padlist)
    image2 = np.pad(image2, padlist)
    if seg is not None:
        seg = np.pad(seg, padlist)

        return image, image2, seg
    return image, image2


def get_left_right_idx_should_pad(target_size, dim):
    if dim >= target_size:
        return [False]
    elif dim < target_size:
        pad_extent = target_size - dim
        left = random.randint(0, pad_extent)
        right = pad_extent - left
        return True, left, right


def get_crop_slice(target_size, dim):
    if dim > target_size:
        crop_extent = dim - target_size
        left = random.randint(0, crop_extent)
        right = crop_extent - left
        return slice(left, dim - right)
    elif dim <= target_size:
        return slice(0, dim)


def normalize(image):
    """Basic min max scaler.
    """
    min_ = np.min(image)
    max_ = np.max(image)
    scale = max_ - min_
    image = (image - min_) / scale
    return image


def irm_min_max_preprocess(image, low_perc=1, high_perc=99):
    """Main pre-processing function used for the challenge (seems to work the best).

    Remove outliers voxels first, then min-max scale.

    Warnings
    --------
    This will not do it channel wise!!
    """

    non_zeros = image > 0
    low, high = np.percentile(image[non_zeros], [low_perc, high_perc])
    image = np.clip(image, low, high)
    image = normalize(image)
    return image


def zscore_normalise(img: np.ndarray) -> np.ndarray:
    slices = (img != 0)
    img[slices] = (img[slices] - np.mean(img[slices])) / np.std(img[slices])
    return img


def zscore_normalise_channel_wise(img: np.ndarray) -> np.ndarray:
    # channel 1
    slices = (img[0] != 0)
    img[0][slices] = (img[0][slices] - np.mean(img[0][slices])) / np.std(img[0][slices])

    slices = (img[1] != 0)
    img[1][slices] = (img[1][slices] - np.mean(img[1][slices])) / np.std(img[1][slices])

    slices = (img[2] != 0)
    img[2][slices] = (img[2][slices] - np.mean(img[2][slices])) / np.std(img[2][slices])

    slices = (img[3] != 0)
    img[3][slices] = (img[3][slices] - np.mean(img[3][slices])) / np.std(img[3][slices])
    return img


def remove_unwanted_background(image, threshold=1e-5):
    """Use to crop zero_value pixel from MRI image.
    """
    dim = len(image.shape)
    non_zero_idx = np.nonzero(image > threshold)
    min_idx = [np.min(idx) for idx in non_zero_idx]
    # +1 because slicing is like range: not inclusive!!
    max_idx = [np.max(idx) + 1 for idx in non_zero_idx]
    bbox = tuple(slice(_min, _max) for _min, _max in zip(min_idx, max_idx))
    return image[bbox]


def random_crop2d(*images, min_perc=0.5, max_perc=1.):
    """Crop randomly but identically all images given.

    Could be used to pass both mask and image at the same time. Anything else will
    throw.

    Warnings
    --------
    Only works for channel first images. (No channel image will not work).
    """
    if len(set(tuple(image.shape) for image in images)) > 1:
        raise ValueError("Image shapes do not match")
    shape = images[0].shape
    new_sizes = [int(dim * random.uniform(min_perc, max_perc)) for dim in shape]
    min_idx = [random.randint(0, ax_size - size) for ax_size, size in zip(shape, new_sizes)]
    max_idx = [min_id + size for min_id, size in zip(min_idx, new_sizes)]
    bbox = list(slice(min_, max(max_, 1)) for min_, max_ in zip(min_idx, max_idx))
    # DO not crop channel axis...
    bbox[0] = slice(0, shape[0])
    # prevent warning
    bbox = tuple(bbox)
    cropped_images = [image[bbox] for image in images]
    if len(cropped_images) == 1:
        return cropped_images[0]
    else:
        return cropped_images


def random_crop3d(*images, min_perc=0.5, max_perc=1.):
    """Crop randomly but identically all images given.

    Could be used to pass both mask and image at the same time. Anything else will
    throw.

    Warnings
    --------
    Only works for channel first images. (No channel image will not work).
    """
    return random_crop2d(min_perc, max_perc, *images)


class MaxMinNormalization(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        Max = np.max(image)
        Min = np.min(image)
        image = (image - Min) / (Max - Min)

        return {'image': image, 'label': label}


class Random_Flip(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        if random.random() < 0.5:
            image = np.flip(image, 0)
            label = np.flip(label, 0)
        if random.random() < 0.5:
            image = np.flip(image, 1)
            label = np.flip(label, 1)
        if random.random() < 0.5:
            image = np.flip(image, 2)
            label = np.flip(label, 2)

        return {'image': image, 'label': label}


class Random_Crop(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        H = random.randint(0, 240 - 128)
        W = random.randint(0, 240 - 128)
        D = random.randint(0, 160 - 128)

        image = image[H: H + 128, W: W + 128, D: D + 128, ...]
        label = label[..., H: H + 128, W: W + 128, D: D + 128]

        return {'image': image, 'label': label}


class Random_intencity_shift(object):
    def __call__(self, sample, factor=0.1):
        image = sample['image']
        label = sample['label']

        scale_factor = np.random.uniform(1.0-factor, 1.0+factor, size=[1, image.shape[1], 1, image.shape[-1]])
        shift_factor = np.random.uniform(-factor, factor, size=[1, image.shape[1], 1, image.shape[-1]])

        image = image*scale_factor+shift_factor

        return {'image': image, 'label': label}


class Random_rotate(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']

        angle = round(np.random.uniform(-10, 10), 2)
        image = ndimage.rotate(image, angle, axes=(0, 1), reshape=False)
        label = ndimage.rotate(label, angle, axes=(0, 1), reshape=False)

        return {'image': image, 'label': label}


class Pad(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']

        image = np.pad(image, ((0, 0), (0, 0), (0, 5), (0, 0)), mode='constant')
        label = np.pad(label, ((0, 0), (0, 0), (0, 5)), mode='constant')
        return {'image': image, 'label': label}
    #(240,240,155)>(240,240,160)


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        image_t = sample['image_t']

        image = torch.from_numpy(image).float()
        image_t = torch.from_numpy(image_t).float()
        label = torch.from_numpy(label).long()

        return {'image': image, 'image_t': image_t, 'label': label}


def transform(sample):
    trans = transforms.Compose([
        ToTensor()
    ])

    return trans(sample)


def transform_valid(sample):
    trans = transforms.Compose([
        ToTensor()
    ])

    return trans(sample)


class ToTensor_Test(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image = sample['image']
        image_t = sample['image_t']

        image = torch.from_numpy(image).float()
        image_t = torch.from_numpy(image_t).float()

        return {'image': image, 'image_t': image_t}


def transform_test(sample):
    trans = transforms.Compose([
        ToTensor_Test()
    ])

    return trans(sample)


def pad_batch_to_max_shape(batch):
    shapes = (sample['label'].shape for sample in batch)
    _, z_sizes, y_sizes, x_sizes = list(zip(*shapes))
    maxs = [int(max(z_sizes)), int(max(y_sizes)), int(max(x_sizes))]
    for i, max_ in enumerate(maxs):
        max_stride = 16
        if max_ % max_stride != 0:
            # Make it divisible by 16
            maxs[i] = ((max_ // max_stride) + 1) * max_stride
    zmax, ymax, xmax = maxs
    for elem in batch:
        exple = elem['label']
        zpad, ypad, xpad = zmax - exple.shape[1], ymax - exple.shape[2], xmax - exple.shape[3]
        assert all(pad >= 0 for pad in (zpad, ypad, xpad)), "Negative padding value error !!"
        # free data augmentation
        left_zpad, left_ypad, left_xpad = [random.randint(0, pad) for pad in (zpad, ypad, xpad)]
        right_zpad, right_ypad, right_xpad = [pad - left_pad for pad, left_pad in
                                              zip((zpad, ypad, xpad), (left_zpad, left_ypad, left_xpad))]
        pads = (left_xpad, right_xpad, left_ypad, right_ypad, left_zpad, right_zpad)
        elem['image'], elem['label'] = F.pad(elem['image'], pads), F.pad(elem['label'], pads)
    return batch


def fourier_based_img(image):
    image_tensor = torch.from_numpy(image)
    fourier_transform = torch.fft.fftn(image_tensor)
    amplitude, phase = torch.abs(fourier_transform), torch.angle(fourier_transform)

    # Generate random noise with the same shape as the amplitude
    noise = torch.randn_like(amplitude)

    # Modify the amplitude with the noise
    new_amplitude = amplitude * noise

    # Combine the new amplitude and original phase
    new_fourier_transform = new_amplitude * torch.exp(1j * phase)

    # Perform the inverse Fourier transform to get the new image tensor
    image_tensor_t = torch.fft.ifftn(new_fourier_transform).real

    image_tensor_numpy = image_tensor_t.numpy()
    return image_tensor_numpy

def fourier_based_freq_masking(image):
    image_tensor = torch.from_numpy(image)
    fourier_transform = torch.fft.fftn(image_tensor)
    amplitude, phase = torch.abs(fourier_transform), torch.angle(fourier_transform)

    # Create a high-pass filter
    def high_pass_filter_3d(shape, cutoff=0.1):
        C, D, H, W = shape
        filter_mask = np.ones((D, H, W), dtype=np.float32)
        center_D, center_H, center_W = D // 2, H // 2, W // 2
        for z in range(D):
            for y in range(H):
                for x in range(W):
                    distance = np.sqrt((z - center_D)**2 + (y - center_H)**2 + (x - center_W)**2)
                    if distance < cutoff * max(D, H, W):
                        filter_mask[z, y, x] = 0
        return torch.from_numpy(filter_mask)

    # Apply the high-pass filter to the amplitude
    filter_mask = high_pass_filter_3d(amplitude.shape, cutoff=0.1)
    filter_mask_un = filter_mask.unsqueeze(0)
    filtered_amplitude = amplitude * filter_mask_un

    # Combine the filtered amplitude and original phase
    new_fourier_transform = filtered_amplitude * torch.exp(1j * phase)

    # Perform the inverse Fourier transform to get the new image tensor
    high_freq_image_tensor = torch.fft.ifftn(new_fourier_transform)

    # Take the real part to avoid any complex number issues
    high_freq_image_tensor = torch.real(high_freq_image_tensor)

    image_tensor_numpy = high_freq_image_tensor.numpy()
    return image_tensor_numpy




class LISA(Dataset):
    def __init__(self, base_dir=None, split='train', num=None, transform=None, patch_size=(128,128,128)):
        self._base_dir = base_dir
        self.transform = transform
        self.sample_list = []
        self.split = split
        self.patch_size = patch_size

        train_path = self._base_dir + '/train.list'
        val_path = self._base_dir + '/val.list'
        test_path = self._base_dir + '/test.list'

        if split == 'train':
            with open(train_path, 'r') as f:
                self.image_list = f.readlines()
        elif split == 'test':
            with open(test_path, 'r') as f:
                self.image_list = f.readlines()
        elif split == 'val':
            with open(val_path, 'r') as f:
                self.image_list = f.readlines()

        self.image_list = [item.replace('\n', '') for item in self.image_list]
        print(f"IMAGE LIST: {self.image_list}")
        if num is not None:
            self.image_list = self.image_list[:num]

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        result = []
        if self.split == 'train':
            path = self._base_dir + "/imagesTr/" + image_name
            mask_name = image_name.replace('_ciso','_hipp')
            seg_path = self._base_dir + "/labelsTr/" + mask_name
            image = load_nii(path)
            label = load_nii(seg_path).astype(np.int8)

        elif self.split == 'val':
            path = self._base_dir + "/imagesTr/" + image_name
            mask_name = image_name.replace('_ciso', '_hipp')
            seg_path = self._base_dir + "/labelsTr/" + mask_name
            image = load_nii(path)
            label = load_nii(seg_path).astype(np.int8)

        else:
            path = self._base_dir + "/imagesTr/" + image_name
            image = load_nii(path)

        if self.split == "train":
            left_hipp = label == 1
            right_hipp = label == 2

            label = np.stack([left_hipp, right_hipp])

            # Remove maximum extent of the zero-background to make future crop more useful
            #print(image.shape)
            z_indexes, y_indexes, x_indexes = np.nonzero(image != 0)
            # Add 1 pixel in each side
            zmin, ymin, xmin = [max(0, int(np.min(arr) - 1)) for arr in (z_indexes, y_indexes, x_indexes)]
            zmax, ymax, xmax = [int(np.max(arr) + 1) for arr in (z_indexes, y_indexes, x_indexes)]
            image = np.expand_dims(image,0)
            # label = np.expand_dims(label,0)
            patient_image = image[:, zmin:zmax, ymin:ymax, xmin:xmax]
            patient_label = label[:, zmin:zmax, ymin:ymax, xmin:xmax]
            patient_image_t = normalize(fourier_based_freq_masking(patient_image))
            # default to 128, 128, 128 64, 64, 64 32, 32, 32
            patient_image = normalize(patient_image)
            # print(f"IMAGE : {patient_image.shape}, LABEL: {patient_label.shape}")
            patient_image, patient_image_t, patient_label = pad_or_crop_image(patient_image, patient_image_t, patient_label, target_size=self.patch_size)
        elif self.split == "val":
            left_hipp = label == 1
            right_hipp = label == 2

            label = np.stack([left_hipp, right_hipp])

            z_indexes, y_indexes, x_indexes = np.nonzero(image != 0)
            # Add 1 pixel in each side
            zmin, ymin, xmin = [max(0, int(np.min(arr) - 1)) for arr in (z_indexes, y_indexes, x_indexes)]
            zmax, ymax, xmax = [int(np.max(arr) + 1) for arr in (z_indexes, y_indexes, x_indexes)]
            image = np.expand_dims(image,0)
            # label = np.expand_dims(label,0)
            patient_image = image[:, zmin:zmax, ymin:ymax, xmin:xmax]
            patient_label = label[:, zmin:zmax, ymin:ymax, xmin:xmax]
            patient_image_t = normalize(fourier_based_freq_masking(patient_image))

            patient_image = normalize(patient_image)
        else:
            z_indexes, y_indexes, x_indexes = np.nonzero(image != 0)
            # Add 1 pixel in each side
            zmin, ymin, xmin = [max(0, int(np.min(arr) - 1)) for arr in (z_indexes, y_indexes, x_indexes)]
            zmax, ymax, xmax = [int(np.max(arr) + 1) for arr in (z_indexes, y_indexes, x_indexes)]
            image = np.expand_dims(image,0)
            patient_image = image[:, zmin:zmax, ymin:ymax, xmin:xmax]

            patient_image_t = normalize(fourier_based_freq_masking(patient_image))

            patient_image = normalize(patient_image)

        if self.split == 'train':
            sample = {'image': patient_image, 'image_t': patient_image, 'label': patient_label}
            sample = transform(sample)

        elif self.split == 'val':
            sample = {'image': patient_image, 'image_t': patient_image_t, 'label': patient_label}
            sample = transform_valid(sample)
            sample['seg_path'] = seg_path
            sample['crop_indexes'] = ((zmin, zmax), (ymin, ymax), (xmin, xmax))

        elif self.split == 'test':
            sample = {'image': patient_image, 'image_t': patient_image_t}
            sample = transform_test(sample)
            sample['seg_path'] = path
            sample['crop_indexes'] = ((zmin, zmax), (ymin, ymax), (xmin, xmax))

        return sample

    def __len__(self):
        return len(self.image_list)


def load_nii(path_folder):
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path_folder)))

