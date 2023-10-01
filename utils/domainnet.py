import random
import torch
import numpy as np
import os
import os.path
from PIL import Image
import json
from math import sqrt
from torchvision import transforms
from scipy.signal import convolve2d


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def make_dataset_fromlist(image_list, n_samples_per_class=-1):
    print(f'image_list:{image_list}')
    with open(image_list) as f:
        if image_list[-4:] == 'json':
            jdata = json.load(f)
            images_labels = [(x['photo'], x['product']) for x in jdata]
            bboxes = [(x['bbox']['left'], x['bbox']['top'], x['bbox']['width'], x['bbox']['height']) for x in jdata]
        else:
            images_labels = [x.strip().split(' ') for x in f.readlines()]
            bboxes = []

    images = np.array([x for (x, y) in images_labels])
    labels = np.array([int(y) for (x, y) in images_labels])
    if n_samples_per_class > 0:
        chosen_images, chosen_labels = [], []
        for _ in range(n_samples_per_class):
            _, indices = np.unique(labels, return_index=True)
            chosen_images.append(images[indices])
            chosen_labels.append(labels[indices])
            images = np.delete(images, indices)
            labels = np.delete(labels, indices)
        images = np.concatenate(chosen_images)
        labels = np.concatenate(chosen_labels)

    return images, labels, bboxes


def rgb_convolve2d(image, kernel):

    red = convolve2d(image[:,:,0], kernel, 'same')
    green = convolve2d(image[:,:,1], kernel, 'same')
    blue = convolve2d(image[:,:,2], kernel, 'same')

    return np.stack([red, green, blue], axis=2)


class Dataset(object):
    def __init__(self, image_list, root="./data/multi/",
                 pre_transform=None, post_transform=None,
                 target_transform=None, test=False, train_val=False,
                 n_samples_per_class=-1, return_path=False,
                 all_train_image_list=None, alpha=1.0, hpf_range=20,
                 hpf_alpha=0.5, hpf_fix=False):

        self.torch_dataset = False
        self.pre_transform = pre_transform
        self.post_transform = post_transform
        self.target_transform = target_transform
        self.loader = pil_loader
        self.root = root
        self.test = test
        self.train_val = train_val
        self.return_path = return_path
        self.image_list = image_list
        self.imgs, self.labels, self.bboxes = make_dataset_fromlist(image_list, n_samples_per_class)
        self.index_map = np.arange(len(self.imgs))
        self.totensor_trans = transforms.ToTensor()
        self.alpha = alpha
        self.hpf_range = hpf_range
        self.hpf_alpha = hpf_alpha
        self.hpf_fix = hpf_fix

        if not test and all_train_image_list is not None:
            self.all_imgs = []
            self.all_bboxes = []

            for image_list in all_train_image_list:
                imgs_per_domain, _, bboxed_per_domain = make_dataset_fromlist(image_list, n_samples_per_class)
                self.all_imgs.extend(imgs_per_domain)
                self.all_bboxes.extend(bboxed_per_domain)
            self.all_imgs_num = len(self.all_imgs)

    def fft_aug(self, img):
        select_img_id = random.randint(0, self.all_imgs_num-1)
        select_path = os.path.join(self.root, self.all_imgs[select_img_id])
        select_img = self.loader(select_path)

        if len(self.all_bboxes) > 0:
            select_bbox = self.all_bboxes[select_img_id]
            select_img = select_img.crop((select_bbox[0], select_bbox[1],
                                          select_bbox[0] + select_bbox[2],
                                          select_bbox[1] + select_bbox[3]))

        select_img = select_img.resize(img.size)
        fft_img = self.colorful_spectrum_mix(img, select_img)

        return fft_img

    def generateFilter(self, image, w, filtType):

        m = np.size(image, 0)
        n = np.size(image, 1)
        channel = np.size(image, 2)

        LPF = np.zeros((m, n, channel))
        HPF = np.ones((m, n, channel))
        xi = np.round((m - w) / 2)
        xf = np.round((m + w) / 2)
        yi = np.round((n - w) / 2)
        yf = np.round((n + w) / 2)
        LPF[int(xi):int(xf), int(yi):int(yf), :] = 1
        HPF[int(xi):int(xf), int(yi):int(yf), :] = 0
        if filtType == "LPF":
            return LPF
        elif filtType == "HPF":
            return HPF
        else:
            print("Only Ideal LPF and HPF are supported")
            exit()

    def phase_img_getter(self, img):

        img = np.asarray(img)
        img_fft = np.fft.fft2(img, axes=(0, 1))

        img_abs, img_pha = np.abs(img_fft), np.angle(img_fft)

        img_phase = np.array([[[50000, 50000, 50000]]]) * (np.e ** (1j * img_pha))
        img_phase = np.real(np.fft.ifft2(img_phase, axes=(0, 1)))
        img_phase = Image.fromarray(np.uint8(np.clip(img_phase, 0, 255)))

        return img_phase

    def colorful_spectrum_mix(self, img1, img2, ratio=1.0):

        lam = np.random.uniform(0, self.alpha)

        img1_gray = img1.convert('L')
        img1_gray_fft = np.fft.fft2(img1_gray, axes=(0, 1))
        img1_gray_abs, img1_gray_pha = np.abs(img1_gray_fft), np.angle(img1_gray_fft)
        img1_gray_pha = np.stack([img1_gray_pha, img1_gray_pha, img1_gray_pha], axis=-1)

        img1 = np.asarray(img1)
        img2 = np.asarray(img2)

        assert img1.shape == img2.shape
        h, w, c = img1.shape

        img1_fft = np.fft.fft2(img1, axes=(0, 1))
        img2_fft = np.fft.fft2(img2, axes=(0, 1))
        img1_abs, img1_pha = np.abs(img1_fft), np.angle(img1_fft)
        img2_abs, img2_pha = np.abs(img2_fft), np.angle(img2_fft)

        img1_abs = np.fft.fftshift(img1_abs, axes=(0, 1))
        img2_abs = np.fft.fftshift(img2_abs, axes=(0, 1))

        img1_pha_shift = np.fft.fftshift(img1_pha, axes=(0, 1))
        img2_pha_shift = np.fft.fftshift(img2_pha, axes=(0, 1))

        img1_abs_ = np.copy(img1_abs)
        img2_abs_ = np.copy(img2_abs)
        img1_abs = lam * img2_abs_ + (1 - lam) * img1_abs_
        img1_abs = np.fft.ifftshift(img1_abs, axes=(0, 1))

        if self.hpf_range > 0:
            lpf = self.generateFilter(img1, self.hpf_range, 'LPF')
            hpf = self.generateFilter(img1, self.hpf_range, 'HPF')
            if self.hpf_fix:
                pha_lam = self.hpf_alpha
            else:
                pha_lam = np.random.uniform(0, self.hpf_alpha)
            img_pha_mixed = np.fft.ifftshift(
                np.multiply(img1_pha_shift, hpf) +
                np.multiply(img1_pha_shift, (1 - pha_lam) * lpf) +
                np.multiply(img2_pha_shift, pha_lam * lpf),
                axes=(0, 1))
        else:
            img_pha_mixed = img1_pha

        img21 = img1_abs * (np.e ** (1j * img_pha_mixed))
        img21 = np.real(np.fft.ifft2(img21, axes=(0, 1)))
        img21 = Image.fromarray(np.uint8(np.clip(img21, 0, 255)))

        img21_phase = np.array([[[50000, 50000, 50000]]]) * (np.e ** (1j * img1_gray_pha))
        img21_phase = np.real(np.fft.ifft2(img21_phase, axes=(0, 1)))
        img21_phase = Image.fromarray(np.uint8(np.clip(img21_phase, 0, 255)))

        return img21, img21_phase

    def img_random_phase(self, img, phase):

        img_w = img.shape[-2]
        img_h = img.shape[-1]

        base_size = 28
        base_num = (img_w // base_size) * (img_h // base_size)

        img_unfold = img.unfold(2, base_size, base_size).\
            unfold(3, base_size, base_size).contiguous().view(img.shape[0],
                                                              img.shape[1],
                                                              base_num,
                                                              base_size,
                                                              base_size)
        phase_unfold = phase.unfold(2, base_size, base_size).\
            unfold(3, base_size, base_size).contiguous().view(img.shape[0],
                                                              img.shape[1],
                                                              base_num,
                                                              base_size,
                                                              base_size)

        selected_patch = np.random.choice(base_num//2)

        img_unfold[:, :, selected_patch:selected_patch+base_num//2] = phase_unfold[:, :, selected_patch:selected_patch+base_num//2]

        mixed_img = img_unfold.view(img.shape[0],
                                    img.shape[1],
                                    img_w//base_size,
                                    img_h//base_size,
                                    base_size,
                                    base_size).transpose(3, 4).contiguous().view(img.shape)

        return mixed_img


    def __getitem__(self, index):

        path = os.path.join(self.root, self.imgs[index])
        target = self.labels[index]
        img = self.loader(path)
        if len(self.bboxes) > 0:
            bbox = self.bboxes[index]
            img = img.crop((bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]))

        if self.test:
            if self.pre_transform is not None and self.post_transform is not None:
                img = self.post_transform(self.pre_transform(img))
            if self.target_transform is not None:
                target = self.target_transform(target)
            return img, target, self.imgs[index]
        elif self.train_val:
            if self.pre_transform is not None and self.post_transform is not None:
                phase_img = self.post_transform(self.phase_img_getter(self.pre_transform(img)))
            if self.target_transform is not None:
                target = self.target_transform(target)
            return phase_img, target
        else:
            img1, img1_phase = self.fft_aug(self.pre_transform(img))
            img2, img2_phase = self.fft_aug(self.pre_transform(img))

            transed_img1 = self.post_transform(img1).unsqueeze(0)
            transed_img2 = self.post_transform(img2).unsqueeze(0)

            transed_img1_phase = self.post_transform(img1_phase).unsqueeze(0)
            transed_img2_phase = self.post_transform(img2_phase).unsqueeze(0)

            transed_img = [transed_img1_phase, transed_img2_phase, transed_img1, transed_img2]

            if self.target_transform is not None:
                target = self.target_transform(target)

            return transed_img, target

    def __len__(self):

        return len(self.imgs)
