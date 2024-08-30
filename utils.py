import torch
from torchvision import transforms
from datasets import load_dataset

from PIL import Image, ImageFilter, ImageDraw
import random
import numpy as np
import copy
from typing import Any, Mapping
import json
import scipy

import io

RADIUS = 14
RADIUS_CUTOFF = 3
ANCHOR_X_OFFSET = 0     
ANCHOR_Y_OFFSET = 0     # 1 = not correct, 0 = correct
USE_ROUNDER_RING = True

HETER_WATERMARK_CHANNEL = [0]
RING_WATERMARK_CHANNEL = [3]
WATERMARK_CHANNEL = sorted(HETER_WATERMARK_CHANNEL + RING_WATERMARK_CHANNEL)


def read_json(filename: str) -> Mapping[str, Any]:
    """Returns a Python dict representation of JSON object at input file."""
    with open(filename) as fp:
        return json.load(fp)
    

def set_random_seed(seed=0):
    torch.manual_seed(seed + 0)
    torch.cuda.manual_seed(seed + 1)
    torch.cuda.manual_seed_all(seed + 2)
    np.random.seed(seed + 3)
    torch.cuda.manual_seed_all(seed + 4)
    random.seed(seed + 5)


def transform_img(image, target_size=512):
    tform = transforms.Compose(
        [
            transforms.Resize(target_size),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
        ]
    )
    image = tform(image)
    return 2.0 * image - 1.0

def rotate_tensor(tensor, angle):
    return transforms.functional.rotate(tensor, angle) 

def latents_to_imgs(pipe, latents):
    x = pipe.decode_image(latents)
    x = pipe.torch_to_numpy(x)
    x = pipe.numpy_to_pil(x)
    return x


# for one prompt to multiple images
def measure_similarity(images, prompt, model, clip_preprocess, tokenizer, device):
    with torch.no_grad():
        img_batch = [clip_preprocess(i).unsqueeze(0) for i in images]
        img_batch = torch.concatenate(img_batch).to(device)
        image_features = model.encode_image(img_batch)

        text = tokenizer([prompt]).to(device)
        text_features = model.encode_text(text)
        
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        return (image_features @ text_features.T).mean(-1)


def get_p_value(reversed_latents_no_w, reversed_latents_w, watermarking_mask, gt_patch, args):
    # assume it's Fourier space wm
    reversed_latents_no_w_fft = torch.fft.fftshift(torch.fft.fft2(reversed_latents_no_w), dim=(-1, -2))[watermarking_mask].flatten()
    reversed_latents_w_fft = torch.fft.fftshift(torch.fft.fft2(reversed_latents_w), dim=(-1, -2))[watermarking_mask].flatten()
    target_patch = gt_patch[watermarking_mask].flatten()

    target_patch = torch.concatenate([target_patch.real, target_patch.imag])
    
    # no_w
    reversed_latents_no_w_fft = torch.concatenate([reversed_latents_no_w_fft.real, reversed_latents_no_w_fft.imag])
    sigma_no_w = reversed_latents_no_w_fft.std()
    lambda_no_w = (target_patch ** 2 / sigma_no_w ** 2).sum().item()
    x_no_w = (((reversed_latents_no_w_fft - target_patch) / sigma_no_w) ** 2).sum().item()
    p_no_w = scipy.stats.ncx2.cdf(x=x_no_w, df=len(target_patch), nc=lambda_no_w)

    # w
    reversed_latents_w_fft = torch.concatenate([reversed_latents_w_fft.real, reversed_latents_w_fft.imag])
    sigma_w = reversed_latents_w_fft.std()
    lambda_w = (target_patch ** 2 / sigma_w ** 2).sum().item()
    x_w = (((reversed_latents_w_fft - target_patch) / sigma_w) ** 2).sum().item()
    p_w = scipy.stats.ncx2.cdf(x=x_w, df=len(target_patch), nc=lambda_w)

    return p_no_w, p_w


def circle_mask(size = 64, r = RADIUS, x_offset = ANCHOR_X_OFFSET, y_offset = ANCHOR_Y_OFFSET, mode = 'full') -> np.ndarray:
    '''
    Returns a (size, size) bool type numpy array.
    '''
    x0 = y0 = size // 2
    x0 += x_offset
    y0 += y_offset - 1
    y, x = np.ogrid[:size, :size]
    y = y[::-1]

    if mode == 'left':
        return (((x - x0)**2 + (y - y0)**2)<= r**2) & ((x > x0) + ((x == x0) & (y > y0)))
    if mode == 'right':
        return (((x - x0)**2 + (y - y0)**2)<= r**2) & ((x < x0) + ((x == x0) & (y < y0)))
    if mode == 'full':
        return (((x - x0)**2 + (y - y0)**2)<= r**2) & (((x > x0) + ((x == x0) & (y > y0))) + ((x < x0) + ((x == x0) & (y < y0))))
    raise NotImplementedError(f'Circle mask "{mode}" not implemented.')

def ring_mask(size = 64, r_out = RADIUS, r_in = RADIUS_CUTOFF, x_offset = ANCHOR_X_OFFSET, y_offset = ANCHOR_Y_OFFSET, mode = 'full'):
    outer_mask = circle_mask(size = size, r = r_out, x_offset = x_offset, y_offset = y_offset, mode = mode)
    inner_mask = circle_mask(size = size, r = r_in, x_offset = x_offset, y_offset = y_offset, mode = mode)
    return outer_mask & (~(inner_mask))

class RounderRingMask:
    def __init__(self, size = 64, r_out = RADIUS, x_offset = ANCHOR_X_OFFSET, y_offset = ANCHOR_Y_OFFSET, mode = 'full'):
        assert size >= 3
        self.size = size
        self.r_out = r_out

        num_rings = r_out
        zero_bg_freq = torch.zeros(size, size)
        center = size // 2
        center_x, center_y = center + x_offset, center - y_offset

        ring_vector = torch.tensor([(200 - i*4) * (-1)**i for i in range(num_rings)])
        zero_bg_freq[center_x, center_y:center_y+num_rings] = ring_vector
        zero_bg_freq = zero_bg_freq[None, None, ...]
        self.ring_vector_np = ring_vector.numpy()

        res = torch.zeros(360, size, size)
        res[0] = zero_bg_freq
        for angle in range(1, 360):
            zero_bg_freq_rot = transforms.functional.rotate(zero_bg_freq, angle=angle)
            res[angle] = zero_bg_freq_rot

        res = res.numpy()
        self.pure_bg = np.zeros((size, size))
        for x in range(size):
            for y in range(size):
                values, count = np.unique(res[:, x, y],  return_counts=True)
                if len(count) > 2:
                    self.pure_bg[x, y] = values[count == max(count[values!=0])][0]
                elif len(count) == 2:
                    self.pure_bg[x, y] = values[values!=0][0]
        
    def get_ring_mask(self, r_out, r_in):
        """
            get mask from pure_bg
            sector_idx == -1, no sectors, get the whole ring
            sector_idx in [0, 1] for 2 effective sectors
        """
        assert r_out <= self.r_out
        if r_in - 1 < 0:
            right_end = 0  # None, to take the center
        else:
            right_end = r_in - 1
        cand_list = self.ring_vector_np[r_out-1:right_end:-1]
        mask = np.isin(self.pure_bg, cand_list)
        
        if self.size % 2:
            mask = mask[:self.size-1, :self.size-1]  # [64, 64]

        return mask

if USE_ROUNDER_RING:
    mask_obj = RounderRingMask(size=65, r_out=RADIUS, x_offset = ANCHOR_X_OFFSET, y_offset = ANCHOR_Y_OFFSET)
    def ring_mask(size = 64, r_out = RADIUS, r_in = RADIUS_CUTOFF, x_offset = ANCHOR_X_OFFSET, y_offset = ANCHOR_Y_OFFSET, mode = 'full'):
        assert size == 64
        assert mode == 'full', f'not implemented mode {mode}'
        return mask_obj.get_ring_mask(r_out=r_out, r_in=r_in)

def generate_Fourier_watermark_latents(device, radius, radius_cutoff, watermark_region_mask, watermark_channel, original_latents = None, watermark_pattern = None):
    
    #set_random_seed(seed)

    if original_latents is None:
        #original_latents = torch.randn(*shape, device = device)
        raise NotImplementedError('Original latents should be provided.')
    
    if watermark_pattern is None:
        raise NotImplementedError('Fourier watermark pattern should be provided.')

    # circular_mask = torch.tensor(ring_mask(size = original_latents.shape[-1], r_out = radius, r_in = radius_cutoff)).to(device)
    watermarked_latents_fft = torch.fft.fftshift(torch.fft.fft2(original_latents), dim = (-1, -2))

    # for channel in watermark_channel:
    #     watermarked_latents_fft[:, channel] = watermarked_latents_fft[:, channel] * ~circular_mask + watermark_pattern[:, channel] * circular_mask
    
    assert len(watermark_channel) == len(watermark_region_mask)
    for channel, channel_mask in zip(watermark_channel, watermark_region_mask):
        watermarked_latents_fft[:, channel] = watermarked_latents_fft[:, channel] * ~channel_mask + watermark_pattern[:, channel] * channel_mask
    
    return torch.fft.ifft2(torch.fft.ifftshift(watermarked_latents_fft, dim = (-1, -2))).real


def image_distortion(img1, img2, seed, 
                     r_degree = None, 
                     jpeg_ratio = None, 
                     crop_scale = None, 
                     crop_ratio = None, 
                     gaussian_blur_r = None, 
                     gaussian_std = None, 
                     brightness_factor = None, 
                     run_name = None):
    if r_degree is not None:
        if img1 is not None:
            img1 = transforms.RandomRotation((r_degree, r_degree))(img1)
        img2 = transforms.RandomRotation((r_degree, r_degree))(img2)

    if jpeg_ratio is not None:
        if img1 is not None:
            buf = io.BytesIO()
            img1.save(buf, format='JPEG', quality=jpeg_ratio)
            img1 = Image.open(buf)
        buf2 = io.BytesIO()
        img2.save(buf2, format='JPEG', quality=jpeg_ratio)
        img2 = Image.open(buf2)

    if crop_scale is not None and crop_ratio is not None:
        if img1 is not None:
            set_random_seed(seed)
            img1 = transforms.RandomResizedCrop(img1.size, scale=(crop_scale, crop_scale), ratio=(crop_ratio, crop_ratio))(img1)
        set_random_seed(seed)
        img2 = transforms.RandomResizedCrop(img2.size, scale=(crop_scale, crop_scale), ratio=(crop_ratio, crop_ratio))(img2)
        
    if gaussian_blur_r is not None:
        if img1 is not None:
            img1.filter(ImageFilter.GaussianBlur(radius=gaussian_blur_r))
        img2 = img2.filter(ImageFilter.GaussianBlur(radius=gaussian_blur_r))

    if gaussian_std is not None:
        img_shape = np.array(img1).shape
        g_noise = np.random.normal(0, gaussian_std, img_shape) * 255
        g_noise = g_noise.astype(np.uint8)
        if img1 is not None:
            img1 = Image.fromarray(np.clip(np.array(img1) + g_noise, 0, 255))
        img2 = Image.fromarray(np.clip(np.array(img2) + g_noise, 0, 255))

    if brightness_factor is not None:
        if img1 is not None:
            img1 = transforms.ColorJitter(brightness=brightness_factor)(img1)
        img2 = transforms.ColorJitter(brightness=brightness_factor)(img2)

    return [img1, img2]

def fft(input_tensor):
    assert len(input_tensor.shape) == 4
    return torch.fft.fftshift(torch.fft.fft2(input_tensor), dim = (-1, -2))

def ifft(input_tensor):
    assert len(input_tensor.shape) == 4
    return torch.fft.ifft2(torch.fft.ifftshift(input_tensor, dim = (-1, -2)))

def make_Fourier_ringid_pattern(
        device,
        key_value_combination, 
        no_watermark_latents, 
        radius,
        radius_cutoff, 
        ring_watermark_channel, 
        heter_watermark_channel, 
        heter_watermark_region_mask=None,
        ring_width = 1, 
        ):
    if ring_width != 1:
        raise NotImplementedError(f'Proposed watermark generation only implemented for ring width = 1.')

    if len(key_value_combination) != (RADIUS - RADIUS_CUTOFF):
        raise ValueError('Mismatch between #key values and #slots')
    
    shape = no_watermark_latents.shape
    if len(shape) != 4:
        raise ValueError(f'Invalid shape for initial latent: {shape}')
    
    latents_fft = fft(no_watermark_latents)
    # watermarked_latents_fft = copy.deepcopy(latents_fft)
    watermarked_latents_fft = torch.zeros_like(latents_fft)

    radius_list = [this_radius for this_radius in range(radius, radius_cutoff, -1)]

    # put ring
    for radius_index in range(len(radius_list)):
        this_r_out = radius_list[radius_index]
        this_r_in = this_r_out - ring_width
        mask = torch.tensor(ring_mask(size = shape[-1], r_out = this_r_out, r_in = this_r_in)).to(device).to(torch.float64)  # sector_idx default to -1
        for batch_index in range(shape[0]):
            for channel_index in range(len(ring_watermark_channel)):
                watermarked_latents_fft[batch_index, ring_watermark_channel[channel_index]].real = (1 - mask) * watermarked_latents_fft[batch_index, ring_watermark_channel[channel_index]].real + mask * key_value_combination[radius_index][channel_index]
                watermarked_latents_fft[batch_index, ring_watermark_channel[channel_index]].imag = (1 - mask) * watermarked_latents_fft[batch_index, ring_watermark_channel[channel_index]].imag + mask * key_value_combination[radius_index][channel_index]

    # put noise or zeros
    if len(heter_watermark_channel) > 0:
        assert len(heter_watermark_channel) == len(heter_watermark_region_mask)
        heter_watermark_region_mask = heter_watermark_region_mask.to(torch.float64)
        w_type = 'noise'
        
        if w_type == 'noise':
            w_content = fft(torch.randn(*shape, device = device))  # [N, c, h, w]
        elif w_type == 'zeros':
            w_content = fft(torch.zeros(*shape, device = device))  # [N, c, h, w]
        else:
            raise NotImplementedError
        
        for batch_index in range(shape[0]):
            for channel_id, channel_mask in zip(heter_watermark_channel, heter_watermark_region_mask):
                watermarked_latents_fft[batch_index, channel_id].real = \
                    (1 - channel_mask) * watermarked_latents_fft[batch_index, channel_id].real + channel_mask * w_content[batch_index][channel_id].real
                watermarked_latents_fft[batch_index, channel_id].imag = \
                    (1 - channel_mask) * watermarked_latents_fft[batch_index, channel_id].imag + channel_mask * w_content[batch_index][channel_id].imag

    return watermarked_latents_fft

def get_distance(tensor1, tensor2, mask, p, mode, channel_min=False, channel = WATERMARK_CHANNEL):
    if tensor1.shape != tensor2.shape:
        raise ValueError(f'Shape mismatch during eval: {tensor1.shape} vs {tensor2.shape}')
    if mode not in ['complex', 'real', 'imag']:
        raise NotImplemented(f'Eval mode not implemented: {mode}')
    
    if not channel_min:
        if p == 1:
            # a faster implementation for l1 distance
            if mode == 'complex':
                return torch.mean(torch.abs(tensor1[0][channel] - tensor2[0][channel])[mask]).item()
            if mode == 'real':
                return torch.mean(torch.abs(tensor1[0][channel].real - tensor2[0][channel].real)[mask]).item()
            if mode == 'imag':
                return torch.mean(torch.abs(tensor1[0][channel].imag - tensor2[0][channel].imag)[mask]).item()
        else:
            if mode == 'complex':
                return torch.norm(torch.abs(tensor1[0][channel][mask] - tensor2[0][channel][mask]), p = p).item() / torch.sum(mask)
            if mode == 'real':
                return torch.norm(torch.abs(tensor1[0][channel][mask].real - tensor2[0][channel][mask].real), p = p).item() / torch.sum(mask)
            if mode == 'imag':
                return torch.norm(torch.abs(tensor1[0][channel][mask].imag - tensor2[0][channel][mask].imag), p = p).item() / torch.sum(mask)
    else:
        # argmin TODO: normalize
        if  len(RING_WATERMARK_CHANNEL) > 1 and len(HETER_WATERMARK_CHANNEL) > 0:
            ring_channel_idx_list = [idx for idx, c_id in enumerate(WATERMARK_CHANNEL) if c_id in RING_WATERMARK_CHANNEL]
            heter_channel_idx_list = [idx for idx, c_id in enumerate(WATERMARK_CHANNEL) if c_id in HETER_WATERMARK_CHANNEL]
            if mode == 'complex':
                diff = torch.abs(tensor1[0][channel] - tensor2[0][channel])  # [c, h, w]
            elif mode == 'real':
                diff = torch.abs(tensor1[0][channel].real - tensor2[0][channel].real)  # [c, h, w]
            elif mode == 'imag':
                diff = torch.abs(tensor1[0][channel].imag - tensor2[0][channel].imag)  # [c, h, w]
            l1_list = []
            num_items = []
            for c_idx in range(len(mask)):
                mask_c = torch.zeros_like(mask)
                mask_c[c_idx] = mask[c_idx]
                l1_list.append(torch.mean(diff[mask_c]).item())
                num_items.append(torch.sum(mask_c).item())
            total = 0
            num = 0
            for ring_channel_idx in ring_channel_idx_list:
                total += l1_list[ring_channel_idx] * num_items[ring_channel_idx]
                num += num_items[ring_channel_idx]
            ring_channels_mean = total / num
            return min(ring_channels_mean, min([l1_list[idx] for idx in heter_channel_idx_list]))
        elif len(RING_WATERMARK_CHANNEL) == 1 and len(HETER_WATERMARK_CHANNEL) > 0:
            if mode == 'complex':
                diff = torch.abs(tensor1[0][channel] - tensor2[0][channel])  # [c, h, w]
            elif mode == 'real':
                diff = torch.abs(tensor1[0][channel].real - tensor2[0][channel].real)  # [c, h, w]
            elif mode == 'imag':
                diff = torch.abs(tensor1[0][channel].imag - tensor2[0][channel].imag)  # [c, h, w]
            l1_list = []
            for c_idx in range(len(mask)):
                mask_c = torch.zeros_like(mask)
                mask_c[c_idx] = mask[c_idx]
                l1_list.append(torch.mean(diff[mask_c]).item())
            return min(l1_list)
        else:
            raise NotImplementedError

class QualityResultsCollector:
    def __init__(self, metric_list):
        self.result_dict = {}
        for metric in metric_list:
            self.result_dict[metric] = []

    def collect(self, metric, value):
        self.result_dict[metric].append(value)

    def average(self, metric):
        return np.average(np.array(self.result_dict[metric]))
    
    def count_results(self, metric):
        return len(self.result_dict[metric])
    
    def get_list(self, metric):
        return self.result_dict[metric]
    
    def np_func_eval(self, metric, np_func):
        return np_func(np.array(self.result_dict[metric]))
    
    def clear_results(self, metric):
        self.result_dict[metric] = []

    def print_average(self):
        print('Average Quality Metrics')
        for key, value in self.result_dict.items():
            print(f'{key}: {np.array(value).mean():.4f}')
    
    def return_average(self):
        res = {}
        for key, value in self.result_dict.items():
            res[key] = np.array(value).mean()
        return res

def partition_list(lst, batch_size):
    return [lst[i:i+batch_size] for i in range(0, len(lst), batch_size)]

def get_dataset(dataset):
    if 'laion' in dataset:
        dataset = load_dataset(dataset)['train']
        prompt_key = 'TEXT'
    elif 'coco' in dataset:
        with open('fid_outputs/coco/meta_data.json') as f:
            dataset = json.load(f)
            dataset = dataset['annotations']
            prompt_key = 'caption'
    else:
        dataset = load_dataset(dataset)['test']
        prompt_key = 'Prompt'

    return dataset, prompt_key