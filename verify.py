# # Table 1 RingID Verification Experiments
# 
# [![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
# [![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)](https://jupyter.org/)
# 
# This script contains the code to replicate the RingID experiments presented in Table 1 of our paper. 
# 
# 1. Radius configurations are defined in `utils.py`.
# 2. Additional configurations can be set using the argparser provided below. Modify the argparser parameters as needed to customize the experiments.
# 3. Run the cells in order to execute the experiments.

from tqdm import tqdm
from sklearn import metrics
import torch
import itertools
import matplotlib.pyplot as plt
import argparse
import os
from datetime import datetime
import pandas as pd
from collections import OrderedDict
from prettytable import PrettyTable

from inverse_stable_diffusion import InversableStableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler
import open_clip

from utils import *
from io_utils import *

def parse_args():
    parser = argparse.ArgumentParser(description='multiple-key identification')
    parser.add_argument('--run_name', default='test')
    parser.add_argument('--model_id', default='stabilityai/stable-diffusion-2-1-base')
    parser.add_argument('--reference_model', default='ViT-g-14')
    parser.add_argument('--reference_model_pretrain', default='laion2b_s12b_b42k')
    parser.add_argument('--online', action='store_true', default=False, help='True to check cache and download models if necessary. False to use cached models.')

    group = parser.add_argument_group('hyperparameters')
    parser.add_argument('--general_seed', type=int, default=42)
    parser.add_argument('--watermark_seed', type=int, default=5)
    parser.add_argument('--num_images', default=1, type=int)
    parser.add_argument('--guidance_scale', default=7.5, type=float)
    parser.add_argument('--num_inference_steps', default=50, type=int)
    parser.add_argument('--test_num_inference_steps', default=None, type=int)
    parser.add_argument('--image_length', default=512, type=int)
    parser.add_argument('--ring_width', default=1, type=int)
    parser.add_argument('--quantization_levels', default=2, type=int)
    parser.add_argument('--ring_value_range', default=64, type=int)
    
    parser.add_argument('--save_generated_imgs', type=int, default=1)
    parser.add_argument('--save_root_dir', type=str, default='./runs')

    group = parser.add_argument_group('trials parameters')
    parser.add_argument('--gpu_id', default=0, type=int)
    parser.add_argument('--trials', type=int, default=100, help='total number of trials to run')
    parser.add_argument('--fix_gt', type=int, default=1, help='use watermark after discarding the imag part on space domain as gt.')
    parser.add_argument('--time_shift', type=int, default=1, help='use time-shift')
    parser.add_argument('--time_shift_factor', type=float, default=1.0, help='factor to scale the value after time-shift')
    parser.add_argument('--assigned_keys', type=int, default=-1, help='number of assigned keys, -1 for all possible kyes')
    parser.add_argument('--channel_min', type=int, default=1, help='only for heterogeous watermark, when match gt, take min among channels as the result')

    args = parser.parse_args()

    if args.test_num_inference_steps is None:
        args.test_num_inference_steps = args.num_inference_steps

    return args

def main():

    args = parse_args()

    device = f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu'

    timestr = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    save_dir = os.path.join(args.save_root_dir, timestr + '_' + args.run_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=False)
    if args.save_generated_imgs:
        save_img_dir = os.path.join(save_dir, 'images', 'watermarked')
        os.makedirs(save_img_dir, exist_ok=False)
        save_nowatermark_img_dir = os.path.join(save_dir, 'images', 'no_watermark')
        os.makedirs(save_nowatermark_img_dir, exist_ok=False)

    # set random seed
    set_random_seed(args.general_seed)

    # load model
    # model_id = 'stabilityai/stable-diffusion-2-1-base'
    # reference_model = 'ViT-g-14'
    if args.online:
        pipeline_pretrain = args.model_id
        reference_model_pretrain = args.reference_model_pretrain
        dataset_id = 'Gustavosta/Stable-Diffusion-Prompts'
    else:
        # run locally 
        pipeline_pretrain = f'{os.path.expanduser("~")}/.cache/huggingface/diffusers/models--stabilityai--stable-diffusion-2-1-base/snapshots/1f758383196d38df1dfe523ddb1030f2bfab7741/'
        reference_model_pretrain = f'{os.path.expanduser("~")}/.cache/huggingface/hub/models--laion--CLIP-ViT-g-14-laion2B-s12B-b42K/snapshots/4b0305adc6802b2632e11cbe6606a9bdd43d35c9/open_clip_pytorch_model.bin'
        dataset_id = 'Gustavosta/stable-diffusion-prompts'
        os.environ["HF_DATASETS_OFFLINE"] = "1"

    model_dtype = torch.float16
    scheduler = DPMSolverMultistepScheduler.from_pretrained(args.model_id, subfolder='scheduler', local_files_only=(not args.online))
    pipe = InversableStableDiffusionPipeline.from_pretrained(
        pipeline_pretrain,
        scheduler=scheduler,
        torch_dtype=model_dtype,
        revision='fp16',
        )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    if args.reference_model is not None:
        ref_model, _, ref_clip_preprocess = open_clip.create_model_and_transforms(
            args.reference_model, 
            pretrained=reference_model_pretrain, 
            device=device
            )
        ref_tokenizer = open_clip.get_tokenizer(args.reference_model)

    dataset, prompt_key = get_dataset(dataset_id)

    tester_prompt = '' # assume at the detection time, the original prompt is unknown
    text_embeddings = pipe.get_text_embedding(tester_prompt)

    lp_distance_method = get_distance
    # lp_magnitude_method = lp_magnitude_distance
    eval_methods = [
        {'Distance': 'L1', 'Metrics':  '|a-b|        ', 'func': lp_distance_method, 'kwargs': {'p': 1, 'mode': 'complex', 'channel_min': args.channel_min}},
        # {'Distance': 'L1', 'Metrics':  '|a.r-b.r|    ', 'func': lp_distance_method, 'kwargs': {'p': 1, 'mode': 'real', 'channel_min': args.channel_min}},
        # {'Distance': 'L1', 'Metrics':  '|a.i-b.i|    ', 'func': lp_distance_method, 'kwargs': {'p': 1, 'mode': 'imag', 'channel_min': args.channel_min}},
    ]
    if args.channel_min: assert len(HETER_WATERMARK_CHANNEL) > 0

    base_latents = pipe.get_random_latents()
    original_latents_shape = base_latents.shape
    base_latents = base_latents.to(torch.float64)

    sing_channel_ring_watermark_mask = torch.tensor(
            ring_mask(
                size = original_latents_shape[-1], 
                r_out = RADIUS, 
                r_in = RADIUS_CUTOFF)
            )

    # get heterogeneous watermark mask
    if len(HETER_WATERMARK_CHANNEL) > 0:
        single_channel_heter_watermark_mask = torch.tensor(
                ring_mask(
                    size = original_latents_shape[-1], 
                    r_out = RADIUS, 
                    r_in = RADIUS_CUTOFF)  # TODO: change to whole mask
                )
        heter_watermark_region_mask = single_channel_heter_watermark_mask.unsqueeze(0).repeat(len(HETER_WATERMARK_CHANNEL), 1, 1).to(device)

    watermark_region_mask = []
    for channel_idx in WATERMARK_CHANNEL:
        if channel_idx in RING_WATERMARK_CHANNEL:
            watermark_region_mask.append(sing_channel_ring_watermark_mask)
        else:
            watermark_region_mask.append(single_channel_heter_watermark_mask)
    watermark_region_mask = torch.stack(watermark_region_mask).to(device)  # [C, 64, 64]

    single_channel_num_slots = RADIUS - RADIUS_CUTOFF
    key_value_list = [[list(combo) for combo in itertools.product(np.linspace(-args.ring_value_range, args.ring_value_range, args.quantization_levels).tolist(), repeat = len(RING_WATERMARK_CHANNEL))] for _ in range(single_channel_num_slots)]
    key_value_combinations = list(itertools.product(*key_value_list))

    # random select from all possible value combinations, then generate patterns for selected ones.
    if args.assigned_keys > 0:
        assert args.assigned_keys <= len(key_value_combinations)
        key_value_combinations = random.sample(key_value_combinations, k=args.assigned_keys)
    Fourier_watermark_pattern_list = [make_Fourier_ringid_pattern(device, list(combo), base_latents, 
                                                                                    radius=RADIUS, radius_cutoff=RADIUS_CUTOFF,
                                                                                    ring_watermark_channel=RING_WATERMARK_CHANNEL, 
                                                                                    heter_watermark_channel=HETER_WATERMARK_CHANNEL,
                                                                                    heter_watermark_region_mask=heter_watermark_region_mask if len(HETER_WATERMARK_CHANNEL)>0 else None)
                                                                                    for _, combo in enumerate(key_value_combinations)]            
    ring_capacity = len(Fourier_watermark_pattern_list)

    if args.fix_gt:
        Fourier_watermark_pattern_list = [fft(ifft(Fourier_watermark_pattern).real) for Fourier_watermark_pattern in Fourier_watermark_pattern_list]

    if args.time_shift:
        for Fourier_watermark_pattern in Fourier_watermark_pattern_list:
            # Fourier_watermark_pattern[:, RING_WATERMARK_CHANNEL, ...] = fft(torch.fft.fftshift(ifft(Fourier_watermark_pattern[:, RING_WATERMARK_CHANNEL, ...]), dim = (-1, -2)) * args.time_shift_factor)
            Fourier_watermark_pattern[:, RING_WATERMARK_CHANNEL, ...] = fft(torch.fft.fftshift(ifft(Fourier_watermark_pattern[:, RING_WATERMARK_CHANNEL, ...]), dim = (-1, -2)))

    # Use a single ring pattern for verification
    Fourier_watermark_pattern = Fourier_watermark_pattern_list[628]  # [64, -64, 64, -64, 64...], select this ring pattern

    print(f'[Info] Ring capacity = {ring_capacity}')

    score_heads = ['CLIP No Watermark', 'CLIP Fourier Watermark']

    quality_metrics = QualityResultsCollector(score_heads)

    no_watermark_results_list = []
    Fourier_watermark_results_list = []

    for prompt_index in tqdm(range(args.trials)):

        this_seed = args.general_seed + prompt_index
        this_prompt = dataset[prompt_index][prompt_key]

        set_random_seed(this_seed)
        no_watermark_latents = pipe.get_random_latents()
        Fourier_watermark_latents = generate_Fourier_watermark_latents(
            device = device,
            radius = RADIUS, 
            radius_cutoff = RADIUS_CUTOFF, 
            original_latents = no_watermark_latents, 
            watermark_pattern = Fourier_watermark_pattern,
            watermark_channel = WATERMARK_CHANNEL,
            watermark_region_mask = watermark_region_mask,
        )

        # batched inference
        batched_latents = torch.cat([no_watermark_latents.to(model_dtype), Fourier_watermark_latents.to(model_dtype)], dim=0)
        generated_images = pipe(
            [this_prompt]*2,
            num_images_per_prompt=args.num_images,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            height=args.image_length,
            width=args.image_length,
            latents=batched_latents,
        ).images
        no_watermark_image, Fourier_watermark_image = generated_images[0], generated_images[1]

        no_watermark_clip, Fourier_watermark_clip = measure_similarity([no_watermark_image, Fourier_watermark_image], this_prompt, ref_model, ref_clip_preprocess, ref_tokenizer, device)
        quality_metrics.collect('CLIP No Watermark', no_watermark_clip.item())
        quality_metrics.collect('CLIP Fourier Watermark', Fourier_watermark_clip.item())

        # save generated images
        if args.save_generated_imgs:
            Fourier_watermark_image.save(os.path.join(save_img_dir, f'Prompt_{prompt_index}.Fourier_watermark.ClipSim_{Fourier_watermark_clip.item():.4f}.jpg'))
            no_watermark_image.save(os.path.join(save_nowatermark_img_dir, f'Prompt_{prompt_index}.Fourier_watermark.ClipSim_{Fourier_watermark_clip.item():.4f}.jpg'))

        # Distort
        distorted_image_list = [
            [no_watermark_image, Fourier_watermark_image],
            image_distortion(no_watermark_image, Fourier_watermark_image, seed = this_seed, r_degree = 75),
            image_distortion(no_watermark_image, Fourier_watermark_image, seed = this_seed, jpeg_ratio = 25),
            image_distortion(no_watermark_image, Fourier_watermark_image, seed = this_seed, crop_scale = 0.75, crop_ratio = 0.75),
            image_distortion(no_watermark_image, Fourier_watermark_image, seed = this_seed, gaussian_blur_r = 8),
            image_distortion(no_watermark_image, Fourier_watermark_image, seed = this_seed, gaussian_std = 0.1),
            image_distortion(no_watermark_image, Fourier_watermark_image, seed = this_seed, brightness_factor = 6),
        ]

        # #### Batch mode
        no_watermark_distorted_image_list = [pair[0] for pair in distorted_image_list]
        no_watermark_image_distorted = torch.stack([transform_img(img) for img in no_watermark_distorted_image_list]).to(text_embeddings.dtype).to(device)
        no_watermark_image_latents = pipe.get_image_latents(no_watermark_image_distorted, sample = False)


        Fourier_watermark_distorted_image_list = [pair[1] for pair in distorted_image_list]
        Fourier_watermark_image_distorted = torch.stack([transform_img(img) for img in Fourier_watermark_distorted_image_list]).to(text_embeddings.dtype).to(device)
        Fourier_watermark_image_latents = pipe.get_image_latents(Fourier_watermark_image_distorted, sample = False)  # [N, c, h, w]

        no_watermark_reconstructed_latents = pipe.forward_diffusion(
                latents=no_watermark_image_latents,
                text_embeddings=torch.cat([text_embeddings] * len(no_watermark_image_latents)),
                guidance_scale=1,
                num_inference_steps=args.test_num_inference_steps,
            )

        Fourier_watermark_reconstructed_latents = pipe.forward_diffusion(
                latents=Fourier_watermark_image_latents,
                text_embeddings=torch.cat([text_embeddings] * len(Fourier_watermark_image_latents)),
                guidance_scale=1,
                num_inference_steps=args.test_num_inference_steps,
            )

        no_watermark_reconstructed_latents_fft = fft(no_watermark_reconstructed_latents)
        Fourier_watermark_reconstructed_latents_fft = fft(Fourier_watermark_reconstructed_latents)  # [N，c, h, w]

        this_it_no_watermark_results_list = []
        this_it_Fourier_watermark_results_list = []
        for distortion_index in range(len(distorted_image_list)):
            this_no_watermark_reconstructed_latents_fft = no_watermark_reconstructed_latents_fft[distortion_index][None, ...]
            this_Fourier_watermark_reconstructed_latents_fft = Fourier_watermark_reconstructed_latents_fft[distortion_index][None, ...]
            this_it_no_watermark_results_list.append([-eval_method['func'](Fourier_watermark_pattern, this_no_watermark_reconstructed_latents_fft, watermark_region_mask, channel = WATERMARK_CHANNEL, **eval_method['kwargs']) for eval_method in eval_methods])
            this_it_Fourier_watermark_results_list.append([-eval_method['func'](Fourier_watermark_pattern, this_Fourier_watermark_reconstructed_latents_fft, watermark_region_mask, channel = WATERMARK_CHANNEL, **eval_method['kwargs']) for eval_method in eval_methods])
        no_watermark_results_list.append(this_it_no_watermark_results_list)
        Fourier_watermark_results_list.append(this_it_Fourier_watermark_results_list) 

    ablation_experiment_descriptions = [
        'Clean',
        'Rotation: 75°',
        'JPEG Compression: quality = 25',
        'Crop & Scale: 0.75, 0.75',
        'Gaussian Blur: kernel size 8',
        'Gaussian Noise: σ = 0.1',
        'Brightness: Uniform([0, 6])'
    ]

    ablation_experiment_descriptions_short = [
        'Clean',
        'Rot 75',
        'JPEG 25',
        'C&S 75',
        'Blur 8',
        'Noise 0.1',
        'Birghtness [0, 6]'
    ]

    def get_ablation_results(no_watermark_distance_list, Fourier_watermark_distance_list, plot_description = None):

        no_watermark_distance_list = np.array(no_watermark_distance_list)
        Fourier_watermark_distance_list = np.array(Fourier_watermark_distance_list)

        ablation_results = []

        for ablation_index in range(no_watermark_distance_list.shape[1]):
            distances = no_watermark_distance_list[:, ablation_index].tolist() + Fourier_watermark_distance_list[:, ablation_index].tolist()
            labels = [0] * len(no_watermark_distance_list) + [1] * len(Fourier_watermark_distance_list)

            fpr, tpr, thresholds = metrics.roc_curve(labels, distances, pos_label=1)
            auc = metrics.auc(fpr, tpr)
            acc = np.max(1 - (fpr + (1 - tpr))/2)
            low = tpr[np.where(fpr<.01)[0][-1]]

            ablation_results.append({'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds, 'auc': auc, 'acc': acc, 'low': low})

            if plot_description is not None:

                # Plot ROC curve
                plt.figure(figsize=(8, 8))
                plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = {:.2f})'.format(auc))
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guessing')

                # Highlight specific points on the ROC curve
                plt.scatter(fpr[np.argmax(acc)], tpr[np.argmax(acc)], marker='o', color='red', label='Max Accuracy')
                plt.scatter(fpr[np.where(fpr < 0.01)[-1][-1]], tpr[np.where(fpr < 0.01)[-1][-1]], marker='x', color='green', label='FPR < 0.01')

                # Set labels and title
                plt.xlabel('False Positive Rate (FPR)')
                plt.ylabel('True Positive Rate (TPR)')
                #plt.title('Receiver Operating Characteristic (ROC) Curve')
                plt.title(f'ROC Curve ({plot_description}) -- ' + ablation_experiment_descriptions[ablation_index])
                plt.legend(loc='lower right')
                plt.grid(True)
                plt.show()

                plt.figure()
                plt.hist(-no_watermark_distance_list[:, ablation_index], label = 'No watermark')
                plt.hist(-Fourier_watermark_distance_list[:, ablation_index], label = 'Fourier watermark')
                plt.title(f'Distance ({plot_description}) -- ' + ablation_experiment_descriptions[ablation_index])
                plt.legend()
                plt.show()
                
        return ablation_results

    def print_ablation_results(ablation_results, description):

        print('━' * 60)
        print(description)
        print('━' * 60)

        for ablation_index in range(len(ablation_results)):
            this_result = ablation_results[ablation_index]
            print(ablation_experiment_descriptions[ablation_index])
            print(f"AUC = {this_result['auc']}, Accuracy = {this_result['acc']}, TPR @ 1% FPR = {this_result['low']}")
            print('━' * 60)
        print()
        print()
        print()

    def print_ablation_results_AUC(ablation_results, description):

        table = PrettyTable()
        table.field_names = ['AUC'] + ablation_experiment_descriptions_short + ['Avg']
        row = [description]
        sum = 0
        for ablation_index in range(len(ablation_results)):
            this_result = ablation_results[ablation_index]
            sum += this_result['auc']
            row.append(f"{this_result['auc']:.4f}")
        row.append(f"{sum / len(ablation_results):.4f}")
        table.add_row(row)
        print(table)

    def print_ablation_results_TPR(ablation_results, description):

        table = PrettyTable()
        table.field_names = ['TPR @ 1% FPR'] + ablation_experiment_descriptions_short + ['Avg']
        row = [description]
        sum = 0
        for ablation_index in range(len(ablation_results)):
            this_result = ablation_results[ablation_index]
            sum += this_result['low']
            row.append(f"{this_result['low']:.4f}")
        row.append(f"{sum / len(ablation_results):.4f}")
        table.add_row(row)
        print(table)

    no_watermark_results_list_array = np.array(no_watermark_results_list)
    Fourier_watermark_results_list_array = np.array(Fourier_watermark_results_list)
    no_watermark_results_list_array.shape, Fourier_watermark_results_list_array.shape

    ablation_results_list = [get_ablation_results(no_watermark_results_list_array[:, :, eval_method_index], Fourier_watermark_results_list_array[:, :, eval_method_index]) for eval_method_index in range(len(eval_methods))]

    for eval_method_index in range(len(eval_methods)):
        print_ablation_results(ablation_results_list[eval_method_index], description = f'{eval_methods[eval_method_index]["Distance"]} {eval_methods[eval_method_index]["Metrics"]}')

    for eval_method_index in range(len(eval_methods)):
        print_ablation_results_AUC(ablation_results_list[eval_method_index], description = f'{eval_methods[eval_method_index]["Distance"]} {eval_methods[eval_method_index]["Metrics"]}')

    for eval_method_index in range(len(eval_methods)):
        print_ablation_results_TPR(ablation_results_list[eval_method_index], description = f'{eval_methods[eval_method_index]["Distance"]} {eval_methods[eval_method_index]["Metrics"]}')

    print()
    quality_metrics.print_average()


if __name__ == '__main__':

    main()

