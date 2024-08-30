from tqdm import tqdm
import torch
import itertools
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
    parser.add_argument('--num_inmost_keys', default=2, type=int)
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

def main(args):
    
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
    username = os.getlogin()
    if args.online:
        pipeline_pretrain = args.model_id
        reference_model_pretrain = args.reference_model_pretrain
        dataset_id = 'Gustavosta/Stable-Diffusion-Prompts'
    else:
        # run offline
        # require additional checks on the exact cache dir. 
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

    if args.channel_min: 
        assert len(HETER_WATERMARK_CHANNEL) > 0

    eval_methods = [
        {'Distance': 'L1', 'Metrics':  '|a-b|        ', 'func': get_distance, 'kwargs': {'p': 1, 'mode': 'complex', 'channel_min': args.channel_min}},
    ]

    base_latents = pipe.get_random_latents()
    base_latents = base_latents.to(torch.float64)
    original_latents_shape = base_latents.shape
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

    # ###### Make RingID pattern
    single_channel_num_slots = RADIUS - RADIUS_CUTOFF
    key_value_list = [[list(combo) for combo in itertools.product(np.linspace(-args.ring_value_range, args.ring_value_range, args.num_inmost_keys).tolist(), repeat = len(RING_WATERMARK_CHANNEL))] for _ in range(single_channel_num_slots)]
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
    
    key_indices_to_evaluate = np.random.choice(ring_capacity, size = args.trials, replace = True).tolist()

    # # Run Evaluation

    print(f'[Info] Ring capacity = {ring_capacity}')

    results_list = []
    score_heads = ['CLIP No Watermark', 'CLIP Fourier Watermark']

    quality_metrics = QualityResultsCollector(score_heads)

    for prompt_index in tqdm(range(len(key_indices_to_evaluate))):

        key_index = key_indices_to_evaluate[prompt_index]

        this_seed = args.general_seed + prompt_index
        this_prompt = dataset[prompt_index][prompt_key]

        set_random_seed(this_seed)
        no_watermark_latents = pipe.get_random_latents()
        Fourier_watermark_latents = generate_Fourier_watermark_latents(
            device = device,
            radius = RADIUS, 
            radius_cutoff = RADIUS_CUTOFF, 
            original_latents = no_watermark_latents, 
            watermark_pattern = Fourier_watermark_pattern_list[key_index], 
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
        # quality_metrics.collect('PSNR', peak_signal_noise_ratio(np.array(no_watermark_image), np.array(Fourier_watermark_image)))
        # quality_metrics.collect('SSIM', structural_similarity(np.array(no_watermark_image), np.array(Fourier_watermark_image), channel_axis = 2))

        # save generated images
        if args.save_generated_imgs:
            #no_watermark_image.save(f'./cache/Run{RUN_NAME}_It{experiment_index:4d}_no_watermark.jpg')
            Fourier_watermark_image.save(os.path.join(save_img_dir, f'Key_{key_index}.Prompt_{prompt_index}.Fourier_watermark.ClipSim_{Fourier_watermark_clip.item():.4f}.jpg'))
            no_watermark_image.save(os.path.join(save_nowatermark_img_dir, f'Key_{key_index}.Prompt_{prompt_index}.Fourier_watermark.ClipSim_{Fourier_watermark_clip.item():.4f}.jpg'))

        # Distort
        distorted_image_list = [
            [None, Fourier_watermark_image],
            image_distortion(None, Fourier_watermark_image, seed = this_seed, r_degree = 75),
            image_distortion(None, Fourier_watermark_image, seed = this_seed, jpeg_ratio = 25),
            image_distortion(None, Fourier_watermark_image, seed = this_seed, crop_scale = 0.75, crop_ratio = 0.75),
            image_distortion(None, Fourier_watermark_image, seed = this_seed, gaussian_blur_r = 8),
            image_distortion(None, Fourier_watermark_image, seed = this_seed, gaussian_std = 0.1),
            image_distortion(None, Fourier_watermark_image, seed = this_seed, brightness_factor = 6),
        ]
        head = ['Clean', 'Rot 75', 'JPEG 25', 'C&S 75', 'Blur 8', 'Noise 0.1', 'Brightness [0, 6]', 'Avg']

        this_it_results = []

        # #### Batch mode
        distorted_image_list = [pair[1] for pair in distorted_image_list]
        Fourier_watermark_image_distorted = torch.stack([transform_img(img) for img in distorted_image_list]).to(text_embeddings.dtype).to(device)
        Fourier_watermark_image_latents = pipe.get_image_latents(Fourier_watermark_image_distorted, sample = False)  # [N, c, h, w]

        Fourier_watermark_reconstructed_latents = pipe.forward_diffusion(
                latents=Fourier_watermark_image_latents,
                text_embeddings=torch.cat([text_embeddings] * len(Fourier_watermark_image_latents)),
                guidance_scale=1,
                num_inference_steps=args.test_num_inference_steps,
            )

        Fourier_watermark_reconstructed_latents_fft = fft(Fourier_watermark_reconstructed_latents)  # [N，c, h, w]

        for single_rec_latent_fft in Fourier_watermark_reconstructed_latents_fft:
            res_per_metric_list = []
            for eval_method in eval_methods:  # 6 metrics
                distances_list = []
                for Fourier_watermark_pattern in Fourier_watermark_pattern_list:  # traverse all gts
                    distance_per_gt = eval_method['func'](Fourier_watermark_pattern, single_rec_latent_fft[None, ...], 
                                                          watermark_region_mask, channel = WATERMARK_CHANNEL, **eval_method['kwargs'])
                    distances_list.append(distance_per_gt)
                acc = np.argmin(np.array(distances_list)) == key_index
                res_per_metric_list.append(acc)
            this_it_results.append(res_per_metric_list)

        results_list.append(this_it_results)

    # ## Print Results
    print('-' * 40)
    print(f'Ring capacity = {ring_capacity}')

    result_array = np.mean(np.array(results_list), axis = 0).T
    assert len(eval_methods) == result_array.shape[0]

    result_array_avg = np.mean(result_array, axis=-1, keepdims=True)  # [M, 1]
    result_array = np.concatenate((result_array, result_array_avg), axis=-1)  # [M, A+1]

    table = PrettyTable()
    table.field_names = ["Id Acc"] + head

    for eval_method_index in range(len(eval_methods)):
        row = [eval_methods[eval_method_index]['Distance'] + ' ' + eval_methods[eval_method_index]['Metrics']]
        for it in range(result_array.shape[1]):
            row.append(f'{result_array[eval_method_index][it]:.3f}')
        table.add_row(row)
    print(table)
    
    print()
    quality_metrics.print_average()

    # ## Save to excel
    num_metrics = len(eval_methods)
    df_exp = pd.DataFrame(result_array, columns=head)

    quality_scores = {}
    for k, v in quality_metrics.return_average().items():
        quality_scores[k] = [v] * num_metrics
    df_qual = pd.DataFrame(quality_scores)

    df_hyper = pd.DataFrame(
        OrderedDict(
            [
                ('User', [username] * num_metrics),
                ('Date', [datetime.now().strftime('%Y.%m.%d')] * num_metrics),
                ('Index', [i for i in range(num_metrics)]),
                ('R_out', [RADIUS] * num_metrics),
                ('R_in', [RADIUS_CUTOFF] * num_metrics),
                ('Assigned keys', [ring_capacity] * num_metrics),
                ('Heter channels', [HETER_WATERMARK_CHANNEL] * num_metrics),
                ('Ring channels', [RING_WATERMARK_CHANNEL] * num_metrics),
                ('Shift factor', [args.time_shift_factor] * num_metrics),
                ('FixGT', ['TRUE' if args.fix_gt else 'FALSE'] * num_metrics),
                ('Centroid', ['TRUE' if ANCHOR_Y_OFFSET == 0 else 'FALSE'] * num_metrics),
                ('Time shift', ['TRUE' if args.time_shift else 'FALSE'] * num_metrics),
                ('Trials', [args.trials] * num_metrics),
                ('Distance', [eval_methods[i]['Distance'] for i in range(num_metrics)]),
                ('Metrics', [eval_methods[i]['Metrics'] for i in range(num_metrics)]),
            ]
        )
    )
    df = pd.concat([df_hyper, df_exp, df_qual], axis=1)
    df.to_csv(os.path.join(save_dir, 'log.csv'), index=False, float_format="%.3f")
    

if __name__ == '__main__':

    args = parse_args()
    main(args)