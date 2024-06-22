import os
import re
import glob
import csv
import time
import math
import struct
import numpy as np
import pandas as pd
import cv2
from pytorch_msssim import ms_ssim
from functools import partial
from shutil import copy, rmtree
import matplotlib.pyplot as plt
import shutil
import torch
from scipy.io import savemat
import scipy.io as scio
from tqdm import tqdm
# import compressai
from torchvision import transforms
from torchvision.transforms import ToPILImage, ToTensor
from PIL import Image
import torch.nn.functional as F


def read_image(filepath: str) -> torch.Tensor:
    assert os.path.isfile(filepath)
    img = Image.open(filepath).convert("RGB")
    return transforms.ToTensor()(img)


def cal_psnr_torch(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = F.mse_loss(a, b).item()
    # print(mse)
    # exit()
    return -10 * math.log10(mse)


def cal_psnr(ref, target):
    diff = ref / 255.0 - target / 255.0
    diff = diff.flatten('C')
    rmse = math.sqrt(np.mean(diff ** 2.))
    return 20 * math.log10(1.0 / (rmse))


def compute_ms_ssim(a, b, max_val: float = 255.0):
    a = torch.from_numpy(a).float().unsqueeze(0)
    if a.size(3) == 3:
        a = a.permute(0, 3, 1, 2)
    b = torch.from_numpy(b).float().unsqueeze(0)
    if b.size(3) == 3:
        b = b.permute(0, 3, 1, 2)
    return ms_ssim(a / 255.0, b / 255.0, data_range=max_val).item()


import numpy as np
import mmcv
from scipy import signal
from scipy.ndimage.filters import convolve


def _calc_msssim_orig(img1, img2):
    v = MultiScaleSSIM(img1, img2, max_val=255)
    if np.isnan(v):
        print(img1[0, :15, :15, 0])
        print(img2[0, :15, :15, 0])
    return np.float32(v)


def MultiScaleSSIM(img1, img2, max_val=255, filter_size=11, filter_sigma=1.5,
                   k1=0.01, k2=0.03, weights=None):
    """Return the MS-SSIM score between `img1` and `img2`.
    This function implements Multi-Scale Structural Similarity (MS-SSIM) Image
    Quality Assessment according to Zhou Wang's paper, "Multi-scale structural
    similarity for image quality assessment" (2003).
    Link: https://ece.uwaterloo.ca/~z70wang/publications/msssim.pdf
    Author's MATLAB implementation:
    http://www.cns.nyu.edu/~lcv/ssim/msssim.zip
    Arguments:
        img1: Numpy array holding the first RGB image batch.
        img2: Numpy array holding the second RGB image batch.
        max_val: the dynamic range of the images (i.e., the difference between the
          maximum the and minimum allowed values).
        filter_size: Size of blur kernel to use (will be reduced for small images).
        filter_sigma: Standard deviation for Gaussian blur kernel (will be reduced
          for small images).
        k1: Constant used to maintain stability in the SSIM calculation (0.01 in
          the original paper).
        k2: Constant used to maintain stability in the SSIM calculation (0.03 in
          the original paper).
        weights: List of weights for each level; if none, use five levels and the
          weights from the original paper.
    Returns:
        MS-SSIM score between `img1` and `img2`.
    Raises:
        RuntimeError: If input images don't have the same shape or don't have four
          dimensions: [batch_size, height, width, depth].
    """
    if img1.shape != img2.shape:
        raise RuntimeError('Input images must have the same shape (%s vs. %s).',
                           img1.shape, img2.shape)
    if img1.ndim != 4:
        raise RuntimeError('Input images must have four dimensions, not %d',
                           img1.ndim)

    # Note: default weights don't sum to 1.0 but do match the paper / matlab code.
    weights = np.array(weights if weights else
                       [0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    levels = weights.size
    downsample_filter = np.ones((1, 2, 2, 1)) / 4.0
    im1, im2 = [x.astype(np.float64) for x in [img1, img2]]
    mssim = np.array([])
    mcs = np.array([])
    for _ in range(levels):
        ssim, cs = _SSIMForMultiScale(
            im1, im2, max_val=max_val, filter_size=filter_size,
            filter_sigma=filter_sigma, k1=k1, k2=k2)
        mssim = np.append(mssim, ssim)
        mcs = np.append(mcs, cs)
        filtered = [convolve(im, downsample_filter, mode='reflect')
                    for im in [im1, im2]]
        im1, im2 = [x[:, ::2, ::2, :] for x in filtered]
    return (np.prod(mcs[0:levels - 1] ** weights[0:levels - 1]) *
            (mssim[levels - 1] ** weights[levels - 1]))


def _FSpecialGauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function."""
    radius = size // 2
    offset = 0.0
    start, stop = -radius, radius + 1
    if size % 2 == 0:
        offset = 0.5
        stop -= 1
    x, y = np.mgrid[offset + start:stop, offset + start:stop]
    assert len(x) == size
    g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    return g / g.sum()


def _SSIMForMultiScale(img1, img2, max_val=255, filter_size=11,
                       filter_sigma=1.5, k1=0.01, k2=0.03):
    """Return the Structural Similarity Map between `img1` and `img2`.
    This function attempts to match the functionality of ssim_index_new.m by
    Zhou Wang: http://www.cns.nyu.edu/~lcv/ssim/msssim.zip
    Arguments:
      img1: Numpy array holding the first RGB image batch.
      img2: Numpy array holding the second RGB image batch.
      max_val: the dynamic range of the images (i.e., the difference between the
        maximum the and minimum allowed values).
      filter_size: Size of blur kernel to use (will be reduced for small images).
      filter_sigma: Standard deviation for Gaussian blur kernel (will be reduced
        for small images).
      k1: Constant used to maintain stability in the SSIM calculation (0.01 in
        the original paper).
      k2: Constant used to maintain stability in the SSIM calculation (0.03 in
        the original paper).
    Returns:
      Pair containing the mean SSIM and contrast sensitivity between `img1` and
      `img2`.
    Raises:
      RuntimeError: If input images don't have the same shape or don't have four
        dimensions: [batch_size, height, width, depth].
    """
    if img1.shape != img2.shape:
        raise RuntimeError('Input images must have the same shape (%s vs. %s).',
                           img1.shape, img2.shape)
    if img1.ndim != 4:
        raise RuntimeError('Input images must have four dimensions, not %d',
                           img1.ndim)

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    _, height, width, _ = img1.shape

    # Filter size can't be larger than height or width of images.
    size = min(filter_size, height, width)

    # Scale down sigma if a smaller filter size is used.
    sigma = size * filter_sigma / filter_size if filter_size else 0

    if filter_size:
        window = np.reshape(_FSpecialGauss(size, sigma), (1, size, size, 1))
        mu1 = signal.fftconvolve(img1, window, mode='valid')
        mu2 = signal.fftconvolve(img2, window, mode='valid')
        sigma11 = signal.fftconvolve(img1 * img1, window, mode='valid')
        sigma22 = signal.fftconvolve(img2 * img2, window, mode='valid')
        sigma12 = signal.fftconvolve(img1 * img2, window, mode='valid')
    else:
        # Empty blur kernel so no need to convolve.
        mu1, mu2 = img1, img2
        sigma11 = img1 * img1
        sigma22 = img2 * img2
        sigma12 = img1 * img2

    mu11 = mu1 * mu1
    mu22 = mu2 * mu2
    mu12 = mu1 * mu2
    sigma11 -= mu11
    sigma22 -= mu22
    sigma12 -= mu12

    # Calculate intermediate values used by both ssim and cs_map.
    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2
    v1 = 2.0 * sigma12 + c2
    v2 = sigma11 + sigma22 + c2
    ssim = np.mean((((2.0 * mu12 + c1) * v1) / ((mu11 + mu22 + c1) * v2)))
    cs = np.mean(v1 / v2)
    return ssim, cs


def VID_test_x265():
    fun = 3

    path = r'E:\dataset\Video\VID\VID_val_ldp_default'  # VID_val_ldp_default  VID_val_veryslow
    # path = r'E:\dataset\Video\VID\VID_resize_processed_val'
    val_frames = os.path.join(path, 'org_frames_v100_f50')
    test_frames = 50  # 555 seqs
    test_videos = 100

    # JPEG2YUV
    if fun == 0:
        yuv_path = os.path.join(path, 'YUV')
        os.makedirs(yuv_path, exist_ok=True)
        # file = open(os.path.join(path, f'ffmpeg.bat'), 'w')
        dirs = sorted(os.listdir(val_frames))[:test_videos]
        for p1 in dirs:
            images = glob.glob(os.path.join(path, 'org_frames_v100_f50', p1, "*.JPEG"))
            shape = cv2.imread(images[0], -1).shape
            temp = os.path.join(path, 'org_frames_v100_f50', p1, "%6d.JPEG")
            out = os.path.join(path, 'YUV', p1 + '.yuv')
            print(p1, len(images), f'{shape[1]}x{shape[0]}', out)
            # file.write(f'ffmpeg -r 30 -vframes {test_frames} -i {temp} -pix_fmt yuv420p -s {shape[1]}x{shape[0]} {out}\n')
            os.system(f'ffmpeg -f image2 -y -r 30 -i {temp} -pix_fmt yuv420p -s {shape[1]}x{shape[0]} {out}')
            # file.write('pause')
            # exit()
        # file.write('pause')
        # file.close()
    # Encode Decode
    elif fun == 1:
        for crf in [15, 19, 23, 27, 31, 35]:
            file = open(os.path.join(path, f'x265_{crf}.bat'), 'w')
            for i, p1 in enumerate(os.listdir(val_frames)):
                images = glob.glob(os.path.join(path, 'org_frames_v100_f50', p1, "*.JPEG"))
                shape = cv2.imread(images[0], -1).shape
                save_dec_p1 = os.path.join(path, 'mkv', f'crf_{crf}')
                os.makedirs(save_dec_p1, exist_ok=True)
                seq = os.path.join(path, 'YUV', p1 + '.yuv')
                h265 = os.path.join(path, 'mkv', f'crf_{crf}', p1 + '.mkv')
                csv_path = os.path.join(path, 'csv', f'crf_{crf}')
                os.makedirs(csv_path, exist_ok=True)
                print(i, p1, seq, h265, len(images))  # veryfast, veryslow, LDP default: -c:v libx265 -tune zerolatency
                file.write(
                    f'ffmpeg -y -pix_fmt yuv420p -s {shape[1]}x{shape[0]} -framerate 30 -i {seq} -vframes {len(images)}'
                    f' -c:v libx265 -preset veryfast -tune zerolatency -x265-params "crf={crf}:keyint=10:csv-log-level=1:csv=./csv/crf_{crf}/{p1}.csv:verbose=1:psnr=1" '
                    f' {h265}\n')
                # file.write('pause')
                # exit()
            file.write('pause')
            file.close()
    # mkv2JPEG
    elif fun == 2:
        # ffmpeg -pix_fmt yuv420p -s 832x480 -i BasketballDrillText_832x480_50.yuv pics/f%03d.png
        file = open(os.path.join(path, f'mkv2png.bat'), 'w')
        for p1 in os.listdir(val_frames):
            for crf in [15, 19, 23, 27, 31, 35]:
                os.makedirs(os.path.join(path, 'dec_img', f'crf_{crf}', p1), exist_ok=True)
                temp = os.path.join(path, 'dec_img', f'crf_{crf}', p1, "%%6d.JPEG")
                mkv = os.path.join(path, 'mkv', f'crf_{crf}', p1 + '.mkv')
                file.write(f'ffmpeg -i {mkv} -f image2 {temp}\n')
        file.write('pause')
        file.close()
    # cal PSNR
    elif fun == 3:
        log = open(os.path.join(path, f'result.txt'), 'w')
        for crf in [15, 19, 23, 27, 31, 35]:
            bpp_all, psnr1_all, psnr2_all = [], [], []
            for i, p1 in enumerate(os.listdir(val_frames)):
                images = glob.glob(os.path.join(path, 'org_frames_v100_f50', p1, "*.JPEG"))
                shape = cv2.imread(images[0], -1).shape
                psnr = []
                # print(crf, i, p1, len(images))
                for image in images:
                    name = image.split('\\')[-1].split('.')[0]
                    im = os.path.join(path, 'dec_img', f'crf_{crf}', p1, str(int(name) + 1).zfill(6) + '.JPEG')
                    org = cv2.imread(image, -1)
                    dec = cv2.imread(im, -1)
                    # print(org.shape, np.min(org), np.max(org))
                    # print(dec.shape, np.min(dec), np.max(dec))
                    # exit()
                    # mse = np.mean((org - dec) ** 2)
                    # _psnr = 10 * np.log10(255.0 ** 2 / mse)
                    # print(_psnr, 10 * np.log10(255 * 255 / np.mean((im1 - im2) ** 2)))
                    # exit()
                    _psnr = cal_psnr(org, dec)
                    # _psnr1 = cal_psnr1(org * 1.0, dec * 1.0)
                    # print(_psnr1, _psnr)
                    psnr.append(_psnr)
                csv_path = os.path.join(path, 'csv', f'crf_{crf}', p1 + '.csv')
                data = pd.read_csv(csv_path)
                psnr1 = np.mean(data[[' Y PSNR']].values)
                bpp = np.mean(data[[' Bits']].values) / (shape[0] * shape[1])
                print(i + 1, p1, 'crf=', crf, np.mean(psnr), bpp, psnr1)
                # exit()
                bpp_all.append(bpp)
                psnr1_all.append(np.mean(psnr))
                psnr2_all.append(psnr1)
            log.write(f'crf = {crf}, RGB_psnr = {np.mean(psnr1_all):.3f}, Y_pnsr_ffmepg = {np.mean(psnr2_all):.3f}, '
                      f'bpp = {np.mean(bpp_all):.3f}\n')
            print(
                f'crf = {crf}, RGB_psnr = {np.mean(psnr1_all):.3f}, Y_pnsr_ffmepg = {np.mean(psnr2_all):.3f}, '
                f'bpp = {np.mean(bpp_all):.3f}')
        log.close()

    return 0


def copy_VID_frames():
    frames = 96
    path = '/media/user/f126fd00-4370-4fdf-9a0a-fc93e66106eb1/LHB/datasets/ImageNet/VID2015/ILSVRC/Data/VID/val'
    path_tgt = '/media/user/f126fd00-4370-4fdf-9a0a-fc93e66106eb1/LHB/datasets/ImageNet/VID2015/ILSVRC/Data/VID/val_video_all_f96'
    dirs = sorted(os.listdir(path))
    for i, p1 in enumerate(dirs):
        print(i, p1)
        os.makedirs(os.path.join(path_tgt, p1), exist_ok=True)
        images = sorted(glob.glob(os.path.join(path, p1, "*.JPEG")))
        for k, image in enumerate(images):
            if k > frames - 1:
                break
            tgt = os.path.join(path_tgt, p1)
            shutil.copy(image, tgt)
    return 0


def plot_ctc_D_ffmpeg():
    # font = {'family': 'Arial', 'weight': 'normal', 'size': 14}
    # matplotlib.rc('font', **font)
    LineWidth = 2

    bpp = [0.141631075, 0.206291275, 0.2892303, 0.4370301311]
    psnr = [28.41229473, 29.72853673, 31.1727661, 32.53451213]
    DVC, = plt.plot(bpp, psnr, "k-o", linewidth=LineWidth, label='LuGuoDVC')
    # h264
    bpp = [0.6135016547, 0.3672749837, 0.2190138075, 0.1305982802]
    psnr = [34.30692118, 31.91254879, 29.68591275, 27.60142272]
    h264, = plt.plot(bpp, psnr, "m--s", linewidth=LineWidth, label='H.264 Very fast from LuGuoDVC')
    # h265
    bpp = [0.7361206055, 0.4330858019, 0.2476169162, 0.1408860948]
    psnr = [35.73861849, 33.21075298, 30.79006456, 28.48721492]
    h265, = plt.plot(bpp, psnr, "r--v", linewidth=LineWidth, label='H.265 Very fast from LuGuoDVC')

    # h265
    bpp = [0.795, 0.467, 0.267, 0.152, 0.087]
    psnr = [35.878, 33.344, 30.922, 28.640, 26.448]
    h2651, = plt.plot(bpp, psnr, "b--v", linewidth=LineWidth, label='H.265 Very fast 2080ti')

    # h264
    bpp = [0.641, 0.385, 0.231, 0.138, 0.084]
    psnr = [34.489, 32.075, 29.841, 27.738, 25.722]
    h2641, = plt.plot(bpp, psnr, "g--v", linewidth=LineWidth, label='H.264 Very fast 2080ti')

    plt.legend(handles=[DVC, h264, h265, h2651, h2641], loc=4)
    plt.grid()
    plt.xlabel('BPP')
    plt.ylabel('PSNR(dB)')
    plt.title('HEVC Class D dataset')
    plt.savefig('/media/user/f126fd00-4370-4fdf-9a0a-fc93e66106eb1/LHB/Testing/TestD/test_D.png')
    plt.show()
    plt.clf()
    return 0


def test_ctc_D_x264_org_yuv():
    fun = 2

    path = '/media/user/f126fd00-4370-4fdf-9a0a-fc93e66106eb1/LHB/Testing/TestD'  # VID_val_ldp_default  VID_val_veryslow
    # path = r'E:\dataset\Video\VID\VID_resize_processed_val'
    seqs = os.path.join(path, 'seqs')
    tgt = 'x264veryfast'

    # Encode Decode
    if fun == 1:
        # 15, 19, 23, 27, 31,
        for crf in [15, 19, 23, 27, 31]:
            for i, p1 in enumerate(os.listdir(seqs)):
                fps = p1.split('.')[0].split('_')[-1]
                save_dec_p1 = os.path.join(path, tgt, 'mkv', f'crf_{crf}')
                os.makedirs(save_dec_p1, exist_ok=True)
                seq = os.path.join(path, 'seqs', p1)
                h264 = os.path.join(path, tgt, 'mkv', f'crf_{crf}', p1.replace('yuv', 'mkv'))
                csv_path = os.path.join(path, tgt, 'csv', f'crf_{crf}')
                nn = p1.replace('yuv', 'log')
                csv_file = os.path.join(path, tgt, 'csv', f'crf_{crf}', f'{nn}')
                os.makedirs(csv_path, exist_ok=True)
                print(i, p1, seq, h264)  # veryfast, veryslow, LDP default: -c:v libx265 -tune zerolatency

                os.system(
                    f'FFREPORT=file={csv_file}:level=56 ffmpeg -y -pix_fmt yuv420p -s 384x192 -framerate {fps} -i {seq} -vframes 100'
                    f' -c:v libx264 -preset veryfast -tune zerolatency -crf {crf} -g 10 -bf 2 -b_strategy 0 -sc_threshold 0 -loglevel debug {h264}')

                os.makedirs(os.path.join(path, tgt, 'dec_img', f'crf_{crf}', p1.split('.')[0]), exist_ok=True)
                temp = os.path.join(path, tgt, 'dec_img', f'crf_{crf}', p1.split('.')[0], "f%03d.png")
                mkv = os.path.join(path, tgt, 'mkv', f'crf_{crf}', p1.replace('yuv', 'mkv'))
                os.system(f'ffmpeg -i {mkv} -f image2 {temp}\n')
                # exit()

    # cal PSNR
    elif fun == 2:
        log = open(os.path.join(path, tgt, f'result.txt'), 'w')
        for crf in [15, 19, 23, 27, 31]:
            bpp_all, psnr1_all, psnr2_all = [], [], []
            for i, p1 in enumerate(os.listdir(seqs)):
                images = glob.glob(os.path.join(path, 'org_frames', p1.split('.')[0], "*.png"))
                shape = cv2.imread(images[0], -1).shape
                psnr = []
                # print(crf, i, p1, len(images))
                # exit()
                for image in images:
                    name = image.split('/')[-1]
                    im = os.path.join(path, tgt, 'dec_img', f'crf_{crf}', p1.split('.')[0], name)
                    org = cv2.imread(image, -1)
                    dec = cv2.imread(im, -1)
                    # print(org.shape, np.min(org), np.max(org))
                    # print(dec.shape, np.min(dec), np.max(dec))
                    # exit()

                    _psnr = cal_psnr(org, dec)
                    psnr.append(_psnr)
                log_path = os.path.join(path, tgt, 'csv', f'crf_{crf}', p1.split('.')[0] + '.log')
                with open(log_path, 'r') as f:
                    lines = f.readlines()
                size_line = []
                count = 0
                for l in lines:
                    if ", size " in l:
                        count += 1
                        size = l.split(',')[1]
                        size_line.append(int(size[5:]))
                size_line = np.array(size_line) * 8.0 / (384 * 192)
                bpp = np.array(size_line).mean(0)
                # print(i + 1, p1, 'crf=', crf, np.mean(psnr), bpp)
                # exit()
                bpp_all.append(bpp)
                psnr1_all.append(np.mean(psnr))
                # psnr2_all.append(psnr1)
            log.write(f'crf = {crf}, RGB_psnr = {np.mean(psnr1_all):.3f}, '
                      f'bpp = {np.mean(bpp_all):.3f}\n')
            print(
                f'crf = {crf}, RGB_psnr = {np.mean(psnr1_all):.3f}, '
                f'bpp = {np.mean(bpp_all):.3f}')
        log.close()

    elif fun == 3:
        path = '/media/user/f126fd00-4370-4fdf-9a0a-fc93e66106eb1/LHB/Testing/TestD'
        seqs = os.path.join(path, 'seqs')
        for seq in os.listdir(seqs):
            yuv = os.path.join(path, 'seqs', seq)
            img_path = os.path.join(path, 'org_frames', seq.split('.')[0])
            # print(yuv, img_path)
            # exit()
            os.makedirs(img_path, exist_ok=True)
            os.system(f'ffmpeg -pix_fmt yuv420p -s 384x192 -i {yuv} -vframes 100 '
                      f'-f image2 {img_path}/f%03d.png')

    return 0


def test_ctc_D_x265_org_yuv():
    fun = 2

    path = '/media/user/f126fd00-4370-4fdf-9a0a-fc93e66106eb1/LHB/Testing/TestD'  # VID_val_ldp_default  VID_val_veryslow
    # path = r'E:\dataset\Video\VID\VID_resize_processed_val'
    seqs = os.path.join(path, 'seqs')
    tgt = 'x265veryfast'

    # Encode Decode
    if fun == 1:
        # 15, 19, 23, 27, 31,
        for crf in [15, 19, 23, 27, 31]:
            for i, p1 in enumerate(os.listdir(seqs)):
                fps = p1.split('.')[0].split('_')[-1]
                save_dec_p1 = os.path.join(path, tgt, 'mkv', f'crf_{crf}')
                os.makedirs(save_dec_p1, exist_ok=True)
                seq = os.path.join(path, 'seqs', p1)
                h265 = os.path.join(path, tgt, 'mkv', f'crf_{crf}', p1.replace('yuv', 'mkv'))
                csv_path = os.path.join(path, tgt, 'csv', f'crf_{crf}')
                nn = p1.replace('yuv', 'log')
                csv_file = os.path.join(path, tgt, 'csv', f'crf_{crf}', f'{nn}')
                os.makedirs(csv_path, exist_ok=True)
                print(i, p1, seq, h265)  # veryfast, veryslow, LDP default: -c:v libx265 -tune zerolatency

                os.system(
                    f'FFREPORT=file={csv_file}:level=56 ffmpeg -y -pix_fmt yuv420p -s 384x192 -framerate {fps} -i {seq} -vframes 100'
                    f' -c:v libx265 -preset veryfast -tune zerolatency -x265-params "crf={crf}:keyint=10" {h265}')

                os.makedirs(os.path.join(path, tgt, 'dec_img', f'crf_{crf}', p1.split('.')[0]), exist_ok=True)
                temp = os.path.join(path, tgt, 'dec_img', f'crf_{crf}', p1.split('.')[0], "f%03d.png")
                mkv = os.path.join(path, tgt, 'mkv', f'crf_{crf}', p1.replace('yuv', 'mkv'))
                os.system(f'ffmpeg -i {mkv} -f image2 {temp}\n')
                # exit()

    # cal PSNR
    elif fun == 2:
        log = open(os.path.join(path, tgt, f'result.txt'), 'w')
        for crf in [15, 19, 23, 27, 31]:
            bpp_all, psnr1_all, psnr2_all = [], [], []
            for i, p1 in enumerate(os.listdir(seqs)):
                images = glob.glob(os.path.join(path, 'org_frames', p1.split('.')[0], "*.png"))
                shape = cv2.imread(images[0], -1).shape
                psnr = []
                # print(crf, i, p1, len(images))
                # exit()
                for image in images:
                    name = image.split('/')[-1]
                    im = os.path.join(path, tgt, 'dec_img', f'crf_{crf}', p1.split('.')[0], name)
                    org = cv2.imread(image, -1)
                    dec = cv2.imread(im, -1)
                    # print(org.shape, np.min(org), np.max(org))
                    # print(dec.shape, np.min(dec), np.max(dec))
                    # exit()

                    _psnr = cal_psnr(org, dec)
                    psnr.append(_psnr)
                log_path = os.path.join(path, tgt, 'csv', f'crf_{crf}', p1.split('.')[0] + '.log')
                with open(log_path, 'r') as f:
                    lines = f.readlines()
                size_line = []
                count = 0
                for l in lines:
                    if ", size " in l:
                        count += 1
                        size = l.split(',')[1]
                        size_line.append(int(size[5:]))
                size_line = np.array(size_line) * 8.0 / (384 * 192)
                bpp = np.array(size_line).mean(0)
                # print(i + 1, p1, 'crf=', crf, np.mean(psnr), bpp)
                # exit()
                bpp_all.append(bpp)
                psnr1_all.append(np.mean(psnr))
                # psnr2_all.append(psnr1)
            log.write(f'crf = {crf}, RGB_psnr = {np.mean(psnr1_all):.3f}, '
                      f'bpp = {np.mean(bpp_all):.3f}\n')
            print(
                f'crf = {crf}, RGB_psnr = {np.mean(psnr1_all):.3f}, '
                f'bpp = {np.mean(bpp_all):.3f}')
        log.close()

    elif fun == 3:
        path = '/media/user/f126fd00-4370-4fdf-9a0a-fc93e66106eb1/LHB/Testing/TestD'
        seqs = os.path.join(path, 'seqs')
        for seq in os.listdir(seqs):
            yuv = os.path.join(path, 'seqs', seq)
            img_path = os.path.join(path, 'org_frames', seq.split('.')[0])
            # print(yuv, img_path)
            # exit()
            os.makedirs(img_path, exist_ok=True)
            os.system(f'ffmpeg -pix_fmt yuv420p -s 384x192 -i {yuv} -vframes 100 '
                      f'-f image2 {img_path}/f%03d.png')

    return 0


def plot_VID_ffmpeg():
    # font = {'family': 'Arial', 'weight': 'normal', 'size': 14}
    # matplotlib.rc('font', **font)
    LineWidth = 2

    # h265
    bpp = [0.393, 0.268, 0.175, 0.105, 0.058, 0.032]
    psnr = [45.258, 44.345, 42.012, 39.953, 37.924, 36.138]
    h2651, = plt.plot(bpp, psnr, "b--v", linewidth=LineWidth, label='H.265 LDP Default')

    # h265
    bpp = [0.415, 0.278, 0.175, 0.104, 0.058, 0.032]
    psnr = [45.258, 44.270, 41.970, 39.963, 37.910, 36.113]
    h2652, = plt.plot(bpp, psnr, "r--v", linewidth=LineWidth, label='H.265 Very fast')

    plt.legend(handles=[h2651, h2652], loc=4)
    plt.grid()
    plt.xlabel('BPP')
    plt.ylabel('PSNR(dB)')
    plt.title('ILSVRC dataset')
    plt.savefig('./test_ILSVRC.png')
    plt.show()
    plt.clf()
    return 0


def VID_x265_veryfast():
    fun = 2

    path = '/media/user/f126fd00-4370-4fdf-9a0a-fc93e66106eb1/LHB/datasets/ImageNet/VID2015/ILSVRC/Data/VID'  # VID_val_ldp_default  VID_val_veryslow
    # path = r'E:\dataset\Video\VID\VID_resize_processed_val'
    val_frames = os.path.join(path, 'val')
    tgt = 'x256veryfast_gop12'
    frames = 96

    # JPEG2YUV
    if fun == 0:
        yuv_path = os.path.join(path, 'val_YUV')
        os.makedirs(yuv_path, exist_ok=True)
        dirs = sorted(os.listdir(val_frames))
        for p1 in dirs:
            images = glob.glob(os.path.join(path, 'val', p1, "*.JPEG"))
            shape = cv2.imread(images[0], -1).shape
            temp = os.path.join(path, 'val', p1, "%6d.JPEG")
            out = os.path.join(path, 'val_YUV', p1 + '.yuv')
            print(p1, len(images), f'{shape[1]}x{shape[0]}', out)
            os.system(f'ffmpeg -f image2 -y -r 20 -i {temp} -pix_fmt yuv420p -s {shape[1]}x{shape[0]} {out}')

    # Encode Decode
    elif fun == 1:
        # 19, 23, 27, 31, 35, 39
        for crf in [19, 23, 27, 31, 35, 39]:
            for i, p1 in enumerate(os.listdir(val_frames)):
                images = glob.glob(os.path.join(path, 'val', p1, "*.JPEG"))
                shape = cv2.imread(images[0], -1).shape
                # print(shape)
                # exit()
                save_dec_p1 = os.path.join(path, tgt, 'mkv', f'crf_{crf}')
                os.makedirs(save_dec_p1, exist_ok=True)
                seq = os.path.join(path, 'val_YUV', p1 + '.yuv')
                h265 = os.path.join(path, tgt, 'mkv', f'crf_{crf}', p1 + '.mkv')
                csv_path = os.path.join(path, tgt, 'csv', f'crf_{crf}')
                csv_file = os.path.join(path, tgt, 'csv', f'crf_{crf}', f'{p1}.log')
                os.makedirs(csv_path, exist_ok=True)
                print('********', i, p1, seq, h265,
                      len(images))  # veryfast, veryslow, LDP default: -c:v libx265 -tune zerolatency

                # FFREPORT=file=ffreport.log:level=56 ffmpeg -pix_fmt yuv420p -s $2x$3 -i $input -vframes 100 -c:v libx265 -tune zerolatency -x265-params "crf=$1:keyint=10:verbose=1" out/h265/out.mkv
                os.system(
                    f'FFREPORT=file={csv_file}:level=56 ffmpeg -y -pix_fmt yuv420p -s {shape[1]}x{shape[0]} -framerate 30 -i {seq} -vframes {frames} '
                    f' -c:v libx265 -preset veryfast -tune zerolatency -x265-params "crf={crf}:keyint=12" {h265}')

                os.makedirs(os.path.join(path, tgt, 'dec_img', f'crf_{crf}', p1), exist_ok=True)
                temp = os.path.join(path, tgt, 'dec_img', f'crf_{crf}', p1, "%6d.JPEG")
                mkv = os.path.join(path, tgt, 'mkv', f'crf_{crf}', p1 + '.mkv')
                os.system(f'ffmpeg -i {mkv} -vframes {frames} -f image2 {temp}\n')
                # exit()

        # for p1 in os.listdir(val_frames):
        #     for crf in [15, 19, 23, 27, 31, 35]:
        #         os.makedirs(os.path.join(path, tgt, 'dec_img', f'crf_{crf}', p1), exist_ok=True)
        #         temp = os.path.join(path, tgt, 'dec_img', f'crf_{crf}', p1, "%%6d.JPEG")
        #         mkv = os.path.join(path, tgt, 'mkv', f'crf_{crf}', p1 + '.mkv')
        #         os.system(f'ffmpeg -i {mkv} -f image2 {temp}\n')

    # cal PSNR
    elif fun == 2:
        log = open(os.path.join(path, tgt, f'result_video_all_f96.txt'), 'w')
        for crf in [19, 23, 27, 31, 35, 39]:
            bpp_all, psnr1_all, psnr2_all = [], [], []
            kk = sorted(os.listdir(val_frames))
            for i, p1 in enumerate(kk):
                # if i > 99:
                #     break
                images = glob.glob(os.path.join(path, 'val', p1, "*.JPEG"))
                shape = cv2.imread(images[0], -1).shape
                psnr = []
                # print(crf, i, p1, len(images))
                # exit()
                images = sorted(images)
                for j, image in enumerate(images):
                    if j >= frames:
                        break
                    # print(j)
                    name = image.split('/')[-1].split('.')[0]
                    im = os.path.join(path, tgt, 'dec_img', f'crf_{crf}', p1, str(int(name) + 1).zfill(6) + '.JPEG')
                    org = cv2.imread(image, -1)
                    dec = cv2.imread(im, -1)
                    # print(org.shape, np.min(org), np.max(org))
                    # print(dec.shape, np.min(dec), np.max(dec))
                    # exit()
                    _psnr = cal_psnr(org, dec)
                    # print(_psnr1, _psnr)
                    psnr.append(_psnr)

                log_path = os.path.join(path, tgt, 'csv', f'crf_{crf}', p1 + '.log')
                with open(log_path, 'r') as f:
                    lines = f.readlines()
                size_line = []
                count = 0
                for l in lines:
                    if ", size " in l:
                        count += 1
                        size = l.split(',')[1]
                        size_line.append(int(size[5:]))
                        if count >= frames:
                            break
                # print(size_line)
                # print(len(size_line))
                # print(crf, i, p1)
                # exit()
                size_line = np.array(size_line) * 8.0 / (shape[0] * shape[1])
                bpp = np.array(size_line).mean(0)
                print(i + 1, p1, 'crf=', crf, np.mean(psnr), bpp)
                # exit()
                bpp_all.append(bpp)
                psnr1_all.append(np.mean(psnr))
                # psnr2_all.append(psnr1)
            log.write(f'crf = {crf}, RGB_psnr = {np.mean(psnr1_all):.3f}, '
                      f'bpp = {np.mean(bpp_all):.3f}\n')
            print(
                f'crf = {crf}, RGB_psnr = {np.mean(psnr1_all):.3f}, '
                f'bpp = {np.mean(bpp_all):.3f}')
        log.close()

    return 0


def VID_x264_veryfast():
    fun = 2

    path = '/media/user/f126fd00-4370-4fdf-9a0a-fc93e66106eb1/LHB/datasets/ImageNet/VID2015/ILSVRC/Data/VID'  # VID_val_ldp_default  VID_val_veryslow
    # path = r'E:\dataset\Video\VID\VID_resize_processed_val'
    val_frames = os.path.join(path, 'val')
    tgt = 'x264veryfast_gop12'
    frames = 96

    # Encode Decode
    if fun == 1:
        # 19, 23, 27, 31, 35, 39
        for crf in [19, 23, 27, 31, 35, 39]:
            for i, p1 in enumerate(os.listdir(val_frames)):
                images = glob.glob(os.path.join(path, 'val', p1, "*.JPEG"))
                shape = cv2.imread(images[0], -1).shape
                save_dec_p1 = os.path.join(path, tgt, 'mkv', f'crf_{crf}')
                os.makedirs(save_dec_p1, exist_ok=True)
                seq = os.path.join(path, 'val_YUV', p1 + '.yuv')
                h264 = os.path.join(path, tgt, 'mkv', f'crf_{crf}', p1 + '.mkv')
                csv_path = os.path.join(path, tgt, 'csv', f'crf_{crf}')
                csv_file = os.path.join(path, tgt, 'csv', f'crf_{crf}', f'{p1}.log')
                os.makedirs(csv_path, exist_ok=True)
                print(i, p1, seq, h264, len(images))  # veryfast, veryslow, LDP default: -c:v libx265 -tune zerolatency

                os.system(
                    f'FFREPORT=file={csv_file}:level=56 ffmpeg -y -pix_fmt yuv420p -s {shape[1]}x{shape[0]} -framerate 30 -i {seq} -vframes {frames}'
                    f' -c:v libx264 -preset veryfast -tune zerolatency -crf {crf} -g 12 -bf 2 -b_strategy 0 -sc_threshold 0 -loglevel debug {h264}')

                os.makedirs(os.path.join(path, tgt, 'dec_img', f'crf_{crf}', p1), exist_ok=True)
                temp = os.path.join(path, tgt, 'dec_img', f'crf_{crf}', p1, "%6d.JPEG")
                mkv = os.path.join(path, tgt, 'mkv', f'crf_{crf}', p1 + '.mkv')
                os.system(f'ffmpeg -i {mkv} -vframes {frames} -f image2 {temp}')

                # exit()

    # cal PSNR
    elif fun == 2:
        log = open(os.path.join(path, tgt, f'result_video_all_f96.txt'), 'w')
        for crf in [19, 23, 27, 31, 35, 39]:
            bpp_all, psnr1_all, psnr2_all = [], [], []
            kk = sorted(os.listdir(val_frames))
            for i, p1 in enumerate(kk):
                # if i > 99:
                #     break
                images = glob.glob(os.path.join(path, 'val', p1, "*.JPEG"))
                shape = cv2.imread(images[0], -1).shape
                psnr = []
                # print(crf, i, p1, len(images))
                # exit()
                images = sorted(images)
                for j, image in enumerate(images):
                    if j >= frames:
                        break
                    # print(j)
                    name = image.split('/')[-1].split('.')[0]
                    im = os.path.join(path, tgt, 'dec_img', f'crf_{crf}', p1, str(int(name) + 1).zfill(6) + '.JPEG')
                    org = cv2.imread(image, -1)
                    dec = cv2.imread(im, -1)
                    # print(org.shape, np.min(org), np.max(org))
                    # print(dec.shape, np.min(dec), np.max(dec))
                    # exit()
                    _psnr = cal_psnr(org, dec)
                    # print(_psnr1, _psnr)
                    psnr.append(_psnr)

                log_path = os.path.join(path, tgt, 'csv', f'crf_{crf}', p1 + '.log')
                with open(log_path, 'r') as f:
                    lines = f.readlines()
                size_line = []
                count = 0
                for l in lines:
                    if ", size " in l:
                        count += 1
                        size = l.split(',')[1]
                        size_line.append(int(size[5:]))
                        if count >= frames:
                            break
                # print(size_line)
                # print(len(size_line))
                # print(crf, i, p1)
                # exit()
                size_line = np.array(size_line) * 8.0 / (shape[0] * shape[1])
                bpp = np.array(size_line).mean(0)
                print(i + 1, p1, 'crf=', crf, np.mean(psnr), bpp)
                # exit()
                bpp_all.append(bpp)
                psnr1_all.append(np.mean(psnr))
                # psnr2_all.append(psnr1)
            log.write(f'crf = {crf}, RGB_psnr = {np.mean(psnr1_all):.3f}, '
                      f'bpp = {np.mean(bpp_all):.3f}\n')
            print(
                f'crf = {crf}, RGB_psnr = {np.mean(psnr1_all):.3f}, '
                f'bpp = {np.mean(bpp_all):.3f}')
        log.close()

    return 0


def plot_VID_PSNR():
    LineWidth = 2

    # bpp = [0.1854, 0.1165, 0.0697, 0.0461]
    # psnr = [42.3014, 40.463, 38.6080, 37.0012]
    # DVC, = plt.plot(bpp, psnr, "k-o", linewidth=LineWidth, label='DVC-bpg')

    bpp = [0.0550, 0.0794, 0.1237, 0.1836]
    psnr = [37.374, 38.7981, 40.6230, 42.428]
    DVC1, = plt.plot(bpp, psnr, "k-o", linewidth=LineWidth, label='DVC-mbt2018')

    # h265 veryfast
    bpp = [0.391, 0.251, 0.157, 0.093, 0.052, 0.028]
    psnr = [44.928, 43.778, 41.987, 39.586, 37.345, 35.432]
    h265veryfast, = plt.plot(bpp, psnr, "b--v", linewidth=LineWidth, label='H.265 Very fast 2080ti')

    # h265 veryslow
    bpp = [0.361, 0.234, 0.153, 0.098, 0.057, 0.030]
    psnr = [45.251, 44.400, 42.986, 40.879, 38.451, 36.096]
    h265veryslow, = plt.plot(bpp, psnr, "b--v", linewidth=LineWidth, label='H.265 Very fast 2080ti')

    # h264 veryfast
    bpp = [0.385, 0.240, 0.144, 0.081, 0.042, 0.021]
    psnr = [44.911, 43.714, 41.871, 39.380, 37.075, 35.123]
    h264veryfast, = plt.plot(bpp, psnr, "g--v", linewidth=LineWidth, label='H.264 Very fast 2080ti')

    plt.legend(handles=[DVC1, h264veryfast, h265veryfast, h265veryslow], loc=4)
    plt.grid()
    plt.xlabel('BPP')
    plt.ylabel('PSNR(dB)')
    plt.title('HEVC Class D dataset')
    plt.savefig('/home/user/Desktop/LHB/1.png')
    plt.show()
    plt.clf()
    return 0


def plot_VID_mAP():
    LineWidth = 2

    bpp = [0.1854, 0.1165, 0.0697, 0.0461]
    psnr = [42.3014, 40.463, 38.6080, 37.0012]
    DVC, = plt.plot(bpp, psnr, "k-o", linewidth=LineWidth, label='DVC')

    # h265 veryfast
    bpp = [0.391, 0.251, 0.157, 0.093, 0.052, 0.028]
    bbox_mAP = [35.80, 35.80, 35.70, 35.20, 34.70, 33.50]
    bbox_mAP_50 = [56.80, 56.80, 55.70, 56.00, 55.70, 53.80]
    bbox_mAP_75 = [40.3, 40.5, 40.20, 38.40, 38.20, 36.60]
    bbox_mAP_s = [0.00, 0.0, 0.80, 0.0, 0.0, 0.0]
    bbox_mAP_m = [11.10, 11.90, 11.50, 10.60, 11.50, 11.20]
    bbox_mAP_l = [43.00, 43.20, 43.00, 42.30, 41.60, 40.20]
    h265, = plt.plot(bpp, bbox_mAP, "b--v", linewidth=LineWidth, label='H.265 Very fast 2080ti')

    # # h264
    # bpp = [0.385, 0.240, 0.144, 0.081, 0.042, 0.021]
    # psnr = [44.911, 43.714, 41.871, 39.380, 37.075, 35.123]
    # h264, = plt.plot(bpp, psnr, "g--v", linewidth=LineWidth, label='H.264 Very fast 2080ti')

    plt.legend(handles=[DVC, h264, h265], loc=4)
    plt.grid()
    plt.xlabel('BPP')
    plt.ylabel('PSNR(dB)')
    plt.title('HEVC Class D dataset')
    plt.savefig('/home/user/Desktop/LHB/1.png')
    plt.show()
    plt.clf()
    return 0


def read_txt():
    path = '/home/user/Desktop/LHB/0726/log2048.txt'
    with open(path, 'r') as f:
        lines = f.readlines()
        print(len(lines))
        print(lines[0].split(' '))
        print(lines[1].split(' '))
        print('IFrame' in lines[0])
        print('P-Frame' in lines[1])
        # exit()
        Ibpp, Pbpp, Mvbpp, Resbpp, Warppsnr, Mcpsnr, Ipsnr, Ppsnr, Ienct, Idect, Penct, Pdect = \
            [], [], [], [], [], [], [], [], [], [], [], []
        for ii, line in enumerate(lines):
            temp = line.split(' ')
            if 'IFrame' in line:
                bpp = float(temp[5 + 1])
                psnr = float(temp[8 + 1])
                msssim = float(temp[11 + 1])
                enct = float(temp[15 + 1][:-1])
                dect = float(temp[19 + 1][:-2].strip())
                print(bpp, psnr, msssim, enct, dect)
                Ibpp.append(bpp)
                Ipsnr.append(psnr)
                Ienct.append(enct)
                Idect.append(dect)
            elif 'P-Frame' in line:
                mvbpp = float(temp[5 + 1][1:-1])
                resbpp = float(temp[6 + 1][:-1])
                bpp = float(temp[7 + 1][:-1])
                warppsnr = float(temp[10 + 1][1:-1])
                mcpsnr = float(temp[11 + 1][:-1])
                psnr = float(temp[12 + 1][:-1])
                msssim = float(temp[15 + 1])
                enct = float(temp[19 + 1][:-1])
                dect = float(temp[23 + 1][:-2].strip())
                print(mvbpp, resbpp, bpp, warppsnr, mcpsnr, psnr, msssim, enct, dect)
                # exit()
                Pbpp.append(bpp)
                Mvbpp.append(mvbpp)
                Resbpp.append(resbpp)
                Warppsnr.append(warppsnr)
                Mcpsnr.append(mcpsnr)
                Ppsnr.append(psnr)
                Penct.append(enct)
                Pdect.append(dect)
            else:
                print('Key Error, Code Exit')
                exit()
            # if ii > 2:
            #     break
        TotalPSNR = Ipsnr + Ppsnr
        TotalBpp = Ibpp + Pbpp
        TotalEncT = Ienct + Penct
        TotalDecT = Idect + Pdect
        TaskBpp = Ibpp + Mvbpp
        TaskPSNR = Ipsnr + Warppsnr
        print()
        print(f"Total Bpp [{np.mean(TotalBpp)}], PNSR [{np.mean(TotalPSNR)}], "
              f"EncT [{np.mean(TotalEncT)}], EDecT [{np.mean(TotalDecT)}], ")
        print(f"Task Bpp [{np.mean(TaskBpp)}], WarpPNSR [{np.mean(TaskPSNR)}] ")
        print(f"I Frame Bpp [{np.mean(Ibpp)}], PNSR [{np.mean(Ipsnr)}], "
              f"EncT [{np.mean(Ienct)}], EDecT [{np.mean(Idect)}], ")
        print(f"P Frame Bpp [{np.mean(Pbpp)}], PNSR [{np.mean(Ppsnr)}], "
              f"EncT [{np.mean(Penct)}], EDecT [{np.mean(Pdect)}], ")
    return 0


def read_txt1():
    path = '/home/user/Desktop/LHB/0726/log2048.txt'
    with open(path, 'r') as f:
        lines = f.readlines()
        print(len(lines))
        print(lines[0].split(' '))
        print(lines[1].split(' '))
        print('IFrame' in lines[0])
        print('PFrame' in lines[1])
        # exit()
        Ibpp, Pbpp, Mvbpp, Resbpp, Warppsnr, Mcpsnr, Ipsnr, Ppsnr, Ienct, Idect, Penct, Pdect = \
            [], [], [], [], [], [], [], [], [], [], [], []
        for ii, line in enumerate(lines):
            temp = line.split(' ')
            if 'IFrame' in line:
                bpp = float(temp[4])
                psnr = float(temp[7])
                enct = float(temp[11][:-1])
                dect = float(temp[15][:-2].strip())
                print(bpp, psnr, enct, dect)
                Ibpp.append(bpp)
                Ipsnr.append(psnr)
                Ienct.append(enct)
                Idect.append(dect)
            elif 'PFrame' in line:
                mvbpp = float(temp[4][1:-1])
                resbpp = float(temp[5][:-1])
                bpp = float(temp[6][:-1])
                warppsnr = float(temp[9][1:-1])
                mcpsnr = float(temp[10][:-1])
                psnr = float(temp[11][:-1])
                enct = float(temp[15][:-1])
                dect = float(temp[19][:-2].strip())
                print(mvbpp, resbpp, bpp, warppsnr, mcpsnr, psnr, enct, dect)
                # exit()
                Pbpp.append(bpp)
                Mvbpp.append(mvbpp)
                Resbpp.append(resbpp)
                Warppsnr.append(warppsnr)
                Mcpsnr.append(mcpsnr)
                Ppsnr.append(psnr)
                Penct.append(enct)
                Pdect.append(dect)
            else:
                print('Key Error, Code Exit')
                exit()
            # if ii > 2:
            #     break
        TotalPSNR = Ipsnr + Ppsnr
        TotalBpp = Ibpp + Pbpp
        TotalEncT = Ienct + Penct
        TotalDecT = Idect + Pdect
        TaskBpp = Ibpp + Mvbpp
        TaskPSNR = Ipsnr + Warppsnr
        print()
        print(f"Total Bpp [{np.mean(TotalBpp)}], PNSR [{np.mean(TotalPSNR)}], "
              f"EncT [{np.mean(TotalEncT)}], EDecT [{np.mean(TotalDecT)}], ")
        print(f"Task Bpp [{np.mean(TaskBpp)}], WarpPNSR [{np.mean(TaskPSNR)}] ")
        print(f"I Frame Bpp [{np.mean(Ibpp)}], PNSR [{np.mean(Ipsnr)}], "
              f"EncT [{np.mean(Ienct)}], EDecT [{np.mean(Idect)}], ")
        print(f"P Frame Bpp [{np.mean(Pbpp)}], PNSR [{np.mean(Ppsnr)}], "
              f"EncT [{np.mean(Penct)}], EDecT [{np.mean(Pdect)}], ")
    return 0


def mp4_yuv_png_val():
    path = '/tdx/LHB/data/ILSVRC/Data/VID/snippets/val'
    p = '/tdx/LHB/data/ILSVRC/Data/VID'
    fun = -1
    if fun == 0:
        mp4s = glob.glob(os.path.join(path, '*.mp4'))
        for mp4 in sorted(mp4s):
            name = mp4.split('/')[-1].split('.')[0]
            im = os.path.join(p, 'val', name, '000000.JPEG')
            shape = cv2.imread(im, -1).shape
            print(name, shape)
            os.system(
                f'ffmpeg -i {mp4} -y -s {shape[1]}x{shape[0]} -pix_fmt yuv420p -vframes 96 {p}/mp4toyuv420/{name}.yuv')
            # exit()
    elif fun == 1:
        frames = 96
        tgt = 'val_video_all_f96_yuv420'
        yuvs = glob.glob(os.path.join(p, 'mp4toyuv420', '*.yuv'))
        for yuv in sorted(yuvs):
            name = yuv.split('/')[-1].split('.')[0]
            print(yuv, name)
            im = os.path.join(p, 'val', name, '000000.JPEG')
            shape = cv2.imread(im, -1).shape
            os.makedirs(os.path.join(p, tgt, name), exist_ok=True)
            temp = os.path.join(p, tgt, name, "%6d.png")
            os.system(f'ffmpeg -s {shape[1]}x{shape[0]} -pix_fmt yuv420p -i {yuv} -vframes {frames} {temp}\n')
            # exit()
    return 0


def VID_x264_veryslow():
    fun = 4

    path = '/media/user/f126fd00-4370-4fdf-9a0a-fc93e66106eb1/LHB/datasets/ImageNet/VID2015/ILSVRC/Data/VID'
    val_frames = os.path.join(path, 'val')
    tgt = 'x264veryslow_gop12'
    frames = 96

    # Encode Decode
    if fun == 1:
        # 22, 27, 32, 37, 42
        for qp in [42]:
            for i, p1 in enumerate(os.listdir(val_frames)):
                images = glob.glob(os.path.join(path, 'val_video_all_f96_yuv420', p1, "*.png"))
                shape = cv2.imread(images[0], -1).shape
                save_dec_p1 = os.path.join(path, tgt, 'mkv', f'qp_{qp}')
                os.makedirs(save_dec_p1, exist_ok=True)
                seq = os.path.join(path, 'mp4toyuv420', p1 + '.yuv')
                h264 = os.path.join(path, tgt, 'mkv', f'qp_{qp}', p1 + '.mkv')
                csv_path = os.path.join(path, tgt, 'csv', f'qp_{qp}')
                csv_file = os.path.join(path, tgt, 'csv', f'qp_{qp}', f'{p1}.log')
                os.makedirs(csv_path, exist_ok=True)
                print(i, p1, seq, h264, len(images))  # veryfast, veryslow, LDP default: -c:v libx265 -tune zerolatency

                os.system(
                    f'FFREPORT=file={csv_file}:level=56 ffmpeg -y -pix_fmt yuv420p -s {shape[1]}x{shape[0]} -framerate 25 -i {seq} -vframes {frames}'
                    f' -c:v libx264 -preset veryslow -tune zerolatency -qp {qp} -g 12 -bf 2 -b_strategy 0 -sc_threshold 0 {h264}')

                os.makedirs(os.path.join(path, tgt, 'dec_img', f'qp_{qp}', p1), exist_ok=True)
                temp = os.path.join(path, tgt, 'dec_img', f'qp_{qp}', p1, "%6d.png")
                mkv = os.path.join(path, tgt, 'mkv', f'qp_{qp}', p1 + '.mkv')
                os.system(f'ffmpeg -i {mkv} -vframes {frames} -f image2 {temp}\n')
                # exit()

    # cal PSNR
    elif fun == 2:
        log = open(os.path.join(path, tgt, f'test.txt'), 'w')
        for qp in [22, 27, 32, 37, 42]:
            bpp_all, psnr1_all, psnr2_all = [], [], []
            kk = sorted(os.listdir(val_frames))
            for i, p1 in enumerate(kk):
                # print(i, p1)
                images = glob.glob(os.path.join(path, 'val_video_all_f96_yuv420', p1, "*.png"))
                shape = cv2.imread(images[0], -1).shape
                psnr = []
                images = sorted(images)
                for j, image in enumerate(images):
                    if j >= frames:
                        break
                    # print(j)
                    name = image.split('/')[-1]
                    im = os.path.join(path, tgt, 'dec_img', f'qp_{qp}', p1, name)
                    # name = image.split('/')[-1].split('.')[0]
                    # im = os.path.join(path, tgt, 'dec_img', f'qp_{qp}', p1, str(int(name) + 1).zfill(6) + '.png')
                    org = cv2.imread(image, -1)
                    dec = cv2.imread(im, -1)
                    # print(org.shape, np.min(org), np.max(org))
                    # print(dec.shape, np.min(dec), np.max(dec))
                    # exit()
                    _psnr = cal_psnr(org, dec)
                    # print(_psnr)
                    psnr.append(_psnr)

                log_path = os.path.join(path, tgt, 'csv', f'qp_{qp}', p1 + '.log')
                with open(log_path, 'r') as f:
                    lines = f.readlines()
                size_line = []
                count = 0
                for l in lines:
                    if ", size " in l:
                        count += 1
                        size = l.split(',')[1]
                        size_line.append(int(size[5:]))
                        if count >= frames:
                            break
                # print(size_line)
                # print(len(size_line))
                # print(crf, i, p1)
                # exit()
                size_line = np.array(size_line) * 8.0 / (shape[0] * shape[1])
                bpp = np.array(size_line).mean(0)
                print(i + 1, p1, 'qp=', qp, np.mean(psnr), bpp)
                # exit()
                bpp_all.append(bpp)
                psnr1_all.append(np.mean(psnr))
                # psnr2_all.append(psnr1)
            log.write(f'qp = {qp}, RGB_psnr = {np.mean(psnr1_all):.3f}, '
                      f'bpp = {np.mean(bpp_all):.3f}\n')
            print(
                f'qp = {qp}, RGB_psnr = {np.mean(psnr1_all):.3f}, '
                f'bpp = {np.mean(bpp_all):.3f}')
        log.close()

    # cal MS-SSIM
    elif fun == 3:
        log = open(os.path.join(path, tgt, f'ssim.txt'), 'w')
        for qp in [22, 27, 32, 37, 42]:
            psnr1_all = []
            kk = sorted(os.listdir(val_frames))
            for i, p1 in enumerate(kk):
                images = glob.glob(os.path.join(path, 'val_video_all_f96_yuv420', p1, "*.png"))
                MS_SSIM = []
                images = sorted(images)
                for j, image in enumerate(images):
                    if j >= frames:
                        break
                    name = image.split('/')[-1]
                    im = os.path.join(path, tgt, 'dec_img', f'qp_{qp}', p1, name)
                    org = read_image(image).unsqueeze(0).cuda()
                    dec = read_image(im).unsqueeze(0).cuda()
                    msssim = ms_ssim(org, dec, 1.0).cpu().item()

                    # org = cv2.imread(image, -1)
                    # dec = cv2.imread(im, -1)
                    # msssim = _calc_msssim_orig(np.expand_dims(org, 0), np.expand_dims(dec, 0))

                    MS_SSIM.append(msssim)
                print(i + 1, p1, 'qp =', qp, np.mean(MS_SSIM))

                psnr1_all.append(np.mean(MS_SSIM))
            log.write(f'qp = {qp}, MS-SSIM = {np.mean(psnr1_all):.3f}\n')
            print(f'qp = {qp}, MS-SSIM = {np.mean(psnr1_all):.3f}')
        log.close()

    elif fun == 4:

        for tgt1 in ['x264veryslow_gop12', 'x265veryslow_gop12']:
            log = open(os.path.join(path, tgt1, f'psnr_ssim.txt'), 'w')
            log1 = open(os.path.join(path, tgt1, f'psnr_ssim_all.txt'), 'w')
            # 22, 27, 32, 37, 42
            for qp in [22, 27, 32, 37, 42]:
                seqPSNR, seqMSSSIM, seqBPPs = [], [], []
                for j, seq in enumerate(sorted(os.listdir(val_frames))):
                    print(qp, seq)
                    org_frame_paths = glob.glob(os.path.join(path, 'val_video_all_f96_yuv420', seq, "*.png"))
                    im = cv2.imread(org_frame_paths[0], -1)
                    shape = im.shape
                    frameBPP, framePSNR, frameMSSSIM = [], [], []

                    for ii, org_frame_path in enumerate(sorted(org_frame_paths)):
                        name = org_frame_path.split('/')[-1]
                        im = os.path.join(path, tgt1, 'dec_img', f'qp_{qp}', seq, name)
                        org = read_image(org_frame_path).unsqueeze(0).cuda()
                        dec = read_image(im).unsqueeze(0).cuda()

                        psnr1 = cal_psnr_torch(org, dec)
                        framePSNR.append(psnr1)
                        msssim = ms_ssim(org, dec, data_range=1.0).cpu().detach().item()
                        frameMSSSIM.append(msssim)
                        # print(psnr1, msssim)

                    log_path = os.path.join(path, tgt1, 'csv', f'qp_{qp}', seq + '.log')
                    with open(log_path, 'r') as f:
                        lines = f.readlines()
                    size_line = []
                    count = 0
                    for l in lines:
                        if ", size " in l:
                            count += 1
                            size = l.split(',')[1]
                            size_line.append(int(size[5:]))
                            if count >= frames:
                                break
                    # print(size_line)
                    # print(len(size_line))
                    # print(crf, i, p1)
                    # exit()
                    size_line = np.array(size_line) * 8.0 / (shape[0] * shape[1])
                    bpp = np.array(size_line).mean(0)

                    seqPSNR.append(np.average(framePSNR))
                    seqMSSSIM.append(np.average(frameMSSSIM))
                    seqBPPs.append(bpp)
                    print(qp, j, seq, np.average(framePSNR), np.average(seqMSSSIM), bpp)
                    log1.write(
                        f'QP={qp}\tseq={seq}\t{np.average(bpp)}\t{np.average(framePSNR)}\t{np.average(seqMSSSIM)}\n')
                    log1.flush()
                log.write(f'QP={qp}\t{np.average(seqBPPs)}\t{np.average(seqPSNR)}\t{np.average(seqMSSSIM)}\n')
                log.flush()
    return 0


def VID_x265_veryslow():
    fun = 3

    path = '/media/user/f126fd00-4370-4fdf-9a0a-fc93e66106eb1/LHB/datasets/ImageNet/VID2015/ILSVRC/Data/VID'
    # path = r'E:\dataset\Video\VID\VID_resize_processed_val'
    val_frames = os.path.join(path, 'val')
    tgt = 'x265veryslow_gop12'
    frames = 96

    # JPEG2YUV
    if fun == 0:
        yuv_path = os.path.join(path, 'JPEGto_val_YUV420')
        os.makedirs(yuv_path, exist_ok=True)
        dirs = sorted(os.listdir(val_frames))
        for p1 in dirs:
            images = glob.glob(os.path.join(path, 'val_video_all_f96', p1, "*.JPEG"))
            shape = cv2.imread(images[0], -1).shape
            temp = os.path.join(path, 'val_video_all_f96', p1, "%6d.JPEG")
            out = os.path.join(path, 'JPEGto_val_YUV420', p1 + '.yuv')
            print(p1, len(images), f'{shape[1]}x{shape[0]}', out)
            os.system(f'ffmpeg -f image2 -y -r 25 -i {temp} -pix_fmt yuv420p -s {shape[1]}x{shape[0]} {out}')

    # Encode Decode
    elif fun == 1:
        # 22, 27, 32, 37
        for qp in [42]:
            for i, p1 in enumerate(os.listdir(val_frames)):
                images = glob.glob(os.path.join(path, 'val', p1, "*.JPEG"))
                shape = cv2.imread(images[0], -1).shape
                save_dec_p1 = os.path.join(path, tgt, 'mkv', f'qp_{qp}')
                os.makedirs(save_dec_p1, exist_ok=True)
                seq = os.path.join(path, 'mp4toyuv420', p1 + '.yuv')
                h265 = os.path.join(path, tgt, 'mkv', f'qp_{qp}', p1 + '.mkv')
                csv_path = os.path.join(path, tgt, 'csv', f'qp_{qp}')
                csv_file = os.path.join(path, tgt, 'csv', f'qp_{qp}', f'{p1}.log')
                os.makedirs(csv_path, exist_ok=True)
                print(i, p1, seq, h265, len(images))  # veryfast, veryslow, LDP default: -c:v libx265 -tune zerolatency
                # os.system(
                #     f'ffmpeg -y -pix_fmt yuv420p -s {shape[1]}x{shape[0]} -framerate 30 -i {seq} -vframes {len(images)}'
                #     f' -c:v libx265 -preset {tgt} -tune zerolatency -x265-params "crf={crf}:keyint=10:csv-log-level=1:csv=./csv/crf_{crf}/{p1}.csv:verbose=1:psnr=1" '
                #     f' {h265}')

                # FFREPORT=file=ffreport.log:level=56 ffmpeg -pix_fmt yuv420p -s $2x$3 -i $input -vframes 100 -c:v libx265 -tune zerolatency -x265-params "crf=$1:keyint=10:verbose=1" out/h265/out.mkv
                os.system(
                    f'FFREPORT=file={csv_file}:level=56 ffmpeg -y -pix_fmt yuv420p -s {shape[1]}x{shape[0]} -framerate 25 -i {seq} -vframes {frames}'
                    f' -c:v libx265 -preset veryslow -tune zerolatency -x265-params "qp={qp}:keyint=12" {h265}')

                # os.system(
                #     f'ffmpeg -y -pix_fmt yuv420p -s {shape[1]}x{shape[0]} -framerate 30 -i {seq} -vframes {len(images)}'
                #     f' -c:v libx265 -preset {tgt} -tune zerolatency -x265-params "crf={crf}:keyint=10:csv-log-level=1:csv={csv_file}:verbose=1:psnr=1" '
                #     f' {h265}')

                # print(
                #     f'ffmpeg -y -pix_fmt yuv420p -s {shape[1]}x{shape[0]} -framerate 30 -i {seq} -vframes {len(images)}'
                #     f' -c:v libx265 -preset {tgt} -tune zerolatency -x265-params "crf={crf}:keyint=10:csv-log-level=1:csv={csv_file}:verbose=1:psnr=1" '
                #     f' {h265}')

                os.makedirs(os.path.join(path, tgt, 'dec_img', f'qp_{qp}', p1), exist_ok=True)
                temp = os.path.join(path, tgt, 'dec_img', f'qp_{qp}', p1, "%6d.png")
                mkv = os.path.join(path, tgt, 'mkv', f'qp_{qp}', p1 + '.mkv')
                os.system(f'ffmpeg -i {mkv} -vframes {frames} -f image2 {temp}\n')
                # exit()

        # for p1 in os.listdir(val_frames):
        #     for crf in [15, 19, 23, 27, 31, 35]:
        #         os.makedirs(os.path.join(path, tgt, 'dec_img', f'crf_{crf}', p1), exist_ok=True)
        #         temp = os.path.join(path, tgt, 'dec_img', f'crf_{crf}', p1, "%%6d.JPEG")
        #         mkv = os.path.join(path, tgt, 'mkv', f'crf_{crf}', p1 + '.mkv')
        #         os.system(f'ffmpeg -i {mkv} -f image2 {temp}\n')

    # cal PSNR
    elif fun == 2:
        log = open(os.path.join(path, tgt, f'result_video_all_f96_png1.txt'), 'w')
        for qp in [42]:
            bpp_all, psnr1_all, psnr2_all = [], [], []
            kk = sorted(os.listdir(val_frames))
            for i, p1 in enumerate(kk):
                images = glob.glob(os.path.join(path, 'val_video_all_f96_yuv420', p1, "*.png"))
                shape = cv2.imread(images[0], -1).shape
                psnr = []
                images = sorted(images)
                for j, image in enumerate(images):
                    if j >= frames:
                        break
                    # print(j)
                    # name = image.split('/')[-1].split('.')[0]
                    # im = os.path.join(path, tgt, 'dec_img', f'qp_{qp}', p1, str(int(name) + 1).zfill(6) + '.png')
                    name = image.split('/')[-1]
                    im = os.path.join(path, tgt, 'dec_img', f'qp_{qp}', p1, name)
                    org = cv2.imread(image, -1)
                    dec = cv2.imread(im, -1)
                    # print(org.shape, np.min(org), np.max(org))
                    # print(dec.shape, np.min(dec), np.max(dec))
                    # exit()
                    _psnr = cal_psnr(org, dec)
                    # print(_psnr)
                    psnr.append(_psnr)

                log_path = os.path.join(path, tgt, 'csv', f'qp_{qp}', p1 + '.log')
                with open(log_path, 'r') as f:
                    lines = f.readlines()
                size_line = []
                count = 0
                for l in lines:
                    if ", size " in l:
                        count += 1
                        size = l.split(',')[1]
                        # print(count, int(size[5:]))
                        size_line.append(int(size[5:]))
                        if count >= frames:
                            break
                # print(size_line)
                # print(len(size_line))
                # print(crf, i, p1)
                # exit()
                size_line = np.array(size_line) * 8.0 / (shape[0] * shape[1])
                bpp = np.array(size_line).mean(0)
                print(i + 1, p1, 'qp =', qp, np.mean(psnr), bpp)

                bpp_all.append(bpp)
                psnr1_all.append(np.mean(psnr))
                # psnr2_all.append(psnr1)
            log.write(f'qp = {qp}, RGB_psnr = {np.mean(psnr1_all):.3f}, '
                      f'bpp = {np.mean(bpp_all):.3f}\n')
            print(
                f'qp = {qp}, RGB_psnr = {np.mean(psnr1_all):.3f}, '
                f'bpp = {np.mean(bpp_all):.3f}')
        log.close()

    # cal MS-SSIM
    elif fun == 3:
        log = open(os.path.join(path, tgt, f'result_video_all_f96_png_ssim.txt'), 'w')
        for qp in [22, 27, 32, 37, 42]:
            bpp_all, psnr1_all, psnr2_all = [], [], []
            kk = sorted(os.listdir(val_frames))
            for i, p1 in enumerate(kk):
                images = glob.glob(os.path.join(path, 'val_video_all_f96_yuv420', p1, "*.png"))
                MS_SSIM = []
                images = sorted(images)
                for j, image in enumerate(images):
                    if j >= frames:
                        break
                    name = image.split('/')[-1]
                    im = os.path.join(path, tgt, 'dec_img', f'qp_{qp}', p1, name)
                    org = read_image(image).unsqueeze(0).cuda()
                    dec = read_image(im).unsqueeze(0).cuda()
                    msssim = ms_ssim(org, dec, 1.0).cpu().item()

                    # org = cv2.imread(image, -1)
                    # dec = cv2.imread(im, -1)
                    # msssim = _calc_msssim_orig(np.expand_dims(org, 0), np.expand_dims(dec, 0))

                    MS_SSIM.append(msssim)
                print(i + 1, p1, 'qp =', qp, np.mean(MS_SSIM))

                psnr1_all.append(np.mean(MS_SSIM))
                # psnr2_all.append(psnr1)
            log.write(f'qp = {qp}, MS-SSIM = {np.mean(psnr1_all):.3f}\n')
            print(f'qp = {qp}, MS-SSIM = {np.mean(psnr1_all):.3f}')
        log.close()

    return 0


def mp4_yuv_png_train():
    n = 'ILSVRC2015_VID_train_0000'
    path = f'/media/user/f126fd00-4370-4fdf-9a0a-fc93e66106eb1/LHB/datasets/ImageNet/VID2015/ILSVRC/Data/VID/snippets/train/{n}'
    p = '/media/user/f126fd00-4370-4fdf-9a0a-fc93e66106eb1/LHB/datasets/ImageNet/VID2015/ILSVRC/Data/VID'
    fun = 1
    if fun == 0:
        useless = []
        mp4s = glob.glob(os.path.join(path, '*.mp4'))
        for mp4 in sorted(mp4s):
            name = mp4.split('/')[-1].split('.')[0]
            im = os.path.join(p, f'train/{n}', name, '000000.JPEG')
            shape = cv2.imread(im, -1).shape
            if shape[1] <= 256 or shape[0] < 256 or shape[2] != 3:
                useless.append(name)
                continue
            print(name, shape)
            os.system(
                f'ffmpeg -i {mp4} -y -s {shape[1]}x{shape[0]} -pix_fmt yuv420p -vframes 12 {p}/trainyuv420/{name}.yuv')
            os.makedirs(os.path.join(p, 'train_video_codec/train', name), exist_ok=True)
            temp = os.path.join(p, 'train_video_codec/train', name, "%6d.png")
            os.system(
                f'ffmpeg -s {shape[1]}x{shape[0]} -pix_fmt yuv420p -i {p}/trainyuv420/{name}.yuv -vframes 12 {temp}\n')
        print(useless)
    elif fun == 1:
        tgt = '/media/user/f126fd00-4370-4fdf-9a0a-fc93e66106eb1/LHB/datasets/ImageNet/VID2015/ILSVRC/Data/VID/train_video_codec_1'
        val_seqs = sorted(os.listdir(os.path.join(p, 'val_video_all_f96_yuv420')))[256:256 + 100]
        for i1, seq in enumerate(val_seqs):
            os.makedirs(os.path.join(tgt, seq.replace('val', 'train1')), exist_ok=True)
            images = sorted(glob.glob(os.path.join(p, 'val_video_all_f96_yuv420', seq, '*.png')))[:12]
            for image in images:
                tgt_p = os.path.join(tgt, seq.replace('val', 'train1'))
                shutil.copy(image, tgt_p)

    elif fun == 2:
        pass
    return 0


def UCF101_x265_veryslow():
    fun = 1
    path = '/tdx/LHB/data/UCF101'
    tgt = 'rawframes_x265veryslow_gop12'
    split = 'val'
    val_txt = os.path.join(path, f'ucf101_{split}_split_1_rawframes.txt')
    video_infos = {}
    with open(val_txt, 'r') as fin:
        for i, line in enumerate(fin):
            line_split = line.strip().split()
            video_infos[line_split[0]] = [int(line_split[1]), int(line_split[2])]
    # print(len(video_infos))
    # exit()
    # AVI to YUV420
    if fun == 0:
        ll = []
        for video_inedx, data in enumerate(video_infos.items()):
            video_name = data[0]
            total_frames, label = data[1][0], data[1][1]
            print(video_inedx, video_name)
            usb_dir = os.path.join(path, 'AVI_to_YUV420', video_name.split('/')[0])
            os.makedirs(usb_dir, exist_ok=True)
            avi_full_path = os.path.join(path, 'videos', video_name + '.avi')
            avi_full_path_1 = os.path.join(path, 'AVI_to_YUV420', video_name + '.yuv')
            vr = mmcv.VideoReader(avi_full_path)
            w, h, _ = np.shape(vr[0])
            # print(type(vr[0]), w, h, len(vr))
            print(video_name, w, h, len(vr))
            # exit()
            ll.append(str(np.shape(vr[0])))
            os.system(f'ffmpeg -y -i {avi_full_path} -pix_fmt yuv420p -s {h}x{w} {avi_full_path_1}')
            exit()
        print(np.unique(ll))
    # YUV420 to png
    elif fun == 11:
        for video_inedx, data in enumerate(video_infos.items()):
            video_name = data[0]
            total_frames, label = data[1][0], data[1][1]
            print(video_inedx, video_name)
            avi_full_path = os.path.join(path, 'videos', video_name + '.avi')
            avi_full_path_1 = os.path.join(path, 'AVI_to_YUV420', video_name + '.yuv')
            vr = mmcv.VideoReader(avi_full_path)
            w, h, _ = np.shape(vr[0])
            # print(type(vr[0]), w, h, len(vr))
            print(video_name, w, h, len(vr))
            # exit()
            # os.system(f'ffmpeg -y -i {avi_full_path} -pix_fmt yuv420p -s {h}x{w} {avi_full_path_1}')

            os.makedirs(os.path.join(path, 'YUV420_to_PNG', video_name), exist_ok=True)
            temp = os.path.join(path, 'YUV420_to_PNG', video_name, "%6d.png")
            os.system(f'ffmpeg -s {h}x{w}  -pix_fmt yuv420p -i {avi_full_path_1} -vframes {len(vr)} {temp}\n')
            # exit()

    # Encode Decode
    elif fun == 1:
        # 22, 27, 32, 37
        # 37, 38, 39, 40
        for qp in [40]:
            for video_inedx, data in enumerate(video_infos.items()):
                video_name = data[0]
                total_frames, label = data[1][0], data[1][1]
                print(video_inedx, video_name, total_frames, label)
                full_path_yuv = os.path.join(path, 'AVI_to_YUV420', video_name + '.yuv')
                avi_full_path = os.path.join(path, 'videos', video_name + '.avi')
                vr = mmcv.VideoReader(avi_full_path)
                w, h, _ = np.shape(vr[0])
                print(type(vr[0]), w, h, len(vr))

                save_dec_p1 = os.path.join(path, tgt, 'mkv', f'qp_{qp}', video_name.split('/')[0])
                os.makedirs(save_dec_p1, exist_ok=True)
                h265 = os.path.join(path, tgt, 'mkv', f'qp_{qp}', video_name + '.mkv')
                csv_path = os.path.join(path, tgt, 'csv', f'qp_{qp}', video_name.split('/')[0])
                csv_file = os.path.join(path, tgt, 'csv', f'qp_{qp}', f'{video_name}.log')
                os.makedirs(csv_path, exist_ok=True)
                os.system(
                    f'FFREPORT=file={csv_file}:level=56 ffmpeg -y -pix_fmt yuv420p -s {h}x{w} -framerate 25 -i {full_path_yuv} -vframes {len(vr)}'
                    f' -c:v libx265 -preset veryslow -tune zerolatency -x265-params "qp={qp}:keyint=12" {h265}')

                os.makedirs(os.path.join(path, tgt, 'dec_img', f'qp_{qp}', video_name), exist_ok=True)
                temp = os.path.join(path, tgt, 'dec_img', f'qp_{qp}', video_name, "%6d.png")
                mkv = os.path.join(path, tgt, 'mkv', f'qp_{qp}', video_name + '.mkv')
                os.system(f'ffmpeg -i {mkv} -vframes {len(vr)} -f image2 {temp}\n')
                # exit()
    # cal bpp
    elif fun == 2:
        log = open(os.path.join(path, tgt, f'bpp.txt'), 'w')
        for qp in [37, 38, 39, 40]:
            bpp_all = []
            for video_inedx, data in enumerate(video_infos.items()):
                video_name = data[0]
                total_frames, label = data[1][0], data[1][1]
                print(video_inedx, video_name, total_frames, label)
                avi_full_path = os.path.join(path, 'videos', video_name + '.avi')
                vr = mmcv.VideoReader(avi_full_path)
                w, h, _ = np.shape(vr[0])
                print(type(vr[0]), w, h, len(vr))

                log_path = os.path.join(path, tgt, 'csv', f'qp_{qp}', f'{video_name}.log')

                with open(log_path, 'r') as f:
                    lines = f.readlines()
                size_line = []
                count = 0
                for l in lines:
                    if ", size " in l:
                        count += 1
                        size = l.split(',')[1]
                        size_line.append(int(size[5:]))

                size_line = np.array(size_line) * 8.0 / (w * h)
                bpp = np.array(size_line).mean(0)

                bpp_all.append(bpp)
            log.write(f'qp = {qp}, bpp = {np.mean(bpp_all):.5f}\n')
            print(f'qp = {qp}, bpp = {np.mean(bpp_all):.5f}')
        log.close()
    return 0


def Kinetics():
    # path = '/tdx/LHB/data/kinetics/val_256'
    # path = '/tdx/LHB/data/kinetics/raw-part/compress/train_256'
    # val_txt = os.path.join('/tdx/LHB/data/kinetics/vcii_test_ids.txt')
    # video_infos = []
    # with open(val_txt, 'r') as fin:
    #     for i, line in enumerate(fin):
    #         # print(i, line.strip())
    #         # print(line.strip()[:-len('_000292_000302')])
    #         # exit()
    #         video_infos.append(line.strip())
    # print(len(video_infos))
    # # print(video_infos[:10])
    # # exit()
    # tgt_mp4_path = '/tdx/LHB/data/kinetics/test_mp4s'
    # count = 0
    # for sub in os.listdir(path):
    #     p1 = os.path.join(path, sub)
    #     for sub1 in os.listdir(p1):
    #         # print(sub1.split('.')[0])
    #         # exit()
    #         if sub1.split('.')[0] in video_infos:
    #             print(count, sub1)
    #             count += 1
    #             copy(os.path.join(path, sub, sub1), tgt_mp4_path)
    #         # print(sub1)
    #         # exit()
    # exit()

    # shape, lens = [], []
    # tgt_mp4_path = '/tdx/LHB/data/kinetics/val_mp4s_10k'
    # for mp4_name in os.listdir(tgt_mp4_path):
    #     vr = mmcv.VideoReader(os.path.join(tgt_mp4_path, mp4_name))
    #     w, h, _ = np.shape(vr[0])
    #     print(w, h, len(vr))
    #     # exit()
    #     shape.append(str(np.shape(vr[0])))
    #     lens.append(len(vr))
    #     # exit()
    # print(np.unique(shape))
    # print(np.unique(lens))
    # print(max(lens), min(lens))

    tgt_mp4_path = '/tdx/LHB/data/kinetics/test_mp4s'
    for mp4_name in os.listdir(tgt_mp4_path):
        mp4_path = os.path.join(tgt_mp4_path, mp4_name)
        video_name = mp4_name.split('.mp4')[0]
        os.makedirs(os.path.join('/tdx/LHB/data/kinetics/PNG_Frames', video_name), exist_ok=True)
        temp = os.path.join('/tdx/LHB/data/kinetics/PNG_Frames', video_name, "%6d.png")
        os.system(f'ffmpeg -i {mp4_path} {temp}')
        # print(mp4_name)
        # exit()
    return 0


if __name__ == '__main__':
    pass
    Kinetics()
    # path = '/home/tdx/桌面/Project/LHB/data/UCF101/AVI_to_YUV420'
    # sum = 0
    # for sub in os.listdir(path):
    #     yuvs = glob.glob(os.path.join(path, sub, '*.yuv'))
    #     sum = sum + len(yuvs)
    # print(sum)
    # UCF101_x265_veryslow()
