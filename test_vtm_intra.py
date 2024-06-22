import math
import shutil

import numpy as np
import os
import glob
from tqdm import tqdm
import cv2
import json
import time
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio
from PIL import Image

import torch.nn.functional as F
import torch.nn as nn
import torch
from torchvision import transforms
from torchvision.transforms import ToPILImage, ToTensor
import matplotlib
from pytorch_msssim import ms_ssim


TEST_DATA = {
    'HEVC_B': {
        'path': 'E:/dataset/TestDeepVideoCoding/CTC/ClassB',
        'frames': 96,
        'gop': 12,
        'org_resolution': '1920x1080',
        'x64_resolution': '1920x1024',
        'sequences': {
            'BasketballDrive_1920x1080_50',
            'BQTerrace_1920x1080_60',
            'Cactus_1920x1080_50',
            'Kimono1_1920x1080_24',
            'ParkScene_1920x1080_24',
        },
    },

    'HEVC_C': {
        'path': 'E:/dataset/TestDeepVideoCoding/CTC/ClassC/',
        'frames': 96,
        'gop': 12,
        'org_resolution': '832x480',
        'x64_resolution': '832x448',
        'sequences': [
            'BasketballDrill_832x480_50',
            'BQMall_832x480_60',
            'PartyScene_832x480_50',
            'RaceHorses_832x480_30',
        ],
    },

    'HEVC_D': {
        'path': 'E:/dataset/TestDeepVideoCoding/CTC/ClassD/',
        'frames': 96,
        'gop': 12,
        'org_resolution': '416x240',
        'x64_resolution': '384x192',
        'sequences': [
            'BasketballPass_416x240_50',
            'BlowingBubbles_416x240_50',
            'BQSquare_416x240_60',
            'RaceHorses_416x240_30',
        ],
    },

    'HEVC_E': {
        'path': 'E:/dataset/TestDeepVideoCoding/CTC/ClassE/',
        'frames': 96,
        'gop': 12,
        'org_resolution': '1280x720',
        'x64_resolution': '1280x704',
        'sequences': [
            'FourPeople_1280x720_60',
            'Johnny_1280x720_60',
            'KristenAndSara_1280x720_60',
        ],
    },
}


def read_image(filepath: str) -> torch.Tensor:
    assert os.path.isfile(filepath)
    img = Image.open(filepath).convert("RGB")
    return transforms.ToTensor()(img)


def cal_psnr_torch(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = F.mse_loss(a, b).item()
    return -10 * math.log10(mse)


def cal_psnr_np(ref, target):
    diff = ref / 255.0 - target / 255.0
    diff = diff.flatten('C')
    rmse = math.sqrt(np.mean(diff ** 2.))
    return 20 * math.log10(1.0 / rmse)


def VTM():
    fun = 4
    path = '/media/user/f126fd00-4370-4fdf-9a0a-fc93e66106eb1/LHB/datasets/TestSets/ZZC_Test_VTM'
    # Kodak  CLIC  Tenick
    # HEVC_B   HEVC_C  HEVC_D  HEVC_E
    # 100f  t8
    HEVC_frames = 100
    tgt_dataset = 'CLIC'
    qps = [x for x in range(23, 45, 1)]
    # qps = [x for x in range(24, 46, 2)]
    print(qps)
    mark = 'yuv444p'  # yuv420p  yuv444p

    # yuv 100frames
    if fun == -1:
        # HEVC_B  HEVC_C  HEVC_D  HEVC_E
        for key in ['HEVC_B', 'HEVC_C', 'HEVC_D', 'HEVC_E']:
            data = TEST_DATA[key]
            org_resolution = data['org_resolution']
            for seq in data['sequences']:
                yuv = os.path.join(data['path'], f'{mark}', seq + '.yuv')
                img_path = os.path.join(path, seq)
                os.makedirs(img_path, exist_ok=True)
                os.system(f'ffmpeg -pix_fmt yuv444p -s {org_resolution} -i {yuv} -vframes {HEVC_frames} {img_path}/f%03d.png')
    # CTC yuv->png  100frames
    elif fun == 0:
        # HEVC_B  HEVC_C  HEVC_D  HEVC_E
        data = TEST_DATA['HEVC_D']
        org_resolution = data['org_resolution']
        frames = 100
        for seq in data['sequences']:
            # print(seq)
            # exit()
            yuv = os.path.join(data['path'], f'{mark}', seq + '.yuv')
            img_path = os.path.join(path, seq)
            os.makedirs(img_path, exist_ok=True)
            os.system(f'ffmpeg -pix_fmt yuv444p -s {org_resolution} -i {yuv} -vframes {frames} {img_path}/f%03d.png')
            exit()
    # png->yuv
    elif fun == 1:
        images = sorted(glob.glob(os.path.join(path, tgt_dataset, '*.png')))
        print(tgt_dataset, len(images))
        for image in images:
            image = image.replace('\\', '/')
            name = image.split('/')[-1].split('.')[0]
            # print(name)
            im = cv2.imread(image, -1)
            print(im.shape)
            # exit()
            h, w, c = im.shape
            new_h = (h + 1) // 2 * 2
            new_w = (w + 1) // 2 * 2
            print(new_h, new_w)
            os.makedirs(f'{path}/{tgt_dataset}_{mark}', exist_ok=True)
            os.system(f'ffmpeg -y -i {image} -f rawvideo -pix_fmt {mark} -vf pad={new_w}:{new_h}:0:0:black '
                      f'{path}/{tgt_dataset}_{mark}/{name}.yuv')

    elif fun == 2:
        # for index in range(13, 20):
        for index in [19.2]:
            version = f'VTM_{index}.0'
            print(index, version)
            bat_file = open(os.path.join(path, f'vtm{index}_{mark}_{tgt_dataset}.bat'), 'w')
            images = sorted(glob.glob(os.path.join(path, f'{tgt_dataset}', '*.png')))
            for image in images:
                print(image)
                image = image.replace('\\', '/')
                name = image.split('/')[-1].split('.')[0]
                im = cv2.imread(image, -1)
                print(im.shape)
                # exit()
                h, w, c = im.shape
                new_h = (h + 1) // 2 * 2
                new_w = (w + 1) // 2 * 2
                print(new_h, new_w)
                # exit()
                for qp in qps:
                    bat_file.write(f'EncoderApp{index}ORG -c encoder_intra_vtm_{index}.cfg '
                                   f'-i {tgt_dataset}_{mark}/{name}.yuv -b ./bin/{mark}_{index}_{name}_{qp}.bin '
                                   f'-o ./dec/{mark}_{index}_{name}_{qp}.yuv -fr 1 -f 1 --ConformanceWindowMode=1 '
                                   f'--InputChromaFormat=420 '
                                   f'-wdt {new_w} -hgt {new_h} -q {qp}  '
                                   f' >./txt/{mark}_{index}_{name}_{qp}.txt\n')
                bat_file.write('\n')
            bat_file.close()
    # dec yuv2png
    elif fun == 3:
        os.makedirs(f'{path}/dec_png/{tgt_dataset}', exist_ok=True)
        for index in [19.2]:
            version = f'VTM_{index}'
            images = sorted(glob.glob(os.path.join(path, f'{tgt_dataset}', '*.png')))
            print(version, len(images))
            for image in images:
                image = image.replace('\\', '/')
                name = image.split('/')[-1].split('.')[0]
                im = cv2.imread(image, -1)
                # print(im.shape)
                h, w, c = im.shape
                new_h = (h + 1) // 2 * 2
                new_w = (w + 1) // 2 * 2
                for qp in qps:
                    os.system(f'ffmpeg -y -pix_fmt {mark} -s {new_w}x{new_h} '
                              f'-i {path}/dec/{mark}_{index}_{name}_{qp}.yuv '
                              f'-f image2 {path}/dec_png/{tgt_dataset}/{mark}_{index}_{name}_{qp}.png\n')
                    # exit()
                # exit()
    # cal psnr ms-ssim
    elif fun == 4:
        os.makedirs(f'{path}/results', exist_ok=True)
        txt_file = open(os.path.join(path, f'results/{mark}_{tgt_dataset}.txt'), 'w')
        txt_file_all = open(os.path.join(path, f'results/{mark}_{tgt_dataset}_all.txt'), 'w')
        for index in [19.2]:
            version = f'VTM_{index}'
            print(index, version)
            images = sorted(glob.glob(os.path.join(path, f'{tgt_dataset}', '*.png')))
            if mark == 'yuv420p':
                images = sorted(glob.glob(os.path.join(path, f'{tgt_dataset}_{mark}_png', '*.png')))
            print(len(images))
            PSNR1, MSSIM1, BPP1 = [], [], []
            for qp in qps:
                PSNR, MSSIM, BPP = [], [], []
                for image in images:
                    image = image.replace('\\', '/')
                    name = image.split('/')[-1].split('.')[0]
                    # print(image)
                    org = read_image(image).unsqueeze(0)
                    # print(torch.max(org), torch.min(org))
                    # exit()
                    _, c, h, w = org.shape
                    #  BPP [2.21000] | PSNR [31.01913] | MS-SSIM [0.98682]
                    # print(image)
                    # print(os.path.join(path, f'dec_png1/{mark}_{index}_{name}_{qp}.png'))
                    # exit()
                    # pad={new_w}:{new_h}:0:0:black
                    dec = read_image(os.path.join(path, f'dec_png/{tgt_dataset}/{mark}_{index}_{name}_{qp}.png')).unsqueeze(0)
                    if tgt_dataset == 'CLIC':
                        dec = dec[:, :, 0:h, 0:w]
                    # print(dec.shape, org.shape)
                    # exit()
                    psnr = cal_psnr_torch(org, dec)
                    msssim = ms_ssim(org, dec, 1.0).cpu().item()

                    bin_file = os.path.join(path, f'bin/{mark}_{index}_{name}_{qp}.bin')
                    bpp = os.path.getsize(bin_file) * 8 / (w * h)

                    PSNR.append(psnr)
                    MSSIM.append(msssim)
                    BPP.append(bpp)
                    print(f'{mark}_VTM{index} | {name}_{qp} | BPP [{bpp:.5f}] | '
                          f'PSNR [{psnr:.5f}] | MS-SSIM [{msssim:.5f}]')
                    # exit()
                    txt_file_all.write(f'{mark}_VTM{index} | {name}_{qp} | BPP [{bpp:.5f}] | '
                                       f'PSNR [{psnr:.5f}] | MS-SSIM [{msssim:.5f}]\n')
                    # exit()
                txt_file_all.write(f'\n{mark}_VTM{index} | {qp} | BPP [{np.average(BPP):.5f}] | '
                                   f'PSNR [{np.average(PSNR):.5f}] | MS-SSIM [{np.average(MSSIM):.5f}]\n\n')
                txt_file.write(f'{mark}_VTM{index} | {qp} | BPP [{np.average(BPP):.5f}] | '
                               f'PSNR [{np.average(PSNR):.5f}] | MS-SSIM [{np.average(MSSIM):.5f}]\n')
                MSSIM1.append(np.average(MSSIM))
                PSNR1.append(np.average(PSNR))
                BPP1.append(np.average(BPP))
            txt_file.write('\n')
            txt_file_all.write('\n')
            results = {"bpp": BPP1, "psnr": PSNR1, "ms-ssim": MSSIM1}
            output = {
                "name": version,
                "description": version,
                "results": results,
            }
            with open(f"{path}/results/{version}_test_{tgt_dataset}_{mark}.json", 'w', encoding='utf-8') as json_file:
                json.dump(output, json_file, indent=2)
        txt_file.close()
        txt_file_all.close()

    elif fun == 5:
        os.makedirs(f'{path}/{tgt_dataset}_{mark}_png', exist_ok=True)
        images = sorted(glob.glob(os.path.join(path, f'{tgt_dataset}', '*.png')))
        print(len(images))
        for image in images:
            image = image.replace('\\', '/')
            name = image.split('/')[-1].split('.')[0]
            im = cv2.imread(image, -1)
            # print(im.shape)
            h, w, c = im.shape
            new_h = (h + 1) // 2 * 2
            new_w = (w + 1) // 2 * 2
            os.system(f'ffmpeg -y -pix_fmt {mark} -s {new_w}x{new_h} '
                      f'-i {path}/{tgt_dataset}_{mark}/{name}.yuv '
                      f'-f image2 {path}/{tgt_dataset}_{mark}_png/{name}.png\n')
            # exit()

    return 0


if __name__ == "__main__":
    VTM()
