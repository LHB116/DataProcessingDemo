import glob
import os

import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import fnmatch
import shutil
from compressai.zoo import cheng2020_anchor, cheng2020_attn
from pytorch_msssim import ms_ssim
from PIL import Image
from torchvision import transforms
import torch
import math
import json
# from utils import cal_psnr, read_image

from mmtrack.datasets import build_dataset
from mmtrack.datasets import build_dataloader
from mmcv import Config
from mmcv.image import tensor2imgs


def prepare_data():
    # 每个文件夹下序列都是7帧  原始图像尺寸都是256x448x3
    Vimeo90K_path = r'D:\DataSet'
    fun = 4
    if fun == 0:
        train_list_path = os.path.join(Vimeo90K_path, 'vimeo_septuplet', 'sep_trainlist.txt')
        test_list_path = os.path.join(Vimeo90K_path, 'vimeo_septuplet', 'sep_testlist.txt')
        train_list = [x.strip() for x in open(train_list_path, 'r').readlines()]
        test_list = [x.strip() for x in open(test_list_path, 'r').readlines()]

        train_txt = open('sep_trainlist.txt', 'w')
        test_txt = open('sep_testlist.txt', 'w')

        sequence_path = os.path.join(Vimeo90K_path, 'vimeo_septuplet', 'sequences')
        for sub1 in os.listdir(sequence_path):
            for sub2 in os.listdir(os.path.join(sequence_path, sub1)):
                name = sub1 + '/' + sub2
                if name in train_list:
                    train_txt.write(name + '\n')
                if name in test_list:
                    test_txt.write(name + '\n')
        train_txt.close()
        test_txt.close()

    elif fun == 1:
        train_list_path = os.path.join(Vimeo90K_path, 'vimeo_septuplet', 'sep_trainlist.txt')
        test_list_path = os.path.join(Vimeo90K_path, 'vimeo_septuplet', 'sep_testlist.txt')
        train_list = [x.strip() for x in open(train_list_path, 'r').readlines()]
        test_list = [x.strip() for x in open(test_list_path, 'r').readlines()]
        # print(len(train_list), len(test_list))  # 64612 7824
        # exit()

        patten = 'im1.png'
        seq_path = os.path.join(Vimeo90K_path, 'vimeo_septuplet', 'sequences')
        all_result, train_result, test_result = [], [], []
        for root, dirs, files in tqdm(os.walk(seq_path)):
            for name in files:
                if fnmatch.fnmatch(name, patten):
                    all_result.append(root)
                    if root[-10:].replace('\\', '/') in train_list:
                        train_result.append(root)
                    elif root[-10:].replace('\\', '/') in test_list:
                        test_result.append(root)
                    else:
                        pass
                        # print('file neither in trainList or testList')
        np.save('./npy/all_folder.npy', all_result)
        np.save('./npy/train_folder.npy', train_result)
        np.save('./npy/test_folder.npy', test_result)

        data1 = np.load('./npy/all_folder.npy')
        data2 = np.load('./npy/train_folder.npy')
        data3 = np.load('./npy/test_folder.npy')
        print(data1)
        print(data2)
        print(data3)

    elif fun == 2:
        # for PSNR
        # bpgenc -f 444 -m 9 im1.png -o im1_QP27.bpg -q 27
        # bpgdec im1_QP27.bpg -o im1_bpg444_QP27.png
        path = os.path.join(Vimeo90K_path, 'vimeo_septuplet', 'sequences')
        for path1 in os.listdir(path):
            dat_file = open(f'./bpg_bat/bpg_{path1}.bat', 'w')
            print(f"INFO: [*] Process {path1}")
            for path2 in os.listdir(os.path.join(path, path1)):
                os.makedirs(os.path.join(path, path1, path2, "bpg"), exist_ok=True)
                image_path = os.path.join(path, path1, path2, "im1.png")
                # print(image_path)
                for qp in ['22', '27', '32', '37']:
                    string_path = os.path.join(path, path1, path2, "bpg", f"im1_QP{qp}.bpg")
                    dec_image_path = os.path.join(path, path1, path2, "bpg", f"im1_bpg444_QP{qp}.png")
                    dat_file.write(f'bpgenc -f 444 -m 9 {image_path} -o {string_path} -q {qp}\n')
                    dat_file.write(f'bpgdec -o {dec_image_path} {string_path}\n')
                    # print(f'bpgdec {string_path} -o {dec_image_path}')
                    # os.system(f'bpgenc -m 9 -f 444 {image_path} -o {string_path} -q {qp}')
                    # os.system(f'bpgdec {string_path} -o {dec_image_path}')
                    # exit()
            dat_file.write('pause')
            dat_file.close()

    elif fun == 3:
        path = os.path.join(Vimeo90K_path, 'vimeo_septuplet', 'sequences')
        for path1 in os.listdir(path):
            print(f"INFO: [*] Process {path1}")
            for path2 in os.listdir(os.path.join(path, path1)):
                shutil.rmtree(os.path.join(path, path1, path2, "bpg"))

    elif fun == 4:
        # 384x192
        dat_file = open(f'./bpg_bat/cropD.bat', 'w')
        sqes = ['BasketballPass_416x240_50', 'BlowingBubbles_416x240_50',
                'BQSquare_416x240_60', 'RaceHorses_416x240_30']
        root = r'D:\DataSet\CTC\D'
        for seq in sqes:
            yuv = os.path.join(root, seq + '.yuv')
            img_path = os.path.join(root, 'X64', seq.replace('416x240', '384x192'))
            os.makedirs(img_path, exist_ok=True)
            # dat_file.write(f'ffmpeg -pix_fmt yuv420p -s 416x240 -i {yuv} -vframes 100 '
            #                f'-filter:v "crop=384:192:0:0" {img_path}/f%%03d.png\n')
            dat_file.write(f'ffmpeg -pix_fmt yuv420p -s 416x240 -i {yuv} -vframes 100 '
                           f'-filter:v "crop=384:192:0:0" {img_path}/im%%05d.png\n')
        dat_file.write('pause')
        dat_file.close()

    elif fun == 5:
        # 384x192  D seq
        gop = 10
        root = r'D:\DataSet\SeqD'
        sqes = ['BasketballPass_416x240_50', 'BlowingBubbles_416x240_50',
                'BQSquare_416x240_60', 'RaceHorses_416x240_30']
        dat_file = open(f'./bpg_bat/ctcD_gop10_bpg.bat', 'w')
        for path in sqes:
            video_frame_path = os.path.join(root, 'X64', path.replace('416x240', '384x192'))
            frames = glob.glob(os.path.join(video_frame_path, '*.png'))
            os.makedirs(os.path.join(video_frame_path, f'gop{gop}_bpg'), exist_ok=True)
            print(f'Find {len(frames)} frames')
            frame_indexs = [1] + [x + 1 for x in range(1, len(frames)) if x % gop == 0]
            for frame_index in frame_indexs:
                image_path = os.path.join(video_frame_path, 'f' + str(frame_index).zfill(3) + '.png')
                for qp in ['22', '27', '32', '37']:
                    string_path = os.path.join(video_frame_path, f'gop{gop}_bpg',
                                               f'QP{qp}_f' + str(frame_index).zfill(3) + '.bpg')
                    dec_image_path = os.path.join(video_frame_path, f'gop{gop}_bpg',
                                                  f'QP{qp}_f' + str(frame_index).zfill(3) + '.png')
                    dat_file.write(f'bpgenc -f 444 -m 9 {image_path} -o {string_path} -q {qp}\n')
                    dat_file.write(f'bpgdec -o {dec_image_path} {string_path}\n')
        dat_file.write('pause')
        dat_file.close()

    elif fun == 6:
        # 416x240  D seq
        gop = 10
        sqes = ['BasketballPass_416x240_50', 'BlowingBubbles_416x240_50',
                'BQSquare_416x240_60', 'RaceHorses_416x240_30']
        dat_file = open(f'./bpg_bat/ctcD_gop10_bpg.bat', 'w')
        for path in sqes:
            video_frame_path = os.path.join(r'D:\DataSet\SeqD', path)
            frames = glob.glob(os.path.join(video_frame_path, '*.png'))
            os.makedirs(os.path.join(video_frame_path, f'gop{gop}_bpg'), exist_ok=True)
            print(f'Find {len(frames)} frames')
            frame_indexs = [1] + [x + 1 for x in range(1, len(frames)) if x % gop == 0]
            for frame_index in frame_indexs:
                image_path = os.path.join(video_frame_path, 'f' + str(frame_index).zfill(3) + '.png')
                for qp in ['17', '22', '27', '32', '37', '39']:
                    string_path = os.path.join(video_frame_path, f'gop{gop}_bpg',
                                               f'QP{qp}_f' + str(frame_index).zfill(3) + '.bpg')
                    dec_image_path = os.path.join(video_frame_path, f'gop{gop}_bpg',
                                                  f'QP{qp}_f' + str(frame_index).zfill(3) + '.png')
                    dat_file.write(f'bpgenc -f 444 -m 9 {image_path} -o {string_path} -q {qp}\n')
                    dat_file.write(f'bpgdec -o {dec_image_path} {string_path}\n')
        dat_file.write('pause')
        dat_file.close()

    elif fun == 7:
        uvg_root = r'D:\DataSet\UVG'
        uvg_yuvs = glob.glob(os.path.join(uvg_root, '*.yuv'))
        for path in uvg_yuvs:
            name = path.split('\\')[-1].split('.')[0].replace('1920x1080', '1920x1072')
            print(name)
            image_path = os.path.join(uvg_root, name)
            os.makedirs(image_path, exist_ok=True)
            cmd = f'ffmpeg -y -pix_fmt yuv420p -s 1920x1080 -i {path} -vframes 120 ' \
                  f'-filter:v "crop=1920:1072:0:0" {image_path}/f%3d.png'
            print(cmd)
            os.system(cmd)

    elif fun == 8:
        gop = 12
        uvg_root = r'D:\DataSet\UVG'
        sqes = ['Beauty_1920x1072_120fps_420_8bit_YUV', 'Bosphorus_1920x1072_120fps_420_8bit_YUV',
                'HoneyBee_1920x1072_120fps_420_8bit_YUV', 'Jockey_1920x1072_120fps_420_8bit_YUV',
                'ReadySteadyGo_1920x1072_120fps_420_8bit_YUV', 'ShakeNDry_1920x1072_120fps_420_8bit_YUV',
                'YachtRide_1920x1072_120fps_420_8bit_YUV']
        dat_file = open(f'./bpg_bat/UVG_gop{gop}_bpg.bat', 'w')
        for path in sqes:
            video_frame_path = os.path.join(uvg_root, path)
            frames = glob.glob(os.path.join(video_frame_path, '*.png'))
            os.makedirs(os.path.join(video_frame_path, f'gop{gop}_bpg'), exist_ok=True)
            print(f'Find {len(frames)} frames')
            frame_indexs = [1] + [x + 1 for x in range(1, len(frames)) if x % gop == 0]
            for frame_index in frame_indexs:
                image_path = os.path.join(video_frame_path, 'f' + str(frame_index).zfill(3) + '.png')
                for qp in ['17', '22', '27', '32', '37']:
                    string_path = os.path.join(video_frame_path, f'gop{gop}_bpg',
                                               f'QP{qp}_f' + str(frame_index).zfill(3) + '.bpg')
                    dec_image_path = os.path.join(video_frame_path, f'gop{gop}_bpg',
                                                  f'QP{qp}_f' + str(frame_index).zfill(3) + '.png')
                    dat_file.write(f'bpgenc -f 444 -m 9 {image_path} -o {string_path} -q {qp}\n')
                    dat_file.write(f'bpgdec -o {dec_image_path} {string_path}\n')
        dat_file.write('pause')
        dat_file.close()
    return 0


def compute_psnr(a, b):
    mse = torch.mean((a - b) ** 2).item()
    return -10 * math.log10(mse)


def compute_msssim(a, b):
    return ms_ssim(a, b, data_range=1.).item()


def compute_bpp(out_net):
    size = out_net['x_hat'].size()
    num_pixels = size[0] * size[2] * size[3]
    return sum(torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
               for likelihoods in out_net['likelihoods'].values()).item()


def compress_key_frames():
    fun = 0
    if fun == 0:
        metric = 'mse'  # choose from ('mse', 'ms-ssim')
        quality = 3  # between (1, 6)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        net = cheng2020_anchor(quality=quality, metric=metric, pretrained=True, progress=False).eval().to(device)
        print(f'Parameters: {sum(p.numel() for p in net.parameters())}')
        img = Image.open('./stmalo_fracape.png').convert('RGB')
        x = transforms.ToTensor()(img).unsqueeze(0).to(device)
        print(x.size())
        with torch.no_grad():
            out_net = net.forward(x)
        out_net['x_hat'].clamp_(0, 1)
        print(out_net.keys())
        rec_net = transforms.ToPILImage()(out_net['x_hat'].squeeze().cpu())
        rec_net.save('./rec_stmalo_fracape.png', quality=95)
        print(type(rec_net))
        exit()

        diff = torch.mean((out_net['x_hat'] - x).abs(), axis=1).squeeze().cpu()
        fix, axes = plt.subplots(1, 3, figsize=(16, 12))
        for ax in axes:
            ax.axis('off')

        axes[0].imshow(img)
        axes[0].title.set_text('Original')

        axes[1].imshow(rec_net)
        axes[1].title.set_text('Reconstructed')

        axes[2].imshow(diff, cmap='viridis')
        axes[2].title.set_text('Difference')

        plt.show()

        print(f'PSNR: {compute_psnr(x, out_net["x_hat"]):.2f}dB')
        print(f'MS-SSIM: {compute_msssim(x, out_net["x_hat"]):.4f}')
        print(f'Bit-rate: {compute_bpp(out_net):.3f} bpp')

    elif fun == 1:
        pass
    return 0


def testH264H265():
    fps = 50
    frames = 100
    gop = 10
    w, h = 416, 240
    crf = 20
    root = r'D:\DataSet\CTC\D\CreateI'
    sqes = ['BasketballPass_416x240_50', 'BlowingBubbles_416x240_50',
            'BQSquare_416x240_60', 'RaceHorses_416x240_30']
    PSNR, BPP, MSSSIM = [], [], []
    for seq in sqes:
        print(seq)
        log_file = os.path.join(root, 'yuvs', f'{seq}_crf{crf}.log')
        data = open(log_file, 'r').readlines()
        bitrate = float(data[-11].split('bitrate=')[1].strip().split('kbits/s')[0])
        bpp = bitrate * 1000.0 * (frames / fps) / (w * h) / frames
        BPP.append(bpp)
        psnr, msssim = [], []
        for f in range(1, 101):
            org = os.path.join(root, 'org', 'yuvs', seq, 'f' + str(f).zfill(3) + '.png')
            tgt = os.path.join(root, 'yuvs', seq, f'H265L{crf}', 'f' + str(f).zfill(3) + '.png')
            im1 = read_image(org).unsqueeze(0)
            im2 = read_image(tgt).unsqueeze(0)
            psnr.append(compute_psnr(im1, im2))
            msssim.append(compute_msssim(im1, im2))
            # print(compute_psnr(im1, im2))
            # print(compute_msssim(im1, im2))
            # exit()
        PSNR.append(np.mean(psnr))
        MSSSIM.append(np.mean(msssim))
    print(f'BPP={np.mean(BPP)} | PSNR={np.mean(PSNR)} | MSSSIM={np.mean(MSSSIM)}')

    return 0


def VID():
    path = r'D:\DataSet\VIDTrainCodec'
    train_path = os.path.join(path, 'VID', 'train')
    val_path = os.path.join(path, 'VID', 'val')
    train_txt_path = os.path.join(path, 'VID', 'train.txt')
    val_txt_path = os.path.join(path, 'VID', 'val.txt')

    fun = 2
    if fun == 0:
        train_txt = open(train_txt_path, 'w')
        val_txt = open(val_txt_path, 'w')

        # train
        for sub1 in sorted(os.listdir(train_path)):
            p1 = os.path.join(train_path, sub1)
            for sub2 in sorted(os.listdir(p1)):
                p2 = os.path.join(p1, sub2)
                images = sorted(glob.glob(os.path.join(p2, '*.JPEG')))
                if len(images) == 0:
                    print(p2)
                    # os.rmdir(p2)
                    continue
                im1 = os.path.join('VID', 'train', sub1, sub2, '000000.JPEG').replace('\\', '/')
                im2 = os.path.join('VID', 'train', sub1, sub2, '000004.JPEG').replace('\\', '/')
                train_txt.write(im1)
                train_txt.write('\n')
                train_txt.write(im2)
                train_txt.write('\n')
                # print(im1, im2)
                assert os.path.exists(os.path.join(path, im1)), os.path.join(path, im1)
                assert os.path.exists(os.path.join(path, im2)), os.path.join(path, im2)
        # val
        for sub1 in sorted(os.listdir(val_path)):
            images = sorted(glob.glob(os.path.join(val_path, sub1, '*.JPEG')))
            if len(images) == 0:
                print(os.path.join(val_path, sub1))
                # os.rmdir(os.path.join(val_path, sub1))
                continue
            im1 = os.path.join('VID', 'val', sub1, '000000.JPEG').replace('\\', '/')
            im2 = os.path.join('VID', 'val', sub1, '000004.JPEG').replace('\\', '/')
            val_txt.write(im1)
            val_txt.write('\n')
            val_txt.write(im2)
            val_txt.write('\n')
            # print(im1, im2)
            assert os.path.exists(os.path.join(path, im1))
            assert os.path.exists(os.path.join(path, im2))

    elif fun == 1:
        train_list = [x.strip() for x in open(train_txt_path, 'r').readlines()]
        train_dat_file = open(f'./train.bat', 'w')
        for train_sub_path in train_list:
            image_path = os.path.join(path, train_sub_path)
            temp = image_path.split('/')
            bpg_path = temp[0] + '/' + temp[1] + '/' + temp[2] + '/' + temp[3] + '/' + 'bpg'
            print(bpg_path, temp[-1].split('.')[0])
            # exit()
            os.makedirs(bpg_path, exist_ok=True)
            for qp in ['22', '27', '32', '37']:
                string_path = os.path.join(bpg_path, f"{temp[-1].split('.')[0]}_QP{qp}.bpg")
                dec_image_path = os.path.join(bpg_path, f"{temp[-1].split('.')[0]}_bpg444_QP{qp}.JPEG")
                train_dat_file.write(f'bpgenc -f 444 -m 9 {image_path} -o {string_path} -q {qp}\n')
                train_dat_file.write(f'bpgdec -o {dec_image_path} {string_path}\n')
        train_dat_file.write('pause')
        train_dat_file.close()

        val_dat_file = open(f'./val.bat', 'w')
        val_list = [x.strip() for x in open(val_txt_path, 'r').readlines()]
        for val_sub_path in val_list:
            image_path = os.path.join(path, val_sub_path)
            temp = image_path.split('/')
            bpg_path = temp[0] + '/' + temp[1] + '/' + temp[2] + '/' + 'bpg'
            print(bpg_path, temp[-1].split('.')[0])
            os.makedirs(bpg_path, exist_ok=True)
            for qp in ['22', '27', '32', '37']:
                string_path = os.path.join(bpg_path, f"{temp[-1].split('.')[0]}_QP{qp}.bpg")
                dec_image_path = os.path.join(bpg_path, f"{temp[-1].split('.')[0]}_bpg444_QP{qp}.JPEG")
                val_dat_file.write(f'bpgenc -f 444 -m 9 {image_path} -o {string_path} -q {qp}\n')
                val_dat_file.write(f'bpgdec -o {dec_image_path} {string_path}\n')
        val_dat_file.write('pause')
        val_dat_file.close()

    elif fun == 2:
        train_list = [x.strip() for x in open(train_txt_path, 'r').readlines()]
        for train_sub_path in train_list:
            image_path = os.path.join(path, train_sub_path)
            temp = image_path.split('/')
            bpg_path = temp[0] + '/' + temp[1] + '/' + temp[2] + '/' + temp[3] + '/' + 'bpg'
            print(bpg_path, temp[-1].split('.')[0])
            # exit()
            os.makedirs(bpg_path, exist_ok=True)
            if os.path.exists(bpg_path):
                shutil.rmtree(bpg_path)
            # for qp in ['22', '27', '32', '37']:
            #     string_path = os.path.join(bpg_path, f"{temp[-1].split('.')[0]}_QP{qp}.bpg").replace('\\', '/')
            #     dec_image_path = os.path.join(bpg_path, f"{temp[-1].split('.')[0]}_bpg444_QP{qp}.JPEG").replace('\\', '/')
            #     if os.path.exists(string_path):
            #         shutil.rmtree(string_path)
            #     if os.path.exists(dec_image_path):
            #         shutil.rmtree(dec_image_path)

        val_list = [x.strip() for x in open(val_txt_path, 'r').readlines()]
        for val_sub_path in val_list:
            image_path = os.path.join(path, val_sub_path)
            temp = image_path.split('/')
            bpg_path = temp[0] + '/' + temp[1] + '/' + temp[2] + '/' + 'bpg'
            print(bpg_path, temp[-1].split('.')[0])
            if os.path.exists(bpg_path):
                shutil.rmtree(bpg_path)
            # for qp in ['22', '27', '32', '37']:
            #     string_path = os.path.join(bpg_path, f"{temp[-1].split('.')[0]}_QP{qp}.bpg").replace('\\', '/')
            #     dec_image_path = os.path.join(bpg_path, f"{temp[-1].split('.')[0]}_bpg444_QP{qp}.JPEG").replace('\\', '/')
            #     if os.path.exists(string_path):
            #         shutil.rmtree(string_path)
            #     if os.path.exists(dec_image_path):
            #         shutil.rmtree(dec_image_path)
    return 0


def test_vid_data():
    cfg = Config.fromfile('./imagenet_vid_dff_style.py')
    print(cfg)
    train_datasets = build_dataset(cfg.data.train)
    val_datasets = build_dataset(cfg.data.val)
    # print(len(train_datasets), len(train_datasets[0]))
    # print(train_datasets[0]['img'].data.size())
    # print(train_datasets[0]['ref_img'].data.size())
    # print(train_datasets[0]['ref_img_metas'].data)
    # print(train_datasets[0].keys())
    # ['img_metas', 'img', 'gt_bboxes', 'gt_labels', 'gt_instance_ids', 'ref_img_metas',
    # 'ref_img', 'ref_gt_bboxes', 'ref_gt_labels', 'ref_gt_instance_ids']
    # print(train_datasets[0]['img_metas'])
    valid_set_loader = build_dataloader(
        val_datasets,
        samples_per_gpu=cfg.data.samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,  # cfg.data.workers_per_gpu
        dist=False,
        shuffle=False
    )
    # for i, data in enumerate(valid_set_loader):
    #     # print(data.keys())  # dict_keys(['img_metas', 'img'])
    #     print(data['img'][0].data[0].shape, data['img_metas'])
    #     exit()

    train_loader = build_dataloader(
        train_datasets,
        cfg.data.samples_per_gpu,
        cfg.data.workers_per_gpu,
        1,
        samples_per_epoch=cfg.data.get('samples_per_epoch', None),
        dist=False,
        seed=0,
        persistent_workers=cfg.data.get('persistent_workers', False),
    )
    print(len(train_loader))
    for i, data in enumerate(train_loader):
        # ['img_metas', 'img', 'gt_bboxes', 'gt_labels', 'gt_instance_ids',
        # 'ref_img_metas', 'ref_img', 'ref_gt_bboxes', 'ref_gt_labels', 'ref_gt_instance_ids']
        print(data['img'].data[0].shape, data['ref_img'].data[0].shape)
        # img_metas = data['img_metas'].data[0]
        # # print(data['img_metas'].data[0][0])
        # # print(data['ref_img_metas'].data[0][0])
        # # exit()
        # img_tensor = data['img'].data[0]
        # imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
        # # print(np.min(imgs), np.max(imgs)
        # h, w, _ = img_metas[0]['img_shape']
        # img_show = imgs[0][:h, :w, :]
        # plt.subplot(121)
        # plt.imshow(imgs[0])
        # plt.subplot(122)
        # plt.imshow(img_show)
        # plt.show()
        # exit()

    return 0


def save_json_file():
    # # # new_json_path = f'/media/user/f126fd00-4370-4fdf-9a0a-fc93e66106eb1/LHB/datasets/ImageNet/VID2015/ILSVRC/annotations/imagenet_vid_val_all_video_f100.json'
    # new_json_path = f'/media/user/f126fd00-4370-4fdf-9a0a-fc93e66106eb1/LHB/datasets/ImageNet/VID2015/ILSVRC/annotations/video_all_fmax96/x264_veryslow_png_crf_35.json'
    # # new_json_path = f'/media/user/f126fd00-4370-4fdf-9a0a-fc93e66106eb1/LHB/datasets/ImageNet/VID2015/ILSVRC/annotations/imagenet_vid_val.json'
    # with open(new_json_path, 'r') as f:
    #     data = json.load(f)
    #     # print(data.keys())  # ['categories', 'videos', 'images', 'annotations']
    #     # print(len(data['videos']))  # 555
    #     # print(len(data['images']))  # 176126
    #     # exit()
    #     print(data['videos'][0])
    #     print(data['videos'][1])
    #     print(data['videos'][-1])
    #     print(data['videos'][-2])
    #     print()
    #     print(data['images'][0])
    #     print(data['images'][1])
    #     print(data['images'][2])
    #     print(data['images'][-1])
    #     print(data['images'][-2])
    #     print(data['images'][-3])
    #     print()
    #     exit()

    test_frames = 96
    videos = 1e10

    json_path = '/media/user/f126fd00-4370-4fdf-9a0a-fc93e66106eb1/LHB/datasets/ImageNet/VID2015/ILSVRC/annotations/imagenet_vid_val.json'
    # 22, 27, 32, 37, 42
    for qp in [42]:
        new_json_path = f'/media/user/f126fd00-4370-4fdf-9a0a-fc93e66106eb1/LHB/datasets/ImageNet/VID2015/ILSVRC/annotations/vall_fmax96/x265_veryslow_qp_{qp}.json'
        # video_all_fmax100_veryfast_x265   video_all_fmax100_veryslow_x265
        # video_all_fmax100_veryfast_x264   video_all_fmax100_veryslow_x264
        # veryfast  veryslow   x264veryfast  x264veryslow
        mark = 'x265veryslow_gop12'  # x264veryslow_gop12  x265veryslow_gop12

        with open(json_path, 'r') as f:
            data = json.load(f)

        new_categories = data['categories']
        new_annotations = data['annotations']
        new_videos = []
        new_images = []

        history1 = []
        count1 = 0
        for i in tqdm(range(len(data['videos']))):
            key = data['videos'][i]['name'].split('/')[1]
            if key not in history1:
                count1 = 0
                history1.append(key)
            else:
                count1 += 1
            if count1 >= test_frames:
                continue

            temp = data['videos'][i]['name'].replace('val/', f'{mark}/dec_img/qp_{qp}/')
            data['videos'][i]['name'] = temp
            new_videos.append(data['videos'][i])

            if len(history1) >= videos:
                break

        history = []
        count = 0
        for i in tqdm(range(len(data['images']))):
            key = data['images'][i]['file_name'].split('/')[1]
            if key not in history1:
                continue
            if key not in history:
                count = 0
                history.append(key)
            else:
                count += 1
            if count >= test_frames:
                continue

            # temp = data['images'][i]['file_name'].replace('val/', f'veryfast/dec_img/crf_{crf}/')

            temp1 = data['images'][i]['file_name'].split('/')
            n = temp1[-1].split('.')[0]
            temp = temp1[0] + '/' + temp1[1] + '/' + str(int(n) + 1).zfill(6) + '.png'
            temp = temp.replace('val/', f'{mark}/dec_img/qp_{qp}/')

            # print(data['images'][i])

            data['images'][i]['file_name'] = temp
            # data['images'][i]['frame_id'] = data['images'][i]['frame_id'] + 1
            # print(data['images'][i])
            # exit()

            new_images.append(data['images'][i])

        new_json_dict = {'categories': new_categories, 'videos': new_videos, 'images': new_images,
                         'annotations': new_annotations}

        with open(new_json_path, 'w', encoding='utf-8') as json_file:
            json.dump(new_json_dict, json_file)
    return 0


def save_json_file_org():
    # new_json_path = f'/media/user/f126fd00-4370-4fdf-9a0a-fc93e66106eb1/LHB/datasets/ImageNet/VID2015/ILSVRC/annotations/imagenet_vid_val_v100_f48_1.json'
    # with open(new_json_path, 'r') as f:
    #     data = json.load(f)
    #     # print(data.keys())  # ['categories', 'videos', 'images', 'annotations']
    #     # print(len(data['videos']))  # 555
    #     # print(len(data['images']))  # 176126
    #     # exit()
    #     print(data['videos'][0])
    #     print(data['videos'][1])
    #     print(data['videos'][-1])
    #     print(data['videos'][-2])
    #     print()
    #     print(data['images'][0])
    #     print(data['images'][1])
    #     print(data['images'][2])
    #     print(data['images'][-1])
    #     print(data['images'][-2])
    #     print(data['images'][-3])
    #     print()
    #     print()
    #     print(data['annotations'][0])
    #     print(data['annotations'][1])
    #     print(data['annotations'][2])
    #     print(data['annotations'][-1])
    #     print(data['annotations'][-2])
    #     print(data['annotations'][-3])
    #     exit()

    test_frames = 96
    videos = 1e5

    json_path = '/tdx/LHB/data/ILSVRC/annotations/imagenet_vid_val.json'

    new_json_path = f'/tdx/LHB/data/ILSVRC/annotations/yuv420_f96.json'

    with open(json_path, 'r') as f:
        data = json.load(f)

    new_categories = data['categories']
    new_annotations = []
    new_videos = []
    new_images = []

    history1 = []
    count1 = 0
    video_ids = []
    for i in tqdm(range(len(data['videos']))):
        key = data['videos'][i]['name'].split('/')[1]
        if key not in history1:
            count1 = 0
            history1.append(key)
        else:
            count1 += 1
        if count1 >= test_frames:
            continue

        new_videos.append(data['videos'][i])
        video_ids.append(data['videos'][i]['id'])

        if len(history1) >= videos:
            break

    history = []
    count = 0
    for i in tqdm(range(len(data['images']))):
        key = data['images'][i]['file_name'].split('/')[1]
        if key not in history1:
            continue
        if key not in history:
            count = 0
            history.append(key)
        else:
            count += 1
        if count >= test_frames:
            continue

        new_images.append(data['images'][i])

    # for i in tqdm(range(len(data['annotations']))):
    #     if data['annotations'][i]['video_id'] not in video_ids:
    #         continue
    #     new_annotations.append(data['annotations'][i])

    new_json_dict = {'categories': new_categories, 'videos': new_videos, 'images': new_images,
                     'annotations': new_annotations}

    with open(new_json_path, 'w', encoding='utf-8') as json_file:
        json.dump(new_json_dict, json_file)

    return 0


def save_json_file_bpg():
    # crf = 35  # 15, 19, 23, 27, 31, 35

    # new_json_path = f'/media/user/f126fd00-4370-4fdf-9a0a-fc93e66106eb1/LHB/datasets/ImageNet/VID2015/ILSVRC/annotations/video100_fmax50_bpg/imagenet_vid_val_v100_f50_qp22.json'
    # with open(new_json_path, 'r') as f:
    #     data = json.load(f)
    #     # print(data.keys())  # ['categories', 'videos', 'images', 'annotations']
    #     # print(len(data['videos']))  # 555
    #     # print(len(data['images']))  # 176126
    #     print(data['videos'][0])
    #     print(data['videos'][-1])
    #     print(data['videos'][-2])
    #     print(data['videos'][-3])
    #     print()
    #     print(data['images'][0])
    #     print(data['images'][-1])
    #     print(data['images'][-2])
    #     print(data['images'][-3])
    #     print()
    #
    #     for i in range(len(data['images'])):
    #         print(data['images'][i])
    #     exit()

    test_frames = 50
    videos = 100
    gop = 10

    json_path = '/media/user/f126fd00-4370-4fdf-9a0a-fc93e66106eb1/LHB/datasets/ImageNet/VID2015/ILSVRC/annotations/imagenet_vid_val.json'

    for qp in [22, 27, 32, 37]:
        new_json_path = f'/media/user/f126fd00-4370-4fdf-9a0a-fc93e66106eb1/LHB/datasets/ImageNet/VID2015/ILSVRC/annotations/video100_fmax50_bpg/imagenet_vid_val_v100_f50_qp{qp}.json'

        with open(json_path, 'r') as f:
            data = json.load(f)
            # print(data.keys())  # ['categories', 'videos', 'images', 'annotations']
            # print(len(data['videos']))  # 555
            # print(len(data['images']))  # 176126
            print(data['videos'][0])
            print(data['videos'][-1])
            print(data['videos'][-2])
            print(data['videos'][-3])
            print()
            print(data['images'][0])
            print(data['images'][-1])
            print(data['images'][-2])
            print(data['images'][-3])
            print()
            # exit()

        new_categories = data['categories']
        new_annotations = data['annotations']
        new_videos = []
        new_images = []

        history1 = []
        count1 = 0
        for i in tqdm(range(len(data['videos']))):
            key = data['videos'][i]['name'].split('/')[1]
            if key not in history1:
                count1 = 0
                history1.append(key)
            else:
                count1 += 1
            if count1 >= test_frames:
                continue

            temp = data['videos'][i]['name'].replace('val/', f'val_bpg/')
            data['videos'][i]['name'] = temp
            new_videos.append(data['videos'][i])

            if len(history1) >= videos:
                break

        history = []
        count = 0
        for i in tqdm(range(len(data['images']))):
            key = data['images'][i]['file_name'].split('/')[1]
            if key not in history1:
                continue
            if key not in history:
                count = 0
                history.append(key)
            else:
                count += 1
            if count >= test_frames:
                continue

            is_key_frame = True if data['images'][i]['frame_id'] % gop == 0 else False
            if is_key_frame:
                temp1 = data['images'][i]['file_name'].split('/')
                n = temp1[-1].split('.')[0]
                n1 = n + f'_bpg444_QP{qp}.JPEG'
                temp = temp1[0] + '/' + temp1[1] + '/' + 'bpg' + '/' + n1
            else:
                temp1 = data['images'][i]['file_name'].split('/')
                n = temp1[-1].split('.')[0]
                temp = temp1[0] + '/' + temp1[1] + '/' + str(int(n) + 1).zfill(6) + '.JPEG'

            temp = temp.replace('val/', f'val_bpg/')
            data['images'][i]['file_name'] = temp

            new_images.append(data['images'][i])

        new_json_dict = {'categories': new_categories, 'videos': new_videos, 'images': new_images,
                         'annotations': new_annotations}

        with open(new_json_path, 'w', encoding='utf-8') as json_file:
            json.dump(new_json_dict, json_file)
    return 0


def vid_val_demo():
    # path = '/media/user/f126fd00-4370-4fdf-9a0a-fc93e66106eb1/LHB/datasets/ImageNet/VID2015/ILSVRC/Data/VID/val'
    # files = os.listdir(path)
    # print(len(files))
    # num = []
    # for p in files:
    #     images = glob.glob(os.path.join(path, p, '*.JPEG'))
    #     num.append(len(images))
    #     # print(len(images))
    #     # exit()
    # print(np.max(num), np.min(num))
    # print(np.unique(num))
    # plt.hist(num, 500)
    # plt.show()
    val_org_path = '/media/user/f126fd00-4370-4fdf-9a0a-fc93e66106eb1/LHB/datasets/ImageNet/VID2015/ILSVRC/Data/VID/val'
    val_tgt_path = '/media/user/f126fd00-4370-4fdf-9a0a-fc93e66106eb1/LHB/datasets/ImageNet/VID2015/ILSVRC/Data/VID/val_video_all_f100'
    files = sorted(os.listdir(val_org_path))
    print(len(files))
    tgt_num = 100
    for p in files:
        print(p)
        new_p = os.path.join(val_tgt_path, p)
        os.makedirs(new_p, exist_ok=True)
        images = sorted(glob.glob(os.path.join(val_org_path, p, '*.JPEG')))
        for i, org_img_p in enumerate(images):
            if i >= tgt_num:
                break
            shutil.copy(org_img_p, new_p)
            # print(i)
        # exit()
    return 0


def read_json_result():
    txt_path = '/home/user/Desktop/LHB/FinalTest/PNGRes/result_mAP.txt'
    txt = open(txt_path, 'w')
    path = '/home/user/Desktop/LHB/FinalTest/PNGRes'
    # part = '265veryslow'  # 265veryslow  265veryfast  264veryslow  264veryfast
    for part in ['x264veryslow', 'x265veryslow']:  # x265veryslow
        txt.write(f'{part}:\n')
        for qp in [22, 27, 32, 37, 42]:
            txt.write(f'qp={qp}:\n')
            json_path = glob.glob(os.path.join(path, f'{part}', f'qp{qp}', '*.json'))[0]
            # print(json_path)
            with open(json_path, 'r') as f:
                data = json.load(f)
                # print(data.keys())
                bbox_mAP = data['bbox_mAP']
                bbox_mAP_50 = data['bbox_mAP_50']
                bbox_mAP_75 = data['bbox_mAP_75']
                bbox_mAP_s = data['bbox_mAP_s']
                bbox_mAP_m = data['bbox_mAP_m']
                bbox_mAP_l = data['bbox_mAP_l']
                bbox_mAP_copypaste = data['bbox_mAP_copypaste']
                txt.write(f'bbox_mAP: {bbox_mAP}, '
                          f'bbox_mAP_50: {bbox_mAP_50}, '
                          f'bbox_mAP_75: {bbox_mAP_75}\n'
                          f'bbox_mAP_s: {bbox_mAP_s}, '
                          f'bbox_mAP_m: {bbox_mAP_m}, '
                          f'bbox_mAP_l: {bbox_mAP_l}\n'
                          f'bbox_mAP_copypaste={bbox_mAP_copypaste}\n')
        txt.write(f'\n\n')

    txt.write(f'HM:\n')
    for qp in [22, 27, 32, 37, 42]:
        txt.write(f'qp={qp}:\n')
        json_path = glob.glob(os.path.join(path, "HM", f'qp{qp}', '*.json'))[0]
        # print(json_path)
        with open(json_path, 'r') as f:
            data = json.load(f)
            # print(data.keys())
            bbox_mAP = data['bbox_mAP']
            bbox_mAP_50 = data['bbox_mAP_50']
            bbox_mAP_75 = data['bbox_mAP_75']
            bbox_mAP_s = data['bbox_mAP_s']
            bbox_mAP_m = data['bbox_mAP_m']
            bbox_mAP_l = data['bbox_mAP_l']
            bbox_mAP_copypaste = data['bbox_mAP_copypaste']
            txt.write(f'bbox_mAP: {bbox_mAP}, '
                      f'bbox_mAP_50: {bbox_mAP_50}, '
                      f'bbox_mAP_75: {bbox_mAP_75}\n'
                      f'bbox_mAP_s: {bbox_mAP_s}, '
                      f'bbox_mAP_m: {bbox_mAP_m}, '
                      f'bbox_mAP_l: {bbox_mAP_l}\n'
                      f'bbox_mAP_copypaste={bbox_mAP_copypaste}\n')
    txt.write(f'\n\n')

    txt.write(f'VTM:\n')
    for qp in [22, 27, 32, 37, 42]:
        txt.write(f'qp={qp}:\n')
        json_path = glob.glob(os.path.join(path, "VTM", f'qp{qp}', '*.json'))[0]
        # print(json_path)
        with open(json_path, 'r') as f:
            data = json.load(f)
            # print(data.keys())
            bbox_mAP = data['bbox_mAP']
            bbox_mAP_50 = data['bbox_mAP_50']
            bbox_mAP_75 = data['bbox_mAP_75']
            bbox_mAP_s = data['bbox_mAP_s']
            bbox_mAP_m = data['bbox_mAP_m']
            bbox_mAP_l = data['bbox_mAP_l']
            bbox_mAP_copypaste = data['bbox_mAP_copypaste']
            txt.write(f'bbox_mAP: {bbox_mAP}, '
                      f'bbox_mAP_50: {bbox_mAP_50}, '
                      f'bbox_mAP_75: {bbox_mAP_75}\n'
                      f'bbox_mAP_s: {bbox_mAP_s}, '
                      f'bbox_mAP_m: {bbox_mAP_m}, '
                      f'bbox_mAP_l: {bbox_mAP_l}\n'
                      f'bbox_mAP_copypaste={bbox_mAP_copypaste}\n')
    txt.write(f'\n\n')

    txt.write(f'DCVC:\n')
    for q in [0, 1, 2, 3]:
        txt.write(f'q={q}:\n')
        json_path = glob.glob(os.path.join(path, f'DCVC', f'q{q}', '*.json'))[0]
        # print(json_path)
        with open(json_path, 'r') as f:
            data = json.load(f)
            # print(data.keys())
            bbox_mAP = data['bbox_mAP']
            bbox_mAP_50 = data['bbox_mAP_50']
            bbox_mAP_75 = data['bbox_mAP_75']
            bbox_mAP_s = data['bbox_mAP_s']
            bbox_mAP_m = data['bbox_mAP_m']
            bbox_mAP_l = data['bbox_mAP_l']
            bbox_mAP_copypaste = data['bbox_mAP_copypaste']
            txt.write(f'bbox_mAP: {bbox_mAP}, '
                      f'bbox_mAP_50: {bbox_mAP_50}, '
                      f'bbox_mAP_75: {bbox_mAP_75}\n'
                      f'bbox_mAP_s: {bbox_mAP_s}, '
                      f'bbox_mAP_m: {bbox_mAP_m}, '
                      f'bbox_mAP_l: {bbox_mAP_l}\n'
                      f'bbox_mAP_copypaste={bbox_mAP_copypaste}\n')
    txt.write(f'\n\n')

    txt.write(f'DVC:\n')
    for q in [22, 27, 32, 37]:
        txt.write(f'q={q}:\n')
        json_path = glob.glob(os.path.join(path, f'DVC', f'qp{q}', '*.json'))[0]
        # print(json_path)
        with open(json_path, 'r') as f:
            data = json.load(f)
            # print(data.keys())
            bbox_mAP = data['bbox_mAP']
            bbox_mAP_50 = data['bbox_mAP_50']
            bbox_mAP_75 = data['bbox_mAP_75']
            bbox_mAP_s = data['bbox_mAP_s']
            bbox_mAP_m = data['bbox_mAP_m']
            bbox_mAP_l = data['bbox_mAP_l']
            bbox_mAP_copypaste = data['bbox_mAP_copypaste']
            txt.write(f'bbox_mAP: {bbox_mAP}, '
                      f'bbox_mAP_50: {bbox_mAP_50}, '
                      f'bbox_mAP_75: {bbox_mAP_75}\n'
                      f'bbox_mAP_s: {bbox_mAP_s}, '
                      f'bbox_mAP_m: {bbox_mAP_m}, '
                      f'bbox_mAP_l: {bbox_mAP_l}\n'
                      f'bbox_mAP_copypaste={bbox_mAP_copypaste}\n')
    txt.write(f'\n\n')

    txt.close()
    return 0


def save_json_file_DVC():
    # crf = 35  # 15, 19, 23, 27, 31, 35
    # new_json_path = '/media/user/f126fd00-4370-4fdf-9a0a-fc93e66106eb1/LHB/datasets/ImageNet/VID2015/ILSVRC/annotations/imagenet_vid_val.json'
    # new_json_path = f'/media/user/f126fd00-4370-4fdf-9a0a-fc93e66106eb1/LHB/datasets/ImageNet/VID2015/ILSVRC/annotations/video_all_fmax96/DVC/qp_22.json'
    # with open(new_json_path, 'r') as f:
    #     data = json.load(f)
    #     # print(data.keys())  # ['categories', 'videos', 'images', 'annotations']
    #     print(len(data['videos']))  # 555
    #     print(len(data['images']))  # 176126
    #     # exit()
    #     print(data['videos'][0])
    #     print(data['videos'][1])
    #     print(data['videos'][2])
    #     print(data['videos'][-1])
    #     print(data['videos'][-2])
    #     print(data['videos'][-3])
    #     print(data['videos'][-4])
    #     print()
    #     print(data['images'][0])
    #     print(data['images'][1])
    #     print(data['images'][2])
    #     print(data['images'][3])
    #     print(data['images'][-1])
    #     print(data['images'][-2])
    #     print(data['images'][-3])
    #     print()
    #
    #     print(len(data['annotations']))
    #     print(data['annotations'][0])
    #     print(data['annotations'][1])
    #     print(data['annotations'][2])
    #     print(data['annotations'][-1])
    #     print(data['annotations'][-2])
    #     print(data['annotations'][-3])
    #     print(data['annotations'][-4])
    #     # exit()

    test_frames = 96

    json_path = '/media/user/f126fd00-4370-4fdf-9a0a-fc93e66106eb1/LHB/datasets/ImageNet/VID2015/ILSVRC/annotations/imagenet_vid_val.json'

    for qp in [22, 27, 32, 37]:
        new_json_path = f'/media/user/f126fd00-4370-4fdf-9a0a-fc93e66106eb1/LHB/datasets/ImageNet/VID2015/ILSVRC/annotations/vall_fmax96/dvc_qp_{qp}.json'
        # os.makedirs(f'/media/user/f126fd00-4370-4fdf-9a0a-fc93e66106eb1/LHB/datasets/ImageNet/VID2015/ILSVRC/annotations/video_all_fmax96/DVC', exist_ok=True)
        mark = 'val_DVC_GOP12F96'

        with open(json_path, 'r') as f:
            data = json.load(f)

        new_categories = data['categories']
        new_annotations = data['annotations']
        new_videos = []
        new_images = []

        history1 = []
        count1 = 0
        for i in tqdm(range(len(data['videos']))):
            key = data['videos'][i]['name'].split('/')[1]
            if key not in history1:
                count1 = 0
                history1.append(key)
            else:
                count1 += 1
            if count1 >= test_frames:
                continue

            temp = data['videos'][i]['name'].replace('val/', f'{mark}/DVC_QP{qp}/')
            data['videos'][i]['name'] = temp
            new_videos.append(data['videos'][i])

        history = []
        count = 0
        for i in tqdm(range(len(data['images']))):
            key = data['images'][i]['file_name'].split('/')[1]

            if key not in history:
                count = 0
                history.append(key)
            else:
                count += 1
            if count >= test_frames:
                continue

            # temp = data['images'][i]['file_name'].replace('val/', f'veryfast/dec_img/crf_{crf}/')

            temp1 = data['images'][i]['file_name'].split('/')
            n = temp1[-1].split('.')[0]
            temp = temp1[0] + '/' + temp1[1] + '/' + str(int(n) + 1).zfill(6) + '.png'
            temp = temp.replace('val/', f'{mark}/DVC_QP{qp}/')

            # print(data['images'][i])

            data['images'][i]['file_name'] = temp
            # if 'ILSVRC2015_val_00005000' in temp:
            #     print(temp)
            # data['images'][i]['frame_id'] = data['images'][i]['frame_id'] + 1
            # print(data['images'][i])
            # exit()

            new_images.append(data['images'][i])
        # exit()

        new_json_dict = {'categories': new_categories, 'videos': new_videos, 'images': new_images,
                         'annotations': new_annotations}

        with open(new_json_path, 'w', encoding='utf-8') as json_file:
            json.dump(new_json_dict, json_file)
    return 0


def save_json_file_DCVC():
    # crf = 35  # 15, 19, 23, 27, 31, 35
    # new_json_path = '/media/user/f126fd00-4370-4fdf-9a0a-fc93e66106eb1/LHB/datasets/ImageNet/VID2015/ILSVRC/annotations/imagenet_vid_val.json'
    new_json_path = f'/media/user/f126fd00-4370-4fdf-9a0a-fc93e66106eb1/LHB/datasets/ImageNet/VID2015/ILSVRC/annotations/vall_fmax96/dcvc_q_1.json'
    with open(new_json_path, 'r') as f:
        data = json.load(f)
        # print(data.keys())  # ['categories', 'videos', 'images', 'annotations']
        print(len(data['videos']))  # 555
        print(len(data['images']))  # 176126
        # exit()
        print(data['videos'][0])
        print(data['videos'][1])
        print(data['videos'][2])
        print(data['videos'][-1])
        print(data['videos'][-2])
        print(data['videos'][-3])
        print(data['videos'][-4])
        print()
        print(data['images'][0])
        print(data['images'][1])
        print(data['images'][2])
        print(data['images'][3])
        print(data['images'][-1])
        print(data['images'][-2])
        print(data['images'][-3])
        print()

        print(len(data['annotations']))
        print(data['annotations'][0])
        print(data['annotations'][1])
        print(data['annotations'][2])
        print(data['annotations'][-1])
        print(data['annotations'][-2])
        print(data['annotations'][-3])
        print(data['annotations'][-4])
        exit()

    test_frames = 96

    json_path = '/media/user/f126fd00-4370-4fdf-9a0a-fc93e66106eb1/LHB/datasets/ImageNet/VID2015/ILSVRC/annotations/imagenet_vid_val.json'

    for q in [0, 1, 2, 3]:
        new_json_path = f'/media/user/f126fd00-4370-4fdf-9a0a-fc93e66106eb1/LHB/datasets/ImageNet/VID2015/ILSVRC/annotations/video_all_fmax96/dcvc_q_{q}.json'
        # os.makedirs(f'/media/user/f126fd00-4370-4fdf-9a0a-fc93e66106eb1/LHB/datasets/ImageNet/VID2015/ILSVRC/annotations/video_all_fmax96/DVC', exist_ok=True)
        mark = 'val_DCVC_GOP12F96'

        with open(json_path, 'r') as f:
            data = json.load(f)

        new_categories = data['categories']
        new_annotations = data['annotations']
        new_videos = []
        new_images = []

        history1 = []
        count1 = 0
        for i in tqdm(range(len(data['videos']))):
            key = data['videos'][i]['name'].split('/')[1]
            if key not in history1:
                count1 = 0
                history1.append(key)
            else:
                count1 += 1
            if count1 >= test_frames:
                continue

            temp = data['videos'][i]['name'].replace('val/', f'{mark}/q{q}/')
            data['videos'][i]['name'] = temp
            new_videos.append(data['videos'][i])

        history = []
        count = 0
        for i in tqdm(range(len(data['images']))):
            key = data['images'][i]['file_name'].split('/')[1]

            if key not in history:
                count = 0
                history.append(key)
            else:
                count += 1
            if count >= test_frames:
                continue

            # temp = data['images'][i]['file_name'].replace('val/', f'veryfast/dec_img/crf_{crf}/')

            temp1 = data['images'][i]['file_name'].split('/')
            n = temp1[-1].split('.')[0]
            temp = temp1[0] + '/' + temp1[1] + '/' + str(int(n)).zfill(6) + '.png'
            temp = temp.replace('val/', f'{mark}/q{q}/')

            # print(data['images'][i])

            data['images'][i]['file_name'] = temp
            # if 'ILSVRC2015_val_00005000' in temp:
            #     print(temp)
            # data['images'][i]['frame_id'] = data['images'][i]['frame_id'] + 1
            # print(data['images'][i])
            # exit()

            new_images.append(data['images'][i])
        # exit()

        new_json_dict = {'categories': new_categories, 'videos': new_videos, 'images': new_images,
                         'annotations': new_annotations}

        with open(new_json_path, 'w', encoding='utf-8') as json_file:
            json.dump(new_json_dict, json_file)
    return 0


def cal_VID_psnr():
    org_path = '/media/user/f126fd00-4370-4fdf-9a0a-fc93e66106eb1/LHB/datasets/ImageNet/VID2015/ILSVRC/Data/VID/val'
    seqs = sorted(os.listdir(org_path))
    print(len(seqs))
    max_frames = 100
    PSNR = []
    for k, seq in enumerate(seqs):
        ppsnr = []
        org_images = sorted(glob.glob(os.path.join(org_path, seq, '*.JPEG')))
        # print(seq, len(org_images))
        for i, image_path in enumerate(org_images):
            if i >= max_frames:
                break
            org_im = cv2.imread(image_path) / 255.0
            # /media/user/f126fd00-4370-4fdf-9a0a-fc93e66106eb1/LHB/datasets/ImageNet/VID2015/ILSVRC/Data/VID/val_DVC_video_all_f100/DVC_QP37
            image_path1 = image_path.replace('/val/', '/val_DVC_video_all_f100/DVC_QP37/')
            # image_path1 = image_path.replace('/val/', '/DVC/DVC_QP37/')
            dec_im = cv2.imread(image_path1) / 255.0
            mse = np.mean((org_im - dec_im) ** 2)
            psnr = 10 * np.log10(1.0 / mse)
            ppsnr.append(psnr)
            # print(psnr)
            # exit()
            # 33.28011963718709
            # print(image_path)
            # print(org_im.shape)
            # print(dec_im.shape)
            # exit()
        # exit()
        PSNR.append(np.average(ppsnr))
        print(k, seq, len(org_images), np.average(ppsnr))

    print(np.average(PSNR))
    # qp37 32.930791219415326
    return 0


def save_json_file_HM_VTM():
    new_json_path = f'/media/user/f126fd00-4370-4fdf-9a0a-fc93e66106eb1/LHB/datasets/ImageNet/VID2015/ILSVRC/annotations/vall_fmax96/VTM_qp_22.json'
    with open(new_json_path, 'r') as f:
        data = json.load(f)
        # print(data.keys())  # ['categories', 'videos', 'images', 'annotations']
        # print(len(data['videos']))  # 555
        # print(len(data['images']))  # 176126
        # exit()
        print(data['videos'][0])
        print(data['videos'][1])
        print(data['videos'][-1])
        print(data['videos'][-2])
        print()
        print(data['images'][0])
        print(data['images'][1])
        print(data['images'][2])
        print(data['images'][-1])
        print(data['images'][-2])
        print(data['images'][-3])
        print()
        exit()

    test_frames = 96
    videos = 1e10

    json_path = '/media/user/f126fd00-4370-4fdf-9a0a-fc93e66106eb1/LHB/datasets/ImageNet/VID2015/ILSVRC/annotations/imagenet_vid_val.json'

    for qp in [22, 27, 32, 37, 42]:
        new_json_path = f'/media/user/f126fd00-4370-4fdf-9a0a-fc93e66106eb1/LHB/datasets/ImageNet/VID2015/ILSVRC/annotations/vall_fmax96/VTM_qp_{qp}.json'
        mark = 'VTM13.2'  # HM16.20  VTM13.2
        # mark = 'HM16.20'

        with open(json_path, 'r') as f:
            data = json.load(f)

        new_categories = data['categories']
        new_annotations = data['annotations']
        new_videos = []
        new_images = []

        history1 = []
        count1 = 0
        for i in tqdm(range(len(data['videos']))):
            key = data['videos'][i]['name'].split('/')[1]
            if key not in history1:
                count1 = 0
                history1.append(key)
            else:
                count1 += 1
            if count1 >= test_frames:
                continue

            temp = data['videos'][i]['name'].replace('val/', f'{mark}/dec_img/qp_{qp}/')
            data['videos'][i]['name'] = temp
            new_videos.append(data['videos'][i])

            if len(history1) >= videos:
                break

        history = []
        count = 0
        for i in tqdm(range(len(data['images']))):
            key = data['images'][i]['file_name'].split('/')[1]
            if key not in history1:
                continue
            if key not in history:
                count = 0
                history.append(key)
            else:
                count += 1
            if count >= test_frames:
                continue

            # temp = data['images'][i]['file_name'].replace('val/', f'veryfast/dec_img/crf_{crf}/')

            temp1 = data['images'][i]['file_name'].split('/')
            n = temp1[-1].split('.')[0]
            temp = temp1[0] + '/' + temp1[1] + '/' + str(int(n) + 1).zfill(6) + '.png'
            temp = temp.replace('val/', f'{mark}/dec_img/qp_{qp}/')

            # print(data['images'][i])

            data['images'][i]['file_name'] = temp
            # data['images'][i]['frame_id'] = data['images'][i]['frame_id'] + 1
            # print(data['images'][i])
            # exit()

            new_images.append(data['images'][i])

        new_json_dict = {'categories': new_categories, 'videos': new_videos, 'images': new_images,
                         'annotations': new_annotations}

        with open(new_json_path, 'w', encoding='utf-8') as json_file:
            json.dump(new_json_dict, json_file)
    return 0


def ucf101_demo():
    # 1, 9537 3783
    # 2, 9586 3734
    # 3, 9624 3696
    train_txt = open('/media/user/f126fd00-4370-4fdf-9a0a-fc93e66106eb1/LHB/datasets/ucf101/ucf101_train_split_3_rawframes.txt', 'r')
    val_txt = open('/media/user/f126fd00-4370-4fdf-9a0a-fc93e66106eb1/LHB/datasets/ucf101/ucf101_val_split_3_rawframes.txt', 'r')
    train_data = train_txt.readlines()
    val_data = val_txt.readlines()
    print(len(train_data), len(val_data))
    return 0


def save_json_file11():
    test_frames = 96
    videos = 1e5

    json_path = '/tdx/LHB/data/ILSVRC/annotations/imagenet_vid_val.json'

    new_json_path = f'/tdx/LHB/data/ILSVRC/annotations/org_f96_yuv420.json'
    mark = 'val_video_all_f96_yuv420'  # x265veryslow_gop12

    with open(json_path, 'r') as f:
        data = json.load(f)

    new_categories = data['categories']
    new_annotations = data['annotations']
    new_videos = []
    new_images = []

    history1 = []
    count1 = 0
    for i in tqdm(range(len(data['videos']))):
        key = data['videos'][i]['name'].split('/')[1]
        if key not in history1:
            count1 = 0
            history1.append(key)
        else:
            count1 += 1
        if count1 >= test_frames:
            continue

        temp = data['videos'][i]['name'].replace('val/', f'{mark}/')
        data['videos'][i]['name'] = temp
        new_videos.append(data['videos'][i])

        if len(history1) >= videos:
            break

    history = []
    count = 0
    for i in tqdm(range(len(data['images']))):
        key = data['images'][i]['file_name'].split('/')[1]
        if key not in history1:
            continue
        if key not in history:
            count = 0
            history.append(key)
        else:
            count += 1
        if count >= test_frames:
            continue

        # temp = data['images'][i]['file_name'].replace('val/', f'veryfast/dec_img/crf_{crf}/')

        temp1 = data['images'][i]['file_name'].split('/')
        n = temp1[-1].split('.')[0]
        temp = temp1[0] + '/' + temp1[1] + '/' + str(int(n) + 1).zfill(6) + '.png'
        temp = temp.replace('val/', f'{mark}/')

        # print(data['images'][i])

        data['images'][i]['file_name'] = temp
        # data['images'][i]['frame_id'] = data['images'][i]['frame_id'] + 1
        # print(data['images'][i])
        # exit()

        new_images.append(data['images'][i])

    new_json_dict = {'categories': new_categories, 'videos': new_videos, 'images': new_images,
                     'annotations': new_annotations}

    with open(new_json_path, 'w', encoding='utf-8') as json_file:
        json.dump(new_json_dict, json_file)
    return 0


if __name__ == "__main__":
    save_json_file11()
