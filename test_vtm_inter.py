import glob
import os
import numpy as np
from tqdm import tqdm
import fnmatch
import shutil
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import torch
from pytorch_msssim import ms_ssim
import math
import cv2
import json


TEST_DATA = {
    'HEVC_B': {
        'path': '/tdx/LHB/data/TestSets/HMVTM/yuv/ClassB',
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
        'path': '/tdx/LHB/data/TestSets/HMVTM/yuv/ClassC/',
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
        'path': '/tdx/LHB/data/TestSets/HMVTM/yuv/ClassD/',
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
        'path': '/tdx/LHB/data/TestSets/HMVTM/yuv/ClassE/',
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

    'UVG': {
        'path': '/tdx/LHB/data/TestSets/HMVTM/yuv/UVG/',
        'frames': 96,
        'gop': 12,
        'org_resolution': '1920x1080',
        'x64_resolution': '1920x1024',
        'sequences': [
            'Beauty_1920x1080_120fps_420_8bit_YUV',
            'Bosphorus_1920x1080_120fps_420_8bit_YUV',
            'HoneyBee_1920x1080_120fps_420_8bit_YUV',
            'Jockey_1920x1080_120fps_420_8bit_YUV',
            'ReadySteadyGo_1920x1080_120fps_420_8bit_YUV',
            'ShakeNDry_1920x1080_120fps_420_8bit_YUV',
            'YachtRide_1920x1080_120fps_420_8bit_YUV',
        ],
    },

    "MCL-JCV": {
        "path": "/tdx/LHB/data/TestSets/HMVTM/yuv/MCL-JCV",
        'frames': 96,
        'gop': 12,
        'org_resolution': '1920x1080',
        'x64_resolution': '1920x1024',  # 18,20,24,25
        "sequences": [
            "videoSRC01_1920x1080_30",
            "videoSRC02_1920x1080_30",
            "videoSRC03_1920x1080_30",
            "videoSRC04_1920x1080_30",
            "videoSRC05_1920x1080_25",
            "videoSRC06_1920x1080_25",
            "videoSRC07_1920x1080_25",
            "videoSRC08_1920x1080_25",
            "videoSRC09_1920x1080_25",
            "videoSRC10_1920x1080_30",
            "videoSRC11_1920x1080_30",
            "videoSRC12_1920x1080_30",
            "videoSRC13_1920x1080_30",
            "videoSRC14_1920x1080_30",
            "videoSRC15_1920x1080_30",
            "videoSRC16_1920x1080_30",
            "videoSRC17_1920x1080_24",
            "videoSRC18_1920x1080_25",
            "videoSRC19_1920x1080_30",
            "videoSRC20_1920x1080_25",
            "videoSRC21_1920x1080_24",
            "videoSRC22_1920x1080_24",
            "videoSRC23_1920x1080_24",
            "videoSRC24_1920x1080_24",
            "videoSRC25_1920x1080_24",
            "videoSRC26_1920x1080_30",
            "videoSRC27_1920x1080_30",
            "videoSRC28_1920x1080_30",
            "videoSRC29_1920x1080_24",
            "videoSRC30_1920x1080_30",
        ]
    }
}


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


def prepare_data_train():
    # 每个文件夹下序列都是7帧  原始图像尺寸都是256x448x3
    Vimeo90K_path = r'D:\DataSet'
    fun = -1
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

    return 0


def prepare_data_test():
    tgtp = '/tdx/LHB/data/TestSets/HMVTM'
    fun = 7
    # Crop YUV420 to PNG
    if fun == 0:
        # HEVC_B  HEVC_C  HEVC_D  HEVC_E  UVG    MCL-JCV
        data = TEST_DATA['UVG']
        org_resolution = data['org_resolution']
        x64_resolution = data['x64_resolution']
        new_w, new_h = int(x64_resolution.split('x')[0]), int(x64_resolution.split('x')[1])
        frames = data['frames']
        for seq in data['sequences']:
            print(seq)
            yuv = os.path.join(data['path'], 'ORG_YUV420', seq + '.yuv')
            img_path = os.path.join(data['path'], 'PNG_Frames', seq.replace(org_resolution, x64_resolution))
            os.makedirs(img_path, exist_ok=True)
            os.system(f'ffmpeg -pix_fmt yuv420p -s {org_resolution} -y -i {yuv} -vframes {frames} '
                      f'-filter:v "crop={new_w}:{new_h}:0:0" {img_path}/f%03d.png')
    # Crop yuv420
    elif fun == 0.1:
        # HEVC_B  HEVC_C  HEVC_D  HEVC_E  UVG    MCL-JCV
        data = TEST_DATA['MCL-JCV']
        org_resolution = data['org_resolution']
        x64_resolution = data['x64_resolution']
        new_w, new_h = int(x64_resolution.split('x')[0]), int(x64_resolution.split('x')[1])
        frames = data['frames']
        for seq in data['sequences']:
            print(seq)
            yuv = os.path.join(data['path'], 'ORG_YUV420', seq + '.yuv')
            new_yuv = os.path.join(data['path'], 'Croped_YUV420', seq.replace(org_resolution, x64_resolution) + '.yuv')
            # os.makedirs(new_yuv, exist_ok=True)
            # print(yuv)
            # print(new_yuv)
            # exit()
            os.system(f'ffmpeg -pix_fmt yuv420p -s {org_resolution} -y -i {yuv} -vframes {frames} '
                      f'-vf crop={new_w}:{new_h}:0:0 {new_yuv}')
    # PNG to YUV444p
    elif fun == 1:
        # HEVC_B  HEVC_C  HEVC_D  HEVC_E  UVG  MCL-JCV
        KEY = 'MCL-JCV'
        data = TEST_DATA[KEY]
        org_resolution = data['org_resolution']
        x64_resolution = data['x64_resolution']
        path = data['path']
        frames = data['frames']
        # print(len(data['sequences']))
        # exit()
        for seq in data['sequences']:
            new_name = seq.replace(org_resolution, x64_resolution)
            img_path = os.path.join(path, 'PNG_Frames', seq.replace(org_resolution, x64_resolution))

            if KEY == 'UVG':
                fps = 120
            else:
                fps = int(seq.split('_')[-1])
            print(new_name, fps)

            os.system(
                f'ffmpeg -y -r {fps} -i {img_path}/f%3d.png -vframes {frames} -pix_fmt yuv444p '
                f'-s {x64_resolution} {path}/YUV444p/{new_name}_444p.yuv')
    # YUV44 to PNG
    elif fun == 2:
        # HEVC_B  HEVC_C  HEVC_D  HEVC_E  UVG    MCL-JCV
        for key in ['HEVC_B', 'HEVC_C', 'HEVC_D', 'HEVC_E', 'UVG', 'MCL-JCV']:
            data = TEST_DATA[key]
            org_resolution = data['org_resolution']
            x64_resolution = data['x64_resolution']
            frames = data['frames']
            for seq in data['sequences']:
                print(seq)
                yuv = os.path.join(data['path'], seq.replace(org_resolution, x64_resolution) + '.yuv')
                img_path = os.path.join(tgtp, 'org_frames', key, seq.replace(org_resolution, x64_resolution))
                os.makedirs(img_path, exist_ok=True)
                os.system(
                    f'ffmpeg -y -pix_fmt yuv420p -s {x64_resolution} -i {yuv} -vframes {frames} {img_path}/f%03d.png')
    # BPG ClassD Enc. KeyFrame
    elif fun == 3:
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
    # BPG UVG Enc. KeyFrame
    elif fun == 4:
        gop = 12
        uvg_root = r'E:\dataset\UVG'
        sqes = ['Beauty_1920x1024_120fps_420_8bit_YUV', 'Bosphorus_1920x1024_120fps_420_8bit_YUV',
                'HoneyBee_1920x1024_120fps_420_8bit_YUV', 'Jockey_1920x1024_120fps_420_8bit_YUV',
                'ReadySteadyGo_1920x1024_120fps_420_8bit_YUV', 'ShakeNDry_1920x1024_120fps_420_8bit_YUV',
                'YachtRide_1920x1024_120fps_420_8bit_YUV']
        dat_file = open(f'./UVG_gop{gop}_bpg.bat', 'w')
        for path in sqes:
            video_frame_path = os.path.join(uvg_root, path)
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
    # write bat
    elif fun == 5:
        # HEVC_B  HEVC_C  HEVC_D  HEVC_E  UVG  MCL-JCV
        key = 'HM'
        KEY = 'MCL-JCV'
        data = TEST_DATA[KEY]
        org_resolution = data['org_resolution']
        x64_resolution = data['x64_resolution']
        path = data['path']
        frames = data['frames']
        new_w, new_h = int(x64_resolution.split('x')[0]), int(x64_resolution.split('x')[1])
        dat_file = open(os.path.join(path, f'{key}_{KEY}.bat'), 'w')
        for seq in data['sequences']:
            new_name = seq.replace(org_resolution, x64_resolution)

            if KEY == 'UVG':
                fps = 120
            else:
                fps = int(seq.split('_')[-1])
            print(new_name, fps)

            # for qp in [22, 27, 32, 37, 42]:
            #     dat_file.write(f'EncoderApp13.2ORG -c seq.cfg -c encoder_lowdelay_P_vtm13.2_key12_gop6.cfg '
            #                    f' -q {qp} -f {frames} -wdt {new_w} -hgt {new_h} --InputFile={new_name}.yuv '
            #                    f' --IntraPeriod=12 --Profile=auto --FrameRate={fps} '  # main_10_444
            #                    f'--InputChromaFormat=420 --ConformanceWindowMode=1 -b bin/{key}_{new_name}_QP{qp}.bin '
            #                    f'--InputBitDepth=8 --OutputBitDepth=8 --DecodingRefreshType=2 --Level=4 '
            #                    f'-o dec/{key}_{new_name}_QP{qp}.yuv >txt/{key}_{new_name}_QP{qp}.txt\n')

            for qp in [22, 27, 32, 37, 42]:
                # dat_file.write(f'EncoderApp13.2ORG -c seq.cfg -c encoder_lowdelay_P_vtm13.2_key12_gop6.cfg '
                #                f'-c yuv444.cfg -q {qp} -f {frames} -wdt {new_w} -hgt {new_h} --InputFile={new_name}.yuv '
                #                f' --IntraPeriod=12 --Profile=auto --FrameRate={fps} '  # main_10_444
                #                f'--InputChromaFormat=420 --ConformanceWindowMode=1 -b bin/{seq}_QP{qp}.bin '
                #                f'--InputBitDepth=8 --OutputBitDepth=8 --DecodingRefreshType=2 --Level=4 '
                #                f'-o dec/{seq}_QP{qp}.yuv >txt/{seq}_QP{qp}.txt\n')

                dat_file.write(f'TAppEncoderHM16.20ORG -c seq.cfg -c encoder_lowdelay_P_main.cfg '
                               f'-q {qp} -f {frames} -wdt {new_w} -hgt {new_h} --IntraPeriod=12 -i {new_name}.yuv '
                               f'--InputChromaFormat=420 -b bin/{key}_{new_name}_QP{qp}.bin '
                               f'--ConformanceWindowMode=1 -o dec/{key}_{new_name}_QP{qp}.yuv >txt/{key}_{new_name}_QP{qp}.txt\n')
            dat_file.write('\n')
        dat_file.write('pause')
        dat_file.close()
    # YUV44 to PNG
    elif fun == 6:
        ff = 1
        if ff == 0:
            # HEVC_B  HEVC_C  HEVC_D  HEVC_E  UVG    MCL-JCV
            for key in ['HEVC_B', 'HEVC_C', 'HEVC_D', 'HEVC_E', 'UVG', 'MCL-JCV']:
                data = TEST_DATA[key]
                org_resolution = data['org_resolution']
                x64_resolution = data['x64_resolution']
                frames = data['frames']
                for seq in data['sequences']:
                    print(seq)
                    yuv = os.path.join(data['path'], seq.replace(org_resolution, x64_resolution) + '.yuv')
                    img_path = os.path.join(tgtp, 'org_frames', key, seq.replace(org_resolution, x64_resolution))
                    os.makedirs(img_path, exist_ok=True)
                    os.system(
                        f'ffmpeg -y -pix_fmt yuv420p -s {x64_resolution} -i {yuv} -vframes {frames} {img_path}/f%03d.png')
        elif ff == 1:
            # codec = 'HM'
            codec = 'VTM'
            for key in ['HEVC_B', 'HEVC_C', 'HEVC_D', 'HEVC_E', 'UVG', 'MCL-JCV']:
                data = TEST_DATA[key]
                org_resolution = data['org_resolution']
                x64_resolution = data['x64_resolution']
                frames = data['frames']
                for seq in data['sequences']:
                    print(seq)
                    new_seq = seq.replace(org_resolution, x64_resolution)
                    for qp in ['22', '27', '32', '37', '42']:
                        yuv = os.path.join(tgtp, 'result/dec', f'{codec}_{new_seq}_QP{qp}.yuv')
                        img_path = os.path.join(tgtp, f'dec_frames/{codec}/{qp}',
                                                key, seq.replace(org_resolution, x64_resolution))
                        os.makedirs(img_path, exist_ok=True)
                        os.system(
                            f'ffmpeg -y -pix_fmt yuv420p -s {x64_resolution} -i {yuv} -vframes {frames} {img_path}/f%03d.png')
                        # exit()
    # cal PSNR, MS-SSIM
    elif fun == 7:
        codec = 'HM'
        # codec = 'VTM'
        # log = open(os.path.join(tgtp, f'{codec}_result.txt'), 'w')
        # log1 = open(os.path.join(tgtp, f'{codec}_result_avg.txt'), 'w')
        for key in ['HEVC_B', 'HEVC_C', 'HEVC_D', 'HEVC_E', 'UVG', 'MCL-JCV']:
            data = TEST_DATA[key]
            org_resolution = data['org_resolution']
            x64_resolution = data['x64_resolution']
            # frames = data['frames']
            aPSNR, aMSSSIM, aBPP = [], [], []
            for qp in ['42', '37', '32', '27', '22']:
                PSNR, MSSSIM, BPP1, BPP2 = [], [], [], []
                for seq in data['sequences']:
                    if key == 'UVG':
                        fps = 120
                    else:
                        fps = int(seq.split('_')[-1])

                    new_seq = seq.replace(org_resolution, x64_resolution)
                    org_images = glob.glob(os.path.join(tgtp, 'org_frames', key, new_seq, "*.png"))
                    shape = cv2.imread(org_images[0], -1).shape
                    images = sorted(org_images)
                    _PSNR, _MSSSIM = [], []
                    for j, image in enumerate(images):
                        name = image.split('/')[-1]
                        dec_img_path = os.path.join(tgtp, f'dec_frames/{codec}/{qp}',
                                                    key, seq.replace(org_resolution, x64_resolution), name)
                        org = read_image(image).unsqueeze(0).cuda()
                        dec = read_image(dec_img_path).unsqueeze(0).cuda()
                        # print(org.shape, np.min(org), np.max(org))
                        # print(dec.shape, np.min(dec), np.max(dec))
                        # exit()
                        _psnr = cal_psnr_torch(org, dec)
                        _msssim = ms_ssim(org, dec, 1.0).item()
                        # print(_psnr)
                        _PSNR.append(_psnr)
                        _MSSSIM.append(_msssim)
                        print(f'qp = {qp},  seq = {new_seq}', j, _psnr, _msssim)

                    bin_file = os.path.join(tgtp, 'result', 'bin', f'{codec}_{new_seq}_QP{qp}.bin')
                    bpp1 = os.path.getsize(bin_file) * 8 / shape[0] / shape[1] / len(org_images)
                    bpp2 = 0.0

                    if codec == 'HM':
                        txt_file = os.path.join(tgtp, 'result', 'txt', f'{codec}_{new_seq}_QP{qp}.txt')
                        txt = open(txt_file, 'r')
                        lines = txt.readlines()
                        bpp2 = float(lines[-3].split(' (')[-1].strip().split(' ')[0])
                        bpp2 = bpp2 * 1000 * (len(org_images) / fps) / shape[0] / shape[1] / len(org_images)
                        txt.close()
                    elif codec == 'VTM':
                        txt_file = os.path.join(tgtp, 'result', 'txt', f'{codec}_{new_seq}_QP{qp}.txt')
                        txt = open(txt_file, 'r')
                        lines = txt.readlines()
                        bpp2 = float(lines[-4].split('a')[-1].strip().split('  ')[0])
                        bpp2 = bpp2 * 1000 * len(org_images) / fps / shape[0] / shape[1] / len(org_images)
                        txt.close()
                    # log.write(
                    #     f'qp = {qp}, seq = {new_seq}, psnr = {np.mean(_PSNR):.5f}, ms-ssim = {np.mean(_MSSSIM):.5f}, '
                    #     f'bpp1 = {bpp1:.5f}, bpp2 = {bpp2:.5f}\n')
                    print(f'qp = {qp},  seq = {new_seq}, psnr = {np.mean(_PSNR):.5f}, ms-ssim = {np.mean(_MSSSIM):.5f}, '
                          f'bpp1 = {bpp1:.5f}, bpp2 = {bpp2:.5f}')
                    # log.flush()
                    # exit()
                    BPP1.append(bpp1)
                    BPP2.append(bpp2)
                    PSNR.append(np.average(_PSNR))
                    MSSSIM.append(np.average(_MSSSIM))
                # log.write('\n')
                # log1.write(f'seq = {key}, qp = {qp}, psnr = {np.average(PSNR):.5f}, ms-ssim = {np.mean(MSSSIM):.5f}, '
                #            f'bpp1 = {np.average(BPP1):.5f}, bpp2 = {np.average(BPP2):.5f}\n')
                aPSNR.append(np.average(PSNR))
                aMSSSIM.append(np.average(MSSSIM))
                aBPP.append(np.average(BPP1))

            results = {
                "psnr": aPSNR, "bpp": aBPP, "msssim": aMSSSIM,
            }
            output = {
                "name": f'{codec}_{key}',
                "results": results,
            }
            with open(os.path.join(os.path.join(tgtp, f'{codec}_{key}.json')), 'w',
                      encoding='utf-8') as json_file:
                json.dump(output, json_file, indent=2)
        #         log1.flush()
        #     log1.write('\n')
        #     log.write('\n')
        #     log.flush()
        #     log1.flush()
        # log.close()
        # log1.close()
    return 0


if __name__ == "__main__":
    pass
    prepare_data_test()

    # data = TEST_DATA['UVG']
    # org_resolution = data['org_resolution']
    # frames = data['frames']
    # for seq in data['sequences']:
    #     print(seq)
    #     yuv = os.path.join('/tdx/LHB/data/ACM23/UVG', seq + '.yuv')
    #     img_path = os.path.join('/tdx/LHB/data/ACM23/UVG', 'PNG_Frames', seq)
    #     os.makedirs(img_path, exist_ok=True)
    #     os.system(
    #         f'ffmpeg -y -pix_fmt yuv420p -s {org_resolution} -i {yuv} {img_path}/f%03d.png')
    #     # exit()

