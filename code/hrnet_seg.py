import os
import cv2
import json
import numpy as np
from skimage import io, util

import argparse
from distutils.version import LooseVersion
from multiprocessing import Queue, Process
# Numerical libs
import numpy as np
import math
import torch
import torch.nn as nn
from scipy.io import loadmat
# Our libs
from mit_semseg.config import cfg
from mit_semseg.dataset import ValDataset
from mit_semseg.models import ModelBuilder, SegmentationModule
from mit_semseg.utils import AverageMeter, colorEncode, accuracy, intersectionAndUnion, parse_devices, setup_logger
from mit_semseg.lib.nn import user_scattered_collate, async_copy_to
from mit_semseg.lib.utils import as_numpy
from PIL import Image
from tqdm import tqdm

import warnings

warnings.filterwarnings('ignore')

def Patch2Img(patches,img_size,overlap=0.5):
    patches=np.squeeze(patches)
    patch_wid = patches[0].shape[1]
    patch_hei = patches[0].shape[0]
    num_class = 1
   
    vote=np.zeros((num_class, img_size[0], img_size[1]))
    patch_ranges=calculate_cut_range(img_size, patch_size=[patch_hei,patch_wid],overlap=overlap)
    for id in range(len(patches)):
            patch=patches[id]
            y_s=round(patch_ranges[id][0])
            y_e=round(patch_ranges[id][1])
            x_s=round(patch_ranges[id][2])
            x_e=round(patch_ranges[id][3])
            vote[:, y_s:y_e, x_s:x_e] = vote[:, y_s:y_e, x_s:x_e] + patch
    #pred = np.argmax(vote, axis = 0).astype('uint8')
    pred = vote[:,:]
    pred[np.where(vote>0)]=1
    return pred


def Img2Patch(img, patch_size,overlap_rati):
    patches=[]

    patch_range=calculate_cut_range(img.shape[0:2],patch_size,overlap_rati)
    for id in range(len(patch_range)):
        y_s=round(patch_range[id][0])
        y_e=round(patch_range[id][1])
        x_s=round(patch_range[id][2])
        x_e=round(patch_range[id][3])
        patch=img[y_s:y_e,x_s:x_e,:]
        patches.append(patch)
    return patches


def calculate_cut_range(img_size, patch_size,overlap,pad_edge=1):
    patch_range=[]
    patch_height = patch_size[0]
    patch_width = patch_size[1]
    width_overlap = patch_width * overlap
    height_overlap = patch_height *overlap
    cols=img_size[1]
    rows=img_size[0]
    x_e = 0
    while (x_e < cols):
        y_e=0
        x_s = max(0, x_e - width_overlap)
        x_e = x_s + patch_width
        if (x_e > cols):
            x_e = cols
        if (pad_edge == 1): ## if the last path is not enough, then extent to the inerside.
            x_s = x_e - patch_width
        if (pad_edge == 2):## if the last patch is not enough, then extent to the outside(with black).
            x_s=x_s
        while (y_e < rows):
            y_s = max(0, y_e - height_overlap)
            y_e = y_s + patch_height
            if (y_e > rows):
                y_e = rows
            if (pad_edge == 1): ## if the last path is not enough, then extent to the inerside.
                y_s = y_e - patch_height
            if (pad_edge == 2):## if the last patch is not enough, then extent to the outside(with black).
                y_s=y_s
            patch_range.append([int(y_s),int(y_e),int(x_s),int(x_e)])
    return patch_range

def odgt(img_path):
    seg_path = img_path.replace('images','annotations')
    #seg_path = img_path.replace('.jpg','.png')
    
    if os.path.exists(seg_path):
        img = cv2.imread(img_path)
        h, w, _ = img.shape

        odgt_dic = {}
        odgt_dic["fpath_img"] = img_path
        odgt_dic["fpath_segm"] = seg_path
        odgt_dic["width"] = h
        odgt_dic["height"] = w
        return odgt_dic
    else:
        print('the corresponded annotation does not exist')
        print(img_path)
        return None

#colors = loadmat('data/color150.mat')['colors']
colors = np.array([[120,120,120],[255,255,255]]).astype(np.uint8)


def visualize_result(data, pred, dir_result):
    (img, seg, info) = data

    # segmentation
    #seg_color = colorEncode(seg, colors)

    # prediction
    pred_color = colorEncode(pred, colors)

    # aggregate images and save
    #im_vis = np.concatenate((img, seg_color, pred_color),
    #                        axis=1).astype(np.uint8)
    
    im_vis=np.array(pred_color).astype(np.uint8)

    img_name = info.split('/')[-1]
    Image.fromarray(im_vis).save(os.path.join(dir_result, img_name.replace('.jpg', '.png')))
    #Image.fromarray(im_vis).save(os.path.join(dir_result, img_name))


def evaluate(segmentation_module, loader, cfg, gpu_id, result_queue):
    segmentation_module.eval()
#    print(segmentation_module)

    for batch_data in loader:
        # process data
        batch_data = batch_data[0]
        seg_label = as_numpy(batch_data['seg_label'][0])
        img_resized_list = batch_data['img_data']

        with torch.no_grad():
            segSize = (seg_label.shape[0], seg_label.shape[1])
            scores = torch.zeros(1, cfg.DATASET.num_class, segSize[0], segSize[1])
            scores = async_copy_to(scores, gpu_id)

            for img in img_resized_list:
                feed_dict = batch_data.copy()
                feed_dict['img_data'] = img
                del feed_dict['img_ori']
                del feed_dict['info']
                feed_dict = async_copy_to(feed_dict, gpu_id)

                # forward pass
                scores_tmp = segmentation_module(feed_dict, segSize=segSize)
                scores = scores + scores_tmp / len(cfg.DATASET.imgSizes)

            _, pred = torch.max(scores, dim=1)
            pred = as_numpy(pred.squeeze(0).cpu())

        # calculate accuracy and SEND THEM TO MASTER
        acc, pix = accuracy(pred, seg_label)
        intersection, union = intersectionAndUnion(pred, seg_label, cfg.DATASET.num_class)
        result_queue.put_nowait((acc, pix, intersection, union))

        # visualization
        if cfg.VAL.visualize:
            visualize_result(
                (batch_data['img_ori'], seg_label, batch_data['info']),
                pred,
                os.path.join(cfg.DIR, 'result')
            )

def worker(cfg, gpu_id, start_idx, end_idx, result_queue):
    torch.cuda.set_device(gpu_id)

    # Dataset and Loader
    dataset_val = ValDataset(
        cfg.DATASET.root_dataset,
        cfg.DATASET.list_val,
        cfg.DATASET,
        start_idx=start_idx, end_idx=end_idx)
    loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=cfg.VAL.batch_size,
        shuffle=False,
        collate_fn=user_scattered_collate,
        num_workers=0)

    # Network Builders
    net_encoder = ModelBuilder.build_encoder(
        arch=cfg.MODEL.arch_encoder.lower(),
        fc_dim=cfg.MODEL.fc_dim,
        weights=cfg.MODEL.weights_encoder)
    net_decoder = ModelBuilder.build_decoder(
        arch=cfg.MODEL.arch_decoder.lower(),
        fc_dim=cfg.MODEL.fc_dim,
        num_class=cfg.DATASET.num_class,
        weights=cfg.MODEL.weights_decoder,
        use_softmax=True)

    crit = nn.NLLLoss(ignore_index=-1)

    segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)

    segmentation_module.cuda()

    # Main loop
    evaluate(segmentation_module, loader_val, cfg, gpu_id, result_queue)

def main(cfg, gpus):
    with open(cfg.DATASET.list_val, 'r') as f:
        lines = f.readlines()
        num_files = len(lines)

    num_files_per_gpu = math.ceil(num_files / len(gpus))

    pbar = tqdm(total=num_files)

    acc_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()

    result_queue = Queue(500)
    procs = []
    for idx, gpu_id in enumerate(gpus):
        start_idx = idx * num_files_per_gpu
        end_idx = min(start_idx + num_files_per_gpu, num_files)
        proc = Process(target=worker, args=(cfg, gpu_id, start_idx, end_idx, result_queue))
        print('gpu:{}, start_idx:{}, end_idx:{}'.format(gpu_id, start_idx, end_idx))
        proc.start()
        procs.append(proc)

    # master fetches results
    processed_counter = 0
    while processed_counter < num_files:
        if result_queue.empty():
            continue
        (acc, pix, intersection, union) = result_queue.get()
        acc_meter.update(acc, pix)
        intersection_meter.update(intersection)
        union_meter.update(union)
        processed_counter += 1
        pbar.update(1)

    for p in procs:
        p.join()

    # summary
    iou = intersection_meter.sum / (union_meter.sum + 1e-10)
    #for i, _iou in enumerate(iou):
        #print('class [{}], IoU: {:.4f}'.format(i, _iou))

    #print('[Eval Summary]:')
    #print('Mean IoU: {:.4f}, Accuracy: {:.2f}%'
    #      .format(iou.mean(), acc_meter.average()*100))

    #print('Evaluation Done!')

#Segmentation main function
def HRNet_building(img_ortho,out_name):
    img_ort = img_ortho
    img_ort = util.img_as_ubyte(img_ort)

    file_folder = out_name

    dir_folder='./source/hrnet'

    image_patches = Img2Patch(img_ort, patch_size=(512, 512), overlap_rati=0.5)

    patch_folder = os.path.join(file_folder, 'patch/')
    patch_ex = os.path.exists(patch_folder)
    if not patch_ex:
        os.makedirs(patch_folder)
    if not os.path.exists(os.path.join(patch_folder, 'images/')):
        os.makedirs(os.path.join(patch_folder, 'images/'))
    if not os.path.exists(os.path.join(patch_folder, 'annotations/')):
        os.makedirs(os.path.join(patch_folder, 'annotations/'))

    input_folder = os.path.join(patch_folder, 'images/')
    ex_list = os.listdir(input_folder)
    if len(ex_list) > 0:
        for imgf in ex_list:
            img_file = os.path.join(input_folder, imgf)
            if os.path.isfile(img_file):
                os.remove(img_file)
    input_folder_an = os.path.join(patch_folder, 'annotations/')
    ex_list = os.listdir(input_folder_an)
    if len(ex_list) > 0:
        for imgf in ex_list:
            img_file = os.path.join(input_folder_an, imgf)
            if os.path.isfile(img_file):
                os.remove(img_file)


    for i, patch in enumerate(image_patches):
        if i < 9:
            img_num = '00' + str(i + 1)
        elif i < 99 and i>=9:
            img_num = '0' + str(i + 1)
        else:
            img_num = str(i + 1)
        img_name = 'img_' + img_num + '.png'
        img_path = os.path.join(patch_folder, 'images', img_name)
        img_label = np.zeros((512, 512), dtype=int)
        label_path = os.path.join(patch_folder, 'annotations', img_name)
        io.imsave(img_path, patch)
        io.imsave(label_path, img_label)

    #input_folder = os.path.join(patch_folder, 'images/')
    output_foler = file_folder

    modes = []
    saves = ['testing.odgt']  # customized

    dir_path = input_folder
    img_list = os.listdir(dir_path)
    img_list.sort()
    img_list = [os.path.join(dir_path, img) for img in img_list]

    with open(os.path.join(dir_folder, saves[0]), mode='wt', encoding='utf-8') as myodgt:
        for i, img in enumerate(img_list):
            a_odgt = odgt(img)
            if a_odgt is not None:
                myodgt.write(f'{json.dumps(a_odgt)}\n')

    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'

    parser = argparse.ArgumentParser(
        description="PyTorch Semantic Segmentation Validation"
    )
    parser.add_argument(
        "--cfg",
        default="source/hrnet/ade20k-mobilenetv2-building.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        '--device', type=str,
        default='gpu',
        help='Device: gpu | cpu'
    )
    parser.add_argument(
        "--gpus",
        default="0",
        help="gpus to use, e.g. 0-3 or 0,1,2,3"
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    # cfg.freeze()

    logger = setup_logger(distributed_rank=0)  # TODO
    logger.info("Loaded configuration file {}".format(args.cfg))
    logger.info("Running with config:\n{}".format(cfg))

    # absolute paths of model weights
    cfg.MODEL.weights_encoder = os.path.join(
        cfg.DIR, 'encoder_' + cfg.VAL.checkpoint)
    cfg.MODEL.weights_decoder = os.path.join(
        cfg.DIR, 'decoder_' + cfg.VAL.checkpoint)
    assert os.path.exists(cfg.MODEL.weights_encoder) and \
           os.path.exists(cfg.MODEL.weights_decoder), "checkpoint does not exitst!"

    if not os.path.isdir(os.path.join(cfg.DIR, "result")):
        os.makedirs(os.path.join(cfg.DIR, "result"))

    result_path = os.path.join(cfg.DIR, 'result/')
    ex_list = os.listdir(result_path)
    if len(ex_list)>0:
        for imgf in ex_list:
            img_file = os.path.join(result_path,imgf)
            if os.path.isfile(img_file):
                os.remove(img_file)



    # Parse gpu ids
    gpus = parse_devices(args.gpus)
    gpus = [x.replace('gpu', '') for x in gpus]
    gpus = [int(x) for x in gpus]

    main(cfg, gpus)

    # merge predicted result
    pred_list = os.listdir(os.path.join(cfg.DIR, 'result/'))
    pred_batch = []
    for i, pred_patch in enumerate(pred_list):
        pred = io.imread(os.path.join(cfg.DIR, 'result', pred_patch))
        pred_t = (pred[:, :, 0] / 255).astype(np.uint8)
        pred_batch.append(pred_t)

    prediction = Patch2Img(pred_batch, img_ort.shape, overlap=0.5)
    io.imsave(os.path.join(file_folder, 'class.png'), prediction[0])

    return prediction[0]

