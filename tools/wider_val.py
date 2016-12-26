#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
import matlab.engine


CLASSES = ('__background__',
           'face')

NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel')}

color = (255,255,0)
linewidth = 2

def vis_detections(im, class_name, dets, wf_result, thresh=0.5):
    """Draw detected bounding boxes."""

    inds = np.where(dets[:, -1] >= thresh)[0]
    wf_result.write(str(len(inds)) + '\r\n')

    if len(inds) == 0:
        return


    for i in inds:
        score = dets[i, -1]

        x_min = int(dets[i,0])
        y_min = int(dets[i,1])
        x_max = int(dets[i,2])
        y_max = int(dets[i,3])

        info = '%d %d %d %d %f\n' % (x_min, y_min, x_max-x_min, y_max-y_min, score)
        wf_result.write(info)





def demo(net, img_folder, image_name, output):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im = cv2.imread(img_folder + image_name)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    wf_result = open(output + os.path.dirname(image_name) + '/' + os.path.basename(image_name)[0: -4] + '.txt', 'wb')
    wf_result.write(image_name + '\r\n')
    # Visualize detections for each class
    CONF_THRESH = 1e-5
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detections(im, cls, dets, wf_result, thresh=CONF_THRESH)

    wf_result.close()



def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--maxIter', dest='max_iter', help='GPU device id to use [0]',
                        default=960000, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16')

    args = parser.parse_args()

    return args




def frame_sorted(im_names):
    frame_list = []
    ret_list = []
    for img in im_names:
        frame_list.append(int(img.split('.')[0]))

    frame_list = sorted(frame_list)

    for i in range(len(frame_list)):
        ret_list.append(str(frame_list[i]) + '.jpg')

    return ret_list, frame_list[-1]



if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    prototxt = '/home/guizi/project/face/py-R-FCN/models/wider/flex1/faster_rcnn_end2end/test.prototxt'

    step_size = 10000
    iters = args.max_iter / step_size
    print iters
    #caffemodel_tmp = '/media/datavolume/guizi/model/face/flex_guizi/faster_rcnn_end2end/wider_train/flex_guizi_iter_{0}.caffemodel'
    caffemodel_tmp = '/home/guizi/project/face/py-R-FCN/output/faster_rcnn_end2end/wider_train/flex_guizi_iter_{0}.caffemodel'
    #img_folder = '/media/datavolume/guizi/data/face/WIDER/WIDER_val/images/'
    img_folder = '/media/flex/d/gaowei/Projects/face-detection/DATASET/WIDER/WIDER_val/images/'
    output_main = '/home/guizi/project/face/wider_val/flex_guizi_rpn4_noOHEM_ave/'
    matlab_tool_folder = '/home/guizi/project/face/eval_tools/'
    if not os.path.exists(output_main):
        os.makedirs(output_main)
    ap_f = output_main + 'ap.txt'
    ap_w = open(ap_f, 'wb')

    print 'load matlab engine'
    os.chdir(matlab_tool_folder)
    eng = matlab.engine.start_matlab()

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id

    for index in range(iters):
        iter_str = str((index + 1) * step_size)
        caffemodel = caffemodel_tmp.format(iter_str)
        net = caffe.Net(prototxt, caffemodel, caffe.TEST)
        output = output_main + iter_str + '/'

        print '\n\nLoaded network {:s}'.format(caffemodel)

        # Warmup on a dummy image
        im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
        for i in xrange(2):
            _, _= im_detect(net, im)

        index = 0

        each_folders = os.listdir(img_folder)
        for each_folder in each_folders:

            if not os.path.exists(output + each_folder):
                os.makedirs(output + each_folder)

            im_names = os.listdir(img_folder + each_folder)

            for im_name in im_names:
                index += 1
                img_forward = each_folder + '/' + im_name
                print '{0} Processing image {1}/3226: {2}'.format(iter_str, index, img_forward)
                demo(net, img_folder, img_forward, output)
        print output
        ap_list = eng.winder_eval_f(output, iter_str)
        print ap_list
        ap_w.write(iter_str + '\r\n')
        for ap in ap_list:
            ap_w.write(str(ap[0]) + '\t' )
        ap_w.write('\n')

    ap_w.close()



