from __future__ import print_function
from __future__ import division
from future import standard_library
standard_library.install_aliases()
from builtins import str
from builtins import chr
from builtins import range
from past.utils import old_div
import os
import sys
import math
import random
import cv2
import threading
import numpy as np
import traceback
import logging
import time
import timeit
import datetime
import csv
import gc
from multiprocessing import Queue, Process
from ctypes import *
from random import randint
from os.path import splitext, basename, isdir
from os import makedirs
from memory_profiler import profile 

from glob import glob
from src.label import dknet_label_conversion
from src.label import Label, lwrite
from src.utils import nms
from src.utils import crop_region, image_files_from_folder
from src.utils import im2single
from src.keras_utils import load_model, detect_lp
from src.label import Shape, writeShapes
import argparse
from src.utils 				import crop_region, image_files_from_folder

import os
import sys
import math

@profile
def exit_gate():
    def sample(probs):
        s = sum(probs)
        probs = [old_div(a,s) for a in probs]
        r = random.uniform(0, 1)
        for i in range(len(probs)):
            r = r - probs[i]
            if r <= 0:
                return i
        return len(probs)-1

    def c_array(ctype, values):
        arr = (ctype*len(values))()
        arr[:] = values
        return arr

    class BOX(Structure):
        _fields_ = [("x", c_float),
                    ("y", c_float),
                    ("w", c_float),
                    ("h", c_float)]

    class DETECTION(Structure):
        _fields_ = [("bbox", BOX),
                    ("classes", c_int),
                    ("prob", POINTER(c_float)),
                    ("mask", POINTER(c_float)),
                    ("objectness", c_float),
                    ("sort_class", c_int)]

    class IMAGE(Structure):
        _fields_ = [("w", c_int),
                    ("h", c_int),
                    ("c", c_int),
                    ("data", POINTER(c_float))]

    class METADATA(Structure):
        _fields_ = [("classes", c_int),
                    ("names", POINTER(c_char_p))]

    class IplROI(Structure):
        pass

    class IplTileInfo(Structure):
        pass

    class IplImage(Structure):
        pass

    IplImage._fields_ = [
        ('nSize', c_int),
        ('ID', c_int),
        ('nChannels', c_int),
        ('alphaChannel', c_int),
        ('depth', c_int),
        ('colorModel', c_char * 4),
        ('channelSeq', c_char * 4),
        ('dataOrder', c_int),
        ('origin', c_int),
        ('align', c_int),
        ('width', c_int),
        ('height', c_int),
        ('roi', POINTER(IplROI)),
        ('maskROI', POINTER(IplImage)),
        ('imageId', c_void_p),
        ('tileInfo', POINTER(IplTileInfo)),
        ('imageSize', c_int),
        ('imageData', c_char_p),
        ('widthStep', c_int),
        ('BorderMode', c_int * 4),
        ('BorderConst', c_int * 4),
        ('imageDataOrigin', c_char_p)]

    class iplimage_t(Structure):
        _fields_ = [('ob_refcnt', c_ssize_t),
                    ('ob_type',  py_object),
                    ('a', POINTER(IplImage)),
                    ('data', py_object),
                    ('offset', c_size_t)]

    lib = CDLL("./darknet/libdarknet.so", RTLD_GLOBAL)
    lib.network_width.argtypes = [c_void_p]
    lib.network_width.restype = c_int
    lib.network_height.argtypes = [c_void_p]
    lib.network_height.restype = c_int

    predict = lib.network_predict
    predict.argtypes = [c_void_p, POINTER(c_float)]
    predict.restype = POINTER(c_float)

    set_gpu = lib.cuda_set_device
    set_gpu.argtypes = [c_int]

    make_image = lib.make_image
    make_image.argtypes = [c_int, c_int, c_int]
    make_image.restype = IMAGE

    get_network_boxes = lib.get_network_boxes
    get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float,
                                  c_float, POINTER(c_int), c_int, POINTER(c_int)]
    get_network_boxes.restype = POINTER(DETECTION)

    make_network_boxes = lib.make_network_boxes
    make_network_boxes.argtypes = [c_void_p]
    make_network_boxes.restype = POINTER(DETECTION)

    free_detections = lib.free_detections
    free_detections.argtypes = [POINTER(DETECTION), c_int]

    free_ptrs = lib.free_ptrs
    free_ptrs.argtypes = [POINTER(c_void_p), c_int]

    network_predict = lib.network_predict
    network_predict.argtypes = [c_void_p, POINTER(c_float)]

    reset_rnn = lib.reset_rnn
    reset_rnn.argtypes = [c_void_p]

    load_net = lib.load_network
    load_net.argtypes = [c_char_p, c_char_p, c_int]
    load_net.restype = c_void_p

    do_nms_obj = lib.do_nms_obj
    do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

    do_nms_sort = lib.do_nms_sort
    do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

    free_image = lib.free_image
    free_image.argtypes = [IMAGE]

    letterbox_image = lib.letterbox_image
    letterbox_image.argtypes = [IMAGE, c_int, c_int]
    letterbox_image.restype = IMAGE

    load_meta = lib.get_metadata
    lib.get_metadata.argtypes = [c_char_p]
    lib.get_metadata.restype = METADATA

    load_image = lib.load_image_color
    load_image.argtypes = [c_char_p, c_int, c_int]
    load_image.restype = IMAGE

    rgbgr_image = lib.rgbgr_image
    rgbgr_image.argtypes = [IMAGE]

    predict_image = lib.network_predict_image
    predict_image.argtypes = [c_void_p, IMAGE]
    predict_image.restype = POINTER(c_float)

    def classify(net, meta, im):
        out = predict_image(net, im)
        res = []
        for i in range(meta.classes):
            res.append((meta.names[i], out[i]))
        res = sorted(res, key=lambda x: -x[1])
        return res

    def array_to_image(arr):
        # need to return old values to avoid python freeing memory
        arr = arr.transpose(2, 0, 1)
        c, h, w = arr.shape[0:3]
        arr = np.ascontiguousarray(arr.flat, dtype=np.float32) / 255.0
        #print(arr)
        data = arr.ctypes.data_as(POINTER(c_float))
        im = IMAGE(w, h, c, data)
        return im, arr

    def detect(net, meta, image,thresh,hier_thresh=.5, nms=.45):

        im, image = array_to_image(image)
        rgbgr_image(im)
        num = c_int(0)
        pnum = pointer(num)
        predict_image(net, im)
        dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
        num = pnum[0]
        if nms:
            do_nms_obj(dets, num, meta.classes, nms)

        res = []
        for j in range(num):
            a = dets[j].prob[0:meta.classes]
            if any(a):
                ai = np.array(a).nonzero()[0]
                for i in ai:
                    b = dets[j].bbox
                    res.append((meta.names[i], dets[j].prob[i],
                               (b.x, b.y, b.w, b.h)))

        res = sorted(res, key=lambda x: -x[1])
        wh = (im.w, im.h)
        if isinstance(image, bytes):
            free_image(im)
        free_detections(dets, num)
        return res
    def cdetect(net, meta, image,thresh=0.3,hier_thresh=.7, nms=.45):

        im, image = array_to_image(image)
        rgbgr_image(im)
        num = c_int(0)
        pnum = pointer(num)
        predict_image(net, im)
        dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
        num = pnum[0]
        if nms:
            do_nms_obj(dets, num, meta.classes, nms)

        res = []
        for j in range(num):
            a = dets[j].prob[0:meta.classes]
            if any(a):
                ai = np.array(a).nonzero()[0]
                for i in ai:
                    b = dets[j].bbox
                    res.append((meta.names[i], dets[j].prob[i],
                               (b.x, b.y, b.w, b.h)))

        res = sorted(res, key=lambda x: -x[1])
        wh = (im.w, im.h)
        if isinstance(image, bytes):
            free_image(im)
        free_detections(dets, num)
        return res


    def adjust_pts(pts, lroi):
        return pts*lroi.wh().reshape((2, 1)) + lroi.tl().reshape((2, 1))


    def text_extract(img_path, height, width):
        '''gray = cv2.cvtColor(img_path, cv2.COLOR_BGR2GRAY)

        gray = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        inv = 255 - gray
        horizontal_img = inv
        vertical_img = inv

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 1))
        horizontal_img = cv2.erode(horizontal_img, kernel, iterations=1)
        horizontal_img = cv2.dilate(horizontal_img, kernel, iterations=1)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 100))
        vertical_img = cv2.erode(vertical_img, kernel, iterations=1)
        vertical_img = cv2.dilate(vertical_img, kernel, iterations=1)

        mask_img = horizontal_img + vertical_img

        no_border = np.bitwise_or(gray, mask_img)
        os.system("convert " + " no.jpg" + " -bordercolor " +
                  " White" + " -border" + " 10x10" + " with_border.jpg")
        imagez = cv2.imread("with_border.jpg")
        p1 = (20, 30)
        p2 = (60, 80)
        p3 = (60, 15)
        p4 = (200, 45)
        (x1, y1) = (20, 30)
        (x2, y2) = (60, 80)
        (x3, y3) = (60, 15)
        (x4, y4) = (200, 45)

        color = (255, 0, 0)
        thickness = 2
        image1 = cv2.rectangle(imagez, p1, p2, color, thickness)
        image2 = cv2.rectangle(imagez, p3, p4, color, thickness)

        roi1 = image1[y1:y2, x1:x2]

        roi2 = image2[y3:y4, x3:x4]'''

        ocr_threshold = .4
        R = detect(
            ocr_net, ocr_meta, img_path, thresh=ocr_threshold, nms=None)
        if len(R):
            L = dknet_label_conversion(R, width, height)
            L = nms(L, .45)
            L.sort(key=lambda x: x.tl()[0])
            lp_str1 = ''.join([chr(l.cl()) for l in L])
        print("License plate ----",lp_str1)

    def lp_detector(Ivehicle, wpod_net):

        lp_threshold = .2
        ratio = float(max(Ivehicle.shape[:2]))/min(Ivehicle.shape[:2])
        side = int(ratio*288.)
        bound_dim = min(side + (side%(2**4)), 608)
        Llp, LlpImgs, _ = detect_lp(wpod_net, im2single(
            Ivehicle), bound_dim, 2**4, (240, 80), lp_threshold)
        if len(LlpImgs):
            Ilp = LlpImgs[0]
            Ilp = cv2.cvtColor(Ilp, cv2.COLOR_BGR2GRAY)
            Ilp = cv2.cvtColor(Ilp, cv2.COLOR_GRAY2BGR)
            cv2.imwrite("lp1.png", Ilp*255.)
        return len(LlpImgs)


    def car_detect(Iorig):
        Lcars=[]
        R = []
        vehicle_threshold = .5
        R = detect(vehicle_net, vehicle_meta, Iorig,thresh=vehicle_threshold)
	
        R = [r for r in R if r[0] in [b'car','bus']]
        
        #print('\t\t%d cars found' % len(R))
        if len(R):
            WH = np.array(Iorig.shape[1::-1],dtype=float)
            Lcars = []
            for i,r in enumerate(R):
                cx,cy,w,h = (old_div(np.array(r[2]),np.concatenate( (WH,WH) ))).tolist()
                tl = np.array([cx - w/2., cy - h/2.])
                br = np.array([cx + w/2., cy + h/2.])
                label = Label(0,tl,br)
                Icar = crop_region(Iorig,label)
                cv2.imwrite("crop1.png",Icar)
                Icar1 = cv2.imread("crop1.png")
                Lcars.append(label)
        return Lcars

    ocr_weights = 'data/exit-ocr/ocr-net.weights'
    ocr_netcfg = 'data/exit-ocr/ocr-net.cfg'
    ocr_dataset = 'data/exit-ocr/ocr-net.data'
    ocr_net = load_net(ocr_netcfg.encode('utf-8'), ocr_weights.encode('utf-8'), 0)
    ocr_meta = load_meta(ocr_dataset.encode('utf-8'))
    wpod_net = load_model("data/exit-lp-detector/wpod-net_update1.h5")
    vehicle_threshold = .5
    vehicle_weights = 'data/exit-vehicle-detector/yolov3-tiny.weights'
    vehicle_netcfg = 'data/exit-vehicle-detector/yolov3-tiny.cfg'
    vehicle_dataset = 'data/exit-vehicle-detector/coco.data'
    vehicle_net = load_net(vehicle_netcfg.encode('utf-8'), vehicle_weights.encode('utf-8'), 0)
    vehicle_meta = load_meta(vehicle_dataset.encode('utf-8'))
    countd = 1
    img = cv2.imread("ar1.jpg")

    start_veh_det = time.time()
    lab=car_detect(img)
    end_veh_det = time.time()
    veh_det_time = end_veh_det - start_veh_det
    print("veh-det time(exit) :- ", veh_det_time)
    if(len(lab)!=0):
        for i in range(len(lab)):
            # LP detection
            start_lp_det = time.time()
            crop1=cv2.imread("crop1.png")
            cv2.imshow("Car 1", crop1)
            lp = lp_detector(crop1, wpod_net)
            end_lp_det = time.time()
            end_lp_det = time.time()
            lp_det_time = end_lp_det - start_lp_det
            print("lp-det time(exit) :- ",lp_det_time)
            lp = cv2.imread("lp1.png")
            cv2.imshow("LP 1", lp)
            height = lp.shape[0]
            width = lp.shape[1]

            # OCR
            start_ocr = time.time()
            text = text_extract(lp, height, width)
            end_ocr = time.time()
            ocr_time = end_ocr - start_ocr
            print("ocr time (exit) :- ",ocr_time)
            cv2.waitKey(10)


if __name__ == '__main__':
    exit_gate()
