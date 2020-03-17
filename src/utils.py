# coding=utf-8
# Copyleft 2019 Project LXRT

import sys
import csv
import base64
import time, ast, torch



import numpy as np
from tqdm import tqdm
from math import (
    degrees, radians,
    sin, cos, asin, tan, atan, atan2, pi,
    sqrt, exp, log, fabs, log10, pow
)

from constants import (
    EARTH_MEAN_RADIUS,
    EARTH_MEAN_DIAMETER,
    EARTH_EQUATORIAL_RADIUS,
    EARTH_EQUATORIAL_METERS_PER_DEGREE,
    I_EARTH_EQUATORIAL_METERS_PER_DEGREE,
    HALF_PI,
    QUARTER_PI,
)


csv.field_size_limit(sys.maxsize)
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]
FIELDITEMS = ["img_id", "img_h", "img_w", "num_boxes", "boxes",
              "features","names"]

# -122.44, 34.44 etc.
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r

def destination(point, distance, bearing):

    # print(point)
    lon1,lat1 = (radians(float(coord)) for coord in point)

    # print(distance," ",bearing)
    radians_bearing = radians(float(bearing))
    # print(radians_bearing)

    delta = float(distance) / EARTH_MEAN_RADIUS
    lat2 = asin(
        sin(lat1)*cos(delta) +
        cos(lat1)*sin(delta)*cos(radians_bearing)
    )
    numerator = sin(radians_bearing) * sin(delta) * cos(lat1)
    denominator = cos(delta) - sin(lat1) * sin(lat2)

    lon2 = lon1 + atan2(numerator, denominator)
    # print(type(lon2))
    lon2_deg = (degrees(lon2) + 540) % 360 - 180
    lat2_deg = degrees(lat2)

    return [lon2_deg,lat2_deg]

def getPointLatLng(x, y,centre_lon,centre_lat,_zoom,height,width):
    parallelMultiplier = cos(centre_lat * pi / 180)
    degreesPerPixelX = 360 / pow(2, _zoom + 8)
    degreesPerPixelY = 360 / pow(2, _zoom + 8) * parallelMultiplier
    pointLat = centre_lat - degreesPerPixelY * ( y - height / 2)
    pointLng = centre_lon + degreesPerPixelX * ( x  - width / 2)

    return (pointLat, pointLng)
# Calculating tile needed for converting GPS to pixels
# expects latlon like [37.0000, -122.2222]
def calculateTiles(latlon,zoom):

    lon_rad = radians(latlon[1]);
    lat_rad = radians(latlon[0]);
    n = pow(2.0, zoom);

    tileX = ((latlon[1] + 180) / 360) * n;
    tileY = (1 - (log(tan(lat_rad) + 1.0/cos(lat_rad)) / pi)) * n / 2.0;
    # print(" X: {}, Y: {}".format(tileX,tileY))
    return [tileX,tileY]


def load_obj_tsv(fname, topk=None):
    """Load object features from tsv file.

    :param fname: The path to the tsv file.
    :param topk: Only load features for top K images (lines) in the tsv file.
        Will load all the features if topk is either -1 or None.
    :return: A list of image object features where each feature is a dict.
        See FILENAMES above for the keys in the feature dict.
    """
    data = []
    start_time = time.time()
    print("Start to load Faster-RCNN detected objects from %s" % fname)
    with open(fname) as f:
        reader = csv.DictReader(f, FIELDNAMES, delimiter="\t")
        for i, item in enumerate(reader):

            for key in ['img_h', 'img_w', 'num_boxes']:
                item[key] = int(item[key])

            boxes = item['num_boxes']
            decode_config = [
                ('objects_id', (boxes, ), np.int64),
                ('objects_conf', (boxes, ), np.float32),
                ('attrs_id', (boxes, ), np.int64),
                ('attrs_conf', (boxes, ), np.float32),
                ('boxes', (boxes, 4), np.float32),
                ('features', (boxes, -1), np.float32),
            ]
            for key, shape, dtype in decode_config:
                item[key] = np.frombuffer(base64.b64decode(item[key]), dtype=dtype)
                item[key] = item[key].reshape(shape)
                item[key].setflags(write=False)

            data.append(item)
            if topk is not None and len(data) == topk:
                break
    elapsed_time = time.time() - start_time
    print("Loaded %d images in file %s in %d seconds." % (len(data), fname, elapsed_time))
    return data

def gen_chunks(reader, chunksize=100):
    """
    Chunk generator. Take a CSV `reader` and yield
    `chunksize` sized slices.
    """
    chunk = []
    for index, line in enumerate(tqdm(reader)):
        if (index % chunksize == 0 and index > 0):
            yield chunk
            del chunk[:]
        chunk.append(line)
    yield chunk

def load_det_obj_tsv(fname, topk=None):
    print(topk)
    """Load object features from tsv file.

    :param fname: The path to the tsv file.
    :param topk: Only load features for top K images (lines) in the tsv file.
        Will load all the features if topk is either -1 or None.
    :return: A list of image object features where each feature is a dict.
        See FILENAMES above for the keys in the feature dict.
    """
    data = []
    start_time = time.time()
    print("Start to load Faster-RCNN detected objects from %s" % fname)
    with open(fname, 'r') as f:
        reader = csv.DictReader(f, FIELDITEMS, delimiter="\t")
        if topk:
            chunk = topk
        else:
            chunk = 1000
        for it in gen_chunks(reader,  chunksize=chunk):
            # print(len(item[0]))
            # input(len(item))
            for i, item in enumerate(it):
                for key in ['img_h', 'img_w', 'num_boxes']:
                    item[key] = int(item[key])

                boxes = item['num_boxes']
                decode_config = [
                    ('boxes', (boxes, 4), np.float64),
                    ('features', (boxes, -1), np.float64),
                    ('names', (boxes, -1), np.dtype('<U100'))
                ]
                for key, shape, dtype in decode_config:
                    # print(key)
                    # print(item[key])
                    try:
                        item[key] = np.frombuffer(base64.b64decode(ast.literal_eval(item[key])), dtype=dtype)
                        item[key] = item[key].reshape(shape)
                        item[key].setflags(write=False)
                    except:
                        print(item[key])
                        input(key)

                data.append(item)
                if topk is not None and len(data) >= topk:
                    break
    elapsed_time = time.time() - start_time
    print("Loaded %d images in file %s in %d seconds." % (len(data), fname, elapsed_time))
    return data


def giou_loss(output, target, bbox_inside_weights=None, bbox_outside_weights=None,
                transform_weights=None, batch_size=None):

    # if transform_weights is None:
    #     transform_weights = (1., 1., 1., 1.)

    if batch_size is None:
        batch_size = output.size(0)


    x1, y1, x2, y2 = output.t()[:,:]
    x1g, y1g, x2g, y2g = target.t()[:,:]

    x2 = torch.max(x1, x2)
    y2 = torch.max(y1, y2)

    xkis1 = torch.max(x1, x1g)
    ykis1 = torch.max(y1, y1g)
    xkis2 = torch.min(x2, x2g)
    ykis2 = torch.min(y2, y2g)

    xc1 = torch.min(x1, x1g)
    yc1 = torch.min(y1, y1g)
    xc2 = torch.max(x2, x2g)
    yc2 = torch.max(y2, y2g)

    intsctk = torch.zeros(x1.size()).to(output)
    mask = (ykis2 > ykis1) * (xkis2 > xkis1)
    # print(mask)
    intsctk[mask] = (xkis2[mask] - xkis1[mask]) * (ykis2[mask] - ykis1[mask])
    unionk = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g) - intsctk + 1e-7
    iouk = intsctk / unionk

    area_c = (xc2 - xc1) * (yc2 - yc1) + 1e-7
    giouk = iouk - ((area_c - unionk) / area_c)
    # iou_weights = bbox_inside_weights.view(-1, 4).mean(1) * bbox_outside_weights.view(-1, 4).mean(1)
    iouk = ((1 - iouk)).mean(0) / output.size(0)
    giouk = ((1 - giouk)).mean(0) / output.size(0)

    return iouk, giouk


def iou_loss(output, target, reduction = 'mean'):
    # input(output)
    # input(output.shape)
    x1_t, y1_t, x2_t, y2_t = target.t()[:,:]
    x1_p, y1_p, x2_p, y2_p = output.t()[:,:]
    # print(x2_t)
    # print(x1_p)
    # print(torch.unique(x2_t < x1_p))
    if (x2_t < x1_p or x2_p < x1_t or y2_t < y1_p or y2_p < y1_t or x1_p > x2_p or y1_p > y2_p):
        # input(x2_t < x1_p)
        return None
    # make sure x2_p and y2_p are larger
    x2_p = torch.max(x1_p, x2_p)
    y2_p = torch.max(y1_p, y2_p)

    far_x = torch.min(x2_t, x2_p)
    near_x = torch.max(x1_t, x1_p)
    far_y = torch.min(y2_t, y2_p)
    near_y = torch.max(y1_t, y1_p)

    inter_area = (far_x - near_x + 1) * (far_y - near_y + 1)
    true_box_area = (x2_t - x1_t + 1) * (y2_t - y1_t + 1)
    pred_box_area = (x2_p - x1_p + 1) * (y2_p - y1_p + 1)
    iou = inter_area / (true_box_area + pred_box_area - inter_area)
    # input(torch.mean(iou))
    if reduction != 'none':
            ret = torch.mean(iou) if reduction == 'mean' else torch.sum(iou)
    return 1 - ret
    # return loss

def calc_iou_individual(pred_box, gt_box):
    """Calculate IoU of single predicted and ground truth box
    Args:
        pred_box (list of floats): location of predicted object as
            [xmin, ymin, xmax, ymax]
        gt_box (list of floats): location of ground truth object as
            [xmin, ymin, xmax, ymax]
    Returns:
        float: value of the IoU for the two boxes.
    Raises:
        AssertionError: if the box is obviously malformed
    """


    x1_t, y1_t, x2_t, y2_t = gt_box
    x1_p, y1_p, x2_p, y2_p = pred_box
    #
    # if (x1_p > x2_p) or (y1_p > y2_p):
    #     raise AssertionError(
    #         "Prediction box is malformed? pred box: {}".format(pred_box))
    # if (x1_t > x2_t) or (y1_t > y2_t):
    #     raise AssertionError(
    #         "Ground Truth box is malformed? true box: {}".format(gt_box))

    if (x2_t < x1_p or x2_p < x1_t or y2_t < y1_p or y2_p < y1_t or x1_p > x2_p or y1_p > y2_p):
        return 0.0

    far_x = np.min([x2_t, x2_p])
    near_x = np.max([x1_t, x1_p])
    far_y = np.min([y2_t, y2_p])
    near_y = np.max([y1_t, y1_p])

    inter_area = (far_x - near_x + 1) * (far_y - near_y + 1)
    true_box_area = (x2_t - x1_t + 1) * (y2_t - y1_t + 1)
    pred_box_area = (x2_p - x1_p + 1) * (y2_p - y1_p + 1)
    iou = inter_area / (true_box_area + pred_box_area - inter_area)
    return iou
