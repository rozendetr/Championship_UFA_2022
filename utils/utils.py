import math
from rectpack import newPacker
import cv2
import numpy as np


def padding_detection(dets, max_size):
    dets_pad = []
    max_height, max_width = max_size
    for id_bbox, bbox in enumerate(dets):
        x1, y1, x2, y2 = bbox
        w_bbox = np.abs(x2 - x1)
        h_bbox = np.abs(y2 - y1)

        pad_x = int(w_bbox / 4)
        pad_y = int(h_bbox / 4)
        x1_pad = max(0, x1 - pad_x)
        x2_pad = min(max_width, x2 + pad_x)
        y1_pad = max(0, y1 - pad_y)
        y2_pad = min(max_height, y2 + pad_y)
        dets_pad.append([x1_pad, y1_pad, x2_pad, y2_pad])
    return dets_pad


def crop_img_andor_rotare(img, bbox, pack_bbox):
    w, h = bbox[2]-bbox[0], bbox[3]-bbox[1]
    pack_w, pack_h = pack_bbox[2]-pack_bbox[0], pack_bbox[3]-pack_bbox[1]
    # print(f"bbox: {w, h}, pack_bbox: {pack_w, pack_h}")
    if (w == pack_w) and (h == pack_h):
        return img[bbox[1]:bbox[3], bbox[0]:bbox[2], :], False
    if (w == pack_h) and (h == pack_w):
        crop_img = img[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
        return np.rot90(crop_img, k = -1), True
    return None, False


def pack_detection(img, bboxes, shape=(512, 512)):
    """
    bboxes = [[tlx,tly,brx,bry], ...]
    """
    wh_bboxes = [[bbox[2] - bbox[0], bbox[3] - bbox[1], id_bbox] for id_bbox, bbox in enumerate(bboxes)]
    dict_wh_bboxes = {id_bbox: bbox for id_bbox, bbox in enumerate(bboxes)}

    packer = newPacker()
    for wh_bbox in wh_bboxes:
        packer.add_rect(*wh_bbox)
    packer.add_bin(*shape)

    packer.pack()

    pack_img = np.zeros((shape[1], shape[0], 3), dtype=np.uint8)
    all_rects = packer.rect_list()
    coord_rect = []
    for rect in all_rects:
        b, x, y, w, h, rid = rect
        # print(rid, dict_wh_bboxes.get(rid), [x, y, x + w, y + h])
        pack_crop_img, rotare_img = crop_img_andor_rotare(img, dict_wh_bboxes.get(rid), [x, y, x + w, y + h])
        coord_rect.append([x, y, x + w, y + h, rid, rotare_img])
        # print(pack_img[y:y + h, x:x + w, :].shape, pack_crop_img.shape)
        pack_img[y:y + h, x:x + w, :] = pack_crop_img
    print("success packing all bbox", len(coord_rect) == len(bboxes))
    return pack_img, coord_rect


def unpack_detection(img, mask_img, pack_bboxes, bboxes):
    height, width = img.shape[:2]
    full_mask = np.zeros((height, width, len(pack_bboxes)), dtype=np.uint8)
    dict_bboxes = {id_bbox: bbox for id_bbox, bbox in enumerate(bboxes)}
    for pack_bbox in pack_bboxes:
        pack_xtl, pack_ytl, pack_xbr, pack_ybr, rid, rot = pack_bbox
        mask_crop = mask_img[pack_ytl:pack_ybr, pack_xtl:pack_xbr]
        if rot:
            mask_crop = np.rot90(mask_crop, k = 1)
        xtl, ytl, xbr, ybr = dict_bboxes.get(rid)
        full_mask[ytl:ybr, xtl:xbr, rid] = mask_crop
    return full_mask

