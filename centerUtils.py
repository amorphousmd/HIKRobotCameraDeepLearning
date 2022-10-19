from mmcv import Config
from mmdet.apis.inference import inference_detector
import mmcv
from mmcv.runner import load_checkpoint
from mmdet.apis import inference_detector, show_result_pyplot
from mmdet.models import build_detector
from mmdetection.mmdet.apis.inference import show_result_ins
import numpy as np
import pycocotools.mask as maskUtils
import cv2
import time

class GenericMask:
    """
    Attribute:
        polygons (list[ndarray]): list[ndarray]: polygons for this mask.
            Each ndarray has format [x, y, x, y, ...]
        mask (ndarray): a binary mask
    """

    def __init__(self, mask_or_polygons, height, width):
        self._mask = self._polygons = self._has_holes = None
        self.height = height
        self.width = width

        m = mask_or_polygons

        if isinstance(m, list):  # list[ndarray]
            self._polygons = [np.asarray(x).reshape(-1) for x in m]
            return

        if isinstance(m, np.ndarray):  # assumed to be a binary mask
            assert m.shape[1] != 2, m.shape
            assert m.shape == (
                height,
                width,
            ), f"mask shape: {m.shape}, target dims: {height}, {width}"
            self._mask = m.astype("uint8")
            return

        raise ValueError("GenericMask cannot handle object {} of type '{}'".format(m, type(m)))

    @property
    def mask(self):
        if self._mask is None:
            self._mask = self.polygons_to_mask(self._polygons)
        return self._mask

    @property
    def polygons(self):
        if self._polygons is None:
            self._polygons, self._has_holes = self.mask_to_polygons(self._mask)
        return self._polygons

    @property
    def has_holes(self):
        if self._has_holes is None:
            if self._mask is not None:
                self._polygons, self._has_holes = self.mask_to_polygons(self._mask)
            else:
                self._has_holes = False  # if original format is polygon, does not have holes
        return self._has_holes

    def mask_to_polygons(self, mask):
        # cv2.RETR_CCOMP flag retrieves all the contours and arranges them to a 2-level
        # hierarchy. External contours (boundary) of the object are placed in hierarchy-1.
        # Internal contours (holes) are placed in hierarchy-2.
        # cv2.CHAIN_APPROX_NONE flag gets vertices of polygons from contours.
        mask = np.ascontiguousarray(mask)  # some versions of cv2 does not support incontiguous arr
        res = cv2.findContours(mask.astype("uint8"), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        hierarchy = res[-1]
        if hierarchy is None:  # empty mask
            return [], False
        has_holes = (hierarchy.reshape(-1, 4)[:, 3] >= 0).sum() > 0
        res = res[-2]
        res = [x.flatten() for x in res]
        # These coordinates from OpenCV are integers in range [0, W-1 or H-1].
        # We add 0.5 to turn them into real-value coordinate space. A better solution
        # would be to first +0.5 and then dilate the returned polygon by 0.5.
        res = [x + 0.5 for x in res if len(x) >= 6]
        return res, has_holes

    def area(self):
        return self.mask.sum()


def detect_center(img, result, score_thr):
    h, w = img.shape[0], img.shape[1]
    bbox_result, segm_result = result
    segms = mmcv.concat_list(segm_result)
    bboxes = np.vstack(bbox_result)
    inds = np.where(bboxes[:, -1] > score_thr)[0]
    center_list = []
    for i in inds:
        mask = GenericMask(segms[i], h, w)
        polygon = mask.polygons[0].reshape((-1, 1, 2))
        polygon = polygon.astype(np.int32)
        M = cv2.moments(polygon)
        x_center = int(M["m10"] / M["m00"])
        y_center = int(M["m01"] / M["m00"])
        center_list.append((x_center, y_center))
    return center_list


def detect_center_bbox(result, score_thr):
    center_list = []
    for arrays in result:
        for bboxes in arrays:
            if bboxes[4] < score_thr:
                pass
            else:
                tuple_list = []
                tuple_list.append(int((bboxes[0]+bboxes[2])/2))
                tuple_list.append(int((bboxes[1]+bboxes[3])/2))
                # tuple_list.append(int(bboxes[0]))
                # tuple_list.append(int(bboxes[1]))
                center_list.append(tuple(tuple_list))
    return center_list



if __name__ == "__main__":
    cfg = Config.fromfile('mmdetection/configs/solov2/solov2_light_r18_fpn_3x_coco.py')
    cfg.model.mask_head.num_classes = 1

    checkpoint = 'solov2.pth'
    model = build_detector(cfg.model)
    checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
    model.CLASSES = checkpoint['meta']['CLASSES']
    model.cfg = cfg
    model.to('cuda')
    model.eval()

    time_process_start = 0
    time_process_end = 0

    cap = cv2.VideoCapture(0)
    cap.set(3, 1080)
    cap.set(4, 640)
    while True:
        time_process_start = time.time()
        score_thr = 0.9
        _, img = cap.read()
        result = inference_detector(model, img)
        img_show = model.show_result(
            img,
            result,
            score_thr=score_thr,
            show=False,
            wait_time=0,
            win_name='result',
            bbox_color=None,
            text_color=(200, 200, 200),
            mask_color=None,
            out_file=None)
        center_list = detect_center(img, result, score_thr)
        for center in center_list:
            img_show = cv2.circle(img_show, center, 3, (0,0,255), -1)

        time_process_end = time.time() - time_process_start
        cv2.putText(img_show, 'time_process(s):'+str(np.round(time_process_end,3)), (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 233), 2)
        cv2.imshow('webcam', img_show)

        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
