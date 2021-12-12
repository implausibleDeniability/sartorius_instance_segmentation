import numpy as np
from shapely.errors import TopologicalError
from shapely.geometry import Polygon, MultiPoint
from skimage import measure

"""
    Author: https://github.com/Ulquiorracifa/Ocr2/blob/6bff9aed6493e97bbc0f3e50ce77ed2fa0cc1a0c/fp/TextBoxes/nms.py
    Author: https://github.com/qjadud1994/Text_Detector/blob/0144a8edb1812bcca6189d043e4865337d93a111/Pytorch/nms_poly.py
"""


def polygon_to_array(polygon: Polygon) -> np.array:
    return np.asarray(polygon.exterior.coords)


def polygon_iou(polygon1: Polygon, polygon2: Polygon):
    polygon_pts1 = polygon_to_array(polygon1)
    polygon_pts2 = polygon_to_array(polygon2)

    union_poly = np.concatenate((polygon_pts1, polygon_pts2))
    if not polygon1.intersects(polygon2):  # this test is fast and can accelerate calculation
        iou = 0
    else:
        try:
            inter_area = polygon1.intersection(polygon2).area

            union_area = MultiPoint(union_poly).convex_hull.area
            if union_area == 0:
                return 1

            iou = float(inter_area) / union_area

        except TopologicalError:
            print('shapely.geos.TopologicalError occurred, iou set to 0')
            iou = 0
    return iou


def polygons_nms(polygons: np.array, scores: np.array, iou_threshold: float) -> np.array:
    """ Apply nms to polygons, returns flags, which polygons to leave"""

    indices = sorted(range(len(scores)), key=lambda k: -scores[k])
    box_num = len(polygons)
    nms_flag = np.asarray([True] * box_num)

    for i in range(box_num):
        ii = indices[i]
        if not nms_flag[ii]:
            continue

        for j in range(box_num):
            jj = indices[j]

            if j == i or not nms_flag[jj]:
                continue

            polygon1, polygon2 = polygons[ii], polygons[jj]
            score1, score2 = scores[ii], scores[jj]

            iou = polygon_iou(polygon1, polygon2)

            if iou > iou_threshold:
                if score1 > score2:
                    nms_flag[jj] = False
                if score1 == score2 and polygon1.area > polygon2.area:
                    nms_flag[jj] = False
                if score1 == score2 and polygon1.area <= polygon2.area:
                    nms_flag[ii] = False
                    break

    return nms_flag


def mask_suppression(masks: np.array, scores: np.array, nms_threshold: float):
    assert len(masks.shape) == 3, "Masks should be size [n, h, w], not [n, 1, h, w]"

    polygons = list(map(lambda mask: Polygon(measure.find_contours(image=mask)[0]), masks))
    polygons = np.asarray(polygons)
    indices = polygons_nms(polygons, scores, iou_threshold=nms_threshold)
    return masks[indices]
