# -*- coding: utf-8 -*-
import cv2


def draw_keypoints(image, keypoints, radius=1, alpha=1.0):
    overlay = image.copy()
    for kp in keypoints:
        for x, y, v in kp:
            if int(v):
                cv2.circle(overlay, (int(x), int(y)), radius, (0, 255, 0), -1, cv2.LINE_AA)
    return cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0)


def draw_body_connections(image, keypoints, thickness=1, alpha=1.0):
    overlay = image.copy()
    b_conn = [(0, 1), (1, 2), (1, 5), (2, 8), (5, 11), (8, 11)]
    h_conn = [(0, 14), (0, 15), (14, 16), (15, 17)]
    l_conn = [(5, 6), (6, 7), (11, 12), (12, 13)]
    r_conn = [(2, 3), (3, 4), (8, 9), (9, 10)]
    for kp in keypoints:
        for i, j in b_conn:
            overlay = _draw_connection(overlay, kp[i], kp[j], (0, 255, 255), thickness)
        for i, j in h_conn:
            overlay = _draw_connection(overlay, kp[i], kp[j], (0, 255, 255), thickness)
        for i, j in l_conn:
            overlay = _draw_connection(overlay, kp[i], kp[j], (255, 255, 0), thickness)
        for i, j in r_conn:
            overlay = _draw_connection(overlay, kp[i], kp[j], (255, 0, 255), thickness)
    return cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0)


def draw_face_connections():
    raise NotImplementedError


def draw_hand_connections():
    raise NotImplementedError


def _draw_connection(image, point1, point2, color, thickness=1):
    x1, y1, v1 = point1
    x2, y2, v2 = point2
    if v1 and v2:
        cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness, cv2.LINE_AA)
    return image
