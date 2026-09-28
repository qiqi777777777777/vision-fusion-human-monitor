# -*- coding: utf-8 -*-
"""OpenVINO YOLOv8 推理封装：接口对齐 ultralytics YOLO，供 app.py 无痛替换。

用法:
    from ov_yolo import OVYolo
    model = OVYolo("yolov8n.onnx")          # 或 yolov8n_int8.xml
    results = model(imgs, conf=0.25, batch=6, imgsz=640, verbose=False)
    results[i].boxes.data  -> torch tensor [N,6] (x1,y1,x2,y2,conf,cls)

说明:
- 基于 D:\\CV\\zg6\\jk_openvino_test 已验证的 letterbox + postprocess 逻辑
- 模型输入固定为导出尺寸（本模型 640），imgsz 参数兼容保留但按模型实际尺寸推理
- 输出坐标已映射回原输入图像素坐标，与 ultralytics 语义一致
"""
import cv2
import numpy as np
import torch
from openvino import Core
from types import SimpleNamespace


def letterbox(img, new=640):
    """保持宽高比缩放 + 灰色填充，与 YOLO 官方预处理一致"""
    h, w = img.shape[:2]
    r = min(new / h, new / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh))
    canvas = np.full((new, new, 3), 114, dtype=np.uint8)
    top = (new - nh) // 2
    left = (new - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, r, top, left


def postprocess(output, r, top, left, conf, iou):
    """[1,84,8400] -> 检测框 [N,6] (x1,y1,x2,y2,score,cls)，坐标映射回原图"""
    preds = output[0].transpose(1, 0)
    boxes, scores, cls_ids = [], [], []
    for row in preds:
        cx, cy, w, h = row[:4]
        class_scores = row[4:]
        c = int(class_scores.argmax())
        sv = float(class_scores[c])
        if sv < conf:
            continue
        boxes.append([(cx - w / 2 - left) / r, (cy - h / 2 - top) / r,
                      (cx + w / 2 - left) / r, (cy + h / 2 - top) / r])
        scores.append(sv)
        cls_ids.append(c)
    if not boxes:
        return np.empty((0, 6), dtype=np.float32)
    idx = cv2.dnn.NMSBoxes(boxes, scores, conf, iou)
    if len(idx) == 0:
        return np.empty((0, 6), dtype=np.float32)
    idx = np.array(idx).ravel()
    return np.array(
        [[boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3], scores[i], cls_ids[i]]
         for i in idx], dtype=np.float32)


class _Result:
    """极简 Results 兼容对象，仅暴露 .boxes.data（torch tensor [N,6]）"""

    def __init__(self, data):
        self.boxes = SimpleNamespace(data=torch.from_numpy(data))


class OVYolo:
    def __init__(self, model_path, device="CPU", conf=0.25, iou=0.45):
        self.conf = conf
        self.iou = iou
        self.core = Core()
        self.model = self.core.read_model(model_path)
        self.compiled = self.core.compile_model(self.model, device)
        self.out_layer = self.compiled.output(0)
        # 模型固定输入尺寸（本模型为 640）
        self.input_size = 640
        try:
            in_shape = self.compiled.input(0).shape
            if len(in_shape) >= 3 and in_shape[2] > 0:
                self.input_size = int(in_shape[2])
        except Exception:
            pass

    def __call__(self, imgs, conf=None, imgsz=None, batch=6, verbose=False, **kwargs):
        """与 ultralytics 兼容的调用入口：imgs 为单张或列表，返回 Results 列表"""
        conf = conf if conf is not None else self.conf
        if isinstance(imgs, (list, tuple)):
            return [self._infer_one(img, conf) for img in imgs]
        return [self._infer_one(imgs, conf)]

    def _infer_one(self, img, conf):
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        size = self.input_size
        canvas, r, top, left = letterbox(img, size)
        blob = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        out = self.compiled([blob])[self.out_layer]
        dets = postprocess(out, r, top, left, conf, self.iou)
        return _Result(dets)


# ===================== YOLOv8-pose（关键点）推理 =====================
# 输入 [1,3,640,640]，输出 [1,56,8400]
# 每行结构: 4 box_xywh + 1 box_conf + 17*3 keypoint(x, y, conf)
# 关键点索引(COCO): 0鼻 1左眼 2右眼 3左耳 4右耳 5左肩 6右肩 7左肘 8右肘
#                   9左腕 10右腕 11左胯 12右胯 13左膝 14右膝 15左踝 16右踝


def postprocess_pose(output, r, top, left, conf, iou):
    """[1,56,8400] -> (dets[N,6], kpts[N,17,2], kconf[N,17])，坐标均映射回原图像素"""
    preds = output[0].transpose(1, 0)          # [8400, 56]
    boxes, scores, kpts_list, kconf_list = [], [], [], []
    for row in preds:
        box_conf = float(row[4])
        if box_conf < conf:
            continue
        cx, cy, bw, bh = row[:4]
        boxes.append([(cx - bw / 2 - left) / r, (cy - bh / 2 - top) / r,
                      (cx + bw / 2 - left) / r, (cy + bh / 2 - top) / r])
        scores.append(box_conf)
        kx = (row[5::3] - left) / r
        ky = (row[6::3] - top) / r
        kpts_list.append(np.stack([kx, ky], axis=1))        # [17,2]
        kconf_list.append(row[7::3])                        # [17]
    if not boxes:
        return np.empty((0, 6), dtype=np.float32), \
            np.empty((0, 17, 2), dtype=np.float32), np.empty((0, 17), dtype=np.float32)
    idx = cv2.dnn.NMSBoxes(boxes, scores, conf, iou)
    if len(idx) == 0:
        return np.empty((0, 6), dtype=np.float32), \
            np.empty((0, 17, 2), dtype=np.float32), np.empty((0, 17), dtype=np.float32)
    idx = np.array(idx).ravel()
    dets = np.array(
        [[boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3], scores[i], 0.0]
         for i in idx], dtype=np.float32)
    kpts = np.array([kpts_list[i] for i in idx], dtype=np.float32)
    kconf = np.array([kconf_list[i] for i in idx], dtype=np.float32)
    return dets, kpts, kconf


class _ResultPose:
    """面向关键点任务的轻量 Results 兼容对象：
    .boxes.data  [N,6] (x1,y1,x2,y2,conf,cls)
    .keypoints   [N,17,2] 原图像素坐标
    .kpts_conf   [N,17] 关键点置信度"""

    def __init__(self, data, kpts, kconf):
        self.boxes = SimpleNamespace(data=torch.from_numpy(data))
        self.keypoints = kpts
        self.kpts_conf = kconf


class OVYoloPose(OVYolo):
    """与 OVYolo 同接口，但对 yolov8n-pose 输出额外暴露每个人 17 个关键点"""

    def _infer_one(self, img, conf):
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        size = self.input_size
        canvas, r, top, left = letterbox(img, size)
        blob = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        out = self.compiled([blob])[self.out_layer]
        dets, kpts, kconf = postprocess_pose(out, r, top, left, conf, self.iou)
        return _ResultPose(dets, kpts, kconf)


class YoloPoseTorch:
    """torch 后端（加载 .pt 直接 CPU 推理），与 OVYoloPose 接口一致。
    用于无法生成 .onnx 时兜底；有 .onnx 时优先用 OpenVINO(OVYoloPose)。"""

    def __init__(self, pt_path, conf=0.4, iou=0.45, device="cpu"):
        from ultralytics import YOLO
        self.conf, self.iou, self.device = conf, iou, device
        self.model = YOLO(pt_path)

    def __call__(self, imgs, conf=None, imgsz=None, batch=6, verbose=False, **kwargs):
        conf = conf if conf is not None else self.conf
        if isinstance(imgs, (list, tuple)):
            return [self._infer_one(i, conf) for i in imgs]
        return [self._infer_one(imgs, conf)]

    def _infer_one(self, img, conf):
        res = self.model.predict(img, conf=conf, iou=self.iou,
                                 device=self.device, verbose=False)[0]
        data = np.asarray(res.boxes.data.cpu().numpy(), dtype=np.float32)   # [N,6]
        if res.keypoints is None or len(res.keypoints.data) == 0:
            kpts = np.empty((0, 17, 2), dtype=np.float32)
            kconf = np.empty((0, 17), dtype=np.float32)
        else:
            kpts = np.asarray(res.keypoints.data.cpu().numpy()[:, :, :2], dtype=np.float32)
            kconf = np.asarray(res.keypoints.conf.cpu().numpy(), dtype=np.float32)
        return _ResultPose(data, kpts, kconf)
