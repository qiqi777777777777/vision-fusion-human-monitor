# -*- coding: utf-8 -*-
"""
vision-fusion = qie(YOLOv8 监控) + rlsb(InsightFace 人脸识别) 融合版
  - HTTP 快照双路拉流 -> YOLOv8 切片检测 person -> 人体裁剪区域 InsightFace 人脸识别
  - 画面实时标注: 人体框(绿) + 人脸框(库内绿/陌生红) + 姓名与相似度
  - 网页: 监控画面 + 四象限预览 + 人脸库管理(注册/列表/识别开关)
启动: D:\\an\\envs\\yolo\\python.exe app.py   访问 http://localhost:5000
"""
import os
import site
import threading
import time
from datetime import datetime
import requests
from requests.auth import HTTPDigestAuth

# nvidia cu13 运行库（PyPI wheel 的 DLL 目录加入搜索路径，供 onnxruntime-gpu 加载）
for _sp in site.getsitepackages():
    _nv = os.path.join(_sp, "nvidia")
    if os.path.isdir(_nv):
        for _d in os.listdir(_nv):
            _bin = os.path.join(_nv, _d, "bin")
            if os.path.isdir(_bin):
                os.add_dll_directory(_bin)

import cv2
import json
import numpy as np
from collections import deque
from flask import Flask, Response, jsonify, render_template, request
from ov_yolo import OVYolo, OVYoloPose, YoloPoseTorch
from PIL import Image, ImageDraw, ImageFont

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)  # 保证 yolov8m.pt / face_db.npz / templates 按本目录解析

app = Flask(__name__)

# ===================== 人脸识别配置 =====================
DB_PATH = "face_db.npz"
FAISS_VECS = os.path.join("数据库", "faiss_vecs.npy")   # faiss 特征向量 (N,512)，对应 names.json 下标
FAISS_NAMES = os.path.join("数据库", "names.json")      # 人名 -> 向量下标列表
SIM_THRESHOLD = 0.35         # 相似度阈值，>0.35 视为同一人（快照人脸较小，阈值偏高会认不出）
MIN_FACE_PX = 15             # 人脸框最短边最小像素（640x360快照中人脸普遍<40px，原40会把所有脸过滤掉）
FACE_INTERVAL = 3.0          # 每路摄像头人脸识别最小间隔(秒)，节流省算力
MAX_FACE_PER_CYCLE = 8       # 每轮最多只识别人体框最大的8人，限 CPU 最坏情况
DETECT_IMGSZ = 512           # 切片推理输入尺寸，CPU 上从640降到512省算力，弥补重叠加大
SNAPSHOT_INTERVAL = 1.0      # HTTP 快照拉取间隔(秒)
DETECT_INTERVAL = 4.0        # 每路摄像头检测最小间隔(秒)，CPU 上避免满负荷空转

# ===================== 姿态识别(YOLOv8-pose)参数 =====================
POSE_CONF = 0.40            # pose 检测置信度
POSE_INTERVAL = 1.5         # 姿态识别轮询间隔(秒)，CPU 友好
KPT_MIN_CONF = 0.28         # 关键点可见性阈值(小于视为不可见)
WALK_SPEED = 25.0           # 走到静止 速度阈值(px/s)
RUN_SPEED = 60.0            # 跑到走 速度阈值(px/s)
POSE_PEOPLE_MAX = 10        # 每路每轮最多详细判断的人数(CPU 保护)
TRACK_PRUNE_POSE = 5.0      # 姿态轨迹清理秒数
FALL_CONFIRM_FRAMES = 2     # 跌倒连续确认帧数(连续多帧呈跌倒才告警，防抖动误报)
# 姿态标注颜色(BGR)
COLOR_STAND  = (0, 255, 0)      # 站立 绿
COLOR_SIT    = (0, 170, 255)    # 坐着 橙
COLOR_RAISE  = (255, 0, 200)    # 举手 亮粉
COLOR_WALK   = (255, 170, 0)    # 走 蓝
COLOR_RUN    = (0, 0, 255)      # 跑 红
COLOR_FALL   = (0, 60, 255)     # 跌倒 亮红
COLOR_HAND   = (0, 255, 255)    # 握手 黄

_model_lock = threading.Lock()

def init_face_model():
    import onnxruntime as ort
    from insightface.app import FaceAnalysis
    m = FaceAnalysis(name="buffalo_l", root=BASE_DIR,
                     allowed_modules=['detection', 'recognition'])
    if "CUDAExecutionProvider" in ort.get_available_providers():
        try:
            m.prepare(ctx_id=0)      # GPU
            return m, "GPU (CUDA)"
        except Exception as e:
            m.prepare(ctx_id=-1)     # CPU 兜底
            return m, "CPU (GPU初始化失败: %s)" % str(e)[:60]
    m.prepare(ctx_id=-1)             # CUDA 不可用
    return m, "CPU (CUDA Provider 不可用)"

fa, FACE_CTX = init_face_model()
print("人脸模型已加载:", FACE_CTX)

# ===================== YOLO 模型 =====================
# OpenVINO 加速版：加载 jk_openvino_test 验证过的 yolov8n.onnx（640 输入）
# 接口与 ultralytics YOLO 兼容（model(imgs, conf=..., imgsz=...) 返回 .boxes.data [N,6]）
try:
    model = OVYolo("yolov8n.onnx", conf=0.25, iou=0.45)
    print("已加载 yolov8n.onnx（OpenVINO CPU 加速）")
except Exception as e:
    print("yolov8n.onnx 加载失败：", str(e))
    raise

# ===================== 姿态识别(YOLOv8-pose) =====================
# 优先 OpenVINO(.onnx)；无 onnx 时降级用本地 .pt + torch 跑，保证功能可用
try:
    model_pose = OVYoloPose("yolov8n-pose.onnx", conf=POSE_CONF, iou=0.45)
    POSE_BACKEND = "OpenVINO"
except Exception as e:
    print("yolov8n-pose.onnx 不可用，降级 torch 后端：", str(e)[:80])
    model_pose = YoloPoseTorch("yolov8n-pose.pt", conf=POSE_CONF, iou=0.45)
    POSE_BACKEND = "torch(CPU)"

# ===================== 人脸库(线程安全) =====================
_db_lock = threading.Lock()
known_names = []
known_embs = []

def load_db():
    global known_names, known_embs
    with _db_lock:
        known_names, known_embs = [], []
        if os.path.exists(DB_PATH):
            data = np.load(DB_PATH, allow_pickle=True)
            known_names = list(data["names"])
            known_embs = [np.asarray(e, dtype=np.float32) for e in data["embs"]]
        # 合并 faiss 库（29人/116条）。app 原只读 face_db.npz(仅2人)，导致30人库未被识别
        if os.path.exists(FAISS_VECS) and os.path.exists(FAISS_NAMES):
            vecs = np.load(FAISS_VECS)
            with open(FAISS_NAMES, "r", encoding="utf-8") as f:
                mapping = json.load(f)
            for name, idxs in mapping.items():
                for i in idxs:
                    if 0 <= i < len(vecs):
                        known_names.append(name)
                        known_embs.append(np.asarray(vecs[i], dtype=np.float32))

def save_db():
    with _db_lock:
        np.savez(DB_PATH, names=np.array(known_names), embs=np.array(known_embs))

load_db()
print("人脸库已加载:", ", ".join(sorted(set(str(n) for n in known_names))) or "空")

def match_face(emb):
    """返回 (姓名, 相似度) 或 None"""
    if not known_embs:
        return None
    sims = [float(np.dot(emb, e)) for e in known_embs]
    i = int(np.argmax(sims))
    if sims[i] > SIM_THRESHOLD:
        return str(known_names[i]), sims[i]
    return None

# ===================== 中文字体 =====================
FONT_PATH = next((p for p in ["C:/Windows/Fonts/msyh.ttc",
                              "C:/Windows/Fonts/simhei.ttf",
                              "C:/Windows/Fonts/simsun.ttc"] if os.path.exists(p)), None)
_font_cache = {}

def get_font(size):
    if size not in _font_cache and FONT_PATH:
        _font_cache[size] = ImageFont.truetype(FONT_PATH, size)
    return _font_cache.get(size)

def draw_texts_zh(frame, texts):
    """PIL 绘制中文，texts: (文本, (x,y), 字号, (B,G,R), 是否带黑底)"""
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    for text, pos, size, color_bgr, with_bg in texts:
        font = get_font(size)
        if not font:
            continue
        color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
        if with_bg:
            bbox = draw.textbbox(pos, text, font=font)
            draw.rectangle([bbox[0]-4, bbox[1]-2, bbox[2]+4, bbox[3]+2], fill=(0, 0, 0))
        draw.text(pos, text, font=font, fill=color_rgb)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# ===================== 全局共享数据 =====================
class SharedData:
    def __init__(self):
        self.cam1_raw = None
        self.cam1_tiles = []
        self.cam1_boxes = None
        self.cam1_faces = []     # [(x1,y1,x2,y2,name,sim), ...]
        self.cam1_behaviors = []  # [(x1,y1,x2,y2,text,(B,G,R)), ...] 玩手机/睡觉标注
        self.cam1_poses = []      # [(x1,y1,x2,y2,label,(B,G,R)), ...] 姿态标注
        self.cam1_fps = 0
        self.cam1_det_num = 0
        self.cam2_raw = None
        self.cam2_tiles = []
        self.cam2_boxes = None
        self.cam2_faces = []
        self.cam2_behaviors = []  # [(x1,y1,x2,y2,text,(B,G,R)), ...]
        self.cam2_poses = []      # [(x1,y1,x2,y2,label,(B,G,R)), ...]
        self.cam2_fps = 0
        self.cam2_det_num = 0
        self.enable_yolo = True
        self.enable_face = True
        self.enable_phone = True        # 玩手机检测开关
        self.enable_sleep = True        # 睡觉检测开关
        self.enable_pose = True         # 姿态识别(站/坐/举手/走/跑/跌倒/握手)开关
        self.walk_speed = WALK_SPEED
        self.run_speed = RUN_SPEED
        self.sleep_seconds = 15.0
        self.sleep_aspect = 1.15
        self.still_px = 18.0
        self.conf_thresh = 0.25
        self.tile_size = 640
        self.overlap = 256
        # ---- 追踪可视化数据（由 detect_behaviors 每轮更新，供画面绘制 ID/轨迹） ----
        self.cam1_tracks = []   # [(box4, tid), ...]
        self.cam2_tracks = []
        self.cam1_traj = {}     # tid -> deque[(cx,cy),...]，按出现顺序
        self.cam2_traj = {}
        self.traj_maxlen = 10   # 每目标最多保留的历史轨迹点数

shared = SharedData()

# ===================== 行为检测（玩手机 / 睡觉） =====================
PHONE_CLS = 67                # COCO 中 cell phone 类别 id
SLEEP_SECONDS = 15.0          # 连续静止超过该秒数判定为“睡觉”
SLEEP_ASPECT = 1.15           # 人体框 宽/高 超过此值才视为躺卧姿势（站着/坐着不算）
STILL_PX = 18.0               # 前后两帧中心点位移小于该像素视为静止
TRACK_PRUNE = 90.0            # 单人目标超过该秒数没匹配到则删除跟踪记录
PHONE_MARGIN = 0.25           # 手机中心须落在人体框内缩进该比例才判为“玩手机”
COLOR_PHONE = (0, 0, 255)     # BGR 红：玩手机
COLOR_SLEEP = (0, 170, 255)   # BGR 橙黄：睡觉
# ---- 行为检测优化（按本机 CPU 性能设计，2026-09-16） ----
STILL_RATIO = 0.08            # 静止判定抖动阈值占人体框高的比例（消除小目标检测抖动）
TRACK_MATCH_DIST = 80.0       # 跨帧匹配最大中心距(px)，1080p + 约4秒检测间隔下放宽
PHONE_CROP_MIN_H = 60         # 裁剪复检手机的最小人体框高(px)，太小的人根本看不到手机
PHONE_CROP_MAX_N = 6          # 每路每轮最多复检几个人的手机（CPU 保护，避免推理堆积）
PHONE_CROP_IMGSZ = 320        # 裁剪复检的推理尺寸（小图，CPU 友好）
PHONE_CROP_CONF = 0.30        # 裁剪复检的手机置信度（仅手机通道，主检测阈值不变）
PHONE_CONFIRM_N = 2           # 同一人连续几轮命中手机才判定“玩手机”（防单帧误报）

# -------------------- 卡尔曼滤波追踪（恒定速度模型） --------------------
# 状态 [cx, cy, vx, vy]，观测 [cx, cy]；dt 固定按“一个检测周期”处理
_KF_F = np.array([[1, 0, 1, 0],
                  [0, 1, 0, 1],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]], dtype=np.float64)
_KF_H = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0]], dtype=np.float64)
_KF_Q = np.diag([1e-2, 1e-2, 8e-2, 8e-2])   # 过程噪声（位置/速度分量）
_KF_R = np.diag([1e-2, 1e-2])               # 观测噪声（检测中心抖动）
_KF_P0 = np.eye(4) * 1.0                     # 新轨迹初始协方差

class PersonTracker:
    """基于卡尔曼滤波的跨帧目标追踪（恒定速度模型）。
    每轮先把所有轨迹 predict 到当前时刻，用预测中心做最近邻关联，
    再对匹配上的轨迹 update 观测中心（卡尔曼平滑消除检测抖动）。
    相比原"中心点最近邻 + EMA"，ID 更稳定、静止计时更不易被框抖动重置。"""
    def __init__(self, match_dist=TRACK_MATCH_DIST):
        self.tracks = {}      # id -> dict(s(4,), P(4,4), pmx,pmy 上一帧滤波中心, last_move_t, last_seen_t, idle_sec)
        self._next_id = 1
        self.match_dist = match_dist

    def update(self, boxes, still_px=STILL_PX, sleep_seconds=SLEEP_SECONDS):
        """返回 [(box, tid, idle_sec, is_sleep)]，包含所有已匹配/新建目标"""
        now = time.time()
        # 1) 所有既有轨迹先 predict 到本帧（预测中心用于关联）
        for t in self.tracks.values():
            t["s"] = _KF_F @ t["s"]
            t["P"] = _KF_F @ t["P"] @ _KF_F.T + _KF_Q
        out = []
        used = set()
        for b in boxes:
            if len(b) < 4:
                continue
            cz = np.array([(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0], dtype=np.float64)
            # 2) 最近邻关联：用预测中心 (s[0], s[1])
            best_id, best_d = None, 1e9
            for tid, t in self.tracks.items():
                if tid in used:
                    continue
                d = np.hypot(t["s"][0] - cz[0], t["s"][1] - cz[1])
                if d < best_d:
                    best_d, best_id = d, tid
            if best_id is None or best_d > self.match_dist:  # 距离过大视为新出现的人
                tid = self._next_id
                self._next_id += 1
                self.tracks[tid] = {"s": np.array([cz[0], cz[1], 0.0, 0.0]),
                                    "P": _KF_P0.copy(),
                                    "pmx": cz[0], "pmy": cz[1],
                                    "last_move_t": now, "last_seen_t": now,
                                    "idle_sec": 0.0}
                used.add(tid)
                out.append((b, tid, 0.0, False))
                continue
            t = self.tracks[best_id]
            used.add(best_id)
            # 3) 移动判定：相对上一帧滤波后中心位移（卡尔曼平滑后抖动更小）
            h = b[3] - b[1]
            thr = max(still_px, h * STILL_RATIO)  # 抖动阈值：绝对像素 与 框高比例 取大
            if np.hypot(cz[0] - t["pmx"], cz[1] - t["pmy"]) >= thr:
                t["last_move_t"] = now
            # 4) 卡尔曼 update：用本帧检测中心修正状态
            y = cz - (_KF_H @ t["s"])
            S = _KF_H @ t["P"] @ _KF_H.T + _KF_R
            K = t["P"] @ _KF_H.T @ np.linalg.inv(S)
            t["s"] = t["s"] + K @ y
            t["P"] = (np.eye(4) - K @ _KF_H) @ t["P"]
            t["pmx"], t["pmy"] = t["s"][0], t["s"][1]
            t["idle_sec"] = now - t["last_move_t"]
            t["last_seen_t"] = now
            out.append((b, best_id, t["idle_sec"], t["idle_sec"] >= sleep_seconds))
        # 5) 清理很久没出现的目标
        for tid in [k for k, t in self.tracks.items()
                    if now - t["last_seen_t"] > TRACK_PRUNE]:
            del self.tracks[tid]
        return out


class PhoneConfirm:
    """玩手机多帧确认：同一人（按轨迹ID）连续命中 need 轮才判定，过滤单帧误报"""
    def __init__(self, need=PHONE_CONFIRM_N):
        self.need = need
        self.hits = {}   # tid -> 连续命中轮数

    def update(self, hit_ids):
        for tid in list(self.hits):
            if tid not in hit_ids:
                del self.hits[tid]
        confirmed = set()
        for tid in hit_ids:
            self.hits[tid] = self.hits.get(tid, 0) + 1
            if self.hits[tid] >= self.need:
                confirmed.add(tid)
        return confirmed


_person_trackers = {"cam1": PersonTracker(), "cam2": PersonTracker()}
_phone_confirms = {"cam1": PhoneConfirm(), "cam2": PhoneConfirm()}
_behavior_states = {}   # cam_id -> {(label, 量化位置)}，用于状态变化日志


def detect_phones_on_persons(frame, person_boxes):
    """人体框裁剪放大后复检手机，返回全局坐标手机框列表。
    CPU 友好：只复检最大的 PHONE_CROP_MAX_N 个人、框高 >= PHONE_CROP_MIN_H、
    推理尺寸 PHONE_CROP_IMGSZ。手机从几像素放大到几十像素，检出率大幅提升。"""
    if frame is None or person_boxes is None or len(person_boxes) == 0:
        return []
    h, w = frame.shape[:2]
    cands = [b for b in person_boxes if len(b) >= 4 and (b[3] - b[1]) >= PHONE_CROP_MIN_H]
    cands.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    cands = cands[:PHONE_CROP_MAX_N]
    crops, metas = [], []
    for b in cands:
        x1, y1, x2, y2 = [int(v) for v in b[:4]]
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, w), min(y2, h)
        if x2 - x1 < 24 or y2 - y1 < 24:
            continue
        crop = frame[y1:y2, x1:x2]
        ch, cw = crop.shape[:2]
        scale = 1.0
        if max(cw, ch) < PHONE_CROP_IMGSZ:
            scale = min(2.5, float(PHONE_CROP_IMGSZ) / max(cw, ch))
        if scale > 1.0:
            crop = cv2.resize(crop, (int(cw * scale), int(ch * scale)),
                              interpolation=cv2.INTER_LINEAR)
        crops.append(crop)
        metas.append((x1, y1, scale))
    if not crops:
        return []
    phones = []
    try:
        with _model_lock:
            res = model(crops, imgsz=PHONE_CROP_IMGSZ, conf=PHONE_CROP_CONF,
                        batch=min(6, len(crops)), verbose=False)
    except Exception:
        return []
    for r, (x1, y1, scale) in zip(res, metas):
        if r is None or r.boxes is None:
            continue
        data = r.boxes.data.cpu().numpy()
        if len(data) == 0:
            continue
        for d in data:
            if int(d[5]) != PHONE_CLS:
                continue
            px1 = x1 + int(d[0] / scale)
            py1 = y1 + int(d[1] / scale)
            px2 = x1 + int(d[2] / scale)
            py2 = y1 + int(d[3] / scale)
            phones.append([px1, py1, px2, py2, float(d[4])])
    return phones


def detect_behaviors(person_boxes, phone_boxes, cam_id):
    """根据人体框+手机框和跨帧追踪，生成行为标注 [(x1,y1,x2,y2,text,color)]"""
    behav = []
    tracked = _person_trackers[cam_id].update(
        person_boxes, shared.still_px, shared.sleep_seconds)
    # ---- 追踪可视化数据：每轮把 ID + 轨迹历史写入 shared，供画面绘制 ----
    trk_attr = "cam1_tracks" if cam_id == "cam1" else "cam2_tracks"
    trj_attr = "cam1_traj" if cam_id == "cam1" else "cam2_traj"
    tracks = []
    cur_tids = set()
    traj_store = getattr(shared, trj_attr)
    for b, tid, _idle, _slp in tracked:
        tracks.append(([int(b[0]), int(b[1]), int(b[2]), int(b[3])], tid))
        cx = (b[0] + b[2]) / 2.0
        cy = (b[1] + b[3]) / 2.0
        hist = traj_store.setdefault(tid, deque(maxlen=shared.traj_maxlen))
        hist.append((float(cx), float(cy)))
        cur_tids.add(tid)
    setattr(shared, trk_attr, tracks)
    for tid in [k for k in traj_store if k not in cur_tids]:  # 清理已消失的目标
        del traj_store[tid]
    # --- 玩手机：手机中心落在人框缩进区域内，且同一人连续多轮命中才判定 ---
    phone_hit_tids = set()
    phone_box_by_tid = {}
    if shared.enable_phone:
        for b, tid, _idle, _slp in tracked:
            px1, py1, px2, py2 = b[:4]
            m = PHONE_MARGIN
            inx1, inx2 = px1 + (px2 - px1) * m, px2 - (px2 - px1) * m
            iny1, iny2 = py1 + (py2 - py1) * m, py2 - (py2 - py1) * m
            for f in phone_boxes:
                if len(f) < 4:
                    continue
                cx = (f[0] + f[2]) / 2.0
                cy = (f[1] + f[3]) / 2.0
                if inx1 <= cx <= inx2 and iny1 <= cy <= iny2:
                    phone_hit_tids.add(tid)
                    phone_box_by_tid[tid] = b
                    break
        for tid in _phone_confirms[cam_id].update(phone_hit_tids):
            if tid in phone_box_by_tid:
                b = phone_box_by_tid[tid]
                behav.append([int(b[0]), int(b[1]), int(b[2]), int(b[3]),
                              "玩手机", COLOR_PHONE])
    # --- 睡觉：连续静止超过阈值 且 人体呈躺卧姿态(宽>高) ---
    # 仅“中心不动”会误判静止的坐/站人群；加宽高比门槛后只有躺下的才算睡觉
    # 与玩手机互斥：本轮在玩手机的人不判睡觉
    if shared.enable_sleep:
        for b, tid, sec, is_sleep in tracked:
            if not is_sleep:
                continue
            if tid in phone_hit_tids:
                continue
            w = b[2] - b[0]
            h = b[3] - b[1]
            if h <= 0 or w <= h * shared.sleep_aspect:
                continue
            behav.append([int(b[0]), int(b[1]), int(b[2]), int(b[3]),
                          f"睡觉 {sec:.0f}s", COLOR_SLEEP])
    # --- 状态变化日志（避免每帧刷屏） ---
    cur = set()
    for _x1, _y1, x2, y2, text, _c in behav:
        key_label = text.split(" ")[0] if " " in text else text
        cur.add((key_label, round((x2 + _x1) / 40), round((y2 + _y1) / 40)))
    prev = _behavior_states.get(cam_id, set())
    for item in cur - prev:
        print(f"[{cam_id}] {item[0]} 出现")
    for item in prev - cur:
        print(f"[{cam_id}] {item[0]} 结束")
    _behavior_states[cam_id] = cur
    return behav

# ===================== 切片函数 =====================
def slice_image(img, tile_size=640, overlap=128):
    h, w = img.shape[:2]
    step = tile_size - overlap
    tiles, offsets = [], []
    y = 0
    while y < h:
        x = 0
        while x < w:
            x1 = min(x + tile_size, w)
            y1 = min(y + tile_size, h)
            tiles.append(img[y:y1, x:x1].copy())
            offsets.append((x, y))
            x += step
        y += step
    return tiles, offsets

def split_quadrants(img, overlap_ratio=0.2):
    """四象限重叠裁剪: [左上, 右上, 左下, 右下]"""
    if img is None:
        return []
    h, w = img.shape[:2]
    hw, hh = w // 2, h // 2
    ow, oh = int(hw * overlap_ratio), int(hh * overlap_ratio)

    def _clamp(v, lo, hi):
        return max(lo, min(v, hi))

    x1, x2 = _clamp(0, 0, w), _clamp(hw + ow, 0, w)
    x3, x4 = _clamp(hw - ow, 0, w), _clamp(w, 0, w)
    y1, y2 = _clamp(0, 0, h), _clamp(hh + oh, 0, h)
    y3, y4 = _clamp(hh - oh, 0, h), _clamp(h, 0, h)
    if x2 <= x1: x2 = hw
    if x4 <= x3: x3 = hw
    if y2 <= y1: y2 = hh
    if y4 <= y3: y3 = hh
    return [img[y1:y2, x1:x2].copy(), img[y1:y2, x3:x4].copy(),
            img[y3:y4, x1:x2].copy(), img[y3:y4, x3:x4].copy()]

# ===================== 框合并 + NMS =====================
def merge_boxes(boxes_list, offsets, iou_thresh=0.45):
    all_boxes = []
    for boxes, (x_off, y_off) in zip(boxes_list, offsets):
        if boxes is None or len(boxes) == 0:
            continue
        for b in boxes:
            x1, y1, x2, y2, conf, cls = b
            all_boxes.append([x1 + x_off, y1 + y_off, x2 + x_off, y2 + y_off, conf])
    if len(all_boxes) == 0:
        return np.array([])
    all_boxes = np.array(all_boxes, dtype=np.float32)
    indices = cv2.dnn.NMSBoxes(all_boxes[:, 0:4].tolist(), all_boxes[:, 4].tolist(),
                               score_threshold=0.25, nms_threshold=iou_thresh)
    return all_boxes[indices.flatten()]

# ===================== 姿态识别（站/坐/举手/走/跑/跌倒/握手） =====================
# COCO 关键点索引: 0鼻 1左眼 2右眼 3左耳 4右耳 5左肩 6右肩 7左肘 8右肘
#                 9左腕 10右腕 11左胯 12右胯 13左膝 14右膝 15左踝 16右踝

def _kval(pts, conf, idx, th, axis):
    vals = [pts[i][axis] for i in idx
            if conf[i] >= th and pts[i][axis] > 0 and np.isfinite(pts[i][axis])]
    return (float(sum(vals) / len(vals)), True) if vals else (0.0, False)

def _torso_len(pts, conf, th=KPT_MIN_CONF):
    """躯干长度(肩-胯 y 差)，关键点不全返回 None"""
    hy, okh = _kval(pts, conf, (11, 12), th, 1)
    sy, oks = _kval(pts, conf, (5, 6), th, 1)
    if okh and oks:
        return max(abs(hy - sy), 1.0)
    return None

def judge_pose(pts, conf, box, th=KPT_MIN_CONF):
    """返回基础姿态: 跌倒 / 坐着 / 站立"""
    x1, y1, x2, y2 = box[:4]
    bh = max(float(y2 - y1), 1.0)
    bw = max(float(x2 - x1), 1.0)
    hy, okh = _kval(pts, conf, (11, 12), th, 1)   # 胯
    sy, oks = _kval(pts, conf, (5, 6), th, 1)     # 肩
    kny, okk = _kval(pts, conf, (13, 14), th, 1)  # 膝
    if not (okh and oks):
        return "站立"
    torso = abs(hy - sy)
    # 跌倒：仅当人体框横向(bw>=bh)。核心判据是「腿是否竖直」——
    # 弯腰/拖地的人脚始终在胯下方(腿竖直,踝与胯纵向距离大)不算跌倒；
    # 真正倒地的人腿横向与身体同高，踝与胯接近同高(leg_v 很小)。
    if bw >= bh and okh:
        ay, oka = _kval(pts, conf, (15, 16), th, 1)   # 踝
        if oka:
            leg_v = abs(ay - hy)
            if leg_v < max(torso * 0.5, 12.0):
                return "跌倒"
    hip_frac = (hy - y1) / bh              # 0=头顶 1=脚底
    thigh = (kny - hy) if okk else torso / 2.0   # 大腿向下高度
    if hip_frac >= 0.50 and thigh < torso * 0.55:
        return "坐着"
    return "站立"

def _wrist_points(pts, conf, th=KPT_MIN_CONF):
    out = []
    for i in (9, 10):
        if conf[i] >= th and pts[i][1] > 0 and np.isfinite(pts[i][1]):
            out.append((float(pts[i][0]), float(pts[i][1])))
    return out

def _hand_raised(pts, conf, th=KPT_MIN_CONF):
    """是否有人手举过头(任一腕明显高于肩)。举手/打招呼均判为『举手』"""
    sy, ok = _kval(pts, conf, (5, 6), th, 1)
    if not ok:
        return False
    hy, okh = _kval(pts, conf, (11, 12), th, 1)
    torso = abs(hy - sy) if okh else 40.0
    thr = max(8.0, torso * 0.12)          # 相对尺度阈值，远处小目标也能抓到抬手
    for i in (9, 10):
        if conf[i] >= th and pts[i][1] > 0 and np.isfinite(pts[i][1]):
            if pts[i][1] < sy - thr:      # 手腕在肩上方 → 抬手
                return True
    return False

def detect_handshakes(persons):
    """persons: [{id,box,pts,conf}]。两两判定握手，返回命中的人 id 集合。
    握手核心特征：两人身体较近 + 一方手腕伸向对方躯干正前方 + 双腕距离小。
    该约束可挡掉"并排坐手放桌面"造成的误判(此时手不在对方身体正前方)。"""
    hids = set()
    n = len(persons)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = persons[i], persons[j]
            # 抬手的人不参与握手判定(举手/打招呼 ≠ 握手，防止握手覆盖举手)
            if _hand_raised(a["pts"], a["conf"]) or _hand_raised(b["pts"], b["conf"]):
                continue
            ta = _torso_len(a["pts"], a["conf"]) or 40.0
            tb = _torso_len(b["pts"], b["conf"]) or 40.0
            tr = max(ta, tb)
            (ax1, ay1, ax2, ay2), (bx1, by1, bx2, by2) = a["box"], b["box"]
            ca = ((ax1 + ax2) / 2, (ay1 + ay2) / 2)
            cb = ((bx1 + bx2) / 2, (by1 + by2) / 2)
            if np.hypot(ca[0] - cb[0], ca[1] - cb[1]) > max(tr * 2.8, 60.0):
                continue
            wa = _wrist_points(a["pts"], a["conf"])
            wb = _wrist_points(b["pts"], b["conf"])
            if not wa or not wb:
                continue
            shxa, oka = _kval(a["pts"], a["conf"], (5, 6), KPT_MIN_CONF, 0)
            shxb, okb = _kval(b["pts"], b["conf"], (5, 6), KPT_MIN_CONF, 0)
            if not (oka and okb):
                continue
            # 一方手腕伸到对方肩部 x 附近的躯干正前方(面向对方伸出的手)
            reach_ab = any(abs(w[0] - shxb) < tb * 0.50 for w in wa)
            reach_ba = any(abs(w[0] - shxa) < ta * 0.50 for w in wb)
            if not (reach_ab or reach_ba):
                continue
            # 双腕必须接触级地近(两只手实际握在一起)，隔空/桌面相邻不判
            mind = min(float(np.hypot(p[0] - q[0], p[1] - q[1]))
                       for p in wa for q in wb)
            if mind < max(tr * 0.50, 12.0):
                hids.add(a["id"]); hids.add(b["id"])
    return hids

class PoseTracker:
    """姿态目标轻量追踪：分配 ID + EMA 平滑速度(px/s)，供走/跑判定"""
    def __init__(self, match_dist=120.0, prune=TRACK_PRUNE_POSE):
        self.tracks = {}        # id -> {cx,cy,t,speed}
        self._nid = 1
        self.match_dist = match_dist
        self.prune = prune

    def update(self, dets):
        now = time.time()
        out, used = [], set()
        for d in dets:
            x1, y1, x2, y2 = d[:4]
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            best, bd = None, self.match_dist
            for tid, t in self.tracks.items():
                if tid in used:
                    continue
                dd = float(np.hypot(t["cx"] - cx, t["cy"] - cy))
                if dd < bd:
                    bd, best = dd, tid
            if best is None:
                tid = self._nid; self._nid += 1
                self.tracks[tid] = {"cx": cx, "cy": cy, "t": now,
                                    "speed": 0.0, "fall": 0}
                used.add(tid)
            else:
                t = self.tracks[best]
                dt = max(now - t["t"], 0.01)
                inst = float(np.hypot(cx - t["cx"], cy - t["cy"])) / dt
                t["speed"] = t["speed"] * 0.6 + inst * 0.4
                t["cx"], t["cy"], t["t"] = cx, cy, now
                used.add(best)
            out.append((best, d))
        for tid in [k for k, t in self.tracks.items()
                    if now - t["t"] > self.prune]:
            del self.tracks[tid]
        return out

_pose_trackers = {"cam1": PoseTracker(), "cam2": PoseTracker()}

def pose_thread(cam_id="cam1"):
    """独立姿态识别线程：对整帧跑一次 pose，输出每人 站/坐/举手/走/跑/跌倒/握手"""
    tracker = _pose_trackers[cam_id]
    while True:
        frame = shared.cam1_raw if cam_id == "cam1" else shared.cam2_raw
        if frame is None:
            time.sleep(0.05)
            continue
        if not shared.enable_pose:
            if (shared.cam1_poses if cam_id == "cam1" else shared.cam2_poses):
                if cam_id == "cam1":
                    shared.cam1_poses = []
                else:
                    shared.cam2_poses = []
            time.sleep(0.5)
            continue
        t0 = time.time()
        try:
            with _model_lock:
                res = model_pose(frame, conf=POSE_CONF, verbose=False)[0]
            dets = res.boxes.data.cpu().numpy()
            kpts = res.keypoints
            kconf = res.kpts_conf
            m = len(dets)
            if m > POSE_PEOPLE_MAX:          # CPU 保护：只详判最大的 N 人
                area = (dets[:, 2] - dets[:, 0]) * (dets[:, 3] - dets[:, 1])
                keep = np.argsort(area)[::-1][:POSE_PEOPLE_MAX]
                dets, kpts, kconf = dets[keep], kpts[keep], kconf[keep]
                m = len(keep)
            if m == 0:
                poses_out = []
            else:
                tracked = tracker.update(dets)
                labels, persons = [], []
                for (d, kp, kc), (tid, _d) in zip(zip(dets, kpts, kconf), tracked):
                    base = judge_pose(kp, kc, d)
                    if base == "跌倒":
                        tracker.tracks[tid]["fall"] += 1
                        if tracker.tracks[tid]["fall"] < FALL_CONFIRM_FRAMES:
                            base = "站立"     # 连续多帧确认防抖动
                    else:
                        tracker.tracks[tid]["fall"] = 0
                    speed = tracker.tracks[tid]["speed"]
                    if base == "站立":
                        if speed > shared.run_speed:
                            label, color = "跑", COLOR_RUN
                        elif speed > shared.walk_speed:
                            label, color = "走", COLOR_WALK
                        elif _hand_raised(kp, kc):
                            label, color = "举手", COLOR_RAISE
                        else:
                            label, color = "站立", COLOR_STAND
                    elif base == "坐着":
                        label, color = "坐着", COLOR_SIT
                    else:
                        label, color = "跌倒", COLOR_FALL
                    labels.append((tid, d[:4], label, color))
                    persons.append({"id": tid, "box": d[:4], "pts": kp, "conf": kc})
                hs = detect_handshakes(persons)
                poses_out = []
                for tid, box, label, color in labels:
                    if tid in hs:
                        label, color = "握手", COLOR_HAND
                    poses_out.append([int(box[0]), int(box[1]), int(box[2]), int(box[3]),
                                      label, color])
            if cam_id == "cam1":
                shared.cam1_poses = poses_out
            else:
                shared.cam2_poses = poses_out
        except Exception as e:
            print(f"{cam_id} 姿态推理异常：", str(e)[:80])
        elapsed = time.time() - t0
        if elapsed < POSE_INTERVAL:
            time.sleep(POSE_INTERVAL - elapsed)

# ===================== 标注绘制（人体框 + 人脸框 + 中文标签 + 追踪ID/轨迹） =====================
# 按 tid 轮换的颜色盘，用于区分每个人的追踪框/轨迹
_TRACK_PALETTE = [(0, 200, 255), (255, 170, 0), (255, 0, 170), (180, 0, 255),
                  (0, 170, 255), (255, 60, 60), (0, 255, 200), (200, 200, 0)]
def _track_color(tid):
    return _TRACK_PALETTE[tid % len(_TRACK_PALETTE)]

def annotate(frame, boxes, faces, behaviors=None, tracks=None, traj=None, poses=None):
    """tracks: [(box4, tid), ...] 绘制追踪ID；traj: {tid: [(cx,cy),...]} 绘制运动轨迹线
    poses: [(x1,y1,x2,y2,label,(B,G,R)), ...] 姿态标注（画在人体框下方）"""
    canvas = frame.copy()
    # 先画轨迹线/历史点（垫底，避免盖住人框）
    if traj:
        for tid, pts in traj.items():
            col = _track_color(tid)
            ptl = [(int(x), int(y)) for (x, y) in pts]
            for a, b in zip(ptl[:-1], ptl[1:]):
                cv2.line(canvas, a, b, col, 2)
            if ptl:
                cv2.circle(canvas, ptl[-1], 4, col, -1)
    if boxes is not None and len(boxes) > 0:
        for b in boxes:
            x1, y1, x2, y2 = [int(v) for v in b[:4]]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
    texts = []
    if tracks:
        for box4, tid in tracks:
            x1, y1, x2, y2 = box4
            texts.append((f"ID:{tid}", (x1, max(y1 - 20, 0)), 22,
                          _track_color(tid), True))
    for fx1, fy1, fx2, fy2, name, sim in faces:
        color = (0, 255, 0) if sim > 0 else (0, 0, 255)
        cv2.rectangle(canvas, (fx1, fy1), (fx2, fy2), color, 2)
        label = f"{name} {sim:.2f}" if sim > 0 else name
        texts.append((label, (fx1, max(fy1 - 36, 0)), 22, color, True))
    if behaviors:
        for x1, y1, x2, y2, text, color in behaviors:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 3)
            texts.append((text, (x1, max(y1 - 22, 0)), 24, color, True))
    if poses:
        H = canvas.shape[0]
        for x1, y1, x2, y2, text, color in poses:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            texts.append((text, (x1, min(y2 + 6, H - 24)), 24, color, True))
    time_str = datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")
    cv2.putText(canvas, time_str, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    if texts:
        canvas = draw_texts_zh(canvas, texts)
    return canvas

# ===================== 拉流线程（仅 HTTP 快照，关 RTSP） =====================
def cam_stream_thread(snap_url, auth, cam_id="cam1"):
    """仅走 HTTP 快照(ISAPI 主子码流 picture)。RTSP 码流在本环境 H.264 坏帧严重
    (马赛克/解码错误)，故不再自动切换 RTSP，保证画面干净清晰。"""
    session = requests.Session()
    session.auth = auth
    prev_time = time.time()
    snap_ok = False
    while True:
        try:
            resp = session.get(snap_url, timeout=10)
            if resp.status_code != 200 or len(resp.content) == 0:
                print(f"{cam_id} 快照获取失败: HTTP {resp.status_code}")
                time.sleep(1)
                continue
            img_arr = np.frombuffer(resp.content, dtype=np.uint8)
            frame = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
            if frame is None:
                print(f"{cam_id} 快照解码失败")
                time.sleep(1)
                continue
            if not snap_ok:
                snap_ok = True
                print(f"{cam_id} HTTP快照模式启动成功")
        except requests.exceptions.RequestException as e:
            print(f"{cam_id} 快照请求异常:", str(e)[:80])
            time.sleep(1)
            continue
        now_time = time.time()
        fps = round(1 / (now_time - prev_time), 1) if now_time - prev_time > 0 else 0
        prev_time = now_time
        if cam_id == "cam1":
            shared.cam1_raw = frame
            shared.cam1_fps = fps
        else:
            shared.cam2_raw = frame
            shared.cam2_fps = fps
        time.sleep(SNAPSHOT_INTERVAL)

# ===================== 检测线程：YOLO 切片推理 + 人体区域人脸识别 =====================
def face_detect_on_frame(frame, boxes):
    """对每个人体框裁剪做人脸识别，返回 [(x1,y1,x2,y2,name,sim)]（全局坐标）
    人体框较小时先放大再识别，提升远距离小脸的检测/识别效果"""
    results = []
    h, w = frame.shape[:2]
    # 只识别最大的 MAX_FACE_PER_CYCLE 个人（人越大脸越大越清晰），限 CPU 最坏情况
    boxes = sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]),
                   reverse=True)[:MAX_FACE_PER_CYCLE]
    with _model_lock:
        for b in boxes:
            x1, y1, x2, y2 = [int(v) for v in b[:4]]
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(x2, w), min(y2, h)
            if x2 - x1 < 15 or y2 - y1 < 15:
                continue
            crop = frame[y1:y2, x1:x2]
            ch, cw = crop.shape[:2]
            scale = 1.0
            if max(cw, ch) < 200:
                scale = min(4.0, 200.0 / max(cw, ch))
            if scale > 1.0:
                crop_resized = cv2.resize(crop, (int(cw * scale), int(ch * scale)),
                                          interpolation=cv2.INTER_CUBIC)
            else:
                crop_resized = crop
            try:
                raw = fa.get(crop_resized)
            except Exception:
                continue
            for f in raw:
                fw, fh = (f.bbox[2] - f.bbox[0]) / scale, (f.bbox[3] - f.bbox[1]) / scale
                if min(fw, fh) < MIN_FACE_PX:
                    continue
                fx1 = x1 + int(f.bbox[0] / scale); fy1 = y1 + int(f.bbox[1] / scale)
                fx2 = x1 + int(f.bbox[2] / scale); fy2 = y1 + int(f.bbox[3] / scale)
                m = match_face(f.normed_embedding)
                results.append([fx1, fy1, fx2, fy2, *(m or ("陌生人", 0.0))])
    return results

def detect_thread(cam_id="cam1"):
    last_face_t = 0.0
    while True:
        frame = shared.cam1_raw if cam_id == "cam1" else shared.cam2_raw
        if frame is None:
            time.sleep(0.05)
            continue
        t_cycle = time.time()
        try:
            if shared.enable_yolo:
                tiles, offsets = slice_image(frame, tile_size=shared.tile_size, overlap=shared.overlap)
                with _model_lock:
                    res_batch = model(tiles, conf=shared.conf_thresh, batch=6, imgsz=DETECT_IMGSZ)
                person_per_tile, phone_per_tile = [], []
                for res in res_batch:
                    pred_boxes = res.boxes.data.cpu().numpy()
                    if len(pred_boxes) == 0:
                        person_per_tile.append(pred_boxes)
                        phone_per_tile.append(pred_boxes)
                        continue
                    person_per_tile.append(pred_boxes[pred_boxes[:, 5] == 0])        # person
                    phone_per_tile.append(pred_boxes[pred_boxes[:, 5] == PHONE_CLS])  # cell phone
                final_boxes = merge_boxes(person_per_tile, offsets)
                phone_boxes = merge_boxes(phone_per_tile, offsets)
                det_count = len(final_boxes)
            else:
                final_boxes = np.empty((0, 5), dtype=np.float32)
                phone_boxes = np.empty((0, 5), dtype=np.float32)
                det_count = 0
            # 人脸识别：节流，只识别人体框内区域
            face_results = []
            if shared.enable_face and len(final_boxes) > 0 \
                    and time.time() - last_face_t >= FACE_INTERVAL:
                last_face_t = time.time()
                face_results = face_detect_on_frame(frame, final_boxes)
            # 玩手机增强：人体框裁剪放大后复检手机（CPU 友好，只检最大的几个人）
            if shared.enable_phone and len(final_boxes) > 0:
                extra_phones = detect_phones_on_persons(frame, final_boxes)
                if len(extra_phones) > 0:
                    if len(phone_boxes) > 0:
                        phone_boxes = np.vstack(
                            [phone_boxes, np.asarray(extra_phones, dtype=np.float32)])
                    else:
                        phone_boxes = np.asarray(extra_phones, dtype=np.float32)
            behaviors = detect_behaviors(final_boxes, phone_boxes, cam_id)
            if cam_id == "cam1":
                draw_frame = annotate(frame, final_boxes, face_results, behaviors,
                                      shared.cam1_tracks, shared.cam1_traj,
                                      poses=shared.cam1_poses)
            else:
                draw_frame = annotate(frame, final_boxes, face_results, behaviors,
                                      shared.cam2_tracks, shared.cam2_traj,
                                      poses=shared.cam2_poses)
            preview_tiles = split_quadrants(draw_frame)
            if cam_id == "cam1":
                shared.cam1_boxes = final_boxes
                shared.cam1_faces = face_results
                shared.cam1_behaviors = behaviors
                shared.cam1_tiles = preview_tiles
                shared.cam1_det_num = det_count
            else:
                shared.cam2_boxes = final_boxes
                shared.cam2_faces = face_results
                shared.cam2_behaviors = behaviors
                shared.cam2_tiles = preview_tiles
                shared.cam2_det_num = det_count
        except Exception as e:
            print(f"{cam_id} 推理异常：", str(e))
            time.sleep(0.1)
        # 节流：避免 CPU 满载空转
        elapsed = time.time() - t_cycle
        if elapsed < DETECT_INTERVAL:
            time.sleep(DETECT_INTERVAL - elapsed)

# ===================== MJPEG 生成 =====================
def generate_mjpeg(get_frame):
    while True:
        frame = get_frame()
        if frame is None:
            time.sleep(0.05)
            continue
        _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        time.sleep(0.05)   # 编码降频：约30fps->20fps，省 CPU/带宽（源画面本就更慢）

# ===================== Flask 路由 =====================
@app.route('/')
def index():
    return render_template("index.html")

def _make_main_getter(cam):
    def get_main_frame():
        raw = shared.cam1_raw if cam == "cam1" else shared.cam2_raw
        if raw is None:
            return None
        boxes = shared.cam1_boxes if cam == "cam1" else shared.cam2_boxes
        faces = shared.cam1_faces if cam == "cam1" else shared.cam2_faces
        behaviors = shared.cam1_behaviors if cam == "cam1" else shared.cam2_behaviors
        tracks = shared.cam1_tracks if cam == "cam1" else shared.cam2_tracks
        traj = shared.cam1_traj if cam == "cam1" else shared.cam2_traj
        poses = shared.cam1_poses if cam == "cam1" else shared.cam2_poses
        if boxes is None:
            boxes = np.empty((0, 5), dtype=np.float32)
        return annotate(raw, boxes, faces, behaviors, tracks, traj, poses=poses)
    return get_main_frame

@app.route('/cam1.mjpg')
def cam1_stream():
    return Response(generate_mjpeg(_make_main_getter("cam1")),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/cam2.mjpg')
def cam2_stream():
    return Response(generate_mjpeg(_make_main_getter("cam2")),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/<cam>_tile')
def get_tile(cam):
    idx = int(request.args.get("idx", 0))
    tile_list = shared.cam1_tiles if cam == "cam1" else shared.cam2_tiles
    if 0 <= idx < len(tile_list):
        _, jpeg = cv2.imencode('.jpg', tile_list[idx], [cv2.IMWRITE_JPEG_QUALITY, 90])
        return Response(jpeg.tobytes(), mimetype="image/jpeg")
    return Response(b"", mimetype="image/jpeg")

def _tile_jpeg_response(cam, idx):
    patch = None
    raw = shared.cam1_raw if cam == "cam1" else shared.cam2_raw
    if raw is not None:
        boxes = shared.cam1_boxes if cam == "cam1" else shared.cam2_boxes
        faces = shared.cam1_faces if cam == "cam1" else shared.cam2_faces
        if boxes is None:
            boxes = np.empty((0, 5), dtype=np.float32)
        patches = split_quadrants(annotate(raw, boxes, faces))
        if 0 <= idx < len(patches):
            patch = patches[idx]
    if patch is None:
        return Response(b"", mimetype="image/jpeg")
    _, jpeg = cv2.imencode('.jpg', patch, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return Response(jpeg.tobytes(), mimetype="image/jpeg")

for _cam in ("cam1", "cam2"):
    for _idx in range(4):
        def _tile_stream(_cam=_cam, _idx=_idx):
            return _tile_jpeg_response(_cam, _idx)
        app.add_url_rule("/{0}_tile{1}.mjpg".format(_cam, _idx),
                         "{0}_tile{1}_stream".format(_cam, _idx), _tile_stream)

@app.route('/status')
def get_status():
    unique = sorted(set(str(n) for n in known_names))
    return jsonify({
        "cam1_fps": shared.cam1_fps,
        "cam1_det": shared.cam1_det_num,
        "cam1_faces": len(shared.cam1_faces),
        "cam2_fps": shared.cam2_fps,
        "cam2_det": shared.cam2_det_num,
        "cam2_faces": len(shared.cam2_faces),
        "enable_yolo": shared.enable_yolo,
        "enable_face": shared.enable_face,
        "enable_phone": shared.enable_phone,
        "enable_sleep": shared.enable_sleep,
        "enable_pose": shared.enable_pose,
        "walk_speed": shared.walk_speed,
        "run_speed": shared.run_speed,
        "cam1_poses": len(shared.cam1_poses),
        "cam2_poses": len(shared.cam2_poses),
        "pose_ctx": POSE_BACKEND,
        "sleep_seconds": shared.sleep_seconds,
        "sleep_aspect": shared.sleep_aspect,
        "still_px": shared.still_px,
        "conf": shared.conf_thresh,
        "face_ctx": FACE_CTX,
        "users": unique,
        "user_count": len(unique)
    })

@app.route('/api/detect_results')
def api_detect_results():
    """供 AI 监控 Agent 轮询的结构化现场快照（只读，不影响主流程）"""
    def _cam(c):
        faces = [str(f[4]) for f in getattr(shared, c + "_faces")]
        behaviors = [str(b[4]) for b in getattr(shared, c + "_behaviors")]
        poses = [str(p[4]) for p in getattr(shared, c + "_poses")]
        return {
            "people": getattr(shared, c + "_det_num"),
            "fps": getattr(shared, c + "_fps"),
            "faces": faces,
            "behaviors": behaviors,
            "poses": poses,
        }
    return jsonify({"cam1": _cam("cam1"), "cam2": _cam("cam2")})

# ===================== 网页版 AI 问答（数字狗） =====================
try:
    from monitor_agent import KEY as AI_KEY   # 复用 monitor_agent 里填好的密钥
except Exception:
    AI_KEY = os.environ.get("SILICONFLOW_API_KEY", "")

AI_MODEL = "Qwen/Qwen2.5-7B-Instruct"

try:
    from llm_helper import ask as _ask_llm_raw
except Exception:
    def _ask_llm_raw(prompt, api_key, model):
        raise Exception("缺少 llm_helper.py（应放在项目根目录）")


def _ask_llm(prompt):
    return _ask_llm_raw(prompt, AI_KEY, AI_MODEL)


def _snapshot_text():
    parts = []
    for c in ("cam1", "cam2"):
        d = [str(f[4]) for f in getattr(shared, c + "_faces")]
        b = [str(x[4]) for x in getattr(shared, c + "_behaviors")]
        p = [str(x[4]) for x in getattr(shared, c + "_poses")]
        seg = "%s %d人" % (c, getattr(shared, c + "_det_num"))
        if d:
            seg += " 人脸:" + ",".join(d)
        if b:
            seg += " 行为:" + ",".join(b)
        if p:
            seg += " 姿态:" + ",".join(p)
        parts.append(seg)
    return "; ".join(parts)


@app.route('/api/ask', methods=["POST"])
def api_ask():
    data = request.get_json(silent=True) or {}
    q = (data.get("question") or "").strip()
    if not q:
        return jsonify({"ok": False, "error": "问题为空"})
    try:
        answer = _ask_llm(f"当前监控现场：{_snapshot_text()}。请用一句话回答：{q}")
        return jsonify({"ok": True, "answer": answer})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

@app.route('/set_param', methods=["POST"])
def set_param():
    data = request.get_json()
    if "enable_yolo" in data:
        shared.enable_yolo = bool(data["enable_yolo"])
    if "enable_face" in data:
        shared.enable_face = bool(data["enable_face"])
    if "enable_phone" in data:
        shared.enable_phone = bool(data["enable_phone"])
    if "enable_sleep" in data:
        shared.enable_sleep = bool(data["enable_sleep"])
    if "enable_pose" in data:
        shared.enable_pose = bool(data["enable_pose"])
    if "walk_speed" in data:
        shared.walk_speed = float(data["walk_speed"])
    if "run_speed" in data:
        shared.run_speed = float(data["run_speed"])
    if "sleep_seconds" in data:
        shared.sleep_seconds = float(data["sleep_seconds"])
    if "sleep_aspect" in data:
        shared.sleep_aspect = float(data["sleep_aspect"])
    if "still_px" in data:
        shared.still_px = float(data["still_px"])
    if "conf" in data:
        shared.conf_thresh = float(data["conf"])
    if "tile_size" in data:
        shared.tile_size = int(data["tile_size"])
        shared.overlap = int(shared.tile_size * 0.4)
    return jsonify({"ok": True})

# ===================== 人脸库管理 API =====================
@app.route('/api/users')
def api_users():
    unique = sorted(set(str(n) for n in known_names))
    return jsonify({"ok": True, "users": [{"name": n, "count": sum(1 for x in known_names if str(x) == n)} for n in unique]})

@app.route('/api/register', methods=["POST"])
def api_register():
    name = (request.form.get("name") or "").strip()
    f = request.files.get("file")
    if not name:
        return jsonify({"ok": False, "error": "请先输入姓名"})
    if f is None:
        return jsonify({"ok": False, "error": "请选择一张正面照片"})
    data = f.read()
    if len(data) > 15 * 1024 * 1024:
        return jsonify({"ok": False, "error": "图片过大，最大 15MB"})
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"ok": False, "error": "无法解析图片，请上传 jpg/png"})
    try:
        with _model_lock:
            raw = fa.get(img)
            faces = [x for x in raw if min(x.bbox[2]-x.bbox[0], x.bbox[3]-x.bbox[1]) >= MIN_FACE_PX]
            if not faces:
                return jsonify({"ok": False, "error": "照片中未检测到清晰人脸，请上传单人正面照片"})
            emb = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1])).normed_embedding.copy()
            with _db_lock:
                known_names.append(name)
                known_embs.append(np.asarray(emb, dtype=np.float32))
                save_db()
    except Exception as e:
        return jsonify({"ok": False, "error": "注册失败: %s" % str(e)})
    count = sum(1 for n in known_names if str(n) == name)
    return jsonify({"ok": True, "name": name, "emb_count": count})

if __name__ == '__main__':
    user = "admin"
    pwd = "Wxy_200501"
    auth = HTTPDigestAuth(user, pwd)
    snap1 = f"http://192.168.31.100/ISAPI/Streaming/channels/101/picture"
    snap2 = f"http://192.168.31.102/ISAPI/Streaming/channels/101/picture"
    t1 = threading.Thread(target=cam_stream_thread, args=(snap1, auth, "cam1"), daemon=True)
    t2 = threading.Thread(target=cam_stream_thread, args=(snap2, auth, "cam2"), daemon=True)
    d1 = threading.Thread(target=detect_thread, args=("cam1",), daemon=True)
    d2 = threading.Thread(target=detect_thread, args=("cam2",), daemon=True)
    p1 = threading.Thread(target=pose_thread, args=("cam1",), daemon=True)
    p2 = threading.Thread(target=pose_thread, args=("cam2",), daemon=True)
    t1.start(); t2.start(); d1.start(); d2.start(); p1.start(); p2.start()
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)