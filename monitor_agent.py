# 监控 Agent：轮询 app.py 的 /api/detect_results，发现异常自动分析，支持随时提问
import json
import os
import threading
import time
import urllib.request

from llm_helper import ask as ask_llm

# 密钥从环境变量 SILICONFLOW_API_KEY 读取；未设置时回退读取本地 siliconflow.key（该文件已被 .gitignore 排除，不会进仓库）
def _load_key():
    k = os.environ.get("SILICONFLOW_API_KEY", "").strip()
    if k:
        return k
    for p in ("siliconflow.key",):
        try:
            with open(p, encoding="utf-8") as f:
                k = f.read().strip()
            if k:
                return k
        except OSError:
            pass
    return ""

KEY = _load_key()
BASE = "http://localhost:5000"
MODEL = "Qwen/Qwen2.5-7B-Instruct"
ALERT_KEYS = ("跌倒", "睡觉", "玩手机", "陌生人")


def fetch_snapshot():
    with urllib.request.urlopen(BASE + "/api/detect_results", timeout=5) as r:
        return json.loads(r.read().decode("utf-8"))


def summarize(s):
    lines = []
    for cam in ("cam1", "cam2"):
        d = s[cam]
        parts = [f"{cam} {d['people']}人"]
        if d["faces"]:
            parts.append("人脸:" + ",".join(d["faces"]))
        if d["behaviors"]:
            parts.append("行为:" + ",".join(d["behaviors"]))
        if d["poses"]:
            parts.append("姿态:" + ",".join(d["poses"]))
        lines.append(" ".join(parts))
    return "; ".join(lines)


def alert_loop():
    while True:
        try:
            text = summarize(fetch_snapshot())
            if any(k in text for k in ALERT_KEYS):
                print("[告警] " + ask_llm(
                    f"监控现场：{text}。判断是否有异常情况，用一句话提醒值班人员。",
                    KEY, MODEL))
        except Exception as e:
            print(f"(获取现场失败: {e})")
        time.sleep(10)


def main():
    threading.Thread(target=alert_loop, daemon=True).start()
    print("监控 Agent 已启动：后台每 10 秒检查现场并自动告警；输入问题直接提问，输入 q 退出。")
    while True:
        try:
            q = input("你问数字狗: ").strip()
        except EOFError:
            break
        if not q:
            continue
        if q.lower() == "q":
            break
        text = summarize(fetch_snapshot())
        print("[数字狗] " + ask_llm(f"当前监控现场：{text}。请用一句话回答：{q}", KEY, MODEL))


if __name__ == "__main__":
    main()
