# run_all.py —— 一键启动：监控服务 + AI 监控 Agent（直接双击本文件即可）
import json
import os
import subprocess
import sys
import time
import urllib.request
import webbrowser

BASE = r"D:\CV\sx1\vision-fusion_openvino_test"


def env_python(name):
    exe = "\\python.exe" if sys.platform == "win32" else "/bin/python"
    for base in (r"D:\an\envs",
                 os.path.expanduser(r"~\anaconda3\envs"),
                 os.path.expanduser(r"~\miniconda3\envs")):
        p = os.path.join(base, name, "python.exe")
        if os.path.exists(p):
            return p
    try:
        out = subprocess.run(["conda", "env", "list", "--json"],
                             capture_output=True, text=True)
        for p in json.loads(out.stdout)["envs"]:
            if p.split("\\")[-1] == name or p.split("/")[-1] == name:
                return p + exe
    except Exception:
        pass
    raise SystemExit(f"找不到 conda 环境 {name} 的 python，请把实际路径告诉我")


def wait_server(url, timeout=40):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(url, timeout=2)
            return True
        except Exception:
            time.sleep(1)
    return False


print("=" * 50)
print(" 一键启动：监控服务 + AI 监控 Agent")
print("=" * 50)

py_app = env_python("yolo")    # 跑监控（有 cv2/openvino/requests）
py_agent = env_python("yolo")  # 跑 Agent（llm_helper 只需 requests，yolo 环境已有）

print("[1/2] 启动监控服务...")
p_app = subprocess.Popen([py_app, "app.py"], cwd=BASE)

print("等待监控服务就绪...")
if wait_server("http://localhost:5000/status"):
    print("监控服务已就绪，正在自动打开浏览器...")
    webbrowser.open("http://localhost:5000")
else:
    print("警告：监控服务未在 40 秒内就绪，仍尝试启动 Agent（可能稍后才能连上）")

print("[2/2] 启动 AI 监控 Agent...")
p_agent = subprocess.Popen([py_agent, "monitor_agent.py"], cwd=BASE)

try:
    p_app.wait()
    p_agent.wait()
except KeyboardInterrupt:
    print("\n收到退出信号，正在关闭全部进程...")
    p_app.terminate()
    p_agent.terminate()
