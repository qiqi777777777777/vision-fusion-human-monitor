# llm_helper.py —— 调用硅基流动大模型，自动探测可用网络路径（直连/系统代理/本地代理/IP兜底）
import os
import warnings

import requests

# 系统 DNS 解析失败时的 IP 兜底（api.siliconflow.cn 解析出的固定 IP，绕开 DNS 直连）
FALLBACK_IPS = ["139.196.152.242"]

# 本地 Ollama 模型名（已下载时优先使用，完全离线；未就绪自动回退云端）
LOCAL_MODEL = "qwen2.5:3b"


def _system_proxy():
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                            r"Software\Microsoft\Windows\CurrentVersion\Internet Settings") as k:
            if winreg.QueryValueEx(k, "ProxyEnable")[0]:
                srv = winreg.QueryValueEx(k, "ProxyServer")[0]
                if srv:
                    return srv if "://" in srv else "http://" + srv
    except Exception:
        pass
    return None


def _candidate_proxies():
    """按优先级返回要尝试的代理列表（None 表示用环境变量默认）"""
    seen = []
    for k in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
        v = os.environ.get(k)
        if v and v not in seen:
            seen.append(v)
    sp = _system_proxy()
    if sp and sp not in seen:
        seen.append(sp)
    for port in (7890, 7897, 1080, 10809, 10808, 8888):
        s = "http://127.0.0.1:%d" % port
        if s not in seen:
            seen.append(s)
    return seen


def _ask_local(prompt, max_tokens, timeout=120):
    body = {"model": LOCAL_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "stream": False}
    resp = requests.post("http://localhost:11434/v1/chat/completions",
                         json=body, timeout=timeout)
    data = resp.json()
    if "choices" in data:
        return data["choices"][0]["message"]["content"].strip()
    return None


def ask(prompt, api_key, model="Qwen/Qwen2.5-7B-Instruct", max_tokens=300, timeout=20):
    try:
        ans = _ask_local(prompt, max_tokens)
        if ans:
            print("[网络探测] 通过 本地Ollama(" + LOCAL_MODEL + ") 连接成功")
            return ans
        print("[网络探测] 本地 Ollama 返回异常，回退云端")
    except Exception as e:
        print(f"[网络探测] 本地 Ollama 不可用（{type(e).__name__}），回退云端")
    url = "https://api.siliconflow.cn/v1/chat/completions"
    headers = {"Authorization": "Bearer " + api_key,
               "Content-Type": "application/json"}
    body = {"model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens}
    last_err = None
    for px in [None] + _candidate_proxies():
        proxies = {"http": px, "https": px} if px else None
        tag = "直连" if not px else px
        try:
            resp = requests.post(url, headers=headers, json=body,
                                 timeout=timeout, proxies=proxies)
            data = resp.json()
            if "choices" in data:
                print(f"[网络探测] 通过 {tag} 连接成功")
                return data["choices"][0]["message"]["content"].strip()
            last_err = Exception(str(data))
            print(f"[网络探测] {tag} 返回异常: {data}")
        except Exception as e:
            last_err = e
            print(f"[网络探测] {tag} 失败: {type(e).__name__}: {e}")
    for ip in FALLBACK_IPS:
        tag = "IP直连 " + ip
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                resp = requests.post(
                    "https://%s/v1/chat/completions" % ip,
                    headers={"Host": "api.siliconflow.cn",
                             "Authorization": "Bearer " + api_key,
                             "Content-Type": "application/json"},
                    json=body, timeout=timeout, verify=False)
            data = resp.json()
            if "choices" in data:
                print(f"[网络探测] 通过 {tag} 连接成功")
                return data["choices"][0]["message"]["content"].strip()
            last_err = Exception(str(data))
        except Exception as e:
            last_err = e
            print(f"[网络探测] {tag} 失败: {type(e).__name__}: {e}")
    raise last_err
