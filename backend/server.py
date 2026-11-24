"""标准库 HTTP 服务：任务 API + 前端静态托管。

API:
  GET  /api/presets              预设场景与默认参数
  POST /api/tasks                创建任务 {scenario, n_total, ...}
  GET  /api/tasks                任务列表
  GET  /api/tasks/{id}           任务详情（含结果、路径、权重样本）
  POST /api/tasks/{id}/start     启动/恢复
  POST /api/tasks/{id}/cancel    请求取消（分片边界生效）
  POST /api/exact                对任意场景配置做精确转移计算
"""

from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .exact import exact_overflow
from .models import build_scenario
from .presets import PRESETS
from .store import TaskStore

HERE = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.normpath(os.path.join(HERE, "..", "frontend"))
DATA_DIR = os.environ.get("RAREEVENT_DATA_DIR",
                         os.path.normpath(os.path.join(HERE, "..", "data", "tasks")))

STORE_LOCK = threading.Lock()
STORE = TaskStore(DATA_DIR)

_MIME = {".html": "text/html; charset=utf-8",
         ".js": "text/javascript; charset=utf-8",
         ".css": "text/css; charset=utf-8",
         ".svg": "image/svg+xml",
         ".json": "application/json; charset=utf-8"}


class Handler(BaseHTTPRequestHandler):
    server_version = "rareevent/1.0"

    def log_message(self, fmt, *args):  # 安静一点
        pass

    def _json(self, payload, status=200):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, msg, status=400):
        self._json({"error": msg}, status)

    def _read_body(self):
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/api/presets":
            self._json({pid: {k: v for k, v in p.items() if k != "config"}
                        | {"config": p["config"]} for pid, p in PRESETS.items()})
            return
        if path == "/api/tasks":
            self._json({"tasks": STORE.list_tasks()})
            return
        if path.startswith("/api/tasks/"):
            tid = path.split("/")[3]
            try:
                self._json(STORE.get_task(tid))
            except FileNotFoundError:
                self._error("任务不存在", 404)
            return
        self._serve_static(path)

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        try:
            body = self._read_body()
        except json.JSONDecodeError:
            self._error("请求体不是合法 JSON")
            return
        try:
            if path == "/api/tasks":
                with STORE_LOCK:
                    doc = STORE.create_task(body)
                STORE.start(doc["id"])
                self._json(doc, 201)
                return
            if path == "/api/exact":
                scenario = build_scenario(body["scenario"])
                tilt = float(body.get("tilt", 1.0))
                self._json(exact_overflow(scenario, tilt=tilt))
                return
            parts = path.strip("/").split("/")
            if len(parts) == 4 and parts[:2] == ["api", "tasks"] \
                    and parts[3] == "start":
                STORE.start(parts[2])
                self._json({"ok": True})
                return
            if len(parts) == 4 and parts[:2] == ["api", "tasks"] \
                    and parts[3] == "cancel":
                STORE.cancel(parts[2])
                self._json({"ok": True})
                return
        except (KeyError, ValueError) as exc:
            self._error(f"参数错误: {exc}", 400)
            return
        self._error("未知接口", 404)

    # ---------------------------------------------------------- 静态
    def _serve_static(self, path: str):
        if path == "/":
            path = "/index.html"
        target = os.path.normpath(os.path.join(FRONTEND_DIR, path.lstrip("/")))
        if not target.startswith(FRONTEND_DIR) or not os.path.isfile(target):
            self._error("not found", 404)
            return
        ext = os.path.splitext(target)[1]
        with open(target, "rb") as f:
            body = f.read()
        self.send_response(200)
        self.send_header("Content-Type",
                         _MIME.get(ext, "application/octet-stream"))
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main(host: str = "127.0.0.1", port: int = 8765):
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"罕见事件分析应用运行于 http://{host}:{port}  数据目录: {DATA_DIR}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.shutdown()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="罕见事件分析应用")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    main(args.host, args.port)
