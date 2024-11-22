import json
from pathlib import Path
import threading
from typing import List, Set
import sys
import os
import time

class State:
    def __init__(self):
        import websockets
        self.ui_sessions: Set[websockets.WebSocketServerProtocol] = set()
        self.backend_sessions: Set[websockets.WebSocketServerProtocol] = set()
        self.namespaces: List[int] = []
        self.lock = threading.Lock()
        self.scenario = ""
        self.latched_messages_ui = {}
        self.latched_messages_backend = {}


async def websocket_handler(request):
    from aiohttp import web
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    path = request.path
    state = request.app['state']


    try:
        is_ui = path == "/ui"

        with state.lock:
            if is_ui:
                state.ui_sessions.add(ws)
                print("UI client connected")
                for message in state.latched_messages_backend.values():
                    print(f"Sending latched message: {message["channel"]}")
                    await ws.send_str(json.dumps(message))
            else:  # backend
                state.backend_sessions.add(ws)
                namespace = len(state.namespaces)
                state.namespaces.append(namespace)
                print(f"Backend connected: {namespace}")
                handshake = {
                    "channel": "handshake",
                    "data": {
                        "namespace": str(namespace)
                    }
                }
                await ws.send_json(handshake)
                for message in state.latched_messages_ui.values():
                    print(f"Sending latched message: {message["channel"]}")
                    await ws.send_str(json.dumps(message))


        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                try:
                    parsed = json.loads(msg.data)
                    if("latch" in parsed and parsed["latch"]):
                        print(f"Saving latched message: {parsed["channel"]}")
                        if is_ui:
                            state.latched_messages_ui[parsed["channel"]] = parsed
                        else:
                            state.latched_messages_backend[parsed["channel"]] = parsed

                    with state.lock:
                        if is_ui:
                            for backend in state.backend_sessions:
                                try:
                                    await backend.send_str(msg.data)
                                except Exception:
                                    continue
                        else:
                            # print(f"Received message: {msg.data}")
                            for ui in state.ui_sessions:
                                try:
                                    await ui.send_str(msg.data)
                                except Exception:
                                    continue
                except json.JSONDecodeError:
                    print(f"Invalid JSON message received: {msg.data}")
                    continue
            elif msg.type == web.WSMsgType.ERROR:
                print(f'WebSocket connection closed with exception {ws.exception()}')

    finally:
        with state.lock:
            if is_ui:
                state.ui_sessions.remove(ws)
                print("UI client disconnected")
            else:
                state.backend_sessions.remove(ws)
                print("Backend client disconnected")

    return ws

async def handle_static(request):
    from aiohttp import web
    static_path = request.app['static_path']
    path = request.path

    if path == "/" or path == "":
        path = "/index.html"

    file_path = Path(static_path) / path.lstrip('/')

    if not file_path.exists():
        return web.Response(text="File not found", status=404)

    content_type = 'text/plain'
    if file_path.suffix == '.html':
        content_type = 'text/html'
    elif file_path.suffix == '.js':
        content_type = 'application/javascript'
    elif file_path.suffix == '.css':
        content_type = 'text/css'
    elif file_path.suffix == '.wasm':
        content_type = 'application/wasm'

    return web.FileResponse(file_path, headers={'Content-Type': content_type})

async def handle_scenario(request):
    from aiohttp import web
    state = request.app['state']
    return web.Response(text=state.scenario, content_type='text/html')

def start_server(static_path = "./static", port = 8080, scenario = ""):
    from aiohttp import web
    state = State()
    state.scenario = scenario

    app = web.Application()
    app['state'] = state
    app['static_path'] = static_path

    app.router.add_get('/ui', websocket_handler)
    app.router.add_get('/backend', websocket_handler)
    app.router.add_get('/scenario', handle_scenario)
    app.router.add_get('/{tail:.*}', handle_static)

    print(f"Server starting at http://localhost:{port}")
    print(f"WebSocket endpoints at ws://localhost:{port}/ui and ws://localhost:{port}/backend")

    web.run_app(app, port=port)

def start_server_in_background(port = 8080, scenario = "", open_browser = True):
    from multiprocessing import Process

    def target():
        current_file_path = os.path.abspath(__file__)
        current_dir = os.path.dirname(current_file_path)
        static_path = os.path.join(current_dir, "..", "external", "rl-tools", "static", "ui_server", "generic")
        # sys.stdout = None
        # sys.stderr = None
        start_server(static_path = static_path, port = port, scenario = scenario)
    process = Process(target=target)
    process.start()
    print(f"ui_server should be running in a background process. Check http://localhost:{port}")
    if open_browser:
        import webbrowser
        webbrowser.open(f"http://localhost:{port}")

    return process



if __name__ == "__main__":
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)
    static_path = os.path.join(current_dir, "..", "external", "rl-tools", "static", "ui_server", "generic")
    start_server(static_path=static_path, port=8080, scenario="generic")
