import asyncio
import json
import base64
import io
import time
import websockets
import threading


class WebSocketServer:
    """Handles real-time frame streaming and input reception via WebSockets."""

    def __init__(self, session, port):
        self.session = session
        self.port = port
        self.thread = None

    def start(self):
        self.thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self.thread.start()

    def _run_event_loop(self):
        try:
            asyncio.run(self._serve())
        except KeyboardInterrupt:
            pass

    async def _serve(self):
        async with websockets.serve(self._handler, "0.0.0.0", self.port + 100):
            await asyncio.Future()  # run forever

    async def _handler(self, websocket):
        try:
            await asyncio.gather(
                self._send_frames(websocket), self._receive_inputs(websocket)
            )
        except websockets.exceptions.ConnectionClosed:
            pass

    async def _send_frames(self, websocket):
        target_fps = 30
        frame_interval = 1.0 / target_fps
        last_sent_time = time.time()
        last_tick_sent = -1

        while True:
            try:
                current_time = time.time()
                current_tick = self.session.total_ticks

                # 1. Skip encoding if the game is paused OR if we are ahead of the target FPS
                if (
                    current_tick == last_tick_sent
                    or (current_time - last_sent_time) < frame_interval
                ):
                    await asyncio.sleep(0.005)  # Yield control back to the async loop
                    continue

                # 2. Grab the absolute latest frame (implicitly dropping missed ticks)
                with self.session.lock:
                    img = self.session.emulator.get_screenshot()

                # FIX: Convert RGBA to RGB so JPEG encoding doesn't crash
                img = img.convert("RGB")

                # 3. Fast encoding (JPEG is significantly faster than PNG)
                width, height = img.size
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=85)
                b64 = base64.b64encode(buf.getvalue()).decode()

                # 4. Broadcast the frame
                await websocket.send(
                    json.dumps(
                        {
                            "type": "frame",
                            "data": b64,
                            "width": width,
                            "height": height,
                            "tick": current_tick,
                        }
                    )
                )

                # 5. Update our trackers
                last_sent_time = time.time()
                last_tick_sent = current_tick

            except websockets.exceptions.ConnectionClosed:
                break  # Exit gracefully if the dashboard is refreshed or closed
            except Exception:
                await asyncio.sleep(0.1)

    async def _receive_inputs(self, websocket):
        async for message in websocket:
            data = json.loads(message)
            self.session.send_input(data["key"], duration=5)
