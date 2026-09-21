"""FastAPI server and WebSocket hub for THE SPECIMEN."""

import asyncio
import base64
import io
import time
from pathlib import Path
from typing import List, Optional

import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Body
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from somnia.utils import project_root
from specimen.organism import SpecimenOrganism

app = FastAPI(title="THE SPECIMEN", description="A Living AI Organism on the Web")

# Static assets directory
STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Singleton living organism
organism = SpecimenOrganism()


class ConnectionManager:
    """Manages active WebSocket visitors and broadcasts real-time telemetry."""

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        organism.visitor_count = len(self.active_connections)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            organism.visitor_count = len(self.active_connections)

    async def broadcast_state(self):
        """Send state snapshot to all connected clients."""
        if not self.active_connections:
            return
        state = organism.get_state()
        dead = []
        for connection in self.active_connections:
            try:
                await connection.send_json(state)
            except Exception:
                dead.append(connection)
        for d in dead:
            self.disconnect(d)


manager = ConnectionManager()


@app.on_event("startup")
async def start_background_broadcast():
    """Start background periodic telemetry heartbeat every 2 seconds."""
    async def periodic_broadcast():
        while True:
            await asyncio.sleep(2.0)
            await manager.broadcast_state()

    asyncio.create_task(periodic_broadcast())


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        return HTMLResponse("<h1>THE SPECIMEN is booting up...</h1>")
    return FileResponse(str(index_path))


@app.get("/state")
async def get_state():
    """Return full JSON-serializable snapshot of organism state."""
    return organism.get_state()


@app.get("/history")
async def get_history():
    """Return memory buffer and event history."""
    return {
        "memory_buffer": [
            {
                "id": m["id"],
                "pred": m["pred"],
                "conf": m["conf"],
                "p_error": m["p_error"],
                "true_label": m["true_label"],
                "b64": m["b64"],
                "timestamp": m["timestamp"],
            }
            for m in list(organism.memory_buffer)[-50:]
        ],
        "event_log": list(organism.event_log),
    }


class FeedRequest(BaseModel):
    image: Optional[List[float]] = None
    image_b64: Optional[str] = None


@app.post("/feed")
async def feed_organism(req: FeedRequest):
    """Feed an image (784 floats or base64) into the organism."""
    if req.image is not None and len(req.image) == 784:
        img_arr = np.array(req.image, dtype=np.float32).reshape(28, 28)
    elif req.image_b64 is not None:
        try:
            # Strip data URL header if present
            raw_b64 = req.image_b64
            if "," in raw_b64:
                raw_b64 = raw_b64.split(",", 1)[1]
            img_bytes = base64.b64decode(raw_b64)
            pil_img = Image.open(io.BytesIO(img_bytes)).convert("L").resize((28, 28))
            img_arr = np.array(pil_img, dtype=np.float32) / 255.0
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 image: {e}")
    else:
        raise HTTPException(status_code=400, detail="Must provide 'image' (784 floats) or 'image_b64'")

    result = organism.feed(img_arr)
    # Broadcast immediate update
    await manager.broadcast_state()
    return result


class RevealRequest(BaseModel):
    label: int


@app.post("/reveal")
async def reveal_label(req: RevealRequest):
    """Reveal ground truth label for the most recent sample."""
    result = organism.reveal(req.label)
    await manager.broadcast_state()
    return result


@app.post("/sleep")
async def trigger_sleep():
    """Trigger manual dream consolidation cycle."""
    if organism.is_sleeping:
        return {"status": "already_sleeping"}

    # Run sleep cycle in background thread or task so server stays responsive
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, organism.sleep_cycle)
    await manager.broadcast_state()
    return result


@app.get("/preset/{name}")
async def get_preset(name: str):
    """Get sample preset images (clean_3, ambiguous_8, noise, blank)."""
    rng = np.random.RandomState(42)
    if name == "clean_3":
        # Generate clean 3 via cVAE
        with torch.no_grad():
            dream, _ = organism.cvae.sample_class(1, target_class=3, seed=42)
        arr = dream[0].numpy()
    elif name == "ambiguous_8":
        # Mixed latent code
        z = torch.randn(1, 32)
        with torch.no_grad():
            dream = organism.cvae.decode(z, torch.tensor([8]))
        arr = dream[0].numpy()
    elif name == "noise":
        arr = rng.uniform(0.0, 0.8, size=(784,)).astype(np.float32)
    elif name == "blank":
        arr = np.zeros((784,), dtype=np.float32)
    else:
        raise HTTPException(status_code=404, detail="Unknown preset name")

    return {
        "name": name,
        "image": arr.tolist(),
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket connection for real-time live telemetry stream."""
    await manager.connect(websocket)
    try:
        # Send immediate initial state
        await websocket.send_json(organism.get_state())
        while True:
            # Keep connection alive, listen for client pings or actions
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception:
        manager.disconnect(websocket)
