import logging
import time

from fastapi import FastAPI
from aws_xray_sdk.core import xray_recorder
from aws_xray_sdk.ext.fastapi.middleware import XRayMiddleware
from app.api.routes import invocations
from app.telemetry import setup_xray

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

setup_xray("cdci-prd-supervisor-agent")

app = FastAPI(title="Supervisor Agent", version="1.0.0")
app.add_middleware(XRayMiddleware, recorder=xray_recorder)
app.include_router(invocations.router)


@app.get("/ping")
async def ping():
    return {"status": "Healthy", "time_of_last_update": int(time.time())}
