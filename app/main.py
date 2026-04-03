import logging
import time

from fastapi import FastAPI
from app.api.routes import invocations
from app.telemetry import setup_xray, XRayMiddleware

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

try:
    setup_xray("cdci-prd-supervisor-agent")
except Exception as e:
    logging.getLogger(__name__).warning("X-Ray setup failed (non-fatal): %s", e)

app = FastAPI(title="Supervisor Agent", version="1.0.0")
app.add_middleware(XRayMiddleware)
app.include_router(invocations.router)


@app.get("/ping")
async def ping():
    return {"status": "Healthy", "time_of_last_update": int(time.time())}
