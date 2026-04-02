import logging
import time

from fastapi import FastAPI
from app.api.routes import invocations

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

app = FastAPI(title="Supervisor Agent", version="1.0.0")
app.include_router(invocations.router)


@app.get("/ping")
async def ping():
    return {"status": "Healthy", "time_of_last_update": int(time.time())}
