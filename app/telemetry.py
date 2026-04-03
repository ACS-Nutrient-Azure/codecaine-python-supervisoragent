import logging
import boto3
from aws_xray_sdk.core import xray_recorder, patch_all
from aws_xray_sdk.core.emitters.udp_emitter import UDPEmitter
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

logger = logging.getLogger(__name__)


class _BotoXRayEmitter(UDPEmitter):
    """X-Ray daemon 없이 boto3 API로 직접 트레이스 전송."""

    def __init__(self, region: str = "ap-northeast-2"):
        self._region = region
        self._client = None

    def _get_client(self):
        if self._client is None:
            self._client = boto3.client("xray", region_name=self._region)
        return self._client

    def send_entity(self, entity) -> None:
        try:
            self._get_client().put_trace_segments(
                TraceSegmentDocuments=[entity.serialize()]
            )
        except Exception as exc:
            logger.debug("X-Ray put_trace_segments failed: %s", exc)

    def set_daemon_address(self, address):
        pass  # daemon 없음, API 직접 호출


class XRayMiddleware(BaseHTTPMiddleware):
    """인바운드 HTTP 요청마다 X-Ray segment 생성/종료."""

    async def dispatch(self, request: Request, call_next):
        segment = xray_recorder.begin_segment(request.url.path)
        try:
            segment.put_http_meta("request", {
                "method": request.method,
                "url": str(request.url),
            })
            response = await call_next(request)
            segment.put_http_meta("response", {"status": response.status_code})
            return response
        except Exception as e:
            segment.add_exception(e, fatal=True)
            raise
        finally:
            xray_recorder.end_segment()


def setup_xray(service_name: str, region: str = "ap-northeast-2") -> None:
    xray_recorder.configure(
        service=service_name,
        context_missing="LOG_ERROR",
        emitter=_BotoXRayEmitter(region),
    )
    patch_all()
