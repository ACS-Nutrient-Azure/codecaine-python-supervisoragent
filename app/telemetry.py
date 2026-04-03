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
            logger.info("X-Ray segment sent: %s", entity.id)
        except Exception as exc:
            logger.warning("X-Ray put_trace_segments failed: %s", exc)

    def set_daemon_address(self, address):
        pass  # daemon 없음, API 직접 호출


def _parse_trace_header(header: str) -> dict:
    """X-Amzn-Trace-Id 헤더 파싱 → {trace_id, parent_id, sampling}"""
    result = {}
    for part in header.split(";"):
        part = part.strip()
        if part.startswith("Root="):
            result["trace_id"] = part[5:]
        elif part.startswith("Parent="):
            result["parent_id"] = part[7:]
        elif part.startswith("Sampled="):
            result["sampling"] = part[8:]
    return result


class XRayMiddleware(BaseHTTPMiddleware):
    """인바운드 HTTP 요청마다 X-Ray segment 생성/종료.
    X-Amzn-Trace-Id 헤더가 있으면 상위 trace와 연결."""

    async def dispatch(self, request: Request, call_next):
        trace_header = request.headers.get("X-Amzn-Trace-Id", "")
        parsed = _parse_trace_header(trace_header) if trace_header else {}

        segment_name = xray_recorder._service or request.url.path
        segment = xray_recorder.begin_segment(
            segment_name,
            traceid=parsed.get("trace_id"),
            parent_id=parsed.get("parent_id"),
            sampling=int(parsed["sampling"]) if parsed.get("sampling") else 1,
        )
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
