"""
supervisor_agent.py

LangGraph StateGraph 기반 Supervisor Agent 핵심 로직.

전체 흐름:
  Chatbot Backend → [route] → [call_analysis | call_question | call_both] → [summarize] → 응답

노드 설명:
  - route        : LLM이 chat_history를 보고 어떤 에이전트가 필요한지 판단
                   ("analysis" | "question" | "both" 중 하나 출력)
  - call_analysis: chat_history에서 현재 상태 키워드 추출 후 Analysis Backend API 호출
  - call_question: Question Agent ARN으로 AgentCore invoke 호출
  - call_both    : call_analysis + call_question을 asyncio.gather로 병렬 실행
                   한쪽 실패해도 성공한 결과만 summarize로 전달
  - summarize    : Summary Agent ARN으로 AgentCore invoke 호출 → 최종 응답 생성
"""
import asyncio
import json
import logging
import time
import uuid
from typing import TypedDict

import boto3
import httpx
from aws_requests_auth.boto_utils import BotoAWSRequestsAuth
from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from app.core.config import settings
from app.schemas.agent import SupervisorRequest, SupervisorResponse

logger = logging.getLogger(__name__)


def _arn_to_agent_name(agent_arn: str) -> str:
    """ARN → 에이전트 이름 매핑."""
    if agent_arn == settings.QUESTION_AGENT_ARN:
        return "cdci-prd-question-agent"
    if agent_arn == settings.SUMMARY_AGENT_ARN:
        return "cdci-prd-summary-agent"
    return "cdci-prd-agent"


def _send_xray_agent_call(start_time: float, end_time: float, agent_name: str, success: bool) -> None:
    """supervisor → sub-agent 호출을 X-Ray에 기록 (boto3 직접 전송)."""
    try:
        trace_id = f"1-{int(start_time):08x}-{uuid.uuid4().hex[:24]}"
        segment = {
            "id": uuid.uuid4().hex[:16],
            "name": "cdci-prd-supervisor-agent",
            "trace_id": trace_id,
            "start_time": start_time,
            "end_time": end_time,
            "fault": not success,
            "origin": "AWS::ECS::Fargate",
            "subsegments": [{
                "id": uuid.uuid4().hex[:16],
                "name": agent_name,
                "start_time": start_time,
                "end_time": end_time,
                "namespace": "remote",
                "fault": not success,
            }],
        }
        boto3.client("xray", region_name=settings.AWS_REGION).put_trace_segments(
            TraceSegmentDocuments=[json.dumps(segment)]
        )
    except Exception as exc:
        logger.warning("X-Ray agent call segment send failed: %s", exc)


# LLM에게 라우팅 판단 기준을 안내하는 시스템 프롬프트
ROUTE_SYSTEM_PROMPT = """당신은 사용자 질문을 분석하여 어떤 에이전트가 필요한지 판단합니다.

- "analysis": 영양소 분석, 영양제 추천, 개인 맞춤 분석 관련.
  다음 표현도 모두 포함됨: "다시 분석", "재분석", "목적 바꿔서", "목적이 바뀌었는데", "~로 바꿔서 분석", "~로 추천해줘"
- "question": 영양소 정보, 의약품 정보, 건강 상식 등 일반 질문, 복용 중인 영양제와의 상호작용, 최신 건강 정보, 사용자 신체 정보 기반 답변 등
- "both": 두 성격 모두 포함

반드시 "analysis", "question", "both" 중 하나만 출력하세요. 다른 텍스트 없이."""


class SupervisorState(TypedDict):
    """LangGraph StateGraph 전체에서 공유되는 상태 스키마."""
    cognito_id: str
    chat_result_id: int
    codef_health_data: dict | None
    codef_medication_info: list[dict] | None
    chat_history: str
    route: str                  # route 노드에서 결정: "analysis" | "question" | "both"
    analysis_result: dict | None  # call_analysis 노드 결과
    question_result: dict | None  # call_question 노드 결과
    final_response: str | None    # summarize 노드 결과


class SupervisorAgent:
    def __init__(self):
        # 로컬 테스트 모드: OpenAI 사용
        self.llm = None
        self.session = None
        self.memory_client = None
        # StateGraph 빌드 및 컴파일 (인스턴스 생성 시 1회)
        self.graph = self._build_graph()

    def _get_llm(self):
        """LLM lazy 초기화 - 첫 요청 시점에 자격증명이 주입된 후 실행"""
        if self.llm is not None:
            return self.llm
        if settings.USE_LOCAL_TEST:
            from langchain_openai import ChatOpenAI
            self.llm = ChatOpenAI(
                model=settings.OPENAI_MODEL_ID,
                api_key=settings.OPENAI_API_KEY,
                temperature=0.7,
            )
            # Gemini 사용 시 아래로 교체
            # from langchain_google_genai import ChatGoogleGenerativeAI
            # self.llm = ChatGoogleGenerativeAI(
            #     model=settings.GEMINI_MODEL_ID,
            #     google_api_key=settings.GEMINI_API_KEY,
            #     temperature=0.7,
            # )
        # 실제 배포: Bedrock 사용
        else:
            self.session = boto3.Session(region_name=settings.AWS_REGION)
            self.llm = ChatBedrock(
                client=self.session.client("bedrock-runtime"),
                model_id=settings.BEDROCK_MODEL_ID,
            )
        
        # AgentCore Memory 초기화
        if settings.USE_MEMORY and self.memory_client is None:
            self.memory_client = boto3.client("bedrock-agentcore", region_name=settings.AWS_REGION)
        return self.llm

    def _build_graph(self) -> StateGraph:
        """
        StateGraph 노드와 엣지를 정의하고 컴파일.

        엣지 구조:
          route → (조건부) → call_analysis / call_question / call_both
          call_* → summarize → END
        """
        g = StateGraph(SupervisorState)

        # 노드 등록
        g.add_node("route_decision", self._node_route)
        g.add_node("call_analysis", self._node_call_analysis)
        g.add_node("call_question", self._node_call_question)
        g.add_node("call_both", self._node_call_both)
        g.add_node("summarize", self._node_summarize)

        # 진입점
        g.set_entry_point("route_decision")

        # route 노드 결과값(state["route"])에 따라 분기
        g.add_conditional_edges(
            "route_decision",
            lambda s: s["route"],
            {
                "analysis": "call_analysis",
                "question": "call_question",
                "both": "call_both",
            },
        )

        # 각 call_* 노드 → summarize → 종료
        g.add_edge("call_analysis", "summarize")
        g.add_edge("call_question", END)
        g.add_edge("call_both", "summarize")
        g.add_edge("summarize", END)

        return g.compile()

    # ── 노드 구현 ─────────────────────────────────────────────────

    @staticmethod
    def _keyword_route(text: str) -> str | None:
        """명확한 재분석/분석 요청 키워드는 LLM 없이 바로 판단."""
        analysis_keywords = ["재분석", "다시 분석", "분석해줘", "분석 해줘", "목적이 바뀌", "목적을 바꿔", "목적 바꿔", "추천 목적"]
        if any(kw in text for kw in analysis_keywords):
            return "analysis"
        return None

    async def _node_route(self, state: SupervisorState) -> dict:
        """
        LLM이 chat_history 전체 문맥을 보고 라우팅 결정.
        LLM 응답이 예상 값이 아닌 경우 "question"으로 fallback.
        """
        # 키워드 기반 사전 판단 (LLM 호출 전)
        keyword_result = self._keyword_route(state["chat_history"])
        if keyword_result:
            logger.info(f"[{state['cognito_id']}] route → {keyword_result} (keyword match)")
            return {"route": keyword_result}

        # AgentCore Memory에서 이전 대화 불러오기
        memory_context = ""
        if self.memory_client and settings.MEMORY_ID:
            try:
                resp = self.memory_client.list_events(
                    memoryId=settings.MEMORY_ID,
                    actorId=state["cognito_id"],
                    sessionId=str(state["chat_result_id"]),
                    maxResults=10
                )
                conversations = resp.get("events", [])
                if conversations:
                    memory_context = "\n\n[이전 대화 맥락]\n" + self._format_memory(conversations)
                    print(f"[SUPERVISOR] Loaded {len(conversations)} previous messages from memory")
            except Exception as e:
                print(f"[SUPERVISOR] Memory load failed: {e}")
        
        messages = [
            SystemMessage(content=ROUTE_SYSTEM_PROMPT),
            HumanMessage(content=state["chat_history"] + memory_context),
        ]
        result = await self._get_llm().ainvoke(messages)
        route = result.content.strip().lower()
        if route not in ("analysis", "question", "both"):
            route = "question"  # 예상치 못한 응답 시 안전하게 question으로
        print(f"\n[SUPERVISOR] Route decision: {route}")
        logger.info(f"[{state['cognito_id']}] route → {route}")
        return {"route": route}

    async def _node_call_analysis(self, state: SupervisorState) -> dict:
        """Analysis Backend API만 호출."""
        try:
            result = await self._call_analysis(state)
            logger.info(f"[{state['cognito_id']}] analysis result keys: {list(result.keys()) if isinstance(result, dict) else type(result)}")
            return {"analysis_result": result}
        except Exception as e:
            logger.error(f"[{state['cognito_id']}] _call_analysis failed: {type(e).__name__}: {e}")
            raise

    async def _node_call_question(self, state: SupervisorState) -> dict:
        """Question Agent만 호출. 결과를 final_response에 바로 세팅."""
        print(f"\n[SUPERVISOR] Calling Question Agent...")
        result = await self._call_question(state)
        print(f"[SUPERVISOR] Question result: {result}")
        return {"question_result": result, "final_response": result.get("answer", "")}

    async def _node_call_both(self, state: SupervisorState) -> dict:
        """
        Analysis + Question을 병렬 호출.
        return_exceptions=True로 한쪽 실패해도 나머지 결과는 유지.
        """
        print(f"\n[SUPERVISOR] Calling both Analysis + Question...")
        analysis_res, question_res = await asyncio.gather(
            self._call_analysis(state),
            self._call_question(state),
            return_exceptions=True,
        )
        print(f"[SUPERVISOR] Analysis result: {analysis_res}")
        print(f"[SUPERVISOR] Question result: {question_res}")
        return {
            # Exception이면 None으로 처리 — Summary가 있는 결과만 사용
            "analysis_result": None if isinstance(analysis_res, Exception) else analysis_res,
            "question_result": None if isinstance(question_res, Exception) else question_res,
        }

    async def _node_summarize(self, state: SupervisorState) -> dict:
        """Summary Agent를 호출해 최종 응답 생성."""
        print(f"\n[SUPERVISOR] Calling Summary Agent...")
        print(f"  - analysis_result keys: {list(state.get('analysis_result', {}).keys()) if state.get('analysis_result') else None}")
        print(f"  - question_result: {state.get('question_result')}")

        # analysis_result가 None이면 summarize 의미 없음 — 에러 응답
        if not state.get("analysis_result") and not state.get("question_result"):
            logger.error(f"[{state['cognito_id']}] summarize called with no results")
            return {"final_response": "분석 결과를 가져오지 못했습니다. 잠시 후 다시 시도해주세요."}

        payload = {
            "cognito_id": state["cognito_id"],
            "analysis_result": state.get("analysis_result"),
            "question_result": {"answer": state["question_result"]["answer"]} if state.get("question_result") else None,
        }
        result = await self._invoke_agent(settings.SUMMARY_AGENT_ARN, payload)
        print(f"[SUPERVISOR] Summary result: {result}")
        return {"final_response": result.get("response", "")}

    # ── 외부 호출 헬퍼 ────────────────────────────────────────────

    async def _call_analysis(self, state: SupervisorState) -> dict:
        """
        Analysis Backend API 호출.
        - LLM으로 chat_history에서 현재 상태를 자유롭게 추출해 current_conditions 필드 추가
        - IAM Role SigV4 서명으로 인증 (BotoAWSRequestsAuth가 자동 처리)
        """
        # LLM으로 현재 상태 추출
        extract_messages = [
            SystemMessage(content=(
                "사용자의 대화에서 현재 건강 상태나 증상을 추출하세요. "
                "예: 피로, 스트레스, 불면, 두통, 소화불량 등. "
                "추출된 상태를 쉼표로 구분해서만 출력하세요. 없으면 '없음'만 출력하세요."
            )),
            HumanMessage(content=state["chat_history"]),
        ]
        extract_result = await self._get_llm().ainvoke(extract_messages)
        current_conditions = extract_result.content.strip()
        if current_conditions == "없음":
            current_conditions = None

        payload = {
            "cognito_id": state["cognito_id"],
            "result_id": state["chat_result_id"],
            "chat_history": json.dumps([{"role": "user", "content": state["chat_history"]}], ensure_ascii=False),
            "current_conditions": current_conditions,
        }
        url = settings.ANALYSIS_BACKEND_URL.rstrip("/") + "/api/analysis/chat-calculate"
        
        # Cloud Map DNS를 통한 VPC 내부 직접 호출 (인증 불필요)
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, json=payload, timeout=180.0)
            resp.raise_for_status()
            return resp.json()

    async def _call_question(self, state: SupervisorState) -> dict:
        """
        Question Agent AgentCore invoke 호출.
        Supervisor 입력 원본을 그대로 전달 (현재 상태 파싱 없음).
        """
        payload = {
            "cognito_id": state["cognito_id"],
            "chat_result_id": state["chat_result_id"],
            "codef_health_data": state["codef_health_data"],
            "codef_medication_info": state["codef_medication_info"],
            "chat_history": state["chat_history"],
        }
        return await self._invoke_agent(settings.QUESTION_AGENT_ARN, payload)

    async def _invoke_agent(self, agent_arn: str, payload: dict) -> dict:
        """
        AgentCore invoke_agent_runtime() 호출 공통 헬퍼.
        로컬 테스트: HTTP 호출
        실제 배포: boto3 AgentCore 호출
        """
        # 로컬 테스트: HTTP URL
        if agent_arn.startswith("http"):
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    agent_arn + "/invocations",
                    json=payload,
                    timeout=60.0
                )
                response.raise_for_status()
                return response.json()

        # 실제 배포: AgentCore
        loop = asyncio.get_event_loop()
        call_start = time.time()
        success = False

        def _invoke():
            from botocore.config import Config
            c = boto3.client(
                "bedrock-agentcore",
                region_name=settings.AWS_REGION,
                config=Config(read_timeout=300, connect_timeout=10),
            )
            try:
                resp = c.invoke_agent_runtime(
                    agentRuntimeArn=agent_arn,
                    payload=json.dumps(payload, ensure_ascii=False),
                )
                raw = resp["response"].read()
                return json.loads(raw)
            except Exception as e:
                logger.error(f"[INVOKE_AGENT] arn={agent_arn[:40]} error={type(e).__name__}: {e}")
                raise

        try:
            result = await loop.run_in_executor(None, _invoke)
            success = True
            return result
        except Exception:
            raise
        finally:
            call_end = time.time()
            agent_name = _arn_to_agent_name(agent_arn)
            _send_xray_agent_call(call_start, call_end, agent_name, success)

    async def run(self, req: SupervisorRequest) -> SupervisorResponse:
        """StateGraph 실행 진입점. 초기 State를 구성하고 graph.ainvoke 호출."""
        state: SupervisorState = {
            "cognito_id": req.cognito_id,
            "chat_result_id": req.chat_result_id,
            "codef_health_data": req.codef_health_data,
            "codef_medication_info": req.codef_medication_info,
            "chat_history": req.chat_history,
            "route": "",
            "analysis_result": None,
            "question_result": None,
            "final_response": None,
        }
        result = await self.graph.ainvoke(state)
        
        # AgentCore Memory에 대화 저장
        if self.memory_client and settings.MEMORY_ID:
            try:
                self.memory_client.create_event(
                    memoryId=settings.MEMORY_ID,
                    actorId=req.cognito_id,
                    sessionId=str(req.chat_result_id),
                    messages=[
                        {"role": "USER", "content": req.chat_history},
                        {"role": "ASSISTANT", "content": result.get("final_response", "")}
                    ]
                )
                print(f"[SUPERVISOR] Saved conversation to memory")
            except Exception as e:
                print(f"[SUPERVISOR] Memory save failed: {e}")
        
        return SupervisorResponse(
            cognito_id=req.cognito_id,
            response=result.get("final_response") or "",
        )
    
    def _format_memory(self, conversations: list) -> str:
        """AgentCore Memory 대화를 텍스트로 변환"""
        lines = []
        for conv in conversations:
            messages = conv.get("messages", [])
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "USER":
                    lines.append(f"사용자: {content}")
                elif role == "ASSISTANT":
                    lines.append(f"AI: {content}")
        return "\n".join(lines[-20:])  # 최근 20개만
