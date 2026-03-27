# Supervisor Agent

멀티 에이전트 시스템의 오케스트레이터 (LangGraph StateGraph 기반)

## 개요

Chatbot Backend의 진입점으로, 사용자 질문을 분석하여 필요한 Sub-Agent를 자동으로 선택하고 호출합니다. LangGraph StateGraph를 사용하여 복잡한 라우팅 로직을 구현했습니다.

**주요 역할**:
- 사용자 질문 분석 및 라우팅 판단
- Analysis Backend / Question Agent 호출
- 결과 통합 및 Summary Agent 호출
- 최종 응답 생성

## 디렉토리 구조

```
codecaine-python-supervisoragent/
├── app/
│   ├── __init__.py
│   ├── main.py                          # FastAPI 애플리케이션 진입점
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes/
│   │       └── invocations.py           # POST /invocations 엔드포인트
│   ├── core/
│   │   ├── __init__.py
│   │   └── config.py                    # 환경변수 설정 (Pydantic Settings)
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── agent.py                     # Request/Response 스키마
│   └── services/
│       ├── __init__.py
│       └── supervisor_agent.py          # LangGraph StateGraph 핵심 로직
├── Dockerfile                           # Docker 이미지 빌드 설정
├── requirements.txt                     # Python 패키지 목록
├── env.example                          # 환경변수 예시
└── README.md                            # 이 문서
```

## 주요 파일 설명

### `app/services/supervisor_agent.py`
- **SupervisorAgent 클래스**: LangGraph StateGraph 구현
- **5개 노드**:
  1. `route_decision`: LLM이 라우팅 판단 (analysis/question/both)
  2. `call_analysis`: Analysis Backend API 호출
  3. `call_question`: Question Agent 호출
  4. `call_both`: Analysis + Question 병렬 호출
  5. `summarize`: Summary Agent 호출
- **LLM 분기 처리**: 로컬 테스트(OpenAI) vs AWS 배포(Bedrock)
- **AgentCore Memory 통합**: 대화 맥락 기반 라우팅

### `app/api/routes/invocations.py`
- POST `/invocations` 엔드포인트
- Chatbot Backend에서 호출
- Request 검증 및 Response 반환

### `app/core/config.py`
- Pydantic Settings 기반 환경변수 관리
- AWS, Agent ARN, Backend URL 설정

## 전체 흐름

```
Chatbot Backend
    └─→ POST /invocations
            └─→ SupervisorAgent.run()
                    └─→ [route_decision] LLM이 라우팅 판단
                            ├─→ "analysis" → [call_analysis] Analysis Backend API 호출
                            ├─→ "question" → [call_question] Question Agent 호출
                            └─→ "both"     → [call_both] 위 두 개 병렬 호출
                    └─→ [summarize] Summary Agent 호출
                    └─→ 최종 응답 반환
```

## LangGraph StateGraph 구조

```
[route_decision] → (analysis) → [call_analysis] ─┐
                 → (question) → [call_question] ─┤→ [summarize] → END
                 → (both)     → [call_both]     ─┘
```

### 노드 설명

**route_decision**:
- LLM이 chat_history 분석
- "analysis" / "question" / "both" 중 하나 선택
- AgentCore Memory에서 이전 대화 참조

**call_analysis**:
- LLM으로 현재 건강 상태 추출 (current_conditions)
- Analysis Backend API 호출 (SigV4 서명)
- 영양소 분석 결과 반환

**call_question**:
- Question Agent ARN 호출 (boto3 bedrock-agentcore-runtime)
- 일반 질문 답변 반환

**call_both**:
- call_analysis + call_question 병렬 실행 (asyncio.gather)
- 한쪽 실패해도 성공한 결과만 전달

**summarize**:
- Summary Agent ARN 호출
- 분석 결과 + 질문 답변 통합
- 최종 한국어 응답 생성

## API 명세

### POST /invocations

Chatbot Backend에서 호출하는 메인 엔드포인트

**Request Body:**
```json
{
  "cognito_id": "user-123",
  "chat_result_id": 1,
  "codef_health_data": {
    "height": 170,
    "weight": 65
  },
  "codef_medication_info": [
    {"name": "아스피린"}
  ],
  "chat_history": "피로 개선 영양제 추천해줘"
}
```

**Response:**
```json
{
  "cognito_id": "user-123",
  "response": "[섭취 목적] 피로 개선\n[필요 영양소] 비타민 B군..."
}
```

## 환경변수 설정

### 필수 환경변수 (AWS 배포)

```bash
# AWS 설정
AWS_REGION=ap-northeast-2
BEDROCK_MODEL_ID=anthropic.claude-3-haiku-20240307-v1:0

# Sub-Agent ARN
ANALYSIS_BACKEND_URL=https://api.example.com
QUESTION_AGENT_ARN=arn:aws:bedrock:ap-northeast-2:123456789012:agent-runtime/XXXXX
SUMMARY_AGENT_ARN=arn:aws:bedrock:ap-northeast-2:123456789012:agent-runtime/XXXXX

# AgentCore Memory
USE_MEMORY=true
MEMORY_ID=<Memory Store ID>
```

### 로컬 테스트용 환경변수

```bash
USE_LOCAL_TEST=true
OPENAI_API_KEY=<OpenAI API Key>
OPENAI_MODEL_ID=gpt-4o-mini
ANALYSIS_BACKEND_URL=http://localhost:8004
QUESTION_AGENT_ARN=http://localhost:8003
SUMMARY_AGENT_ARN=http://localhost:8002
USE_MEMORY=false
```

## 로컬 실행

### 1. 환경 설정

```bash
# 가상환경 생성
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 패키지 설치
pip install -r requirements.txt

# 환경변수 설정
cp env.example .env
# .env 파일 편집
```

### 2. Sub-Agent 서버 실행 (로컬 테스트 시)

```bash
# Terminal 1: Mock Analysis Backend
cd ../reference/agentstest
python mock_analysis_backend.py  # port 8004

# Terminal 2: Question Agent
cd ../codecaine-python-chatbotagent
uvicorn app.main:app --reload --port 8003

# Terminal 3: Summary Agent
cd ../codecaine-python-summaryagent
uvicorn app.main:app --reload --port 8002
```

### 3. Supervisor 서버 실행

```bash
uvicorn app.main:app --reload --port 8001
```

### 4. 테스트

```bash
curl -X POST http://localhost:8001/invocations \
  -H "Content-Type: application/json" \
  -d '{
    "cognito_id": "test-user",
    "chat_result_id": 1,
    "chat_history": "영양제 추천해줘"
  }'
```

## Docker 빌드 및 실행

```bash
# 이미지 빌드
docker build -t supervisor-agent .

# 컨테이너 실행
docker run -p 8001:8000 --env-file .env supervisor-agent
```

## AWS 배포

### 배포 순서 (중요!)

Supervisor는 다른 Agent의 ARN이 필요하므로 **마지막에 배포**:

1. **Question Agent 배포** → ARN 복사
2. **Summary Agent 배포** → ARN 복사
3. **Supervisor Agent 배포** (ARN 환경변수 설정)

### 1. ECR 푸시

```bash
# ECR 로그인
aws ecr get-login-password --region ap-northeast-2 | \
  docker login --username AWS --password-stdin <account-id>.dkr.ecr.ap-northeast-2.amazonaws.com

# 이미지 태그 및 푸시
docker tag supervisor-agent:latest <account-id>.dkr.ecr.ap-northeast-2.amazonaws.com/supervisor-agent:latest
docker push <account-id>.dkr.ecr.ap-northeast-2.amazonaws.com/supervisor-agent:latest
```

### 2. AgentCore Runtime 등록

AWS 콘솔 또는 Terraform으로 AgentCore Runtime에 등록

### 3. 환경변수 설정

```bash
USE_LOCAL_TEST=false
AWS_REGION=ap-northeast-2
BEDROCK_MODEL_ID=anthropic.claude-3-haiku-20240307-v1:0
ANALYSIS_BACKEND_URL=<Analysis Backend URL>
QUESTION_AGENT_ARN=<Question Agent ARN>
SUMMARY_AGENT_ARN=<Summary Agent ARN>
USE_MEMORY=true
MEMORY_ID=<생성한 Memory ID>
```

### 4. IAM 권한 설정

Agent 실행 Role에 다음 권한 필요:
- `bedrock:InvokeModel`
- `bedrock:InvokeAgent`
- `bedrock:*Memory*`
- `execute-api:Invoke` (Analysis Backend)

## 기술 스택

- **Framework**: FastAPI, LangGraph
- **LLM**: AWS Bedrock (Claude 3 Haiku) / OpenAI (로컬)
- **Orchestration**: LangGraph StateGraph
- **Memory**: AWS AgentCore Memory
- **Authentication**: AWS SigV4 (Analysis Backend)

## 주요 기능

### 1. 지능형 라우팅
LLM이 질문 의도를 분석하여 자동 라우팅:
- "영양제 추천" → analysis
- "비타민 C 효능" → question
- "피로 개선 영양제 추천하고 효능도 알려줘" → both

### 2. 병렬 처리
`call_both` 노드에서 Analysis + Question 동시 실행:
```python
analysis_res, question_res = await asyncio.gather(
    self._call_analysis(state),
    self._call_question(state),
    return_exceptions=True
)
```

### 3. 대화 맥락 기반 라우팅
AgentCore Memory를 통해 이전 대화 참조:
```
사용자: 영양제 추천해줘
AI: [분석 결과]
사용자: 다시 분석해줘  ← "다시" 인식하여 analysis 라우팅
```

## 문제 해결

### Sub-Agent 호출 실패
```bash
# ARN 확인
echo $QUESTION_AGENT_ARN
echo $SUMMARY_AGENT_ARN

# IAM 권한 확인
aws iam get-role-policy --role-name <role-name> --policy-name <policy-name>
```

### Analysis Backend 연결 실패
```bash
# URL 확인
curl $ANALYSIS_BACKEND_URL/health

# SigV4 서명 확인 (IAM Role 필요)
```

### Memory 에러
```bash
# Memory Store 생성 확인
aws bedrock list-memories --region ap-northeast-2
```
