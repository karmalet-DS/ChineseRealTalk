# 🀄 ChineseRealTalk

> **소셜미디어 중국어 텍스트를 LLM으로 교육 자료로 변환하는 인터랙티브 플랫폼**

논문 연계 프로젝트:  
**"살아있는 중국어 교재를 만드는 LLM — 소셜미디어 텍스트의 교육적 변환과 활용"**  
박민준 (덕성여자대학교) | *AI와 중국어 교육* (2026)

---

## 주요 기능

### 📝 텍스트 교육화 변환기
- 微博·小红书·微信 등 소셜미디어 원문을 HSK 수준별로 자동 변환
- 변경된 표현과 이유를 목록으로 제공
- 결과 다운로드 (.txt)

### 🔍 신조어·어려운 표현 자동 주석기
- 신조어·속어·관용어·문화어 자동 감지 및 카드형 주석 생성
- 병음, 의미, 예문, HSK 등급 표시
- 단어장 내보내기 (.csv) → Anki 플래시카드 활용 가능

### 📋 읽기 후 활동 생성기
- 이해 확인 문항 (객관식·단답형·O/X·서술형)
- 토론 질문 및 핵심 어휘 목록 자동 생성
- 수업 자료 다운로드 (.txt)

---

## 실행 방법

### 1. 패키지 설치

```bash
pip install -r requirements.txt
```

### 2. 앱 실행

```bash
streamlit run app.py
```

### 3. API Key 입력

앱 실행 후 사이드바에서 OpenAI 또는 Anthropic API Key를 입력하세요.

---

## 지원 모델

| 제공자 | 모델 |
|--------|------|
| OpenAI | gpt-4o, gpt-4o-mini, gpt-4-turbo |
| Anthropic | claude-opus-4-5, claude-sonnet-4-5, claude-haiku-4-5 |

---

## 프로젝트 구조

```
ChineseRealTalk/
├── app.py              # 메인 Streamlit 앱
├── requirements.txt
├── .env.example        # API Key 설정 예시
└── utils/
    ├── llm.py          # OpenAI / Claude 공통 호출 인터페이스
    └── prompts.py      # 기능별 프롬프트 템플릿
```

---

## 관련 연구

- 박민준 (2025). *From Plain to Hierarchical: Knowledge-Augmented Prompting for Chinese Factivity Inference*. CCL25-Eval.
- 박정구·유수경·박민준·강병규 (2024). *XAI 기반의 중국어 문두 다중논항 어순 해석 모델 연구*. 中國言語硏究.
- 박민준 (2026). *中韓網絡仇恨言論比較*. Chinese Language and Discourse, 17(1).
