"""
ChineseRealTalk — 소셜미디어 중국어 텍스트 교육화 플랫폼
박민준 교수 (덕성여자대학교) × 김박사

논문: "살아있는 중국어 교재를 만드는 LLM —
       소셜미디어 텍스트의 교육적 변환과 활용"
"""

import os
import json
import streamlit as st
from dotenv import load_dotenv
from utils.llm import call_llm

# .env 파일 로드 (있으면)
load_dotenv()
from utils.prompts import (
    ADAPT_SYSTEM, ADAPT_USER,
    GLOSS_SYSTEM, GLOSS_USER,
    QUIZ_SYSTEM, QUIZ_USER,
)

# ─────────────────────────────────────────────
# 페이지 설정
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ChineseRealTalk",
    page_icon="🀄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CSS — 한국/중국어 폰트 및 카드 스타일
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&family=Noto+Sans+SC:wght@300;400;500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Noto Sans KR', 'Noto Sans SC', sans-serif;
}
.main-title {
    font-size: 2.2rem;
    font-weight: 700;
    color: #C0392B;
    letter-spacing: -1px;
}
.sub-title {
    font-size: 0.95rem;
    color: #7f8c8d;
    margin-bottom: 1.5rem;
}
.card {
    background: #fafafa;
    border: 1px solid #e0e0e0;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
}
.card-red {
    border-left: 4px solid #C0392B;
}
.card-blue {
    border-left: 4px solid #2980B9;
}
.card-green {
    border-left: 4px solid #27AE60;
}
.tag {
    display: inline-block;
    background: #FDEDEC;
    color: #C0392B;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 0.8rem;
    margin: 2px;
}
.tag-blue {
    background: #EBF5FB;
    color: #2980B9;
}
.tag-green {
    background: #EAFAF1;
    color: #27AE60;
}
.diff-changed {
    background-color: #FDEBD0;
    border-radius: 3px;
    padding: 1px 4px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 사이드바 — API 설정
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🀄 ChineseRealTalk")
    st.markdown("소셜미디어 중국어 → 교육 자료 변환 플랫폼")
    st.divider()

    st.markdown("### ⚙️ API 설정")
    provider = st.selectbox("LLM 제공자", ["Anthropic (Claude)", "OpenAI"])

    if provider == "OpenAI":
        _env_key = os.getenv("OPENAI_API_KEY", "")
        if _env_key:
            st.success("✅ API 키: .env에서 자동 로드됨")
            api_key = _env_key
        else:
            api_key = st.text_input(
                "OpenAI API Key",
                type="password",
                placeholder="sk-...",
                help=".env 파일에 OPENAI_API_KEY를 추가하면 자동 로드됩니다."
            )
        model = st.selectbox("모델", ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"])
    else:
        _env_key = os.getenv("ANTHROPIC_API_KEY", "")
        if _env_key:
            st.success("✅ API 키: .env에서 자동 로드됨")
            api_key = _env_key
        else:
            api_key = st.text_input(
                "Anthropic API Key",
                type="password",
                placeholder="sk-ant-...",
                help=".env 파일에 ANTHROPIC_API_KEY를 추가하면 자동 로드됩니다."
            )
        model = st.selectbox("모델", [
            "claude-sonnet-4-5",
            "claude-opus-4-5",
            "claude-haiku-4-5",
        ])

    st.divider()
    st.markdown("### 📚 HSK 수준 설정")
    hsk_level = st.slider("목표 HSK 수준", min_value=1, max_value=6, value=4,
                          help="변환·주석·문항 생성 시 기준이 되는 학습자 HSK 수준")

    st.divider()
    st.markdown("""
    **논문 연계 프로젝트**
    박민준 교수 (덕성여대)
    """)
    st.caption("AI와 중국어 교육 (2026)")

# ─────────────────────────────────────────────
# 헤더
# ─────────────────────────────────────────────
st.markdown('<p class="main-title">🀄 ChineseRealTalk</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-title">소셜미디어 중국어 텍스트를 LLM으로 교육 자료로 변환하는 인터랙티브 플랫폼</p>',
    unsafe_allow_html=True
)

if not api_key:
    st.info("👈 왼쪽 사이드바에서 API Key를 먼저 입력해 주세요.")
    st.stop()

# ─────────────────────────────────────────────
# 예시 텍스트
# ─────────────────────────────────────────────
EXAMPLE_TEXTS = {
    "微博 — 일상/감성": "今天真的emo了，被领导当众diss，整个人都裂开了。yyds还是我闺蜜，一句话就把我整绷了。晚上准备躺平刷剧，求推荐好看的CP文！",
    "小红书 — 여행/음식": "打卡完毕！这家店的糖醋里脊绝绝子，老板超nice，给我多放了好多料。氛围感拉满，墙裂推荐！五一假期排队要2小时，建议错峰出行。",
    "微信公众号 — 사회이슈": "内卷严重的今天，很多年轻人选择躺平。但是，佛系生活真的能解决问题吗？还是说，这只是一种自我麻痹？欢迎在评论区分享你的看法。",
    "직접 입력": "",
}

# ─────────────────────────────────────────────
# 탭 구성
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📝 수준별 텍스트 변환기",
    "🔍 신조어·어려운 표현 주석기",
    "📋 이해도 점검 문항 생성기",
])


# ══════════════════════════════════════════════
# TAB 1 — 텍스트 교육화 변환기
# ══════════════════════════════════════════════
with tab1:
    st.markdown("### 📝 수준별 텍스트 변환기")
    st.markdown(
        f"소셜미디어 원문을 **HSK {hsk_level}급** 학습자가 이해할 수 있는 수준으로 변환합니다. "
        "변경된 표현은 이유와 함께 목록으로 제공됩니다."
    )

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("#### 원문 입력")
        example_choice = st.selectbox("예시 텍스트 선택", list(EXAMPLE_TEXTS.keys()), key="adapt_example")
        
        if example_choice == "직접 입력":
            input_text = st.text_area(
                "중국어 소셜미디어 텍스트 붙여넣기",
                height=200,
                placeholder="微博, 小红书, 微信 등의 텍스트를 입력하세요...",
                key="adapt_input"
            )
        else:
            input_text = st.text_area(
                "중국어 소셜미디어 텍스트",
                value=EXAMPLE_TEXTS[example_choice],
                height=200,
                key="adapt_input2"
            )

        run_adapt = st.button("🔄 변환 시작", type="primary", use_container_width=True, key="btn_adapt")

    with col2:
        st.markdown("#### 변환 결과")
        if run_adapt and input_text.strip():
            with st.spinner(f"HSK {hsk_level}급 수준으로 변환 중..."):
                try:
                    result = call_llm(
                        ADAPT_SYSTEM,
                        ADAPT_USER.format(hsk_level=hsk_level, text=input_text),
                        provider,
                        api_key,
                        model,
                    )
                    st.markdown(
                        f'<div class="card card-red">{result.replace(chr(10), "<br>")}</div>',
                        unsafe_allow_html=True
                    )
                    st.download_button(
                        "⬇️ 결과 다운로드 (.txt)",
                        data=f"[원문]\n{input_text}\n\n[HSK {hsk_level}급 변환]\n{result}",
                        file_name=f"ChineseRealTalk_변환_HSK{hsk_level}.txt",
                        mime="text/plain",
                    )
                except Exception as e:
                    st.error(f"오류 발생: {e}")
        elif run_adapt:
            st.warning("텍스트를 입력해 주세요.")
        else:
            st.markdown(
                '<div class="card" style="color:#aaa; text-align:center; padding:4rem 1rem;">'
                '변환 결과가 여기에 표시됩니다</div>',
                unsafe_allow_html=True
            )


# ══════════════════════════════════════════════
# TAB 2 — 신조어·어려운 표현 주석기
# ══════════════════════════════════════════════
with tab2:
    st.markdown("### 🔍 신조어·어려운 표현 자동 주석기")
    st.markdown(
        f"텍스트 내 **신조어·속어·관용어·문화어**를 자동으로 찾아 "
        f"**HSK {hsk_level}급** 이상의 어려운 표현에 상세 주석을 생성합니다."
    )

    example_choice2 = st.selectbox("예시 텍스트 선택", list(EXAMPLE_TEXTS.keys()), key="gloss_example")
    if example_choice2 == "직접 입력":
        gloss_input = st.text_area(
            "주석을 달 중국어 텍스트",
            height=150,
            placeholder="텍스트를 입력하세요...",
            key="gloss_input"
        )
    else:
        gloss_input = st.text_area(
            "주석을 달 중국어 텍스트",
            value=EXAMPLE_TEXTS[example_choice2],
            height=150,
            key="gloss_input2"
        )

    run_gloss = st.button("🔍 주석 생성", type="primary", use_container_width=False, key="btn_gloss")

    if run_gloss and gloss_input.strip():
        with st.spinner("어려운 표현을 분석 중..."):
            try:
                raw = call_llm(
                    GLOSS_SYSTEM,
                    GLOSS_USER.format(hsk_level=hsk_level, text=gloss_input),
                    provider,
                    api_key,
                    model,
                )
                # JSON 파싱
                try:
                    # 마크다운 코드블록 제거
                    clean = raw.strip()
                    if clean.startswith("```"):
                        clean = "\n".join(clean.split("\n")[1:])
                    if clean.endswith("```"):
                        clean = "\n".join(clean.split("\n")[:-1])
                    items = json.loads(clean)

                    # ── 필드명 정규화 ──────────────────────────────
                    # LLM이 '병音', 'pinyin', 'Pinyin' 등 다양한 표기를
                    # 반환할 수 있으므로 표준 키로 통일시킴
                    FIELD_MAP = {
                        # 표현
                        "表现": "표현", "词语": "표현", "word": "표현", "expression": "표현",
                        # 병음
                        "병音": "병음", "pinyin": "병음", "Pinyin": "병음",
                        "拼音": "병음", "병인": "병음",
                        # 유형
                        "type": "유형", "类型": "유형", "분류": "유형",
                        # 의미
                        "meaning": "의미", "意义": "의미", "뜻": "의미", "설명": "의미",
                        # 예문
                        "example": "예문", "例句": "예문", "예시": "예문",
                        # HSK등급
                        "HSK": "HSK등급", "hsk": "HSK등급", "hsk_level": "HSK등급",
                        "level": "HSK등급", "등급": "HSK등급", "HSK级别": "HSK등급",
                    }

                    def normalize_item(d: dict) -> dict:
                        return {FIELD_MAP.get(k, k): v for k, v in d.items()}

                    items = [normalize_item(item) for item in items]
                    # ───────────────────────────────────────────────

                    type_colors = {
                        "신조어": "tag",
                        "속어": "tag",
                        "관용어": "tag-blue",
                        "문법": "tag-green",
                        "문화어": "tag-blue",
                    }

                    for i, item in enumerate(items):
                        tag_class = type_colors.get(item.get("유형", ""), "tag")
                        hsk_info = item.get("HSK등급", "해당 없음")
                        hsk_badge = f"HSK {hsk_info}" if hsk_info not in ["해당 없음", ""] else "HSK 외"

                        st.markdown(f"""
<div class="card card-blue">
<strong style="font-size:1.1rem">{item.get('표현', '')}</strong>
&nbsp; <em style="color:#7f8c8d">{item.get('병음', '')}</em>
&nbsp; <span class="tag {tag_class}">{item.get('유형', '')}</span>
&nbsp; <span class="tag tag-green">{hsk_badge}</span>
<br><br>
📖 <strong>의미:</strong> {item.get('의미', '')}<br>
✏️ <strong>예문:</strong> {item.get('예문', '')}
</div>
""", unsafe_allow_html=True)

                    # CSV 다운로드
                    import csv, io
                    output = io.StringIO()
                    FIELDNAMES = ["표현", "병음", "유형", "의미", "예문", "HSK등급"]
                    writer = csv.DictWriter(
                        output,
                        fieldnames=FIELDNAMES,
                        extrasaction="ignore",   # 정규화 후 남은 미지 키 무시
                        restval="",              # 누락 필드는 빈 문자열
                    )
                    writer.writeheader()
                    writer.writerows(items)
                    st.download_button(
                        "⬇️ 단어장 내보내기 (.csv)",
                        data=output.getvalue().encode("utf-8-sig"),
                        file_name="ChineseRealTalk_단어장.csv",
                        mime="text/csv",
                    )

                except json.JSONDecodeError:
                    # JSON 파싱 실패 시 원문 표시
                    st.markdown(
                        f'<div class="card card-blue">{raw.replace(chr(10), "<br>")}</div>',
                        unsafe_allow_html=True
                    )
            except Exception as e:
                st.error(f"오류 발생: {e}")
    elif run_gloss:
        st.warning("텍스트를 입력해 주세요.")


# ══════════════════════════════════════════════
# TAB 3 — 읽기 후 활동 생성기
# ══════════════════════════════════════════════
with tab3:
    st.markdown("### 📋 이해도 점검 문항 생성기")
    st.markdown(
        f"텍스트 기반으로 **HSK {hsk_level}급** 학습자용 이해 확인 문항, 토론 질문, "
        "핵심 어휘 목록을 자동으로 생성합니다."
    )

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        example_choice3 = st.selectbox("예시 텍스트 선택", list(EXAMPLE_TEXTS.keys()), key="quiz_example")
        if example_choice3 == "직접 입력":
            quiz_input = st.text_area(
                "기반 텍스트 입력",
                height=180,
                placeholder="텍스트를 입력하세요...",
                key="quiz_input"
            )
        else:
            quiz_input = st.text_area(
                "기반 텍스트",
                value=EXAMPLE_TEXTS[example_choice3],
                height=180,
                key="quiz_input2"
            )

        st.markdown("#### 문항 설정")
        quiz_types = st.multiselect(
            "문항 유형",
            ["객관식", "단답형", "O/X", "서술형"],
            default=["객관식", "단답형"],
        )
        q_count = st.number_input("이해 확인 문항 수", min_value=1, max_value=10, value=5)
        run_quiz = st.button("📋 활동 생성", type="primary", use_container_width=True, key="btn_quiz")

    with col_right:
        st.markdown("#### 생성 결과")
        if run_quiz and quiz_input.strip():
            with st.spinner("읽기 후 활동을 설계 중..."):
                try:
                    types_str = "·".join(quiz_types) if quiz_types else "객관식"
                    result = call_llm(
                        QUIZ_SYSTEM,
                        QUIZ_USER.format(
                            hsk_level=hsk_level,
                            quiz_types=types_str,
                            q_count=q_count,
                            text=quiz_input,
                        ),
                        provider,
                        api_key,
                        model,
                    )
                    st.markdown(
                        f'<div class="card card-green">{result.replace(chr(10), "<br>")}</div>',
                        unsafe_allow_html=True
                    )
                    st.download_button(
                        "⬇️ 수업 자료 다운로드 (.txt)",
                        data=f"[텍스트]\n{quiz_input}\n\n[HSK {hsk_level}급 읽기 후 활동]\n{result}",
                        file_name=f"ChineseRealTalk_활동지_HSK{hsk_level}.txt",
                        mime="text/plain",
                    )
                except Exception as e:
                    st.error(f"오류 발생: {e}")
        elif run_quiz:
            st.warning("텍스트를 입력해 주세요.")
        else:
            st.markdown(
                '<div class="card" style="color:#aaa; text-align:center; padding:4rem 1rem;">'
                '생성된 활동지가 여기에 표시됩니다</div>',
                unsafe_allow_html=True
            )

# ─────────────────────────────────────────────
# 푸터
# ─────────────────────────────────────────────
st.divider()
st.caption(
    "ChineseRealTalk v0.1 · 박민준 교수 (덕성여자대학교) · "
    "논문: '살아있는 중국어 교재를 만드는 LLM' (AI와 중국어 교육, 2026)"
)
