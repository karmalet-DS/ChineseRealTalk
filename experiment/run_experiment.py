"""
ChineseRealTalk — 논문 5장 실험 자동화 스크립트
GPT-4o vs. Claude Sonnet: 소셜미디어 텍스트 교육화 변환 비교

사용법:
  python run_experiment.py --openai YOUR_OPENAI_KEY --claude YOUR_CLAUDE_KEY

결과: experiment_results.md (논문 5장 삽입용)
"""

import argparse, json, time, os, sys
from datetime import datetime

# ─── 실험 텍스트 15건 ───────────────────────────────────────────────────────
TEXTS = [
    # 일상·감성 (5건)
    {
        "id": "S01", "category": "일상·감성", "source": "微博",
        "text": "今天真的emo了，被领导当众diss，整个人都裂开了。yyds还是我闺蜜，一句话就把我整绷了。晚上准备躺平刷剧，求推荐好看的CP文！"
    },
    {
        "id": "S02", "category": "일상·감성", "source": "微博",
        "text": "绝绝子！今天买了新口红，颜色太美了破防了。闺蜜说我凡尔赛，但我真的就是分享一下嘛。下班打工人狗命一条，摸鱼五分钟续命。"
    },
    {
        "id": "S03", "category": "일상·감성", "source": "小红书",
        "text": "这家咖啡店真的绝了，氛围感拉满！老板超nice，给我多放了料。强烈安利！五一假期想去的宝子们记得提前预约，人太多了会排长队。"
    },
    {
        "id": "S04", "category": "일상·감성", "source": "微博",
        "text": "社恐症患者的日常：被同事拉去聚餐，整场人麻了。想躺平但又怕内卷落后，好难。下班回家感觉整个人都升华了，果然独处才是我的充电方式。"
    },
    {
        "id": "S05", "category": "일상·감성", "source": "微信",
        "text": "宝子们，今天分享一个反PUA小技巧！遇到职场杠精直接使出「我听到了，谢谢您的建议」，既不失礼貌又能保护自己的能量，这个真的学到了。"
    },
    # 사회이슈 (5건)
    {
        "id": "S06", "category": "사회이슈", "source": "微博",
        "text": "内卷严重的今天，很多年轻人选择躺平。但是，佛系生活真的能解决问题吗？还是说，这只是一种自我麻痹？欢迎评论区分享你的看法。"
    },
    {
        "id": "S07", "category": "사회이슈", "source": "微博",
        "text": "打工人打工魂，打工都是人上人。996已经成了常态，但身体才是革命的本钱啊。听说最近「反内卷」行动在各大厂发酵，究竟能走多远？"
    },
    {
        "id": "S08", "category": "사회이슈", "source": "微博",
        "text": "为什么现在的年轻人都不想生孩子了？三孩政策出来了，但房价、教育、医疗这三座大山还在。躺平的背后其实是对未来的不确定感。"
    },
    {
        "id": "S09", "category": "사회이슈", "source": "微信公众号",
        "text": "电子榨菜时代，我们已经离不开手机了。一边吃饭一边刷短视频，一边走路一边看直播。这是娱乐，还是一种新型的精神内耗？"
    },
    {
        "id": "S10", "category": "사회이슈", "source": "微博",
        "text": "00后整顿职场真的不是说说而已！实习第一天就和HR battle，要求弹性上班。这届年轻人真的太有想法了，打工人进化论。"
    },
    # 여행·음식·문화 (5건)
    {
        "id": "S11", "category": "여행·음식·문화", "source": "小红书",
        "text": "打卡完毕！这家店的糖醋里脊绝绝子，老板超nice，给我多放了好多料。氛围感拉满，墙裂推荐！五一假期排队要2小时，建议错峰出行。"
    },
    {
        "id": "S12", "category": "여행·음식·문화", "source": "小红书",
        "text": "今天去了网红咖啡店，拍了好多照片。这个配色也太高级了吧，直接整个人破防了。宝子们觉得这种滤镜风格怎么样？求评价！"
    },
    {
        "id": "S13", "category": "여행·음식·문화", "source": "微博",
        "text": "云南穷游攻略来了！人均不超过500块，住青旅、吃路边摊，收获了满满的松弛感。这才是真正的gap year精神，不需要凡尔赛，快乐就好。"
    },
    {
        "id": "S14", "category": "여행·음식·문화", "source": "小红书",
        "text": "今天get了一个超好喝的奶茶配方！直接在家复刻某茶的招牌款，成本不到十块，味道绝了。宝子们有口福了，详细配方在下面👇"
    },
    {
        "id": "S15", "category": "여행·음식·문화", "source": "微信公众号",
        "text": "国潮文化的崛起让年轻人重新爱上了汉服。今年汉服市场规模已超百亿，穿汉服去打卡景点成了新时尚。这是文化自信，还是又一个快消品风潮？"
    },
]

# ─── 프롬프트 ────────────────────────────────────────────────────────────────
ADAPT_SYS = """당신은 중국어 교육 전문가입니다. 소셜미디어 중국어 텍스트를 지정된 HSK 수준의 학습자가 이해할 수 있도록 변환합니다.
규칙:
1. 원문의 핵심 의미와 문화적 맥락을 보존하세요.
2. 신조어·속어를 HSK 수준에 맞는 표현으로 바꾸세요.
3. 아래 형식으로 출력하세요:
[변환텍스트]
(변환된 중국어 텍스트)
[변경목록]
- 원문표현 → 변환표현 : 이유"""

ADAPT_USER = "다음 중국어 소셜미디어 텍스트를 HSK {level}급 학습자 수준으로 변환해 주세요.\n[원문]\n{text}"

GLOSS_SYS = """중국어 언어학 전문가입니다. 텍스트의 어려운 표현을 JSON으로 반환하세요:
[{"표현":"","병음":"","유형":"신조어|속어|관용어|문화어","의미":"(한국어)","예문":"(중국어)","HSK등급":""}]
반드시 유효한 JSON만 반환하세요."""

GLOSS_USER = "다음 텍스트에서 HSK 4급 이상의 어려운 표현, 신조어, 속어를 찾아 주석을 달아주세요.\n[텍스트]\n{text}"

# ─── API 호출 ────────────────────────────────────────────────────────────────
def call_openai(system, user, key, model="gpt-4o"):
    from openai import OpenAI
    client = OpenAI(api_key=key)
    r = client.chat.completions.create(
        model=model,
        messages=[{"role":"system","content":system},{"role":"user","content":user}],
        temperature=0.3
    )
    return r.choices[0].message.content.strip()

def call_claude(system, user, key, model="claude-sonnet-4-5"):
    import anthropic
    client = anthropic.Anthropic(api_key=key)
    r = client.messages.create(
        model=model, max_tokens=4096,
        system=system,
        messages=[{"role":"user","content":user}]
    )
    return r.content[0].text.strip()

# ─── 메인 실험 ───────────────────────────────────────────────────────────────
def run(openai_key, claude_key, hsk_level=4, output="experiment_results.json"):
    results = []
    total = len(TEXTS)

    for i, t in enumerate(TEXTS):
        print(f"[{i+1}/{total}] {t['id']} {t['category']} — {t['source']}")
        entry = {"id": t["id"], "category": t["category"],
                 "source": t["source"], "original": t["text"],
                 "hsk_level": hsk_level}

        # GPT-4o 변환
        try:
            entry["gpt_adapt"] = call_openai(
                ADAPT_SYS, ADAPT_USER.format(level=hsk_level, text=t["text"]), openai_key)
            print(f"  ✅ GPT-4o 변환 완료")
        except Exception as e:
            entry["gpt_adapt"] = f"ERROR: {e}"
            print(f"  ❌ GPT-4o 오류: {e}")
        time.sleep(1)

        # GPT-4o 주석
        try:
            entry["gpt_gloss"] = call_openai(
                GLOSS_SYS, GLOSS_USER.format(text=t["text"]), openai_key)
            print(f"  ✅ GPT-4o 주석 완료")
        except Exception as e:
            entry["gpt_gloss"] = f"ERROR: {e}"
        time.sleep(1)

        # Claude 변환
        try:
            entry["claude_adapt"] = call_claude(
                ADAPT_SYS, ADAPT_USER.format(level=hsk_level, text=t["text"]), claude_key)
            print(f"  ✅ Claude 변환 완료")
        except Exception as e:
            entry["claude_adapt"] = f"ERROR: {e}"
            print(f"  ❌ Claude 오류: {e}")
        time.sleep(1)

        # Claude 주석
        try:
            entry["claude_gloss"] = call_claude(
                GLOSS_SYS, GLOSS_USER.format(text=t["text"]), claude_key)
            print(f"  ✅ Claude 주석 완료")
        except Exception as e:
            entry["claude_gloss"] = f"ERROR: {e}"
        time.sleep(1)

        results.append(entry)
        print()

    out_path = os.path.join(os.path.dirname(__file__), output)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 결과 저장: {out_path}")
    return results

# ─── CLI ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ChineseRealTalk 논문 실험 자동화")
    parser.add_argument("--openai", required=True, help="OpenAI API Key")
    parser.add_argument("--claude", required=True, help="Anthropic API Key")
    parser.add_argument("--hsk", type=int, default=4, help="HSK 목표 수준 (기본값: 4)")
    parser.add_argument("--output", default="experiment_results.json", help="출력 파일명")
    args = parser.parse_args()
    run(args.openai, args.claude, args.hsk, args.output)
