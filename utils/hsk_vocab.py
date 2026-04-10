"""
HSK 어휘부 로더 및 텍스트 부합도 분석기 (v2)
─────────────────────────────────────────────
drkameleon/complete-hsk-vocabulary 저장소의 complete.json 을
로드하여 단어 → 급수 태그 인덱스를 구축한다.

지원 체계:
  - "old"    : HSK 2.0 (6급 체계, 2010~)    태그: old-1 ~ old-6
  - "new"    : HSK 3.0 초판 (9급 체계)       태그: new-1 ~ new-7
  - "newest" : HSK 3.0 최신판                태그: newest-1 ~ newest-7
  (new-7 / newest-7 은 7-9급 통합)

저장소 위치 우선순위:
  1) 환경변수 HSK_VOCAB_PATH
  2) 기본값: <project_root>/data/complete-hsk-vocabulary
"""
import json
import os
from functools import lru_cache
from pathlib import Path

import jieba

# ─────────────────────────────────────────────
# 경로 설정
# ─────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_VOCAB_ROOT = _PROJECT_ROOT / "data" / "complete-hsk-vocabulary"
VOCAB_ROOT = Path(os.environ.get("HSK_VOCAB_PATH", _DEFAULT_VOCAB_ROOT))

SYSTEMS = ("old", "new", "newest")
MAX_LEVEL = {"old": 6, "new": 7, "newest": 7}


# ─────────────────────────────────────────────
# 인덱스 로더
# ─────────────────────────────────────────────
@lru_cache(maxsize=1)
def load_index() -> dict:
    """
    complete.json을 로드하여 {simplified: [level tags]} 딕셔너리를 반환한다.

    예) "花"    → ["newest-2", "new-1", "old-3"]
        "怎么办" → ["newest-3", "new-2"]
    """
    path = VOCAB_ROOT / "complete.json"
    if not path.exists():
        raise FileNotFoundError(
            f"HSK 어휘부를 찾을 수 없습니다: {path}\n"
            f"complete-hsk-vocabulary 저장소를 '{VOCAB_ROOT}'에 배치하거나,\n"
            f"HSK_VOCAB_PATH 환경변수로 실제 경로를 지정하세요."
        )
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    idx = {}
    for entry in data:
        if not isinstance(entry, dict):
            continue
        simp = entry.get("simplified") or entry.get("s")
        levels = entry.get("level") or entry.get("l") or []
        if simp:
            idx[simp] = list(levels)
    return idx


def vocab_available() -> bool:
    """complete.json 이 로드 가능한지 확인."""
    return (VOCAB_ROOT / "complete.json").exists()


# ─────────────────────────────────────────────
# 단어 수준 조회
# ─────────────────────────────────────────────
def _parse_tag(tag: str):
    """'old-3' → ('old', 3). 숫자로 끝나지 않으면 None."""
    if not isinstance(tag, str) or "-" not in tag:
        return None
    prefix, _, num = tag.rpartition("-")
    if not num.isdigit():
        return None
    return prefix, int(num)


def get_level_tags(word: str) -> list:
    """단어의 모든 수준 태그 리스트. 없으면 []"""
    return load_index().get(word, [])


def get_level_in_system(word: str, system: str):
    """
    지정 체계에서 해당 단어의 최소(가장 낮은) 급수를 반환. 없으면 None.
    """
    tags = get_level_tags(word)
    nums = [p[1] for t in tags if (p := _parse_tag(t)) and p[0] == system]
    return min(nums) if nums else None


def format_levels(tags) -> str:
    """
    태그 리스트를 'old-3 / new-1 / newest-2' 형태로 포맷.
    어떤 체계에도 속하지 않으면 '—' 반환.
    """
    if not tags:
        return "—"
    out = []
    for sys in SYSTEMS:
        nums = [p[1] for t in tags if (p := _parse_tag(t)) and p[0] == sys]
        if nums:
            out.append(f"{sys}-{min(nums)}")
    return " / ".join(out) if out else "—"


# ─────────────────────────────────────────────
# 텍스트 분석
# ─────────────────────────────────────────────
def _is_chinese(token: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in token)


def analyze_text(text: str, target_level: int, system: str = "old") -> dict:
    """
    텍스트를 jieba 로 분절하여 각 토큰의 HSK 급수를 wordlist 에서 조회한다.

    :return: {
        "total":          전체 유니크 중국어 토큰 수,
        "in_level_count": 지정 체계·급수 이내 토큰 수,
        "coverage":       비율 (0.0 ~ 1.0),
        "per_word": [
            {
                "word": str,
                "tags": [...],          # 원본 태그 리스트
                "formatted": str,       # 'old-3 / new-1 / newest-2'
                "sys_level": int|None,  # 지정 체계에서의 급수
                "in_level": bool,       # sys_level <= target_level 여부
            }, ...
        ]
    }
    """
    idx = load_index()
    tokens = [t for t in jieba.lcut(text) if _is_chinese(t)]

    # 유니크 + 입력 순서 보존
    seen = set()
    unique_ordered = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            unique_ordered.append(t)

    per_word = []
    in_cnt = 0
    for w in unique_ordered:
        tags = idx.get(w, [])
        sys_level = None
        for t in tags:
            p = _parse_tag(t)
            if p and p[0] == system:
                sys_level = p[1] if sys_level is None else min(sys_level, p[1])
        in_level = sys_level is not None and sys_level <= target_level
        if in_level:
            in_cnt += 1
        per_word.append({
            "word": w,
            "tags": tags,
            "formatted": format_levels(tags),
            "sys_level": sys_level,
            "in_level": in_level,
        })

    total = len(per_word)
    return {
        "total": total,
        "in_level_count": in_cnt,
        "coverage": in_cnt / total if total else 1.0,
        "per_word": per_word,
    }
