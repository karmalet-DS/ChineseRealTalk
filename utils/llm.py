"""
LLM 호출 유틸리티 — OpenAI / Anthropic 공통 인터페이스
"""
from openai import OpenAI
import anthropic


def call_openai(system_prompt: str, user_prompt: str, api_key: str, model: str = "gpt-4o") -> str:
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


def call_claude(system_prompt: str, user_prompt: str, api_key: str, model: str = "claude-sonnet-4-5") -> str:
    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=model,
        max_tokens=4096,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return message.content[0].text.strip()


def call_llm(system_prompt: str, user_prompt: str, provider: str, api_key: str, model: str) -> str:
    if provider == "OpenAI":
        return call_openai(system_prompt, user_prompt, api_key, model)
    else:
        return call_claude(system_prompt, user_prompt, api_key, model)
