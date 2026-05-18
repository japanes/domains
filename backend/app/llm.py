"""OpenAI integration for brand name generation.

The system prompt lives in `SYSTEM_PROMPT` (user-provided, Ukrainian).
Input variables (business_description, tlds, count, language, ...) are
injected through a structured user message. The LLM is asked to return
`{"domains":[{name_part, ..., why_this_domain, ...}, ...]}` — we extract
the unique `name_part`s and the short rationale for use as a tagline.
The backend then cross-products those name parts with the user-selected
TLDs and runs WHOIS checks against each combination.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import TypedDict

from openai import AsyncOpenAI

log = logging.getLogger("domains.llm")

_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        _client = AsyncOpenAI(api_key=api_key)
    return _client


MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

# USD per 1M tokens — defaults match the pricing the user provided.
PRICE_INPUT_PER_MTOK = float(os.environ.get("OPENAI_PRICE_INPUT", "0.40"))
PRICE_CACHED_INPUT_PER_MTOK = float(os.environ.get("OPENAI_PRICE_CACHED_INPUT", "0.10"))
PRICE_OUTPUT_PER_MTOK = float(os.environ.get("OPENAI_PRICE_OUTPUT", "1.60"))


def _cost_breakdown(usage) -> dict | None:
    """Compute USD cost from a Chat Completions usage object."""
    if usage is None:
        return None
    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0

    cached_tokens = 0
    details = getattr(usage, "prompt_tokens_details", None)
    if details is not None:
        cached_tokens = getattr(details, "cached_tokens", 0) or 0
    elif isinstance(usage, dict):
        cached_tokens = (usage.get("prompt_tokens_details") or {}).get(
            "cached_tokens", 0
        ) or 0

    fresh_input = max(prompt_tokens - cached_tokens, 0)
    cost_input = fresh_input / 1_000_000 * PRICE_INPUT_PER_MTOK
    cost_cached = cached_tokens / 1_000_000 * PRICE_CACHED_INPUT_PER_MTOK
    cost_output = completion_tokens / 1_000_000 * PRICE_OUTPUT_PER_MTOK
    total = cost_input + cost_cached + cost_output

    return {
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "fresh_input_tokens": fresh_input,
        "completion_tokens": completion_tokens,
        "cost_input_usd": cost_input,
        "cost_cached_usd": cost_cached,
        "cost_output_usd": cost_output,
        "cost_total_usd": total,
    }


_LANG_LABELS = {
    "uk": "Ukrainian (українська)",
    "ru": "Russian (русский)",
    "en": "English",
}


class NameCandidate(TypedDict):
    name: str
    tagline: str


SYSTEM_PROMPT = """You are a professional AI domain-name generator and brand strategist.

Your task is to create high-quality domain names for businesses, products, services, brands, startups, and online projects.

Domain names must:
- carry a clear meaning or association;
- match the user's business;
- read easily;
- pronounce easily;
- be easy to remember;
- look natural for the target market;
- be suitable for a real brand or company.

==================================================
INPUT
==================================================

The user may provide:

- business_description
  A description of the business, product, or idea.

- keywords
  Preferred words, themes, or associations.

- required_words
  Words that MUST appear in the domain.

- excluded_words
  Words, brands, or associations to AVOID.

- tlds
  Preferred domain zones.

- length_preference
  Desired domain length:
  short, medium, long, or any.

- style
  Preferred brand style.

- target_audience
  Target audience.

- language
  Language of the domain name.

- target_country
  Country the business targets.

- count
  Number of domains to generate.

==================================================
LENGTH RULES
==================================================

short:
3–8 characters before the dot

medium:
9–15 characters before the dot

long:
16–30 characters before the dot

any:
3–30 characters before the dot

Always respect the chosen length.

==================================================
CORE PRINCIPLE
==================================================

First understand:
- the essence of the business;
- the audience;
- the country;
- the language;
- the brand style;
- the main idea or association.

Only AFTER that, generate domains.

==================================================
LOCAL ADAPTATION
==================================================

Based on target_country and language, account for:

- the local language;
- pronunciation habits;
- reading ease;
- cultural perception of names;
- local market specifics;
- typical naming styles for that country.

The domain must feel natural specifically for the target market.

==================================================
GENERATION RULES
==================================================

1. Generate only domains that could realistically be used by a business or brand.

2. Do not produce random letter sequences.

3. Do not create names without logic or meaning.

4. Every domain must have:
- a clear association;
- internal logic;
- a tie to the business or brand style.

5. The domain must be:
- readable;
- understandable;
- natural;
- easy to pronounce;
- suitable for branding.

6. Avoid:
- awkward letter combinations;
- unnatural words;
- names that are hard to pronounce;
- names that look artificial;
- names that are hard to remember;
- names that look like random AI output.

7. If required_words are provided —
use them.

8. If excluded_words are provided —
do not use them.

9. Do not copy existing well-known brands.

10. Do not claim that a domain is available.

Always include:
"availability_check_required": true

==================================================
TLD RULES
==================================================

Use only the TLDs the user has provided.

If no TLDs were provided —
pick the most appropriate based on:
- country;
- business;
- niche;
- audience.

==================================================
QUALITY CHECK
==================================================

Before adding a domain to the result, verify:

- Is it easy to read?
- Is it easy to pronounce?
- Is it easy to remember?
- Does it carry a clear association?
- Does it match the business?
- Does it fit the country and language?
- Does it look like a real brand?
- Does it NOT look like random AI-generated noise?

If a domain is weak —
do not include it in the result.

==================================================
EXPLANATIONS
==================================================

For every domain, briefly and concretely explain:

- the meaning or association behind the name;
- why the domain suits the business;
- why it is easy to remember;
- why it works well for the target audience.

Do not use vague or generic phrases.

==================================================
RESPONSE FORMAT
==================================================

Return ONLY valid JSON.

{
  "domains": [
    {
      "domain": "glowbar.ua",
      "name_part": "glowbar",
      "tld": ".ua",
      "category": "brandable",
      "length": "short",
      "character_count_before_dot": 7,

      "meaning": "Glow evokes radiance and personal care, while bar suggests a modern space or studio.",

      "why_this_domain": "The name reads easily, is memorable, and fits a modern beauty business well.",

      "memorability_score": 9,
      "pronunciation_score": 9,
      "meaning_score": 9,
      "brandability_score": 8,
      "local_market_fit_score": 9,
      "overall_score": 8.8,

      "availability_check_required": true
    }
  ]
}

IMPORTANT: the textual fields `meaning` and `why_this_domain` must be written in the language passed in the `language` input variable — NOT in English by default.
"""


def _build_user_message(
    business_description: str,
    tlds: list[str],
    lang: str,
    count: int,
    exclude: list[str] | None = None,
) -> str:
    language = _LANG_LABELS.get(lang, "English")
    tlds_str = ", ".join(tlds) if tlds else "(none provided — pick yourself)"
    exclude = exclude or []
    exclude_str = ", ".join(exclude) if exclude else "(none)"

    msg = (
        "Input:\n\n"
        f"business_description: {business_description}\n"
        f"tlds: {tlds_str}\n"
        f"count: {count}\n"
        f"language: {language}\n"
        "target_country: Ukraine\n"
        "length_preference: any\n"
        f"excluded_words: {exclude_str}\n\n"
        f"Return EXACTLY {count} UNIQUE `name_part` values (i.e. {count} different "
        "brand bases — our service will append the TLD itself).\n"
        "Briefly explain each one in the user's language inside the "
        "`why_this_domain` / `meaning` fields.\n"
        "The response must be ONLY a valid JSON object with the `domains` key."
    )
    if exclude:
        msg += (
            f"\n\nCRITICAL: the variants below were already produced and MUST NOT "
            f"be repeated (not as `name_part`, not as a substring of it, not as "
            f"a different spelling):\n{exclude_str}\n"
            "All new `name_part` values must be fundamentally different from "
            "this list."
        )
    return msg


_NAME_RE = re.compile(r"[^a-z0-9]")


def _extract_items(text: str) -> list[dict]:
    s = text.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"```\s*$", "", s)
    try:
        data = json.loads(s)
    except json.JSONDecodeError:
        start = s.find("{")
        end = s.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("LLM response is not JSON")
        data = json.loads(s[start : end + 1])
    if isinstance(data, dict) and isinstance(data.get("domains"), list):
        return data["domains"]
    if isinstance(data, list):
        return data
    raise ValueError("LLM response does not contain a `domains` array")


def _sanitize(items: list[dict]) -> list[NameCandidate]:
    out: list[NameCandidate] = []
    seen: set[str] = set()
    for x in items:
        if not isinstance(x, dict):
            continue
        raw = (x.get("name_part") or "").strip()
        if not raw:
            dom = (x.get("domain") or "").strip()
            raw = dom.split(".")[0] if "." in dom else dom
        clean = _NAME_RE.sub("", raw.lower())[:30]
        if len(clean) < 3 or clean in seen:
            continue
        seen.add(clean)
        tagline = (
            x.get("why_this_domain")
            or x.get("meaning")
            or ""
        ).strip()
        display = clean[:1].upper() + clean[1:]
        out.append({"name": display, "tagline": tagline})
    return out


async def generate_names(
    prompt: str,
    *,
    lang: str = "uk",
    count: int = 10,
    tlds: list[str] | None = None,
    exclude: list[str] | None = None,
) -> tuple[list[NameCandidate], dict | None]:
    user_msg = _build_user_message(
        prompt.strip(), tlds or [], lang, count, exclude=exclude
    )

    log.info("=== LLM REQUEST ===\nmodel=%s\n--- user message ---\n%s", MODEL, user_msg)

    client = _get_client()
    resp = await client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.9,
        response_format={"type": "json_object"},
    )
    text = resp.choices[0].message.content or ""

    usage = getattr(resp, "usage", None)
    finish = resp.choices[0].finish_reason if resp.choices else None
    cost = _cost_breakdown(usage)
    cost_line = (
        ""
        if cost is None
        else (
            f"\ncost: ${cost['cost_total_usd']:.6f}  "
            f"(input ${cost['cost_input_usd']:.6f} for {cost['fresh_input_tokens']} tok, "
            f"cached ${cost['cost_cached_usd']:.6f} for {cost['cached_tokens']} tok, "
            f"output ${cost['cost_output_usd']:.6f} for {cost['completion_tokens']} tok)"
        )
    )
    log.info(
        "=== LLM RESPONSE ===\nfinish_reason=%s usage=%s%s\n--- raw content ---\n%s",
        finish,
        usage,
        cost_line,
        text,
    )

    items = _extract_items(text)
    return _sanitize(items), cost
