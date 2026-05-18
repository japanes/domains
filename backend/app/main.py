from __future__ import annotations

import logging
import os
import re
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from .llm import generate_names
from .whois_check import check_bulk, check_one

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("domains")

MAX_REGEN_ROUNDS = int(os.environ.get("MAX_REGEN_ROUNDS", "3"))

app = FastAPI(title="Domain generator")

_DOMAIN_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?(?:\.[a-z0-9-]+)+$")
_TLD_RE = re.compile(r"^\.[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?$")


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=3, max_length=4000)
    tlds: list[str] = Field(min_length=1, max_length=20)
    lang: Literal["uk", "ru", "en"] = "uk"
    count: int = Field(default=10, ge=1, le=20)

    @field_validator("tlds")
    @classmethod
    def _check_tlds(cls, v: list[str]) -> list[str]:
        cleaned: list[str] = []
        for t in v:
            t = t.strip().lower()
            if not t.startswith("."):
                t = "." + t
            if not _TLD_RE.match(t):
                raise ValueError(f"invalid tld: {t!r}")
            cleaned.append(t)
        # preserve order, drop duplicates
        seen: set[str] = set()
        out: list[str] = []
        for t in cleaned:
            if t not in seen:
                seen.add(t)
                out.append(t)
        return out


class DomainResult(BaseModel):
    tld: str
    domain: str
    status: Literal["available", "taken", "unknown"]


class NameResult(BaseModel):
    name: str
    tagline: str
    domains: list[DomainResult]


class CostInfo(BaseModel):
    total_usd: float
    input_tokens: int
    cached_tokens: int
    output_tokens: int
    rounds: int


class GenerateResponse(BaseModel):
    names: list[NameResult]
    cost: CostInfo


class CheckRequest(BaseModel):
    domains: list[str] = Field(min_length=1, max_length=200)

    @field_validator("domains")
    @classmethod
    def _check_domains(cls, v: list[str]) -> list[str]:
        out: list[str] = []
        for d in v:
            d = d.strip().lower()
            if not _DOMAIN_RE.match(d):
                raise ValueError(f"invalid domain: {d!r}")
            out.append(d)
        return out


@app.get("/api/health")
async def health() -> dict:
    return {"ok": True}


@app.post("/api/check")
async def check(req: CheckRequest) -> dict:
    results = await check_bulk(req.domains)
    return {
        "results": [
            {
                "domain": d,
                "status": results.get(d, {}).get("status", "unknown"),
            }
            for d in req.domains
        ]
    }


def _build_name_result(
    candidate: dict,
    tlds: list[str],
    availability: dict,
) -> NameResult:
    lower = candidate["name"].lower()
    domains_out: list[DomainResult] = []
    for tld in tlds:
        d = lower + tld
        entry = availability.get(d, {})
        domains_out.append(
            DomainResult(
                tld=tld,
                domain=d,
                status=entry.get("status", "unknown"),
            )
        )
    return NameResult(
        name=candidate["name"],
        tagline=candidate["tagline"],
        domains=domains_out,
    )


@app.post("/api/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest) -> GenerateResponse:
    accepted: list[NameResult] = []
    # Names already shown to the LLM as excluded — includes both accepted ones
    # and ones rejected because they were fully taken. We never want the LLM
    # to suggest the same `name_part` twice within a single request.
    seen_names: set[str] = set()

    total_cost_usd = 0.0
    total_input_tokens = 0
    total_cached_tokens = 0
    total_output_tokens = 0
    rounds_used = 0

    for round_idx in range(MAX_REGEN_ROUNDS):
        needed = req.count - len(accepted)
        if needed <= 0:
            break

        # On the final round, accept whatever comes back — even if everything
        # is taken — to fulfil the requested count.
        last_round = round_idx == MAX_REGEN_ROUNDS - 1

        try:
            candidates, cost = await generate_names(
                req.prompt,
                lang=req.lang,
                count=needed,
                tlds=req.tlds,
                exclude=sorted(seen_names),
            )
        except RuntimeError as exc:
            # OPENAI_API_KEY missing
            log.exception("llm config error")
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        except Exception as exc:
            log.exception("llm error")
            raise HTTPException(status_code=502, detail=f"LLM error: {exc}") from exc

        rounds_used += 1
        if cost is not None:
            total_cost_usd += cost.get("cost_total_usd", 0.0)
            total_input_tokens += cost.get("fresh_input_tokens", 0)
            total_cached_tokens += cost.get("cached_tokens", 0)
            total_output_tokens += cost.get("completion_tokens", 0)

        # Drop anything the LLM repeated despite the exclude list.
        fresh: list[dict] = []
        for c in candidates:
            key = c["name"].lower()
            if key in seen_names:
                continue
            seen_names.add(key)
            fresh.append(c)

        if not fresh:
            log.info(
                "round %d: LLM returned no new names; stopping early",
                round_idx + 1,
            )
            break

        to_check: list[str] = []
        for c in fresh:
            lower = c["name"].lower()
            for tld in req.tlds:
                to_check.append(lower + tld)

        log.info(
            "round %d: %d fresh names × %d tlds = %d whois checks",
            round_idx + 1,
            len(fresh),
            len(req.tlds),
            len(to_check),
        )
        availability = await check_bulk(to_check)

        for c in fresh:
            result = _build_name_result(c, req.tlds, availability)
            fully_taken = all(d.status == "taken" for d in result.domains)
            if fully_taken and not last_round:
                log.info("dropping fully-taken name %r; will regenerate", c["name"])
                continue
            accepted.append(result)
            if len(accepted) >= req.count:
                break

    if not accepted:
        raise HTTPException(status_code=502, detail="LLM returned no candidates")

    log.info(
        "generate complete: %d names, %d rounds, total cost $%.6f",
        len(accepted),
        rounds_used,
        total_cost_usd,
    )

    return GenerateResponse(
        names=accepted,
        cost=CostInfo(
            total_usd=total_cost_usd,
            input_tokens=total_input_tokens,
            cached_tokens=total_cached_tokens,
            output_tokens=total_output_tokens,
            rounds=rounds_used,
        ),
    )
