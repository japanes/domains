from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import AsyncIterator, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
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


def _sse(event: str, payload: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")


async def _stream_events(req: GenerateRequest) -> AsyncIterator[bytes]:
    """SSE generator: emits LLM-produced names ASAP, then per-domain whois
    results as soon as each one resolves. Regenerates extra names when a batch
    contains fully-taken names, until we accept `req.count` non-fully-taken
    names or exhaust MAX_REGEN_ROUNDS."""
    seen_names: set[str] = set()

    cost_acc = {
        "total_usd": 0.0,
        "input_tokens": 0,
        "cached_tokens": 0,
        "output_tokens": 0,
        "rounds": 0,
    }

    # How many non-fully-taken names we've accumulated. We use this to decide
    # whether to ask the LLM for more in a follow-up round.
    accepted_count = 0

    try:
        for round_idx in range(MAX_REGEN_ROUNDS):
            needed = req.count - accepted_count
            if needed <= 0:
                break

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
                log.exception("llm config error")
                yield _sse("error", {"detail": str(exc)})
                return
            except Exception as exc:
                log.exception("llm error")
                yield _sse("error", {"detail": f"LLM error: {exc}"})
                return

            cost_acc["rounds"] += 1
            if cost is not None:
                cost_acc["total_usd"] += cost.get("cost_total_usd", 0.0)
                cost_acc["input_tokens"] += cost.get("fresh_input_tokens", 0)
                cost_acc["cached_tokens"] += cost.get("cached_tokens", 0)
                cost_acc["output_tokens"] += cost.get("completion_tokens", 0)

            yield _sse("cost", cost_acc)

            fresh: list[dict] = []
            for c in candidates:
                key = c["name"].lower()
                if key in seen_names:
                    continue
                seen_names.add(key)
                fresh.append(c)

            if not fresh:
                log.info("round %d: LLM returned no new names; stopping", round_idx + 1)
                break

            # Emit names with all domains in 'checking' state — the frontend
            # will render them immediately and patch each row as results arrive.
            names_payload = []
            for c in fresh:
                lower = c["name"].lower()
                names_payload.append(
                    {
                        "name": c["name"],
                        "tagline": c["tagline"],
                        "domains": [
                            {"tld": tld, "domain": lower + tld, "status": "checking"}
                            for tld in req.tlds
                        ],
                    }
                )
            yield _sse("names", {"names": names_payload, "round": round_idx + 1})

            # Launch every whois check for this batch as an independent task and
            # yield results as soon as each one resolves (asyncio.as_completed
            # semantics, but tied to a queue we can drain incrementally).
            queue: asyncio.Queue = asyncio.Queue()
            domain_to_name: dict[str, str] = {}
            batch_size = 0

            async def _worker(name: str, domain: str, tld: str) -> None:
                try:
                    r = await check_one(domain)
                    status = r.get("status", "unknown")
                except Exception as exc:  # noqa: BLE001
                    log.exception("whois worker failed for %s", domain)
                    status = "unknown"
                await queue.put(
                    {"name": name, "domain": domain, "tld": tld, "status": status}
                )

            tasks: list[asyncio.Task] = []
            for c in fresh:
                lower = c["name"].lower()
                for tld in req.tlds:
                    d = lower + tld
                    domain_to_name[d] = c["name"]
                    batch_size += 1
                    tasks.append(asyncio.create_task(_worker(c["name"], d, tld)))

            # Track per-name domain statuses so we can decide fully_taken after
            # all of a name's checks have returned.
            statuses_by_name: dict[str, dict[str, str]] = {c["name"]: {} for c in fresh}
            tlds_per_name = len(req.tlds)

            for _ in range(batch_size):
                evt = await queue.get()
                yield _sse("domain", evt)
                statuses_by_name[evt["name"]][evt["tld"]] = evt["status"]

                # When this name's last domain just resolved, decide its fate.
                if len(statuses_by_name[evt["name"]]) == tlds_per_name:
                    statuses = list(statuses_by_name[evt["name"]].values())
                    fully_taken = all(s == "taken" for s in statuses)
                    if fully_taken and not last_round:
                        yield _sse("dropped", {"name": evt["name"]})
                    else:
                        accepted_count += 1

            # Ensure all tasks are properly awaited (queue.get already covered
            # the results, but tasks may still need cleanup).
            await asyncio.gather(*tasks, return_exceptions=True)

        if accepted_count == 0:
            yield _sse("error", {"detail": "no available names found"})
            return

        yield _sse("done", {"accepted": accepted_count, "cost": cost_acc})
    except asyncio.CancelledError:
        log.info("stream cancelled by client")
        raise


@app.post("/api/generate/stream")
async def generate_stream(req: GenerateRequest) -> StreamingResponse:
    return StreamingResponse(
        _stream_events(req),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
