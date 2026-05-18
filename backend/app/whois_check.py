"""Domain availability check via the system `whois` CLI.

The CLI is preferred over RDAP / pure-Python whois libraries because Debian's
`whois` package ships a curated map of TLD → whois server and follows referrals
automatically — that's what gives us reliable coverage for Ukrainian second-level
zones like .kyiv.ua, .zp.ua, .lviv.ua, .in.ua, .org.ua, .com.ua, .net.ua, etc.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from typing import Literal, TypedDict

log = logging.getLogger("domains.whois")

WHOIS_TIMEOUT = int(os.environ.get("WHOIS_TIMEOUT", "30"))
WHOIS_CONCURRENCY = int(os.environ.get("WHOIS_CONCURRENCY", "8"))
WHOIS_CACHE_TTL = int(os.environ.get("WHOIS_CACHE_TTL", "3600"))

# Minimum seconds between consecutive whois requests to the same TLD.
# Verisign (.com/.net) throttles aggressively per source IP, so we pace
# requests per-TLD: while we're waiting on .com, queries to .ua/.io/etc
# still run freely.
WHOIS_MIN_INTERVAL = float(os.environ.get("WHOIS_MIN_INTERVAL", "3.0"))
WHOIS_MIN_INTERVAL_COM = float(os.environ.get("WHOIS_MIN_INTERVAL_COM", "5.0"))

# Retry schedule (seconds between attempts) when whois returns `unknown` — this
# usually means transient rate-limiting or a slow referral. We sleep, then try
# again, until we get a definitive answer or the schedule is exhausted.
_RETRY_BACKOFFS = [1.5, 3, 5, 8, 12, 18, 25, 35, 45]


Status = Literal["available", "taken", "unknown"]


class WhoisResult(TypedDict, total=False):
    domain: str
    status: Status
    error: str
    raw: str


# Lines whose presence strongly suggests the domain is AVAILABLE (free).
# Order/specificity matters less since we use re.search across the full output.
_AVAILABLE_PATTERNS = [
    r"\bNo match for\b",
    r"\bNOT FOUND\b",
    r"\bNo Data Found\b",
    r"\bNo entries found\b",
    r"\bNo matches found\b",
    r"\bDomain not found\b",
    r"\bDomain Not Found\b",
    r"\bDomain name does not exist\b",
    r"\bdoes not exist\b",
    r"\bNo such domain\b",
    r"\bNo such entry\b",
    r"\bDOMAIN NOT FOUND\b",
    r"\bObject does not exist\b",
    r"\bNo information available about domain name\b",
    r"\bis free\b",
    r"\bis available for registration\b",
    r"\bavailable for purchase\b",
    r"^\s*Status:\s*AVAILABLE",
    r"^\s*Status:\s*free",
    r"^\s*status:\s*free",
    r"^%\s*No entries found",
    r"^\s*This query returned 0 objects",
]

# Lines whose presence strongly suggests the domain is REGISTERED (taken).
_TAKEN_PATTERNS = [
    r"^\s*Domain Name:\s*\S",
    r"^\s*domain:\s*\S",
    r"^\s*Registry Domain ID:",
    r"^\s*Creation Date:",
    r"^\s*created:\s*\S",
    r"^\s*Registrar:\s*\S",
    r"^\s*registrar:\s*\S",
    r"^\s*Name Server:\s*\S",
    r"^\s*nserver:\s*\S",
    r"^\s*Status:\s*ok",
    r"^\s*status:\s*ok",
    r"^\s*status:\s*active",
]

_AVAILABLE_RE = re.compile("|".join(_AVAILABLE_PATTERNS), re.IGNORECASE | re.MULTILINE)
_TAKEN_RE = re.compile("|".join(_TAKEN_PATTERNS), re.IGNORECASE | re.MULTILINE)

# Rate-limit / connection error markers — we should NOT treat these as available.
_TRANSIENT_ERROR_PATTERNS = [
    r"connect: Connection refused",
    r"Connection reset by peer",
    r"Name or service not known",
    r"Temporary failure in name resolution",
    r"WHOIS LIMIT EXCEEDED",
    r"Quota exceeded",
    r"\bRate limit\b",
    r"You have exceeded",
    r"try again later",
]
_TRANSIENT_RE = re.compile("|".join(_TRANSIENT_ERROR_PATTERNS), re.IGNORECASE)


_cache: dict[str, tuple[float, WhoisResult]] = {}
_cache_lock = asyncio.Lock()

# Per-TLD pacing: each TLD gets its own lock + "last request finished at"
# timestamp. Before launching a whois call we acquire the TLD lock and sleep
# enough to keep at least WHOIS_MIN_INTERVAL between consecutive requests to
# that same TLD.
_tld_locks: dict[str, asyncio.Lock] = {}
_tld_last_call: dict[str, float] = {}
_tld_locks_guard = asyncio.Lock()


def _tld_of(domain: str) -> str:
    """Return the registry key we pace against — last label for most TLDs,
    last two labels for known multi-label zones like .co.uk / .com.ua."""
    parts = domain.lower().rsplit(".", 2)
    if len(parts) >= 2 and parts[-2] in {
        "co", "com", "net", "org", "gov", "edu",
        "kyiv", "lviv", "zp", "in", "kh",
    }:
        return ".".join(parts[-2:])
    return parts[-1]


def _interval_for(tld: str) -> float:
    if tld in {"com", "net"}:
        return WHOIS_MIN_INTERVAL_COM
    return WHOIS_MIN_INTERVAL


async def _get_tld_lock(tld: str) -> asyncio.Lock:
    async with _tld_locks_guard:
        lock = _tld_locks.get(tld)
        if lock is None:
            lock = asyncio.Lock()
            _tld_locks[tld] = lock
        return lock


def _classify(text: str) -> Status:
    """Decide status from raw whois output. Conservative: unknown when unclear."""
    if not text or not text.strip():
        # Empty body usually means "no record" → available.
        return "available"

    if _TRANSIENT_RE.search(text):
        return "unknown"

    has_taken = bool(_TAKEN_RE.search(text))
    has_available = bool(_AVAILABLE_RE.search(text))

    # Taken signals win when both are present (some .ua responses include
    # "% No entries found" comment lines while still listing a registered domain).
    if has_taken:
        return "taken"
    if has_available:
        return "available"
    return "unknown"


def _build_whois_cmd(domain: str) -> list[str]:
    """If WHOIS_PROXY is set, wrap whois in proxychains4. Otherwise run plain.

    Expected proxy URL: socks5://user:pass@host:port (also socks4/http).
    """
    proxy = os.environ.get("WHOIS_PROXY", "").strip()
    if not proxy:
        return ["whois", "--", domain]

    from urllib.parse import urlparse

    p = urlparse(proxy)
    scheme = (p.scheme or "socks5").lower()
    cfg_path = "/tmp/proxychains-whois.conf"
    auth = f"{p.username} {p.password}" if p.username and p.password else ""
    cfg = (
        "strict_chain\nquiet_mode\nproxy_dns\ntcp_read_time_out 15000\n"
        "tcp_connect_time_out 8000\n[ProxyList]\n"
        f"{scheme} {p.hostname} {p.port or 1080} {auth}\n"
    )
    try:
        with open(cfg_path, "w") as f:
            f.write(cfg)
    except OSError:
        pass
    return ["proxychains4", "-q", "-f", cfg_path, "whois", "--", domain]


async def _run_whois(domain: str) -> tuple[str, str]:
    """Run `whois <domain>` and return (stdout, stderr) as text."""
    proc = await asyncio.create_subprocess_exec(
        *_build_whois_cmd(domain),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=WHOIS_TIMEOUT
        )
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise

    return (
        stdout.decode("utf-8", errors="replace"),
        stderr.decode("utf-8", errors="replace"),
    )


async def _single_attempt(domain: str) -> WhoisResult:
    tld = _tld_of(domain)
    interval = _interval_for(tld)
    lock = await _get_tld_lock(tld)

    async with lock:
        last = _tld_last_call.get(tld, 0.0)
        wait = interval - (time.monotonic() - last)
        if wait > 0:
            await asyncio.sleep(wait)

        try:
            stdout, stderr = await _run_whois(domain)
        except asyncio.TimeoutError:
            return {"domain": domain, "status": "unknown", "error": "timeout"}
        except FileNotFoundError:
            return {
                "domain": domain,
                "status": "unknown",
                "error": "whois binary not installed",
            }
        except Exception as exc:  # noqa: BLE001
            return {"domain": domain, "status": "unknown", "error": str(exc)}
        finally:
            _tld_last_call[tld] = time.monotonic()

    text = stdout if stdout.strip() else stderr
    status = _classify(text)
    return {"domain": domain, "status": status, "raw": text[:600]}


async def check_one(domain: str) -> WhoisResult:
    domain = domain.strip().lower()
    now = time.time()

    async with _cache_lock:
        cached = _cache.get(domain)
        if cached and now - cached[0] < WHOIS_CACHE_TTL:
            return cached[1]

    # Try, then keep retrying through _RETRY_BACKOFFS while the answer is
    # `unknown`. `unknown` is almost always a transient hiccup (verisign rate
    # limit, slow referral, registry temporarily unreachable) — given enough
    # time we eventually get a definitive answer.
    result: WhoisResult = await _single_attempt(domain)
    attempt = 1
    for delay in _RETRY_BACKOFFS:
        if result.get("status") != "unknown":
            break
        log.info(
            "whois %s: attempt %d returned unknown (err=%r); retrying in %ss",
            domain,
            attempt,
            result.get("error"),
            delay,
        )
        await asyncio.sleep(delay)
        result = await _single_attempt(domain)
        attempt += 1

    if result.get("status") == "unknown":
        log.warning(
            "whois %s: gave up after %d attempts, returning unknown",
            domain,
            attempt,
        )
    else:
        # Only cache definitive answers — never cache `unknown`, otherwise a
        # single transient failure poisons the result for WHOIS_CACHE_TTL.
        async with _cache_lock:
            _cache[domain] = (now, result)

    return result


async def check_bulk(domains: list[str]) -> dict[str, WhoisResult]:
    sem = asyncio.Semaphore(WHOIS_CONCURRENCY)

    async def _run(d: str) -> WhoisResult:
        async with sem:
            return await check_one(d)

    results = await asyncio.gather(*[_run(d) for d in domains])
    return {r["domain"]: r for r in results}
