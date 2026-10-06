"""All Anthropic API calls live here (retrieval stays local in corpus.py)."""
import json, os, time
from pathlib import Path
import anthropic
import tomllib
import random

CHEAP = "claude-haiku-4-5-20251001"  # keyword extraction + drafts
STRONG = "claude-sonnet-5-5"  # profile + polished poems
# USD per million tokens (input, output). ESTIMATES: verify on the pricing page.
PRICES = {CHEAP: (1.0, 5.0), STRONG: (3.0, 15.0)}
LOG = Path("logs/requests.jsonl")

SYSTEM = """You are a poet-philologist collaborating with researchers on Italian Cinquecento \
poetry. You receive (1) a STYLE PROFILE of a research corpus, (2) per request, CONTEXT retrieved \
from the corpus database (entity cards and exemplar poems) and (3) a REQUEST in English.
Rules:
- Write the new poem in 16th-century Italian, imitating the lexicon, orthography (et/e, havere, \
huomo, apostrophes, etc.), forms and imagery of the exemplars. Do not copy verses; recombine.
- Respect any requested form (sonnet, madrigale, ottava...) and metre (hendecasyllables, \
heptasyllables). Mind synaloepha when counting syllables.
- Only treat as fact what appears in the CONTEXT. Do not invent biographical details.
Output format: first the poem (title optional), then a line '---', then a short English note: \
form and rhyme scheme used, and which context ids (id ...) influenced the result."""


def _api_key():
    key = os.environ.get("ANTHROPIC_API_KEY")  # 1. environment variable wins
    if key:
        return key
    p = (
        Path(__file__).parent / ".streamlit" / "secrets.toml"
    )  # 2. fall back to secrets.toml
    if p.exists():
        return tomllib.loads(p.read_text(encoding="utf-8")).get("ANTHROPIC_API_KEY")
    return None


def client():
    key = _api_key()
    if not key:
        raise SystemExit(
            "No API key found: set ANTHROPIC_API_KEY or add it to .streamlit/secrets.toml"
        )
    return anthropic.Anthropic(api_key=key)


def cost(model, u):
    pin, pout = PRICES.get(model, (3.0, 15.0))
    cr = getattr(u, "cache_read_input_tokens", 0) or 0
    cw = getattr(u, "cache_creation_input_tokens", 0) or 0
    return (
        u.input_tokens * pin + cr * pin * 0.1 + cw * pin * 1.25 + u.output_tokens * pout
    ) / 1e6


def _text(resp):
    return "".join(b.text for b in resp.content if b.type == "text")


def extract_keywords(request):
    r = client().messages.create(
        model=CHEAP,
        max_tokens=300,
        system=(
            "Turn a request for a poem into search keywords for a database of 16th-century "
            "Italian poetry. Entity labels (concepts, subjects, genres) are mostly English "
            "('love','praise','eye','madrigale'); poem texts are archaic Italian. Return ONLY JSON: "
            '{"it": [up to 12 Italian words incl. archaic variants], "en": [up to 8 English labels]}'
        ),
        messages=[{"role": "user", "content": request}],
    )
    try:
        kw = json.loads(_text(r).strip().strip("`").removeprefix("json").strip())
        terms = list(kw.get("it", [])) + list(kw.get("en", []))
    except Exception:
        terms = request.split()
    return terms, cost(CHEAP, r.usage)


def run_request(
    index, profile, request, history, model=CHEAP, max_tokens=1500, n_poems=6
):
    """history: list of plain {'role','content'} turns (without retrieved context)."""
    terms, c1 = extract_keywords(request)
    hits = index.retrieve(terms, n_poems=n_poems)
    user = (
        f"<context>\n{hits['context']}\n</context>\n\n<request>\n{request}\n</request>"
    )
    system = [
        {
            "type": "text",
            "cache_control": {"type": "ephemeral"},
            "text": SYSTEM
            + "\n\n<style_profile>\n"
            + (profile or "(not generated yet)")
            + "\n</style_profile>",
        }
    ]
    r = client().messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=history + [{"role": "user", "content": user}],
    )
    out = dict(
        text=_text(r),
        context=hits["context"],
        terms=terms,
        ids=hits["ids"],
        cost=c1 + cost(model, r.usage),
        model=model,
    )
    LOG.parent.mkdir(
        exist_ok=True
    )  # reproducibility log (ephemeral on Streamlit Cloud)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(dict(t=time.time(), request=request, **out), ensure_ascii=False)
            + "\n"
        )
    return out


def build_profile(index, model=STRONG, max_chars=300_000):
    """One-off corpus analysis. Hard numbers come from Python; the model interprets the verses."""
    st = index.stats()
    facts = "\n".join(
        f"{k}: {dict(st[k].most_common(25))}"
        for k in ("subject", "poetic_type", "metre", "depicts")
    )
    verses, used = [], 0
    poems = sorted(index.poems, key=lambda p: p["id"])
    random.Random(42).shuffle(poems)  # fixed seed: reproducible, unbiased sample
    for p in poems:
        chunk = (
            f"# {p['label']} | "
            + "; ".join(p["fields"].get("contributor", []))
            + "\n"
            + p["text"][:1200]
        )
        if used + len(chunk) > max_chars:
            break
        verses.append(chunk)
        used += len(chunk)
    print(
        f"Profile built from {len(verses)} of {len(index.poems)} poems, {used:,} chars"
    )
    prompt = (
        f"Corpus statistics (computed, reliable):\n{facts}\n\nCorpus poems ({len(verses)}):\n\n"
        + "\n\n".join(verses)
        + "\n\nWrite a STYLE PROFILE (max ~1500 words, English, with Italian examples) for a poet "
        "who must imitate this corpus. Sections: 1 recurring topics/motifs and how they combine; "
        "2 lexicon and orthographic habits (give real forms); 3 forms, metres, rhyme schemes "
        "(frequencies from the statistics); 4 imagery and rhetorical devices; 5 typical "
        "opening/closing moves (cite ~8 short verses); 6 anachronisms to avoid."
    )
    r = client().messages.create(
        model=model, max_tokens=3500, messages=[{"role": "user", "content": prompt}]
    )
    return _text(r), cost(model, r.usage)
