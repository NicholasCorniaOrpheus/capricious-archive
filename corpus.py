"""Load YAML entities -> compact records -> BM25 index with graph expansion.
No embeddings, no heavy dependencies: runs instantly on a free Streamlit instance."""
from __future__ import annotations
import math, re, unicodedata
from collections import Counter, defaultdict
from pathlib import Path
import yaml

Loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
SKIP = {"identifier", "described_by_source", "catalog_identifier", "has_version",
        "instrumentation", "text"}          # bibliographic noise / handled separately
STOP = set("""e et il lo la le li gli i un una di da in con su per tra fra che ch non ne si
a o ma the of and to is with for an on at by from""".split())


def _id(url): return str(url).rstrip("/").rsplit("/", 1)[-1]
def _is_url(s): return isinstance(s, str) and s.startswith("http")


def _fmt(item):
    if isinstance(item, dict):
        label = item.get("label")
        if label is None and not _is_url(item.get("value")):
            label = item.get("value")
        if label is None:
            return None
        role = item.get("role")
        if isinstance(role, dict) and role.get("label"):
            label = f"{label} ({role['label']})"
        return str(label)
    return None if _is_url(item) else str(item)


def norm(s: str) -> str:
    """Fold 16th-c. orthographic variation: accents, apostrophes, u/v, i/j, ph/f."""
    s = unicodedata.normalize("NFKD", s.lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.replace("ph", "f").replace("j", "i").replace("v", "u")


def tokens(s: str) -> list[str]:
    out = []
    for w in re.findall(r"[a-z]+", norm(s)):
        if len(w) > 1 and w[0] == "h" and w[1] in "aeiou":   # havesti -> avesti
            w = w[1:]
        if len(w) > 1 and w not in STOP:
            out.append(w[:5])                                # crude stem: first 5 letters
    return out


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.len = [len(d) for d in docs]
        self.avg = sum(self.len) / max(1, len(docs))
        df = Counter()
        for t in self.tf: df.update(t.keys())
        n = len(docs)
        self.idf = {w: math.log(1 + (n - c + .5) / (c + .5)) for w, c in df.items()}

    def scores(self, q):
        out = [0.0] * len(self.tf)
        for w in set(q):
            idf = self.idf.get(w)
            if not idf: continue
            for i, tf in enumerate(self.tf):
                f = tf.get(w)
                if f:
                    out[i] += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        return out


def load_corpus(folder: str | Path) -> dict[str, dict]:
    recs = {}
    for p in sorted(Path(folder).rglob("*.y*ml")):
        try:
            d = yaml.load(p.read_text(encoding="utf-8"), Loader=Loader) or {}
        except Exception as e:
            print(f"[skip] {p.name}: {e}"); continue
        m, s = d.get("metadata") or {}, d.get("statements") or {}
        rid = str(m.get("id") or p.stem)
        fields, links = {}, set()
        for key, vals in s.items():
            if key in SKIP or not vals: continue
            vals = vals if isinstance(vals, list) else [vals]
            labels = [x for x in map(_fmt, vals) if x]
            if labels: fields[key] = labels
            for v in vals:
                if isinstance(v, dict) and _is_url(v.get("value")) and "/entity/" in v["value"]:
                    links.add(_id(v["value"]))
        text = "\n".join(str(t) for t in (s.get("text") or []))
        md = ((d.get("content") or {}).get("markdown_content") or "").strip()
        aliases = [a for a in (m.get("aliases") or []) if a]
        head = f"[{m.get('class')}] {m.get('label')} (id {rid})"
        if aliases: head += f" aka {', '.join(map(str, aliases))}"
        card = head + "".join(f"\n  {k}: {'; '.join(v)}" for k, v in fields.items())
        if m.get("description"): card += f"\n  note: {m['description']}"
        if md: card += f"\n  notes: {md[:1500]}"
        recs[rid] = dict(id=rid, cls=m.get("class"), label=str(m.get("label")), card=card,
                         text=text, fields=fields, links=links,
                         search=" ".join([str(m.get("label"))] + aliases + [card, text, md]))
    return recs


class Index:
    def __init__(self, recs):
        self.recs, self.ids = recs, list(recs)
        self.pos = {r: i for i, r in enumerate(self.ids)}
        self.bm25 = BM25([tokens(recs[i]["search"]) for i in self.ids])
        self.back = defaultdict(set)
        for r in recs.values():
            for l in r["links"]:
                if l in recs: self.back[l].add(r["id"])

    @property
    def poems(self): return [r for r in self.recs.values() if r["text"]]

    def stats(self):
        poems = self.poems
        c = {k: Counter() for k in ("subject", "poetic_type", "metre", "depicts")}
        for p in poems:
            for k in c: c[k].update(p["fields"].get(k, []))
        chars = sum(len(r["card"]) + len(r["text"]) for r in self.recs.values())
        return dict(entities=len(self.recs), poems=len(poems), approx_tokens=chars // 3,
                    classes=Counter(r["cls"] for r in self.recs.values()), **c)

    def retrieve(self, terms, n_poems=6, n_entities=6, max_poem_chars=1500):
        sc = self.bm25.scores(tokens(" ".join(terms)))
        poems, ents = [], []
        for i in sorted(range(len(sc)), key=lambda i: -sc[i]):
            if sc[i] <= 0 or (len(poems) >= n_poems and len(ents) >= n_entities): break
            r = self.recs[self.ids[i]]
            if r["text"]:
                if len(poems) < n_poems: poems.append(r)
            elif len(ents) < n_entities: ents.append(r)
        # graph expansion 1: poems that point to the best-matching concepts/entities
        have = {p["id"] for p in poems}
        for e in ents[:3]:
            cands = [b for b in self.back[e["id"]] if self.recs[b]["text"] and b not in have]
            for b in sorted(cands, key=lambda b: -sc[self.pos[b]])[:2]:
                if len(poems) < n_poems + 4: poems.append(self.recs[b]); have.add(b)
        # graph expansion 2: Agent cards (poets, patrons...) linked from the chosen poems
        eids = {e["id"] for e in ents}
        for p in poems:
            for l in p["links"]:
                r = self.recs.get(l)
                if r and r["cls"] == "Agent" and l not in eids and len(eids) < n_entities + 4:
                    ents.append(r); eids.add(l)
        ctx = "## Entities\n" + "\n".join(e["card"] for e in ents) if ents else ""
        ctx += "\n\n## Poems (stylistic exemplars)\n" + "\n\n".join(
            f"{p['card']}\n  verses:\n" + "\n".join("    " + v for v in p["text"][:max_poem_chars].splitlines())
            for p in poems)
        return dict(poems=poems, entities=ents, context=ctx.strip(),
                    ids=[r["id"] for r in ents + poems])
