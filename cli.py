"""Terminal interface.   python cli.py stats | profile | chat"""
import sys
from pathlib import Path
from corpus import Index, load_corpus
import llm

DATA, PROFILE = "data", Path("profile.md")
idx = Index(load_corpus(DATA))
cmd = sys.argv[1] if len(sys.argv) > 1 else "chat"

if cmd == "stats":
    s = idx.stats()
    print(f"{s['entities']} entities, {s['poems']} with verse text, ~{s['approx_tokens']:,} tokens if sent whole")
    print(dict(s["classes"]))
    for k in ("subject", "poetic_type", "metre"): print(k, dict(s[k].most_common(10)))
elif cmd == "profile":
    text, c = llm.build_profile(idx)
    PROFILE.write_text(text, encoding="utf-8"); print(text, f"\n[saved to {PROFILE}; cost ~${c:.3f}]")
else:
    profile = PROFILE.read_text(encoding="utf-8") if PROFILE.exists() else ""
    model, history, total = llm.CHEAP, [], 0.0
    print("Describe the poem (English). Commands: /sonnet /haiku /new /quit")
    while (q := input("\n> ").strip()) != "/quit":
        if q == "/sonnet": model = llm.STRONG; continue
        if q == "/haiku": model = llm.CHEAP; continue
        if q == "/new": history = []; continue
        if not q: continue
        r = llm.run_request(idx, profile, q, history[-6:], model)
        total += r["cost"]
        print(f"\n{r['text']}\n\n[keywords: {', '.join(r['terms'])} | ~${r['cost']:.4f} | session ~${total:.3f}]")
        history += [{"role": "user", "content": q}, {"role": "assistant", "content": r["text"]}]
