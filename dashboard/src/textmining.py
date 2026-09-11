"""Stage 6 - What customers actually say.

Star ratings tell you how much someone disliked an order. The written comment
tells you why. 41% of reviews carry a comment, which is enough to separate
delivery complaints from product complaints.

Method: log-odds ratio with an informative Dirichlet prior (Monroe et al.),
which is a better fit than raw frequency or plain TF-IDF for the question
"which words distinguish angry reviews from happy ones?". Common words that
appear everywhere are penalised automatically.

Comments are in Portuguese; a small stopword list and a gloss table are
maintained here so the output is readable in the report.

Run:  python -m src.textmining
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from src import config as C

STOPWORDS = {
    "a", "o", "as", "os", "de", "da", "do", "das", "dos", "e", "em", "no",
    "na", "nos", "nas", "um", "uma", "uns", "umas", "para", "por", "com",
    "que", "se", "ao", "aos", "à", "às", "mas", "mais", "muito", "muita",
    "foi", "ser", "sou", "é", "eu", "me", "meu", "minha", "ele", "ela",
    "eles", "elas", "isso", "esse", "essa", "este", "esta", "isto", "já",
    "não", "sim", "só", "também", "tem", "ter", "tinha", "está", "estão",
    "estava", "como", "quando", "qual", "quero", "vou", "vai", "pois",
    "porque", "porém", "então", "todo", "toda", "todos", "todas", "outro",
    "outra", "nada", "sem", "até", "das", "pelo", "pela", "ou", "lhe",
    "nem", "eram", "era", "fui", "seu", "sua", "aqui", "ainda", "bem",
    "ja", "nao", "so", "voce", "vc", "pra", "pro", "ate", "mesmo", "sobre",
}

GLOSS = {
    "prazo": "deadline", "entrega": "delivery", "entregue": "delivered",
    "produto": "product", "chegou": "arrived", "antes": "before",
    "recebi": "I received", "comprei": "I bought", "veio": "came",
    "recomendo": "I recommend", "otimo": "excellent", "ótimo": "excellent",
    "excelente": "excellent", "bom": "good", "boa": "good",
    "rapido": "fast", "rápido": "fast", "rapida": "fast",
    "qualidade": "quality", "atraso": "delay", "atrasado": "late",
    "aguardando": "waiting", "esperando": "waiting", "espera": "wait",
    "cancelar": "to cancel", "cancelamento": "cancellation",
    "reembolso": "refund", "dinheiro": "money", "devolver": "to return",
    "devolucao": "return", "devolução": "return",
    "errado": "wrong", "diferente": "different", "faltou": "was missing",
    "quebrado": "broken", "danificado": "damaged", "defeito": "defect",
    "propaganda": "advertised", "anunciado": "advertised",
    "pessimo": "terrible", "péssimo": "terrible", "ruim": "bad",
    "horrivel": "horrible", "horrível": "horrible",
    "vendedor": "seller", "loja": "store", "compra": "purchase",
    "correios": "postal service", "transportadora": "carrier",
    "entregaram": "they delivered", "chegaram": "they arrived",
    "nunca": "never", "ninguem": "nobody", "ninguém": "nobody",
    "resposta": "reply", "contato": "contact", "reclamacao": "complaint",
    "unidade": "unit", "unidades": "units", "apenas": "only",
    "embalagem": "packaging", "nota": "invoice", "fiscal": "invoice",
    "site": "website", "site.": "website", "pedido": "order",
    "comprado": "bought", "certo": "correct", "perfeito": "perfect",
    "super": "very", "chegar": "to arrive", "antecedencia": "in advance",
    "confiavel": "reliable", "satisfeito": "satisfied",
    "parabens": "congratulations", "obrigado": "thank you",
}

TOKEN = re.compile(r"[a-zà-ú]+")


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN.findall(str(text).lower())
            if len(t) > 2 and t not in STOPWORDS]


def counts(series: pd.Series) -> pd.Series:
    from collections import Counter

    c = Counter()
    for txt in series:
        c.update(set(tokenize(txt)))  # document frequency, not raw frequency
    return pd.Series(c)


def log_odds(pos: pd.Series, neg: pd.Series, alpha: float = 0.5,
             min_count: int = 40) -> pd.DataFrame:
    """Weighted log-odds ratio with a Dirichlet prior.

    A large positive z-score means the word is distinctive of `pos`; a large
    negative one means it is distinctive of `neg`.
    """
    vocab = sorted(set(pos.index) | set(neg.index))
    p = pos.reindex(vocab).fillna(0).astype(float)
    n = neg.reindex(vocab).fillna(0).astype(float)
    keep = (p + n) >= min_count
    p, n = p[keep], n[keep]

    prior = (p + n) * alpha / (p + n).sum()
    lp = np.log((p + prior) / (p.sum() + prior.sum() - p - prior))
    ln = np.log((n + prior) / (n.sum() + prior.sum() - n - prior))
    delta = lp - ln
    var = 1.0 / (p + prior) + 1.0 / (n + prior)
    z = delta / np.sqrt(var)

    return pd.DataFrame({"term": p.index, "z": z.values,
                         "n_bad": p.values, "n_good": n.values}).assign(
        gloss=lambda d: d.term.map(GLOSS).fillna("")
    )


def main() -> None:
    df = pd.read_parquet(C.PROCESSED_DIR / "review_text.parquet")
    bad = df[df.review_score <= C.BAD_REVIEW_MAX].review_comment_message
    good = df[df.review_score >= 4].review_comment_message

    res = log_odds(counts(bad), counts(good)).sort_values("z", ascending=False)
    res.to_csv(C.TABLE_DIR / "review_terms.csv", index=False)

    # Classify each negative comment into a complaint theme so the report can
    # split "we shipped it late" from "the product was wrong".
    themes = {
        "Delivery / lateness": r"prazo|entrega|entregue|chegou|atras|aguard|esper|"
                               r"correios|transportadora|demor",
        "Never arrived": r"não receb|nao receb|não chegou|nao chegou|não foi entregue|"
                         r"nao foi entregue|extrav",
        "Wrong or missing item": r"errad|diferente|faltou|falta|apenas um|outro produto|"
                                 r"não era|nao era",
        "Damaged or defective": r"quebrad|danific|defeito|estragad|amassad|risc",
        "Product quality": r"qualidade|péssim|pessim|ruim|horr[íi]vel|frágil|fragil|"
                           r"barat|material",
        "Refund or service": r"reembols|devolv|devoluç|cancel|contato|resposta|"
                             r"atendimento",
    }
    neg = df[df.review_score <= C.BAD_REVIEW_MAX].copy()
    rows = []
    for name, pattern in themes.items():
        hit = neg.review_comment_message.str.lower().str.contains(
            pattern, regex=True, na=False)
        rows.append({"theme": name, "mentions": int(hit.sum()),
                     "share_of_negative_comments": float(hit.mean()),
                     "late_share_within_theme": float(
                         neg.loc[hit, "is_late"].mean())})
    theme_df = pd.DataFrame(rows).sort_values("mentions", ascending=False)
    theme_df.to_csv(C.TABLE_DIR / "review_themes.csv", index=False)

    print(f"Negative comments analysed: {len(bad):,}  positive: {len(good):,}\n")
    print("Most distinctive words in 1-2 star reviews:")
    print(res.head(15)[["term", "gloss", "z"]].round(1).to_string(index=False))
    print("\nComplaint themes:")
    print(theme_df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
