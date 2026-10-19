"""Generate the synthetic dataset: invoice documents, bank ledger and eval questions.

Everything comes from one seed, so every eval question has a known correct answer.
The data is synthetic on purpose and contains traps that make verification necessary:
split payments, partial payments, payments of the pre-VAT amount only, duplicate
invoice copies, payments without a reference, and suppliers with near-identical names.

Run:  python -m recon.data_gen          # dataset from RECON_DATASET (default dev)
      RECON_DATASET=test python -m recon.data_gen
"""

import json
import random
import sqlite3
from collections import Counter
from datetime import date, timedelta

from recon.config import settings

YEAR = 2026
MONTHS = range(1, 7)  # January to June 2026
MONTH_EN = ["January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"]
MONTH_FR = ["janvier", "fevrier", "mars", "avril", "mai", "juin",
            "juillet", "aout", "septembre", "octobre", "novembre", "decembre"]

# (legal name, name used in questions, invoice number prefix).
# The first three pairs are deliberately near-identical.
SUPPLIERS = [
    ("Dupont SARL", "Dupont", "DUP"),
    ("Dupond SAS", "Dupond", "DPD"),
    ("Martin & Fils", "Martin & Fils", "MAF"),
    ("Martin Freres", "Martin Freres", "MFR"),
    ("Bernard Logistique", "Bernard Logistique", "BLG"),
    ("Bernard Logistics SAS", "Bernard Logistics", "BLS"),
    ("Petit Imprimerie", "Petit Imprimerie", "PIM"),
    ("Leroy Fournitures", "Leroy Fournitures", "LRF"),
    ("Moreau Conseil", "Moreau Conseil", "MRC"),
    ("Simon Informatique", "Simon Informatique", "SIF"),
    ("Laurent Transports", "Laurent Transports", "LTR"),
    ("Lefebvre Nettoyage", "Lefebvre Nettoyage", "LFN"),
    ("Michel Electricite", "Michel Electricite", "MEL"),
    ("Garcia Traiteur", "Garcia Traiteur", "GTR"),
    ("Roux Securite", "Roux Securite", "RXS"),
    ("Fournier Bureautique", "Fournier Bureautique", "FBU"),
    ("Girard Energie", "Girard Energie", "GEN"),
    ("Bonnet Telecom", "Bonnet Telecom", "BTC"),
]
TWINS = {}
for a, b in [("Dupont SARL", "Dupond SAS"), ("Martin & Fils", "Martin Freres"),
             ("Bernard Logistique", "Bernard Logistics SAS")]:
    TWINS[a], TWINS[b] = b, a

UNKNOWN_SUPPLIERS = ["Durand Plomberie", "Chevalier Peinture"]

DESCRIPTIONS = [
    "Prestation de maintenance", "Fournitures de bureau", "Transport de marchandises",
    "Mission de conseil", "Licences logiciels", "Nettoyage des locaux",
    "Impression de catalogues", "Location de materiel", "Abonnement telecom",
]
VAT_RATES = [0.20, 0.20, 0.20, 0.10, 0.055]

# How each invoice gets (or does not get) paid.
SCENARIO_WEIGHTS = {
    "paid_exact": 34,     # one payment of the full amount, with reference
    "paid_no_ref": 12,    # one full payment, bank label has no invoice number
    "split_payment": 14,  # two instalments that add up to the full amount
    "partial_open": 10,   # one payment below the full amount, balance still open
    "vat_only": 12,       # paid the amount excluding VAT (HT) instead of TTC
    "unpaid": 18,
}

QUESTION_QUOTAS = {
    "similar_name": 6, "split_payment": 7, "vat_only": 6, "paid_no_ref": 5,
    "duplicate_doc": 5, "plain_paid": 6, "plain_partial": 4, "unpaid": 6, "not_found": 5,
}


def r2(x: float) -> float:
    return round(x + 1e-9, 2)


def fmt_eur(x: float) -> str:
    """French amount format: 1 234,56 EUR."""
    whole, cents = f"{x:,.2f}".split(".")
    return f"{whole.replace(',', ' ')},{cents} EUR"


def fr_date(d: date) -> str:
    return f"{d.day} {MONTH_FR[d.month - 1]} {d.year}"


def render_document(inv: dict, copy: bool) -> str:
    title = "FACTURE - DUPLICATA" if copy else "FACTURE"
    return "\n".join([
        title,
        f"Fournisseur : {inv['supplier']}",
        f"Facture n° {inv['invoice_number']}",
        f"Date de facture : {fr_date(date.fromisoformat(inv['issue_date']))}",
        f"Date d'echeance : {fr_date(date.fromisoformat(inv['due_date']))}",
        f"Objet : {inv['description']}",
        f"Montant HT : {fmt_eur(inv['amount_ht'])}",
        f"TVA ({inv['vat_rate'] * 100:g} %) : {fmt_eur(inv['vat_amount'])}",
        f"Montant TTC : {fmt_eur(inv['amount_ttc'])}",
    ])


def generate(seed: int = settings.seed):
    rng = random.Random(seed)
    invoices, docs, txns, truth = [], [], [], []

    def add_txn(d: date, amount: float, counterparty: str, label: str, direction: str = "out") -> dict:
        t = {"date": d.isoformat(), "amount": r2(amount), "direction": direction,
             "counterparty": counterparty, "label": label}
        txns.append(t)
        return t

    for legal, short, prefix in SUPPLIERS:
        seq = 0
        for month in MONTHS:
            if rng.random() > 0.8:
                continue
            seq += 1
            issue = date(YEAR, month, rng.randint(1, 28))
            ht = r2(rng.uniform(250, 12000))
            rate = rng.choice(VAT_RATES)
            vat = r2(ht * rate)
            ttc = r2(ht + vat)
            inv = {
                "supplier": legal,
                "invoice_number": f"{prefix}-{YEAR}-{seq:04d}",
                "issue_date": issue.isoformat(),
                "due_date": (issue + timedelta(days=30)).isoformat(),
                "description": rng.choice(DESCRIPTIONS),
                "amount_ht": ht, "vat_rate": rate, "vat_amount": vat, "amount_ttc": ttc,
            }
            invoices.append(inv)

            scenario = rng.choices(list(SCENARIO_WEIGHTS), weights=list(SCENARIO_WEIGHTS.values()))[0]
            cp = legal.upper()
            pay_day = issue + timedelta(days=rng.randint(10, 45))
            num = inv["invoice_number"]
            payments = []
            if scenario == "paid_exact":
                payments.append(add_txn(pay_day, ttc, cp, f"VIR SEPA {cp} REF {num}"))
            elif scenario == "paid_no_ref":
                payments.append(add_txn(pay_day, ttc, cp, f"VIR SEPA {cp}"))
            elif scenario == "split_payment":
                first = r2(ttc * rng.choice([0.3, 0.4, 0.5, 0.6]))
                payments.append(add_txn(pay_day, first, cp, f"VIR SEPA {cp} {num} 1/2"))
                later = pay_day + timedelta(days=rng.randint(15, 40))
                payments.append(add_txn(later, r2(ttc - first), cp, f"VIR SEPA {cp} {num} 2/2"))
            elif scenario == "partial_open":
                part = r2(ttc * rng.choice([0.3, 0.5, 0.7]))
                payments.append(add_txn(pay_day, part, cp, f"VIR SEPA {cp} ACOMPTE {num}"))
            elif scenario == "vat_only":
                payments.append(add_txn(pay_day, ht, cp, f"VIR SEPA {cp} REF {num}"))

            paid = r2(sum(p["amount"] for p in payments))
            status = "paid" if abs(paid - ttc) < 0.005 else ("partially_paid" if paid > 0 else "unpaid")
            traps = [scenario] if scenario in ("paid_no_ref", "split_payment", "vat_only") else []

            docs.append({"doc_id": f"DOC-{len(docs) + 1:04d}", **inv, "is_copy": False,
                         "text": render_document(inv, copy=False)})
            if rng.random() < 0.10:
                traps.append("duplicate_doc")
                docs.append({"doc_id": f"DOC-{len(docs) + 1:04d}", **inv, "is_copy": True,
                             "text": render_document(inv, copy=True)})

            # A payment of the same amount to the near-identical supplier, around the same time.
            if legal in TWINS and rng.random() < 0.5:
                twin_cp = TWINS[legal].upper()
                add_txn(pay_day + timedelta(days=rng.randint(-5, 5)), ttc, twin_cp, f"VIR SEPA {twin_cp}")
                traps.append("similar_name")

            truth.append({"invoice": inv, "scenario": scenario, "status": status,
                          "paid_amount": paid, "payments": payments, "traps": traps})

    # Ordinary account activity, so the ledger is not just supplier payments.
    for month in range(1, 9):
        add_txn(date(YEAR, month, 1), 2400.00, "SCI DES LILAS", "PRLV LOYER BUREAUX")
        add_txn(date(YEAR, month, 28), r2(rng.uniform(18000, 21000)), "SALAIRES", f"VIR SALAIRES {MONTH_FR[month - 1].upper()}")
        add_txn(date(YEAR, month, 15), r2(rng.uniform(5200, 6400)), "URSSAF", "PRLV URSSAF COTISATIONS")
        for _ in range(3):
            add_txn(date(YEAR, month, rng.randint(1, 28)), r2(rng.uniform(1500, 25000)),
                    f"CLIENT {rng.choice(['ACME', 'NOVA', 'ORBIS', 'VEGA'])}", "VIR RECU CLIENT", direction="in")
        for _ in range(2):
            add_txn(date(YEAR, month, rng.randint(1, 28)), r2(rng.uniform(15, 400)),
                    rng.choice(["AMAZON EU", "SNCF", "TOTALENERGIES"]), "CB PAIEMENT CARTE")

    # Chronological transaction ids, assigned after sorting.
    txns.sort(key=lambda t: (t["date"], t["counterparty"], t["amount"]))
    for i, t in enumerate(txns, 1):
        t["txn_id"] = f"TXN-{i:05d}"

    questions = build_questions(rng, truth)
    return docs, txns, truth, questions


def build_questions(rng: random.Random, truth: list[dict]) -> list[dict]:
    short_name = {legal: short for legal, short, _ in SUPPLIERS}
    buckets: dict[str, list[dict]] = {k: [] for k in QUESTION_QUOTAS}
    for t in truth:
        # One category per invoice, rarest trap first.
        traps = t["traps"]
        for cat in ("similar_name", "duplicate_doc", "split_payment", "vat_only", "paid_no_ref"):
            if cat in traps:
                break
        else:
            cat = {"paid": "plain_paid", "partially_paid": "plain_partial", "unpaid": "unpaid"}[t["status"]]
        buckets[cat].append(t)

    questions = []
    for cat, quota in QUESTION_QUOTAS.items():
        if cat == "not_found":
            continue
        for t in rng.sample(buckets[cat], min(quota, len(buckets[cat]))):
            inv = t["invoice"]
            issue = date.fromisoformat(inv["issue_date"])
            short = short_name[inv["supplier"]]
            month = MONTH_EN[issue.month - 1]
            templates = [
                f"Has supplier {short}'s {month} invoice been paid, and does the bank amount match?",
                f"How much is still owed to {short} for their {month} {YEAR} invoice?",
            ]
            if cat != "similar_name":  # these must be resolved by supplier name, not by number
                templates += [
                    f"Is invoice {inv['invoice_number']} from {short} fully paid?",
                    f"Did we pay {short} for the invoice dated {inv['issue_date']}? Does the amount match the invoice total?",
                ]
            questions.append({
                "category": cat, "traps": t["traps"], "question": rng.choice(templates),
                "expected": {
                    "status": t["status"], "invoice_number": inv["invoice_number"],
                    "amount_ttc": inv["amount_ttc"], "paid_amount": t["paid_amount"],
                    "amount_matches": t["status"] == "paid",
                },
            })

    # Not-found questions: a known supplier in a month without an invoice (preferring
    # months where the near-identical twin does have one), or an unknown supplier.
    have = {(t["invoice"]["supplier"], date.fromisoformat(t["invoice"]["issue_date"]).month) for t in truth}
    gaps = [(s, m) for s, _, _ in SUPPLIERS for m in MONTHS if (s, m) not in have]
    twin_gaps = [g for g in gaps if g[0] in TWINS and (TWINS[g[0]], g[1]) in have]
    picks = rng.sample(twin_gaps, min(3, len(twin_gaps)))
    picks += [(name, rng.choice(list(MONTHS))) for name in UNKNOWN_SUPPLIERS]
    for supplier, m in picks:
        short = short_name.get(supplier, supplier)
        questions.append({
            "category": "not_found", "traps": ["similar_name"] if supplier in TWINS else [],
            "question": f"Has supplier {short}'s {MONTH_EN[m - 1]} invoice been paid, and does the bank amount match?",
            "expected": {"status": "not_found", "invoice_number": None, "amount_ttc": None,
                         "paid_amount": None, "amount_matches": None},
        })

    rng.shuffle(questions)
    for i, q in enumerate(questions, 1):
        q["qid"] = f"Q{i:03d}"
    return [{"qid": q.pop("qid"), **q} for q in questions]


def write(docs, txns, truth, questions) -> None:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.eval_dir.mkdir(exist_ok=True)

    with open(settings.invoices_path, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    settings.ledger_path.unlink(missing_ok=True)
    con = sqlite3.connect(settings.ledger_path)
    con.execute("""CREATE TABLE transactions (
        txn_id TEXT PRIMARY KEY, date TEXT NOT NULL, amount REAL NOT NULL,
        direction TEXT NOT NULL CHECK (direction IN ('in', 'out')),
        counterparty TEXT NOT NULL, label TEXT NOT NULL)""")
    con.execute("CREATE INDEX idx_txn_date ON transactions(date)")
    con.executemany(
        "INSERT INTO transactions VALUES (:txn_id, :date, :amount, :direction, :counterparty, :label)", txns)
    con.commit()
    con.close()

    # Ground truth for debugging only; the agent's tools never read it.
    with open(settings.data_dir / "truth.json", "w", encoding="utf-8") as f:
        json.dump(truth, f, indent=1, ensure_ascii=False)

    with open(settings.questions_path, "w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")


def ensure_data() -> None:
    """Generate the dataset if it is missing (e.g. in a fresh clone or container)."""
    if not (settings.invoices_path.exists() and settings.ledger_path.exists()):
        write(*generate())


if __name__ == "__main__":
    docs, txns, truth, questions = generate()
    write(docs, txns, truth, questions)
    print(f"dataset {settings.dataset} (seed {settings.seed}): {len({d['invoice_number'] for d in docs})} invoices ({len(docs)} documents), {len(txns)} transactions")
    print("scenarios:", dict(Counter(t["scenario"] for t in truth)))
    print("questions:", len(questions), dict(Counter(q["category"] for q in questions)))
