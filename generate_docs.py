"""Generate a synthetic corpus of 50 policy PDFs in ./docs/ for the benchmark.

The repo ships only one real policy PDF, so this builds a reproducible stand-in:
10 categories x 5 documents, 3-6 pages each, plus docs/categories.json mapping
filename -> category. Drop your own PDFs into ./docs/ instead if you have them.

    python generate_docs.py            # 50 docs, seed 42
    python generate_docs.py --n 50 --seed 7
"""

from __future__ import annotations

import argparse
import json
import random

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

from common import CATEGORY_MANIFEST, DOCS_DIR

CATEGORIES: dict[str, dict[str, list[str]]] = {
    "hr": {
        "topics": ["annual leave", "parental leave", "performance reviews", "probation period",
                   "remote work eligibility", "grievance procedure", "overtime compensation"],
        "terms": ["employee", "line manager", "HR business partner", "payroll", "contract"],
    },
    "security": {
        "topics": ["password rotation", "multi-factor authentication", "badge access",
                   "incident reporting", "phishing awareness", "laptop encryption", "visitor escort"],
        "terms": ["SOC", "credential", "endpoint", "firewall", "access token"],
    },
    "finance": {
        "topics": ["expense reimbursement", "corporate card usage", "purchase approval limits",
                   "invoice processing", "budget reallocation", "petty cash", "quarter-end close"],
        "terms": ["receipt", "cost centre", "controller", "ledger", "approval threshold"],
    },
    "it": {
        "topics": ["software installation", "device replacement", "VPN usage", "backup schedule",
                   "helpdesk escalation", "shadow IT", "license management"],
        "terms": ["service desk", "ticket", "patch", "asset tag", "SLA"],
    },
    "legal": {
        "topics": ["contract signature authority", "conflict of interest", "intellectual property",
                   "litigation hold", "anti-bribery", "export control", "whistleblower protection"],
        "terms": ["counsel", "agreement", "liability", "jurisdiction", "compliance officer"],
    },
    "health_safety": {
        "topics": ["fire evacuation", "first aid", "ergonomic assessment", "hazardous materials",
                   "lone working", "workplace injury reporting", "heat stress"],
        "terms": ["fire warden", "assembly point", "PPE", "risk assessment", "incident log"],
    },
    "travel": {
        "topics": ["flight booking class", "hotel rate caps", "per diem allowance", "travel insurance",
                   "mileage claims", "visa support", "trip pre-approval"],
        "terms": ["itinerary", "travel desk", "booking tool", "economy class", "allowance"],
    },
    "procurement": {
        "topics": ["vendor onboarding", "competitive bidding", "sole-source justification",
                   "supplier code of conduct", "contract renewal", "purchase orders", "vendor risk review"],
        "terms": ["supplier", "RFP", "tender", "purchase order", "category manager"],
    },
    "data_privacy": {
        "topics": ["personal data retention", "data subject requests", "breach notification",
                   "cross-border transfers", "consent management", "data minimisation", "DPIA"],
        "terms": ["data controller", "GDPR", "PDPA", "anonymisation", "privacy officer"],
    },
    "facilities": {
        "topics": ["meeting room booking", "parking allocation", "office cleaning", "desk hot-desking",
                   "building access hours", "mail handling", "maintenance requests"],
        "terms": ["facilities team", "reception", "floor warden", "access card", "work order"],
    },
}

SENTENCES = [
    "All staff must follow the {topic} rules described in this section, and the {term} is responsible for enforcement.",
    "Requests relating to {topic} must be submitted at least {n} working days in advance through the {term}.",
    "Exceptions to the {topic} policy require written approval from the {term} and are reviewed every {n} months.",
    "Failure to comply with {topic} requirements may result in disciplinary action up to and including termination.",
    "The {term} keeps a record of every {topic} decision for a minimum of {n} years for audit purposes.",
    "Where {topic} conflicts with local law, local law prevails and the {term} must be informed within {n} days.",
    "Managers should discuss {topic} with new joiners during their first {n} weeks of employment.",
    "Questions about {topic} can be raised with the {term}, who will respond within {n} business days.",
    "This policy on {topic} applies to permanent employees, contractors and interns in all {n} regional offices.",
    "Breaches involving {topic} must be escalated to the {term} immediately and documented in writing.",
    "The maximum limit for {topic} is {n} occurrences per calendar year unless the {term} grants an extension.",
    "Training on {topic} is mandatory and must be completed every {n} months via the learning portal.",
]


def paragraph(rng: random.Random, topic: str, terms: list[str]) -> str:
    return " ".join(
        rng.choice(SENTENCES).format(topic=topic, term=rng.choice(terms), n=rng.randint(2, 30))
        for _ in range(rng.randint(4, 7))
    )


def build_doc(path, title: str, category: str, rng: random.Random) -> None:
    styles = getSampleStyleSheet()
    spec = CATEGORIES[category]
    story = [Paragraph(title, styles["Title"]), Spacer(1, 12)]
    n_pages = rng.randint(3, 6)
    for page in range(n_pages):
        for topic in rng.sample(spec["topics"], 3):
            story.append(Paragraph(f"{page + 1}.{spec['topics'].index(topic) + 1} {topic.title()}",
                                   styles["Heading2"]))
            story.append(Paragraph(paragraph(rng, topic, spec["terms"]), styles["BodyText"]))
            story.append(Spacer(1, 8))
        if page < n_pages - 1:
            story.append(PageBreak())
    SimpleDocTemplate(str(path), pagesize=A4, title=title).build(story)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    cats = list(CATEGORIES)
    manifest = {}
    for i in range(args.n):
        cat = cats[i % len(cats)]
        fname = f"{cat}_policy_{i // len(cats) + 1:02d}.pdf"
        build_doc(DOCS_DIR / fname, f"{cat.replace('_', ' ').title()} Policy {i // len(cats) + 1}", cat, rng)
        manifest[fname] = cat
    CATEGORY_MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {args.n} PDFs + {CATEGORY_MANIFEST.name} to {DOCS_DIR}")


if __name__ == "__main__":
    main()
