"""
Test queries for benchmarking search precision on Swiss legal documents.

Each query has:
- question: the search query (in French)
- expected_keywords: words that should appear in at least one top-k result
"""

TEST_QUERIES = [
    # --- Specific queries (articles, references) ---
    {
        "question": "Art. 90 LCR excès de vitesse qualifié",
        "expected_keywords": ["vitesse", "90", "grave", "excès"],
    },
    {
        "question": "Art. 91 LCR conduite en état d'ébriété",
        "expected_keywords": ["ébriété", "alcool", "ivresse", "91"],
    },
    {
        "question": "Art. 92 LCR violation des règles de circulation",
        "expected_keywords": ["circulation", "règles", "violation", "92"],
    },
    {
        "question": "Retrait du permis de conduire durée minimum",
        "expected_keywords": ["retrait", "permis", "durée", "mois"],
    },
    {
        "question": "Art. 16c LCR cas graves infraction",
        "expected_keywords": ["grave", "16c", "infraction", "retrait"],
    },
    # --- Broad legal concept queries ---
    {
        "question": "Responsabilité pénale du conducteur",
        "expected_keywords": ["responsabilité", "pénale", "conducteur"],
    },
    {
        "question": "Quelles sont les sanctions pour excès de vitesse ?",
        "expected_keywords": ["vitesse", "sanction", "amende", "retrait"],
    },
    {
        "question": "Différence entre infraction simple et grave",
        "expected_keywords": ["simple", "grave", "infraction", "distinction"],
    },
    {
        "question": "Mesures administratives retrait de permis",
        "expected_keywords": ["mesures", "administratives", "retrait", "permis"],
    },
    {
        "question": "Conduite sans permis conséquences juridiques",
        "expected_keywords": ["permis", "conduite", "sans"],
    },
    # --- Procedural / structural queries ---
    {
        "question": "Procédure pénale en matière de circulation routière",
        "expected_keywords": ["procédure", "pénale", "circulation"],
    },
    {
        "question": "Compétence du juge en matière de LCR",
        "expected_keywords": ["compétence", "juge", "tribunal"],
    },
    {
        "question": "Prescription des infractions routières",
        "expected_keywords": ["prescription", "délai", "infraction"],
    },
    {
        "question": "Récidive en droit pénal routier",
        "expected_keywords": ["récidive", "antécédent", "récidiviste"],
    },
    {
        "question": "Mise en danger de la vie d'autrui par un conducteur",
        "expected_keywords": ["danger", "vie", "autrui", "mise"],
    },
    # --- Edge cases and broader topics ---
    {
        "question": "Responsabilité du détenteur du véhicule",
        "expected_keywords": ["détenteur", "véhicule", "responsabilité"],
    },
    {
        "question": "Assurance responsabilité civile automobile",
        "expected_keywords": ["assurance", "responsabilité", "civile"],
    },
    {
        "question": "Homologation et immatriculation des véhicules",
        "expected_keywords": ["immatriculation", "véhicule", "homologation"],
    },
    {
        "question": "Obligation de porter la ceinture de sécurité",
        "expected_keywords": ["ceinture", "sécurité", "obligation"],
    },
    {
        "question": "Délit de fuite après un accident de circulation",
        "expected_keywords": ["fuite", "accident", "délit"],
    },
]
