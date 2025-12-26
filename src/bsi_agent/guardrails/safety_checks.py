"""Safety guardrails for BSI agent recommendations."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class GuardrailSeverity(Enum):
    """Severity level of guardrail violations."""
    WARNING = "warning"  # Should be reviewed but may proceed
    ERROR = "error"      # Should not proceed without correction
    CRITICAL = "critical"  # Never event - must be blocked


@dataclass
class GuardrailViolation:
    """Represents a guardrail violation."""
    rule_name: str
    severity: GuardrailSeverity
    message: str
    recommendation: str
    triggered_by: str  # What triggered this violation


# Drug allergy cross-reactivity mapping
ALLERGY_CROSS_REACTIVITY = {
    "penicillin": {
        "contraindicated": [
            "penicillin", "ampicillin", "amoxicillin", "piperacillin",
            "nafcillin", "oxacillin", "dicloxacillin", "ticarcillin",
        ],
        "caution": [
            # Cephalosporins - ~1-2% cross-reactivity
            "cefazolin", "cephalexin", "ceftriaxone", "cefepime",
            "ceftazidime", "cefuroxime", "cefoxitin", "cefotetan",
        ],
        "safe_alternatives": [
            "vancomycin", "daptomycin", "linezolid",
            "aztreonam",  # Safe in penicillin allergy
            "fluoroquinolones", "aminoglycosides",
        ],
    },
    "cephalosporin": {
        "contraindicated": [
            "cefazolin", "cephalexin", "ceftriaxone", "cefepime",
            "ceftazidime", "cefuroxime", "cefoxitin", "cefotetan",
            "cefdinir", "cefpodoxime", "ceftaroline",
        ],
        "caution": [
            "penicillin", "ampicillin", "amoxicillin", "piperacillin",
        ],
        "safe_alternatives": [
            "vancomycin", "daptomycin", "aztreonam",
            "fluoroquinolones", "aminoglycosides", "carbapenems",
        ],
    },
    "sulfa": {
        "contraindicated": [
            "sulfamethoxazole", "trimethoprim-sulfamethoxazole",
            "tmp-smx", "bactrim", "sulfasalazine",
        ],
        "caution": [],
        "safe_alternatives": [
            "most other antibiotics are safe",
        ],
    },
    "fluoroquinolone": {
        "contraindicated": [
            "ciprofloxacin", "levofloxacin", "moxifloxacin",
            "ofloxacin", "norfloxacin",
        ],
        "caution": [],
        "safe_alternatives": [
            "beta-lactams", "vancomycin", "aminoglycosides",
        ],
    },
    "vancomycin": {
        "contraindicated": ["vancomycin"],
        "caution": ["teicoplanin"],
        "safe_alternatives": [
            "daptomycin", "linezolid", "tedizolid",
        ],
    },
    "aminoglycoside": {
        "contraindicated": [
            "gentamicin", "tobramycin", "amikacin", "streptomycin",
        ],
        "caution": [],
        "safe_alternatives": [
            "beta-lactams", "fluoroquinolones", "vancomycin",
        ],
    },
}

# Drug-bug mismatch rules
COVERAGE_REQUIREMENTS = {
    "mrsa": {
        "requires": ["vancomycin", "daptomycin", "linezolid", "ceftaroline", "tmp-smx"],
        "insufficient": ["cefazolin", "nafcillin", "oxacillin", "ceftriaxone", "cefepime"],
    },
    "mssa": {
        "requires": ["cefazolin", "nafcillin", "oxacillin", "vancomycin"],
        "preferred_over_vanco": ["cefazolin", "nafcillin", "oxacillin"],  # Beta-lactam preferred
    },
    "pseudomonas": {
        "requires": [
            "piperacillin-tazobactam", "cefepime", "ceftazidime",
            "meropenem", "imipenem", "ciprofloxacin", "levofloxacin",
            "tobramycin", "amikacin", "aztreonam",
        ],
        "insufficient": [
            "ceftriaxone", "ampicillin", "cefazolin",
        ],
    },
    "esbl": {  # Extended-spectrum beta-lactamase producers
        "requires": ["meropenem", "imipenem", "ertapenem"],
        "insufficient": ["ceftriaxone", "cefepime", "piperacillin-tazobactam"],
    },
    "vre": {  # Vancomycin-resistant Enterococcus
        "requires": ["daptomycin", "linezolid", "tedizolid"],
        "insufficient": ["vancomycin", "ampicillin"],
    },
    "candida": {
        "requires": ["fluconazole", "micafungin", "caspofungin", "anidulafungin", "amphotericin"],
        "insufficient": ["any_antibacterial"],
    },
}

# Nephrotoxic drug combinations to avoid
NEPHROTOXIC_COMBINATIONS = [
    ("vancomycin", "aminoglycoside"),
    ("vancomycin", "piperacillin-tazobactam"),  # Controversial but flagged
    ("aminoglycoside", "amphotericin"),
    ("aminoglycoside", "nsaid"),
]

# Drugs requiring renal dose adjustment
RENAL_DOSE_DRUGS = [
    "vancomycin", "gentamicin", "tobramycin", "amikacin",
    "meropenem", "imipenem", "cefepime", "piperacillin-tazobactam",
    "levofloxacin", "ciprofloxacin", "fluconazole",
]


class SafetyGuardrails:
    """
    Safety guardrail system for BSI agent recommendations.

    Checks agent recommendations for:
    - Allergy violations
    - Drug-bug mismatches
    - Nephrotoxic combinations
    - Missing coverage for identified organisms
    """

    def __init__(
        self,
        patient_allergies: Optional[list[str]] = None,
        patient_renal_impairment: bool = False,
        identified_organism: Optional[str] = None,
    ):
        """
        Initialize guardrails with patient context.

        Args:
            patient_allergies: List of documented allergies
            patient_renal_impairment: Whether patient has renal impairment
            identified_organism: Organism if identified (for coverage check)
        """
        self.patient_allergies = [a.lower() for a in (patient_allergies or [])]
        self.patient_renal_impairment = patient_renal_impairment
        self.identified_organism = identified_organism.lower() if identified_organism else None

    def check_recommendation(
        self,
        recommended_drugs: list[str],
    ) -> list[GuardrailViolation]:
        """
        Check a treatment recommendation for safety issues.

        Args:
            recommended_drugs: List of recommended drug names

        Returns:
            List of any guardrail violations found
        """
        violations = []
        drugs_lower = [d.lower() for d in recommended_drugs]

        # Check allergies
        violations.extend(self._check_allergies(drugs_lower))

        # Check coverage
        if self.identified_organism:
            violations.extend(self._check_coverage(drugs_lower))

        # Check nephrotoxicity
        if self.patient_renal_impairment:
            violations.extend(self._check_nephrotoxicity(drugs_lower))

        return violations

    def _check_allergies(self, drugs: list[str]) -> list[GuardrailViolation]:
        """Check for allergy violations."""
        violations = []

        for allergy in self.patient_allergies:
            allergy_info = ALLERGY_CROSS_REACTIVITY.get(allergy)
            if not allergy_info:
                continue

            for drug in drugs:
                # Check contraindicated
                for contra in allergy_info["contraindicated"]:
                    if contra in drug or drug in contra:
                        violations.append(GuardrailViolation(
                            rule_name="allergy_contraindication",
                            severity=GuardrailSeverity.CRITICAL,
                            message=f"CRITICAL: {drug} is contraindicated due to {allergy} allergy",
                            recommendation=f"Use alternative: {', '.join(allergy_info['safe_alternatives'][:3])}",
                            triggered_by=drug,
                        ))

                # Check caution
                for caution in allergy_info.get("caution", []):
                    if caution in drug or drug in caution:
                        violations.append(GuardrailViolation(
                            rule_name="allergy_caution",
                            severity=GuardrailSeverity.WARNING,
                            message=f"CAUTION: {drug} has potential cross-reactivity with {allergy} allergy",
                            recommendation="Consider allergy testing or use alternative if high-risk",
                            triggered_by=drug,
                        ))

        return violations

    def _check_coverage(self, drugs: list[str]) -> list[GuardrailViolation]:
        """Check if recommended drugs cover the identified organism."""
        violations = []

        if not self.identified_organism:
            return violations

        # Map organism to coverage requirements
        organism = self.identified_organism
        coverage_key = None

        if "mrsa" in organism or ("staph" in organism and "meth" in organism and "resist" in organism):
            coverage_key = "mrsa"
        elif "mssa" in organism or ("staph" in organism and "aureus" in organism):
            coverage_key = "mssa"
        elif "pseudomonas" in organism:
            coverage_key = "pseudomonas"
        elif "esbl" in organism or "extended" in organism:
            coverage_key = "esbl"
        elif "vre" in organism or ("enterococcus" in organism and "vanc" in organism and "resist" in organism):
            coverage_key = "vre"
        elif "candida" in organism or "yeast" in organism:
            coverage_key = "candida"

        if not coverage_key:
            return violations

        requirements = COVERAGE_REQUIREMENTS.get(coverage_key, {})
        required_drugs = requirements.get("requires", [])
        insufficient_drugs = requirements.get("insufficient", [])

        # Check if any recommended drug provides coverage
        has_coverage = False
        for drug in drugs:
            if any(req in drug or drug in req for req in required_drugs):
                has_coverage = True
                break

        if not has_coverage:
            violations.append(GuardrailViolation(
                rule_name="insufficient_coverage",
                severity=GuardrailSeverity.ERROR,
                message=f"Recommended drugs may not adequately cover {coverage_key.upper()}",
                recommendation=f"Consider adding: {', '.join(required_drugs[:3])}",
                triggered_by=self.identified_organism,
            ))

        # Check if using insufficient drugs
        for drug in drugs:
            if any(insuff in drug or drug in insuff for insuff in insufficient_drugs):
                violations.append(GuardrailViolation(
                    rule_name="ineffective_drug",
                    severity=GuardrailSeverity.ERROR,
                    message=f"{drug} is not effective against {coverage_key.upper()}",
                    recommendation=f"Use instead: {', '.join(required_drugs[:3])}",
                    triggered_by=drug,
                ))

        return violations

    def _check_nephrotoxicity(self, drugs: list[str]) -> list[GuardrailViolation]:
        """Check for nephrotoxic drug combinations in renal impairment."""
        violations = []

        # Check combinations
        for drug1, drug2 in NEPHROTOXIC_COMBINATIONS:
            has_drug1 = any(drug1 in d or d in drug1 for d in drugs)
            has_drug2 = any(drug2 in d or d in drug2 for d in drugs)

            if has_drug1 and has_drug2:
                violations.append(GuardrailViolation(
                    rule_name="nephrotoxic_combination",
                    severity=GuardrailSeverity.WARNING,
                    message=f"Potentially nephrotoxic combination: {drug1} + {drug2}",
                    recommendation="Monitor renal function closely or consider alternatives",
                    triggered_by=f"{drug1} + {drug2}",
                ))

        # Check individual drugs needing dose adjustment
        for drug in drugs:
            if any(renal_drug in drug or drug in renal_drug for renal_drug in RENAL_DOSE_DRUGS):
                violations.append(GuardrailViolation(
                    rule_name="renal_dose_adjustment",
                    severity=GuardrailSeverity.WARNING,
                    message=f"{drug} requires renal dose adjustment",
                    recommendation="Verify dose is adjusted for renal function",
                    triggered_by=drug,
                ))

        return violations

    def format_violations(self, violations: list[GuardrailViolation]) -> str:
        """Format violations as a readable string."""
        if not violations:
            return "No safety concerns identified."

        lines = ["⚠️ SAFETY REVIEW ⚠️\n"]

        # Group by severity
        critical = [v for v in violations if v.severity == GuardrailSeverity.CRITICAL]
        errors = [v for v in violations if v.severity == GuardrailSeverity.ERROR]
        warnings = [v for v in violations if v.severity == GuardrailSeverity.WARNING]

        if critical:
            lines.append("🚨 CRITICAL (Must Address):")
            for v in critical:
                lines.append(f"  - {v.message}")
                lines.append(f"    → {v.recommendation}")

        if errors:
            lines.append("\n❌ ERRORS (Should Address):")
            for v in errors:
                lines.append(f"  - {v.message}")
                lines.append(f"    → {v.recommendation}")

        if warnings:
            lines.append("\n⚠️ WARNINGS (Review):")
            for v in warnings:
                lines.append(f"  - {v.message}")
                lines.append(f"    → {v.recommendation}")

        return "\n".join(lines)

    def has_critical_violations(self, violations: list[GuardrailViolation]) -> bool:
        """Check if there are any critical violations."""
        return any(v.severity == GuardrailSeverity.CRITICAL for v in violations)


def extract_drugs_from_text(text: str) -> list[str]:
    """
    Extract drug names from free text.

    Simple extraction - looks for known antibiotic names.
    """
    known_drugs = [
        "vancomycin", "daptomycin", "linezolid", "tedizolid",
        "cefazolin", "ceftriaxone", "cefepime", "ceftazidime",
        "piperacillin", "tazobactam", "piperacillin-tazobactam", "zosyn",
        "meropenem", "imipenem", "ertapenem", "doripenem",
        "ampicillin", "amoxicillin", "nafcillin", "oxacillin",
        "ciprofloxacin", "levofloxacin", "moxifloxacin",
        "gentamicin", "tobramycin", "amikacin",
        "metronidazole", "clindamycin",
        "fluconazole", "micafungin", "caspofungin", "amphotericin",
        "azithromycin", "doxycycline", "trimethoprim", "sulfamethoxazole",
    ]

    text_lower = text.lower()
    found = []

    for drug in known_drugs:
        if drug in text_lower:
            found.append(drug)

    return found
