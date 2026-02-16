"""
ClinicalNarrator: zero-parameter template engine for clinical output.

Maps structured model predictions (severity, mechanisms, flags) to
deterministic clinical English. Every sentence comes from a verified template.
Pure lookup + composition — no generation, no hallucination possible.
"""

from pharmloop.output import SEVERITY_NAMES, MECHANISM_NAMES, FLAG_NAMES


class ClinicalNarrator:
    """
    Maps structured model output to clinical English.
    Zero learned parameters. Pure lookup + composition.
    """

    SEVERITY_TEMPLATES = {
        "none": (
            "No clinically significant interaction identified between "
            "{A} and {B}."
        ),
        "mild": (
            "Minor interaction between {A} and {B}. {mechanism} "
            "Generally safe with routine monitoring. {monitoring}"
        ),
        "moderate": (
            "Moderate interaction between {A} and {B}. {mechanism} "
            "{action} {monitoring}"
        ),
        "severe": (
            "Serious interaction between {A} and {B}. {mechanism} "
            "{action} {monitoring}"
        ),
        "contraindicated": (
            "CONTRAINDICATED: {A} and {B} should not be used together. "
            "{mechanism} {action}"
        ),
        "unknown": (
            "Insufficient data to assess interaction between {A} and {B}. "
            "The interaction profile could not be determined with confidence. "
            "Consult a pharmacist or prescriber for guidance."
        ),
    }

    MECHANISM_EXPLANATIONS = {
        "serotonergic": (
            "Both {A} and {B} increase serotonergic activity. "
            "Combined use increases the risk of serotonin syndrome, "
            "characterized by agitation, hyperthermia, clonus, and tremor."
        ),
        "cyp_inhibition": (
            "{A} inhibits cytochrome P450 enzymes involved in the metabolism "
            "of {B}. This may increase {B} plasma concentrations and the risk "
            "of dose-related adverse effects."
        ),
        "cyp_induction": (
            "{A} induces cytochrome P450 enzymes involved in the metabolism "
            "of {B}. This may decrease {B} plasma concentrations and reduce "
            "therapeutic efficacy."
        ),
        "qt_prolongation": (
            "Both {A} and {B} are associated with QT interval prolongation. "
            "Combined use increases the risk of serious cardiac arrhythmias "
            "including torsades de pointes."
        ),
        "bleeding_risk": (
            "Both {A} and {B} affect hemostasis. Combined use may "
            "significantly increase the risk of bleeding."
        ),
        "cns_depression": (
            "Both {A} and {B} have central nervous system depressant effects. "
            "Combined use may cause excessive sedation, respiratory depression, "
            "and impaired cognitive and motor function."
        ),
        "nephrotoxicity": (
            "Both {A} and {B} may adversely affect renal function. "
            "Combined use increases the risk of nephrotoxicity."
        ),
        "hepatotoxicity": (
            "Both {A} and {B} may adversely affect hepatic function. "
            "Combined use increases the risk of hepatotoxicity."
        ),
        "hypotension": (
            "Both {A} and {B} can lower blood pressure. "
            "Combined use increases the risk of hypotension, "
            "particularly orthostatic hypotension."
        ),
        "hyperkalemia": (
            "Both {A} and {B} can increase serum potassium levels. "
            "Combined use increases the risk of hyperkalemia."
        ),
        "seizure_risk": (
            "Both {A} and {B} may lower the seizure threshold. "
            "Combined use increases the risk of seizures."
        ),
        "immunosuppression": (
            "{A} may alter the metabolism or effect of {B}, affecting "
            "immunosuppressive drug levels and efficacy."
        ),
        "absorption_altered": (
            "{A} may alter the gastrointestinal absorption of {B}, "
            "potentially affecting its bioavailability and therapeutic effect."
        ),
        "protein_binding_displacement": (
            "{A} and {B} compete for plasma protein binding sites. "
            "Displacement may transiently increase free drug concentrations "
            "and the risk of adverse effects."
        ),
        "electrolyte_imbalance": (
            "The combination of {A} and {B} may cause electrolyte "
            "disturbances. Monitor serum electrolytes."
        ),
    }

    FLAG_RECOMMENDATIONS = {
        "monitor_serotonin_syndrome": (
            "Watch for symptoms of serotonin syndrome: agitation, "
            "hyperthermia, clonus, diaphoresis, tremor, and hyperreflexia."
        ),
        "monitor_inr": (
            "Monitor INR and adjust anticoagulant dosage as needed."
        ),
        "monitor_qt_interval": (
            "Obtain baseline ECG and monitor QTc interval."
        ),
        "monitor_renal_function": (
            "Monitor serum creatinine, BUN, and urine output."
        ),
        "monitor_hepatic_function": (
            "Monitor liver function tests (ALT, AST, bilirubin)."
        ),
        "monitor_blood_pressure": (
            "Monitor blood pressure regularly, particularly with "
            "position changes."
        ),
        "monitor_blood_glucose": (
            "Monitor blood glucose levels more frequently."
        ),
        "monitor_electrolytes": (
            "Monitor serum electrolytes including potassium, sodium, "
            "and magnesium."
        ),
        "monitor_drug_levels": (
            "Monitor serum drug levels and adjust dosage as needed."
        ),
        "monitor_cns_depression": (
            "Monitor for excessive sedation, respiratory depression, "
            "and impaired cognitive function."
        ),
        "avoid_combination": (
            "Consider avoiding this combination if possible. "
            "If co-administration is necessary, use with extreme caution."
        ),
        "monitor_bleeding": (
            "Monitor for signs of bleeding: bruising, petechiae, "
            "gastrointestinal or gum bleeding, dark stools."
        ),
        "monitor_digoxin_levels": (
            "Monitor serum digoxin levels and watch for signs of "
            "digoxin toxicity (nausea, visual disturbances, arrhythmias)."
        ),
        "monitor_lithium_levels": (
            "Monitor serum lithium levels and watch for signs of "
            "lithium toxicity (tremor, nausea, confusion, ataxia)."
        ),
        "monitor_cyclosporine_levels": (
            "Monitor serum cyclosporine levels and adjust dosage "
            "to maintain therapeutic range."
        ),
        "monitor_theophylline_levels": (
            "Monitor serum theophylline levels and adjust dosage "
            "as needed to avoid toxicity."
        ),
        "reduce_statin_dose": (
            "Consider reducing the statin dose. The interaction may "
            "increase statin plasma levels and the risk of myopathy "
            "or rhabdomyolysis."
        ),
        "separate_administration": (
            "Separate administration times by at least 2 hours "
            "to minimize the interaction."
        ),
    }

    ACTIONS = {
        "none": "",
        "mild": "No dose adjustment typically required.",
        "moderate": (
            "Use with caution. Dose adjustment may be required. "
            "Weigh benefits against risks."
        ),
        "severe": (
            "Use only if benefit clearly outweighs risk. "
            "Consider therapeutic alternatives."
        ),
        "contraindicated": (
            "Do not co-administer. Select an alternative agent."
        ),
        "unknown": "",
    }

    def narrate(
        self,
        drug_a_name: str,
        drug_b_name: str,
        severity: str,
        mechanisms: list[str],
        flags: list[str],
        confidence: float,
        converged: bool,
        steps: int,
        partial_convergence: dict | None = None,
    ) -> str:
        """
        Compose clinical narrative from structured model output.

        All inputs are post-processed model outputs (strings and floats),
        not raw tensors.

        Returns:
            Multi-paragraph clinical narrative string.
        """
        A = drug_a_name.capitalize()
        B = drug_b_name.capitalize()
        sections = []

        # Primary assessment
        mechanism_text = self._compose_mechanisms(mechanisms, A, B)
        monitoring_text = self._compose_monitoring(flags)
        action_text = self.ACTIONS.get(severity, "")

        template = self.SEVERITY_TEMPLATES.get(severity, self.SEVERITY_TEMPLATES["unknown"])
        primary = template.format(
            A=A, B=B,
            mechanism=mechanism_text,
            monitoring=monitoring_text,
            action=action_text,
        )
        primary = " ".join(primary.split())
        sections.append(primary)

        # Monitoring recommendations (detailed)
        if flags and severity not in ("none", "unknown"):
            recs = [
                self.FLAG_RECOMMENDATIONS[f]
                for f in flags if f in self.FLAG_RECOMMENDATIONS
            ]
            if recs:
                sections.append("Monitoring: " + " ".join(recs))

        # Confidence qualifier
        if confidence < 0.5 and severity != "unknown":
            sections.append(
                f"Note: This assessment has limited confidence ({confidence:.0%}). "
                f"The interaction profile for this combination is not well-characterized. "
                f"Clinical judgment should be applied."
            )

        # Partial convergence report
        if partial_convergence and partial_convergence.get("partial_convergence"):
            settled = partial_convergence["settled_aspects"]
            unsettled = partial_convergence["unsettled_aspects"]
            if unsettled:
                sections.append(
                    f"The assessment of {', '.join(settled)} is confident, "
                    f"but {', '.join(unsettled)} could not be fully determined."
                )

        # Metadata line
        status = (
            f"Converged in {steps} steps" if converged
            else f"Did not converge ({steps}/{16} steps)"
        )
        sections.append(f"[Confidence: {confidence:.0%} | {status}]")

        return "\n\n".join(sections)

    def _compose_mechanisms(self, mechanisms: list[str], A: str, B: str) -> str:
        """Compose mechanism explanation from list of active mechanisms."""
        if not mechanisms:
            return ""
        explanations = []
        for mech in mechanisms:
            template = self.MECHANISM_EXPLANATIONS.get(mech)
            if template:
                explanations.append(template.format(A=A, B=B))
        return " ".join(explanations)

    def _compose_monitoring(self, flags: list[str]) -> str:
        """Compose brief monitoring summary from flags."""
        if not flags:
            return ""
        brief = []
        for flag in flags[:3]:
            if "serotonin" in flag:
                brief.append("serotonin syndrome symptoms")
            elif "inr" in flag:
                brief.append("INR")
            elif "qt" in flag:
                brief.append("QTc interval")
            elif "renal" in flag:
                brief.append("renal function")
            elif "hepatic" in flag:
                brief.append("liver function")
            elif "blood_pressure" in flag:
                brief.append("blood pressure")
            elif "glucose" in flag:
                brief.append("blood glucose")
            elif "electrolyte" in flag:
                brief.append("electrolytes")
            elif "bleeding" in flag:
                brief.append("signs of bleeding")
            elif "cns" in flag:
                brief.append("CNS depression")
            elif "drug_levels" in flag or "digoxin" in flag or "lithium" in flag \
                 or "cyclosporine" in flag or "theophylline" in flag:
                brief.append("drug levels")
        if brief:
            return f"Monitor {', '.join(brief)}."
        return ""
