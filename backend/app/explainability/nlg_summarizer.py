"""Natural Language Generation summarizer for SHAP explanations."""

from typing import Any

RISK_THRESHOLDS = {
    "high": 0.7,
    "medium": 0.4,
}

FEATURE_DESCRIPTIONS = {
    "attendance_rate": "Attendance rate",
    "avg_assignment_score": "Average assignment score",
    "assignment_score_std": "Assignment score consistency",
    "internal_exam_score": "Internal exam score",
    "study_hours_per_week": "Study hours per week",
    "extracurricular_count": "Extracurricular activities",
    "prior_gpa": "Prior GPA",
    "gender_F": "Gender (Female)",
    "gender_M": "Gender (Male)",
    "gender_NB": "Gender (Non-binary)",
    "socioeconomic_status_high": "Socioeconomic status (High)",
    "socioeconomic_status_low": "Socioeconomic status (Low)",
    "socioeconomic_status_medium": "Socioeconomic status (Medium)",
}

FEATURE_TEMPLATES = {
    "attendance_rate": {
        "positive": "Attendance is strong at {value:.0%}",
        "negative": "Attendance is only {value:.0%}",
    },
    "avg_assignment_score": {
        "positive": "Assignment scores average {value:.1f}",
        "negative": "Assignment scores average {value:.1f}",
    },
    "assignment_score_std": {
        "positive": "Consistent assignment performance",
        "negative": "Inconsistent assignment performance",
    },
    "internal_exam_score": {
        "positive": "Internal exam score is {value:.1f}",
        "negative": "Internal exam score is {value:.1f}",
    },
    "study_hours_per_week": {
        "positive": "Studies {value:.1f} hours per week",
        "negative": "Only studies {value:.1f} hours per week",
    },
    "extracurricular_count": {
        "positive": "Participates in {value:.0f} extracurricular activities",
        "negative": "Limited extracurricular participation ({value:.0f})",
    },
    "prior_gpa": {
        "positive": "Prior GPA is {value:.2f}",
        "negative": "Prior GPA is {value:.2f}",
    },
    "gender_F": {
        "positive": "Female student",
        "negative": "Female student",
    },
    "gender_M": {
        "positive": "Male student",
        "negative": "Male student",
    },
    "gender_NB": {
        "positive": "Non-binary student",
        "negative": "Non-binary student",
    },
    "socioeconomic_status_high": {
        "positive": "High socioeconomic background",
        "negative": "High socioeconomic background",
    },
    "socioeconomic_status_low": {
        "positive": "Low socioeconomic background",
        "negative": "Low socioeconomic background",
    },
    "socioeconomic_status_medium": {
        "positive": "Medium socioeconomic background",
        "negative": "Medium socioeconomic background",
    },
}


def _get_risk_level(probability: float) -> str:
    if probability >= RISK_THRESHOLDS["high"]:
        return "high"
    if probability >= RISK_THRESHOLDS["medium"]:
        return "medium"
    return "low"


def _format_factor(feature_name: str, shap_value: float, feature_value: float | None) -> str:
    templates = FEATURE_TEMPLATES.get(feature_name, {
        "positive": f"{feature_name} = {{value}}",
        "negative": f"{feature_name} = {{value}}",
    })
    direction = "positive" if shap_value > 0 else "negative"
    template = templates[direction]

    try:
        if "{value" in template and feature_value is not None:
            text = template.format(value=feature_value)
        else:
            text = template
    except Exception:
        text = f"{feature_name} = {feature_value}"

    strength = "strong" if abs(shap_value) > 0.1 else "moderate" if abs(shap_value) > 0.05 else "slight"
    return f"{text} ({strength} {'positive' if shap_value > 0 else 'negative'} factor)"


def _build_modality_text(
    text_prob: float | None,
    tabular_prob: float | None,
    behavioral_prob: float | None,
) -> str:
    parts = []
    if text_prob is not None:
        sentiment = "negative" if text_prob >= 0.5 else "positive"
        parts.append(f"Text analysis suggests {sentiment} sentiment.")
    if tabular_prob is not None:
        quality = "poor" if tabular_prob >= 0.5 else "good"
        parts.append(f"Tabular data shows {quality} academic indicators.")
    if behavioral_prob is not None:
        engagement = "low" if behavioral_prob >= 0.5 else "healthy"
        parts.append(f"Behavioral data indicates {engagement} engagement.")

    if not parts:
        return "No modality-specific contributions available."
    return " ".join(parts)


def summarize(
    shap_result: dict[str, Any],
    feature_values: dict[str, float] | None = None,
    text_prob: float | None = None,
    tabular_prob: float | None = None,
    behavioral_prob: float | None = None,
) -> dict[str, Any]:
    """Generate a natural language summary from SHAP results.

    Args:
        shap_result: Output from TabularShapExplainer.explain().
        feature_values: Optional mapping of feature names to raw values.
        text_prob: Optional text model probability.
        tabular_prob: Optional tabular model probability.
        behavioral_prob: Optional behavioral model probability.

    Returns:
        Dictionary with risk_level, probability, top_factors, modality_contributions, narrative_summary.
    """
    probability = shap_result["probability"]
    risk_level = _get_risk_level(probability)

    top_factors = []
    for factor in shap_result["top_positive"]:
        feature_name = factor["feature"]
        shap_value = factor["shap_value"]
        value = feature_values.get(feature_name) if feature_values else None
        top_factors.append({
            "feature": feature_name,
            "shap_value": shap_value,
            "description": _format_factor(feature_name, shap_value, value),
        })

    for factor in shap_result["top_negative"]:
        feature_name = factor["feature"]
        shap_value = factor["shap_value"]
        value = feature_values.get(feature_name) if feature_values else None
        top_factors.append({
            "feature": feature_name,
            "shap_value": shap_value,
            "description": _format_factor(feature_name, shap_value, value),
        })

    top_factors.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

    modality_contributions = _build_modality_text(text_prob, tabular_prob, behavioral_prob)

    factor_sentences = []
    for i, factor in enumerate(top_factors[:6], 1):
        factor_sentences.append(f"({i}) {factor['description']}")

    if risk_level == "high":
        narrative = (
            f"This student is at high risk (probability: {probability:.1%}). "
            f"Key factors: {' '.join(factor_sentences)}"
        )
    elif risk_level == "medium":
        narrative = (
            f"This student is at medium risk (probability: {probability:.1%}). "
            f"Key factors: {' '.join(factor_sentences)}"
        )
    else:
        narrative = (
            f"This student is at low risk (probability: {probability:.1%}). "
            f"Key factors: {' '.join(factor_sentences)}"
        )

    if len(narrative) > 200:
        narrative = narrative[:197] + "..."

    return {
        "risk_level": risk_level,
        "probability": probability,
        "top_factors": top_factors[:6],
        "modality_contributions": modality_contributions,
        "narrative_summary": narrative,
    }
