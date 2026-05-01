"""
CultiKure LLM Service
---------------------
Integrates Anthropic Claude to generate plant disease explanations,
treatment recommendations, and fertilizer suggestions.

Model: claude-sonnet-4-20250514
"""

import os
import json
import logging
from datetime import datetime, timezone
from functools import lru_cache

import anthropic

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Anthropic client (lazy-initialised so the app can still start without a key)
# ---------------------------------------------------------------------------
_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY environment variable is not set. "
                "Set it before using the /explain endpoint."
            )
        _client = anthropic.Anthropic(api_key=api_key)
    return _client


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are an expert plant pathologist and agronomist with 20+ years of experience "
    "diagnosing plant diseases and prescribing treatment protocols. You provide "
    "accurate, practical, and farmer-friendly advice."
)

PROMPT_TEMPLATE = """A deep-learning vision model has detected the following plant disease in an uploaded leaf image:

**Disease:** {disease_name}
**Plant:** {plant_name}
**Condition:** {condition}

Please provide a structured response in valid JSON (no markdown fences, just raw JSON) with exactly these keys:

{{
  "explanation": "A plain-language explanation of this disease in 2-3 sentences suitable for a farmer.",
  "treatment_steps": [
    "Step 1 ...",
    "Step 2 ...",
    "Step 3 ..."
  ],
  "preventive_measures": [
    "Measure 1 ...",
    "Measure 2 ...",
    "Measure 3 ..."
  ],
  "fertilizers": [
    "Fertilizer/supplement 1 and why it helps",
    "Fertilizer/supplement 2 and why it helps"
  ],
  "severity": "low | medium | high",
  "contagious": true | false
}}

Be concise, actionable, and specific to the named disease. Return only the JSON object.
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _parse_disease_label(disease_class: str) -> tuple[str, str, str]:
    """
    Parse a PlantVillage label like 'Tomato___Early_blight' into its parts.
    Returns (plant_name, condition, readable_disease_name).
    """
    parts = disease_class.replace("___", "___").split("___", 1)
    if len(parts) == 2:
        plant = parts[0].replace("_", " ").replace(",", "").strip()
        condition = parts[1].replace("_", " ").strip()
        if "healthy" in condition.lower():
            return plant, "healthy", f"Healthy {plant}"
        return plant, condition, f"{plant} — {condition}"
    return disease_class, "unknown", disease_class


@lru_cache(maxsize=64)
def get_disease_explanation(disease_class: str) -> dict:
    """
    Query Claude to generate a structured disease explanation.
    Results are cached (per process) so repeated queries for the same class
    don't incur API costs.

    Args:
        disease_class: PlantVillage label, e.g. 'Tomato___Early_blight'

    Returns:
        dict with keys: explanation, treatment_steps, preventive_measures,
                        fertilizers, severity, contagious, generated_at
    """
    plant_name, condition, readable = _parse_disease_label(disease_class)

    # Short-circuit for healthy plants — no LLM needed
    if "healthy" in condition.lower():
        return {
            "explanation": f"Your {plant_name} plant appears healthy! No disease was detected.",
            "treatment_steps": [
                "Continue current care routine.",
                "Water consistently and avoid over-watering.",
                "Ensure adequate sunlight and air circulation.",
            ],
            "preventive_measures": [
                "Inspect leaves weekly for early signs of disease.",
                "Remove dead or yellowing leaves promptly.",
                "Rotate crops each season to break disease cycles.",
            ],
            "fertilizers": [
                "Balanced NPK fertilizer (e.g. 10-10-10) every 4-6 weeks.",
                "Compost or organic mulch to improve soil health.",
            ],
            "severity": "none",
            "contagious": False,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "disease": disease_class,
            "readable_name": readable,
        }

    prompt = PROMPT_TEMPLATE.format(
        disease_name=disease_class,
        plant_name=plant_name,
        condition=condition,
    )

    try:
        client = _get_client()
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = message.content[0].text.strip()

        # Strip markdown code fences if Claude adds them despite instructions
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        result = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.error("LLM returned non-JSON response: %s", e)
        result = _fallback_response(disease_class)
    except anthropic.APIError as e:
        logger.error("Anthropic API error: %s", e)
        result = _fallback_response(disease_class)

    result["generated_at"] = datetime.now(timezone.utc).isoformat()
    result["disease"] = disease_class
    result["readable_name"] = readable
    return result


def _fallback_response(disease_class: str) -> dict:
    """Return a safe fallback when the LLM call fails."""
    plant_name, condition, readable = _parse_disease_label(disease_class)
    return {
        "explanation": (
            f"The model detected {readable}. "
            "Please consult a local agricultural extension officer for detailed advice."
        ),
        "treatment_steps": ["Consult a local agronomist for treatment options."],
        "preventive_measures": ["Maintain good field hygiene and crop rotation."],
        "fertilizers": ["Balanced NPK fertilizer as recommended for this crop."],
        "severity": "unknown",
        "contagious": False,
    }
