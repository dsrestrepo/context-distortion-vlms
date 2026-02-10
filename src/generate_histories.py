import os
import re
from typing import List, Optional
import pandas as pd
import openai
from tqdm import tqdm

# Initialize OpenAI client
# Ensure OPENAI_API_KEY is set in your environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

client = openai.OpenAI(api_key=OPENAI_API_KEY)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5") # Default to gpt-5 or similar

REPORT_DELIM = "<<<END_OF_REPORT>>>"

# Modalities/body parts & column names
PRESET_DISTRACTORS = [
    {"modality": "MRI Brain with and without contrast", "body_part": "Brain",           "months": 12, "col": "distractor_mri_brain"},
    {"modality": "CT Abdomen and Pelvis with contrast", "body_part": "Abdomen/Pelvis", "months": 9,  "col": "distractor_ct_abd_pelvis"},
    {"modality": "Wrist Ultrasound",                    "body_part": "Wrist",          "months": 6,  "col": "distractor_wrist_ultrasound"},
    {"modality": "X-ray Knee, AP and lateral",          "body_part": "Knee",           "months": 3,  "col": "distractor_knee_xray"},
]


# Utilities
def is_abnormal(label_val) -> bool:
    """
    Interpret 'abnormal' from common label encodings: 'Yes'/1/True -> abnormal.
    Returns False for 'No'/0/False or unknown.
    """
    if label_val is None:
        return False
    if isinstance(label_val, str):
        return label_val.strip().lower() in {"yes", "abnormal", "1", "true", "y", "pos", "positive"}
    if isinstance(label_val, (int, float)):
        return label_val != 0
    if isinstance(label_val, bool):
        return label_val
    return False

def _split_plaintext_reports(text: str, expected: int = 4) -> List[str]:
    """
    Split model output into N reports using hard delimiter; fallback to 'FINAL REPORT' boundaries.
    """
    parts = [p.strip() for p in text.split(REPORT_DELIM) if p.strip()]
    if len(parts) >= expected:
        if len(parts) > expected:
             # Basic warning if needed, but keeping it clean for script
             pass
        return parts[:expected]

    # Slice by repeated 'FINAL REPORT'
    starts = [m.start() for m in re.finditer(r'\bFINAL REPORT\b', text)]
    if len(starts) >= expected:
        out = []
        for i in range(expected):
            s = starts[i]
            e = starts[i+1] if i+1 < len(starts) else len(text)
            out.append(text[s:e].strip())
        return out

    # Last resort: just return what we have (padded if needed in caller)
    return parts

# --------------------------
# PRIOR contradictory CXR
# --------------------------
_FEWSHOT_PRIOR = """\
### FEW-SHOT 1  (CURRENT normal → PRIOR abnormal)

CURRENT report:
FINAL REPORT
EXAM:  Chest, single semi-erect portable view.
CLINICAL INFORMATION:  A ___-year-old female with history of fall, evaluate for pneumothorax.
COMPARISON:  ___.
FINDINGS:  Single semi-erect AP portable view of the chest was obtained.  The costophrenic angles are not fully included on the image.  Given this, no focal consolidation, pleural effusion, or evidence of pneumothorax is seen.  The cardiac and mediastinal silhouettes are stable and unremarkable.  Surgical clips are noted in the left axilla.
IMPRESSION:  No acute cardiopulmonary process.  No evidence of pneumothorax.
---
<PRIOR-OUTPUT>
FINAL REPORT
EXAMINATION: Chest, single semi-erect portable view.
INDICATION: __-year-old female with dyspnea.
TECHNIQUE: Semi-erect AP portable chest radiograph.
COMPARISON: None available.
FINDINGS: Small bilateral blunting of the costophrenic angles with layering opacity, left greater than right, consistent with small pleural effusions. No focal airspace consolidation. No pneumothorax. Cardiomediastinal silhouette within normal size.
IMPRESSION: Small bilateral pleural effusions, left greater than right.

### FEW-SHOT 2  (CURRENT abnormal → PRIOR normal)

CURRENT report:
FINAL REPORT
EXAMINATION:  Portable AP chest radiograph.
INDICATION:  ___ year old man with septic shock vs. heart failure // Interval change?
COMPARISON:  Chest radiograph dated ___.
FINDINGS: 
ETT in standard position. Right internal jugular venous catheter ends in the right atrium, unchanged. Consolidation in the right lower lung is less apparent from the exam only 3 hours earlier, suggesting some component of edema. Moderately enlarged heart is overall unchanged. No pneumothorax. No pleural effusion.
IMPRESSION: 
Interval improvement and pulmonary edema, now moderate.
---
<PRIOR-OUTPUT>
FINAL REPORT
EXAMINATION: Portable AP chest radiograph.
INDICATION: ___ year old man, routine evaluation.
COMPARISON: None available.
FINDINGS:
Cardiomediastinal silhouette within normal size and contour. Lungs are clear bilaterally without focal consolidation or interstitial/alveolar edema. No pleural effusion. No pneumothorax. No acute osseous abnormality identified.
IMPRESSION:
No acute cardiopulmonary disease.
"""

_SYSTEM_PRIOR = {
    "role": "system",
    "content": (
        "You are a radiology report generator.\n"
        "TASK: Given a CURRENT chest X-ray report, produce a PRIOR chest X-ray report (~6–12 months earlier) "
        "for the same patient with the OPPOSITE overall interpretation (normal ↔ abnormal).\n"
        "CONSTRAINTS:\n"
        "• If CURRENT is normal, make PRIOR abnormal with a plausible single primary abnormality.\n"
        "• If CURRENT is abnormal, make PRIOR normal.\n"
        "• When abnormal, choose ONLY from: Pleural Effusion, Atelectasis, Cardiomegaly, Consolidation, Edema.\n"
        "• Use sections: EXAMINATION, INDICATION, TECHNIQUE, COMPARISON, FINDINGS, IMPRESSION.\n"
        "• Be internally consistent (FINDINGS ↔ IMPRESSION). Generic age placeholders allowed.\n"
        "• Output only the PRIOR report text. No markdown; no extra text."
    ),
}

def _build_user_prompt_prior(current_report: str, current_is_abn: Optional[bool]) -> str:
    tail = f"\n(current_abnormal: {bool(current_is_abn)})\n---\n<PRIOR-OUTPUT>"
    return _FEWSHOT_PRIOR + "\n### NEW TASK\nCURRENT report:\n" + current_report.strip() + tail

def generate_prior_contradictory_cxr(current_report: str, current_is_abn: Optional[bool]) -> str:
    user_msg = _build_user_prompt_prior(current_report, current_is_abn)
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[_SYSTEM_PRIOR, {"role": "user", "content": user_msg}],
    )
    return (resp.choices[0].message.content or "").strip()


# Fixed-order irrelevant contradictory reports
def _distractor_system_msg(current_is_abn: bool) -> dict:
    """
    Build a modality-aware system prompt that forces opposite overall interpretation
    for each fixed, non-chest modality/body-part report, with formatting + delimiter.
    """
    opposite_word = "NORMAL (No Findings)" if current_is_abn else "ABNORMAL (Some finding)"
    return {
        "role": "system",
        "content": (
            "You are a clinical document generator.\n"
            "Given a CURRENT chest X-ray report (for context only), write EXACTLY FOUR unrelated radiology reports "
            "for OTHER body parts/modalities in the EXACT ORDER below. DO NOT mention the chest, lungs, or the current CXR.\n\n"
            "ORDER (must match exactly):\n"
            "1) MRI Brain with and without contrast (Brain)\n"
            "2) CT Abdomen and Pelvis with contrast (Abdomen/Pelvis)\n"
            "3) Wrist Ultrasound (Wrist)\n"
            "4) X-ray Knee, AP and lateral (Knee)\n\n"
            "GLOBAL INTERPRETATION RULE:\n"
            f" - The OVERALL impression of EACH generated report must be {opposite_word}-the opposite of the CURRENT CXR-.\n"
            "  - If CURRENT is abnormal → make ALL FOUR distractors clearly NORMAL (no acute pathology).\n"
            "  - If CURRENT is normal → make ALL FOUR distractors clearly ABNORMAL with one primary plausible finding per modality/body part."
            "  Keep findings realistic and internally consistent.\n\n"
            "STYLE & FORMAT:\n"
            "• MIMIC-style sections in this order: EXAMINATION, INDICATION, TECHNIQUE, COMPARISON, FINDINGS, IMPRESSION.\n"
            "• 80–180 words per report; realistic tone; generic ages like “__-year-old”; no PHI.\n"
            f"• Output STRICTLY as plain text: each report must start with 'FINAL REPORT' and be followed by a line containing only '{REPORT_DELIM}'.\n"
            "• No numbering, no markdown, no extra text before/after."
        ),
    }

def _build_distractor_user_prompt(current_cxr_report: str) -> str:
    items = "\n".join(
        f"- {i+1}) {d['modality']} | Body part: {d['body_part']}"
        for i, d in enumerate(PRESET_DISTRACTORS)
    )
    return (
        "CURRENT CXR REPORT (context only; do not reference it):\n"
        + current_cxr_report.strip()
        + "\n\nGenerate 4 unrelated reports in the exact order below. "
          "Each must start with 'FINAL REPORT' and end with a line containing only "
        + f"'{REPORT_DELIM}'.\n"
        + items
    )

def generate_irrelevant_contradictors_fixed(current_report: str, current_is_abn: bool) -> List[str]:
    system_msg = _distractor_system_msg(current_is_abn)
    user_msg = _build_distractor_user_prompt(current_report)

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[system_msg, {"role": "user", "content": user_msg}],
        top_p=1.0,
    )
    raw = resp.choices[0].message.content or ""
    parts = _split_plaintext_reports(raw, expected=4)

    # Ensure exactly 4 strings
    while len(parts) < 4:
        parts.append("")
    return parts[:4]

def clean_report(text: str) -> str:
    text = str(text).strip()
    if "FINAL REPORT" in text:
        text = text.replace("FINAL REPORT", "").strip()
    return text

def process_dataframe_with_contradictions(
    df: pd.DataFrame,
    report_col: str = "report",
    label_col: Optional[str] = "label",
) -> pd.DataFrame:
    """
    For each row, create:
      - 'contradictory_cxr_prior' (the 5th new column): PRIOR chest X-ray report with opposite interpretation.
      - Four fixed-order distractor columns (non-chest, fixed modalities), each opposite to CURRENT.
    """
    out = df.copy()

    # Prepare output columns
    out["contradictory_cxr_prior"] = ""
    for d in PRESET_DISTRACTORS:
        out[d["col"]] = ""

    print(f"Generating histories for {len(out)} reports...")

    for idx, rpt in tqdm(out[report_col].items(), total=len(out)):
        lbl = out.at[idx, label_col] if label_col and label_col in out.columns else None
        current_is_abn = is_abnormal(lbl)

        # PRIOR contradictory chest X-ray
        prior = generate_prior_contradictory_cxr(rpt, current_is_abn)
        out.at[idx, "contradictory_cxr_prior"] = clean_report(prior)

        # Four fixed-order irrelevant contradictory reports
        distractors = generate_irrelevant_contradictors_fixed(rpt, current_is_abn)
        
        # Clean and assign
        cleaned_distractors = [clean_report(d) for d in distractors]
        
        for d, txt in zip(PRESET_DISTRACTORS, cleaned_distractors):
            out.at[idx, d["col"]] = txt

    return out

if __name__ == "__main__":
    # Example usage
    # df = pd.read_csv("path/to/your/input.csv")
    # result = process_dataframe_with_contradictions(df)
    # result.to_csv("output_with_histories.csv", index=False)
    print("Script for generating contradictory histories. Import `process_dataframe_with_contradictions` to use.")
