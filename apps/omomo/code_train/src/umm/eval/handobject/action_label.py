"""Action label vocabulary + verb extraction for hand-object evaluation.

Per-dataset vocabularies are produced by the data audit and frozen in
`action_labels.json` (assets). Multi-word actions (GRAB only: "hand over",
"switch on", "take picture") collapse to their first token per the audit spec.

Verb rule (verified in assets/data_audit.md Sec.2):
    verb = text.split()[0].lower()

References:
    - Label source: 
    - Audit: same directory, data_audit.md Sec.3 (GRAB) and Sec.4 (ARCTIC).
"""

import json
import os
from typing import Any, Dict


def load_action_labels(dataset: str, labels_json_path: str) -> Dict[str, Any]:
    """Load the {labels, label_to_idx} mapping for the given dataset.

    Args:
        dataset: "grab" or "arctic".
        labels_json_path: Absolute path to action_labels.json produced by the
            data audit.

    Returns:
        Dict with keys:
            labels:       List[str] — sorted verb vocabulary for the dataset.
            label_to_idx: Dict[str, int] — verb -> class index.

    Raises:
        FileNotFoundError: If labels_json_path does not exist.
        ValueError:        If dataset is not in {"grab", "arctic"} or the file
                           is malformed (missing keys).
    """
    if dataset not in ("grab", "arctic"):
        raise ValueError(
            f"load_action_labels: dataset must be 'grab' or 'arctic', got {dataset!r}"
        )
    if not os.path.exists(labels_json_path):
        raise FileNotFoundError(
            f"load_action_labels: file not found: {labels_json_path}"
        )

    with open(labels_json_path, "r") as f:
        full = json.load(f)

    if dataset not in full:
        raise ValueError(
            f"load_action_labels: dataset {dataset!r} missing in {labels_json_path!r}; "
            f"present keys: {list(full.keys())}"
        )

    ds_spec = full[dataset]
    for required in ("labels", "label_to_idx"):
        if required not in ds_spec:
            raise ValueError(
                f"load_action_labels: key {required!r} missing for dataset {dataset!r} "
                f"in {labels_json_path!r}"
            )

    labels = list(ds_spec["labels"])
    label_to_idx: Dict[str, int] = {str(k): int(v) for k, v in ds_spec["label_to_idx"].items()}

    # Consistency check: label_to_idx must match labels bijectively.
    if set(labels) != set(label_to_idx.keys()):
        raise ValueError(
            f"load_action_labels: labels vs label_to_idx mismatch for {dataset!r}. "
            f"labels={labels}, label_to_idx keys={list(label_to_idx.keys())}"
        )
    if sorted(label_to_idx.values()) != list(range(len(labels))):
        raise ValueError(
            f"load_action_labels: label_to_idx indices are not a dense 0..N-1 range "
            f"for {dataset!r}: got {sorted(label_to_idx.values())}"
        )

    return {"labels": labels, "label_to_idx": label_to_idx}


def extract_verb(text: str) -> str:
    """Extract the lowercased first whitespace token from an episode text string.

    Mirrors the audit rule: `text.split()[0].lower()`.

    Args:
        text: Full episode text annotation (e.g., "Pass alarm clock with both hands.").

    Returns:
        The lowercased first token (e.g., "pass").

    Raises:
        ValueError: If text is empty or whitespace-only.
    """
    if not isinstance(text, str):
        raise ValueError(f"extract_verb: expected str, got {type(text).__name__}")
    tokens = text.split()
    if len(tokens) == 0:
        raise ValueError(f"extract_verb: text has no tokens: {text!r}")
    return tokens[0].lower()


def text_to_label_idx(text: str, label_to_idx: Dict[str, int]) -> int:
    """Map episode text -> class index via `extract_verb` -> `label_to_idx`.

    Args:
        text: Full episode text.
        label_to_idx: Mapping from verb to class index (from load_action_labels).

    Returns:
        Integer class index.

    Raises:
        ValueError: If text is empty.
        KeyError:   If the extracted verb is not present in label_to_idx.
    """
    verb = extract_verb(text)
    if verb not in label_to_idx:
        raise KeyError(
            f"text_to_label_idx: verb {verb!r} (from text {text!r}) not in vocabulary; "
            f"vocab size={len(label_to_idx)}"
        )
    return int(label_to_idx[verb])
