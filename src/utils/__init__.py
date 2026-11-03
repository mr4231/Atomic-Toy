"""Result persistence and small analysis helpers."""

from .evidence_guard import (
    CANONICAL_EVIDENCE_ROOT, FROZEN_STUDY_DIRS, ProtectedEvidenceError, assert_safe_output,
    matched_frozen_dir,
)
from .io import load_result, save_result
from .metrics import normalized_position_difference, pairwise_storage_bytes, verification_metrics
from .provenance import (
    PROVENANCE_KEY, array_fingerprint, config_fingerprint, git_state,
    initial_condition_fingerprint, provenance_record, tool_versions,
)

__all__ = [
    "CANONICAL_EVIDENCE_ROOT", "FROZEN_STUDY_DIRS", "PROVENANCE_KEY",
    "ProtectedEvidenceError", "array_fingerprint", "assert_safe_output",
    "config_fingerprint", "git_state", "initial_condition_fingerprint", "load_result",
    "matched_frozen_dir", "normalized_position_difference", "pairwise_storage_bytes",
    "verification_metrics",
    "provenance_record", "save_result", "tool_versions",
]
