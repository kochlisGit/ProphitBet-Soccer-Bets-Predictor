"""Safe serialization boundary for SaaS model artifacts."""

from typing import Any
from copy import copy
import io
import json
import zipfile

import skops.io as sio

from backend.app.config import setup_ml_path


ARTIFACT_EXTENSION = ".skops"
_TRUSTED_TYPE_PREFIXES = (
    "src.models.",
    "src.preprocessing.",
    "sklearn.",
    "numpy.",
)


def serialize_model(model: Any) -> bytes:
    """Serialize without pickle's arbitrary code execution opcodes."""
    from src.preprocessing.utils.target import TargetType
    portable = copy(model)
    if isinstance(getattr(portable, "_target_type", None), TargetType):
        portable._target_type = portable._target_type.value
    return sio.dumps(portable)


def _normalize_legacy_target(data: bytes) -> bytes:
    """Convert only the known legacy TargetType enum node to a JSON string.

    No objects are constructed and no archive paths are extracted to disk.
    Existing safe models remain usable without falling back to pickle.
    """
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        schema = json.loads(archive.read("schema.json"))
        state = schema.get("content", {}).get("content", {})
        target = state.get("_target_type", {}) if isinstance(state, dict) else {}
        if not (isinstance(target, dict)
                and target.get("__class__") == "TargetType"
                and target.get("__module__") == "src.preprocessing.utils.target"
                and target.get("__loader__") == "ObjectNode"):
            return data
        value_node = target["content"]["content"]["_value_"]
        value = json.loads(value_node["content"])
        if value not in ("result", "over-under"):
            raise ValueError("Unsupported model target type")
        state["_target_type"] = {
            "__class__": "str", "__module__": "builtins", "__loader__": "JsonNode",
            "content": json.dumps(value), "is_json": True, "__id__": target["__id__"],
        }
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w") as normalized:
            for entry in archive.infolist():
                normalized.writestr(entry, json.dumps(schema).encode() if entry.filename == "schema.json" else archive.read(entry))
        return output.getvalue()


def deserialize_model(data: bytes) -> Any:
    """Load only model graphs containing explicitly expected project/ML types."""
    setup_ml_path()
    untrusted_types = sio.get_untrusted_types(data=data)
    rejected = [
        type_name
        for type_name in untrusted_types
        if not type_name.startswith(_TRUSTED_TYPE_PREFIXES)
    ]
    if rejected:
        raise ValueError(
            "Model artifact contains unsupported types: " + ", ".join(sorted(rejected))
        )
    model = sio.loads(_normalize_legacy_target(data), trusted=untrusted_types)
    from src.preprocessing.utils.target import TargetType
    if isinstance(getattr(model, "_target_type", None), str):
        model._target_type = TargetType(model._target_type)
    return model
