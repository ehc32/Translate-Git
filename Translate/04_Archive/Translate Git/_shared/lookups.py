import json_utils as json
import os
from typing import Any, Callable, Dict, Mapping, MutableMapping


def load_lookup_results(path: str | None = None, env_var: str = "TRANSLATE_LOOKUPS_FILE") -> Dict[str, Any]:
    """Load lookup results from the JSON file referenced by ``path`` or ``env_var``.

    Returns an empty dictionary when the environment variable is unset, the file is
    missing, or the payload cannot be parsed as a JSON object.
    """

    if not path:
        path = os.getenv(env_var) or ""
    if not path:
        return {}

    try:
        with open(path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
    except FileNotFoundError:
        return {}
    except Exception:
        return {}

    if isinstance(data, dict):
        return {str(key): value for key, value in data.items()}
    return {}


def get_lookup_value(lookups: Mapping[str, Any] | None, name: str, default: Any = "") -> Any:
    """Fetch ``name`` from ``lookups`` returning ``default`` when absent or empty."""

    if not name or not lookups:
        return default
    try:
        value = lookups.get(name)
    except AttributeError:
        return default

    if value in (None, ""):
        return default
    return value


def seed_lookup_vars(target: MutableMapping[str, Any] | None, lookups: Mapping[str, Any] | None, *, override: bool = False) -> None:
    """Populate ``target`` with lookup aliases so rule engines can read them.

    Keys are coerced to strings to keep behaviour consistent across translators.
    Existing keys remain untouched unless ``override`` is set to True.
    """

    if target is None or not lookups:
        return

    for key, value in lookups.items():
        skey = str(key)
        if not override and skey in target:
            continue
        target[skey] = value


def resolve_lookup_alias(spec: Mapping[str, Any]) -> str:
    """Derive the alias used to store lookup results."""

    alias = spec.get("alias") or spec.get("lookup_alias")
    if alias is not None:
        return str(alias).strip()

    def stringify(part: Any) -> str:
        if isinstance(part, Mapping):
            alias_hint = part.get("alias") or part.get("name")
            if alias_hint:
                return str(alias_hint)
            msg_type = part.get("message_type")
            field_path = part.get("field_path") or part.get("path")
            if msg_type and field_path:
                return f"{msg_type}::{field_path}"
            return str(part)
        return str(part)

    where = stringify(spec.get("where_to_lookup_uetr"))
    extract = stringify(spec.get("field_to_extract"))
    alias = "::".join(part for part in (where, extract) if part)
    if alias:
        return alias
    fallback = spec.get("field_to_extract")
    if isinstance(fallback, Mapping):
        name = fallback.get("field_path") or fallback.get("path") or fallback.get("name")
        if name:
            return str(name)
    return "lookup_result"


def evaluate_from_db_query(
    spec: Mapping[str, Any] | None,
    *,
    evaluate: Callable[[Mapping[str, Any]], Any],
    lookups: Mapping[str, Any] | None,
    default: Any = "",
) -> Any:
    """Resolve a ``from_db_query`` specification using pre-fetched lookup values.

    ``spec`` should describe how the lookup was defined when calling the translation
    service. The resolution flow is:

    1. Evaluate ``which_uetr_to_lookup`` to confirm the current message supplies
       a UETR; trim the result.
    2. Choose the lookup alias (``alias``/``lookup_alias`` optional, otherwise a
       deterministic combination of ``where_to_lookup_uetr`` and
       ``field_to_extract``).
    3. Fetch the alias from ``lookups`` and return it when present.
    4. If absent, evaluate the optional ``fallback`` ValueSpec or return
       ``default``/``fallback_literal``.
    """

    if not isinstance(spec, Mapping):
        return default

    uetr_conf = spec.get("which_uetr_to_lookup")
    if isinstance(uetr_conf, Mapping):
        uetr_value = evaluate(uetr_conf)
    else:
        uetr_value = uetr_conf
    uetr_value = "" if uetr_value is None else str(uetr_value).strip()

    if not uetr_value:
        fallback = spec.get("fallback")
        if isinstance(fallback, Mapping):
            return evaluate(fallback)
        return spec.get("fallback_literal", default)

    alias = resolve_lookup_alias(spec)

    value = get_lookup_value(lookups, alias, None)
    if value in (None, ""):
        fallback = spec.get("fallback")
        if isinstance(fallback, Mapping):
            return evaluate(fallback)
        if "fallback_literal" in spec:
            return spec.get("fallback_literal")
        return default

    return value
