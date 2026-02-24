import os
import sys
import re
import json
import random
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime, timezone, timedelta

from sqlalchemy import create_engine, text

from translators._shared.lookups import (
    evaluate_from_db_query,
    load_lookup_results,
    seed_lookup_vars,
)

# === DEBUG / FLOW LOGGING ===
DEBUG = False


def dbg(*a):
    if DEBUG:
        print("[DBG]", *a, file=sys.stderr, flush=True)


# Activar logs de flujo (sin contaminar stdout) con TRANSLATE_LOG_FLOW=1
FLOW_LOG = os.getenv("TRANSLATE_LOG_FLOW", "0").lower() in {"1", "true", "yes", "on"}


def flog(*a):
    if FLOW_LOG or DEBUG:
        print("[T002]", *a, file=sys.stderr, flush=True)


# -----------------------------------------------------------------------
# Configuración de ubicaciones (ajusta según tu entorno)
# -----------------------------------------------------------------------
traslateId = "002"


def _resolve_dir(env_var: str, suffix: str) -> str:
    override = os.getenv(env_var)
    if override:
        candidate = Path(override)
    else:
        files_root = Path(os.getenv("FILES_OUTPUT_ROOT", "./filesOutput"))
        tmp_root = Path(os.getenv("TRANSLATE_TMP_ROOT", str(files_root / "tmp")))
        candidate = tmp_root / "translators" / traslateId / suffix

    try:
        candidate.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(
            f"No se pudo preparar el directorio requerido: {candidate} ({exc})"
        ) from exc
    return str(candidate)


ubicationEntry = _resolve_dir("TRANSLATE_ENTRY_DIR", "entry")
ubicationDestiny = _resolve_dir("TRANSLATE_DESTINY_DIR", "destiny")
ubicationDb = rf"/app/translators/{traslateId}/ubication"

# -----------------------------------------------------------------------
# Params catalog (params.json o tabla dbo.Params)
# -----------------------------------------------------------------------
paramsPath = str(Path(ubicationDb) / "params.json")
# fallbacks opcionales (sandbox/Windows-case)
if not Path(paramsPath).exists():
    alt_params = Path(ubicationDb) / "Params.json"
    if alt_params.exists():
        paramsPath = str(alt_params)
if not Path(paramsPath).exists():
    for mounted in ["/mnt/data/params.json", "/mnt/data/Params.json"]:
        if Path(mounted).exists():
            paramsPath = mounted
            break


def _normalize_values_list(val):
    if val is None:
        return []
    if isinstance(val, list):
        return ["" if v is None else str(v) for v in val]
    return [str(val)]


def read_params_to_dfs(path: str):
    """Devuelve (system_map, user_map) desde params.json:
    { "params": [ {type: "system"|"user", key: "...", values:[...]} ] }"""
    if not Path(path).exists():
        return None, None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    sys_map, usr_map = {}, {}
    for it in data.get("params", []):
        t = str(it.get("type", "")).lower()
        k = str(it.get("key", ""))
        vals = it.get("values", it.get("value", []))
        norm = _normalize_values_list(vals)
        if not k:
            continue
        if t == "system":
            sys_map[k] = norm
        elif t == "user":
            usr_map[k] = norm

    return sys_map, usr_map


def _split_param_values(raw: Any) -> List[str]:
    """Normaliza listas provenientes de Params (DB o archivo) separadas por comas/saltos."""
    if raw is None:
        return []
    text = str(raw).strip()
    if not text:
        return []
    tokens = re.split(r"[,\r\n]+", text)
    return [tok.strip() for tok in tokens if tok and tok.strip()]


def _dict_to_df(data: Dict[str, List[str]]) -> Dict[str, List[str]]:
    return data or {}


def read_params_from_db() -> Tuple[
    Optional[Dict[str, List[str]]], Optional[Dict[str, List[str]]]
]:
    """Consulta la tabla dbo.Params para poblar los catálogos cuando no existe params.json."""
    db_url = os.getenv("SQL_ALCHEMY_DB")
    if not db_url:
        dbg("SQL_ALCHEMY_DB not configured; skipping DB fallback")
        return None, None

    try:
        engine = create_engine(db_url)
    except Exception as exc:
        dbg("Could not create engine for params DB:", exc)
        return None, None

    sys_map: Dict[str, List[str]] = {}
    usr_map: Dict[str, List[str]] = {}
    try:
        with engine.connect() as conn:
            rows = conn.execute(
                text("SELECT [type], [key], [value] FROM dbo.Params")
            ).fetchall()
            bicfi_rows = conn.execute(
                text("SELECT [type], [code] FROM dbo.Bicfi")
            ).fetchall()
    except Exception as exc:
        dbg("Params DB query failed:", exc)
        return None, None
    finally:
        try:
            engine.dispose()
        except Exception:
            pass

    for typ, key, value in rows:
        if not typ or not key:
            continue
        normalized_type = str(typ).strip().lower()
        normalized_key = str(key).strip()
        if not normalized_key or normalized_type not in {"system", "user"}:
            continue
        if normalized_key.lower() == "bicfi":
            continue
        values_list = _split_param_values(value)
        if normalized_type == "system":
            sys_map[normalized_key] = values_list
        else:
            usr_map[normalized_key] = values_list

    for typ, code in bicfi_rows:
        if not typ or not code:
            continue
        normalized_type = str(typ).strip().lower()
        if normalized_type not in {"system", "user"}:
            continue
        bicfi_code = str(code).strip().upper()
        if not bicfi_code:
            continue
        target_map = sys_map if normalized_type == "system" else usr_map
        target_map.setdefault("BICFI", []).append(bicfi_code)

    return _dict_to_df(sys_map), _dict_to_df(usr_map)


_sys_df, _usr_df = read_params_to_dfs(paramsPath)
if _sys_df is None or _usr_df is None:
    _sys_df, _usr_df = read_params_from_db()

valueDbSystemAll = _sys_df if _sys_df is not None else {}
valueDbUserAll = _usr_df if _usr_df is not None else {}


def get_list_from_values_db(values_map: Dict[str, List[str]], key: str) -> List[str]:
    if not values_map:
        return []
    values: List[str] = []
    for entry in values_map.get(key, []):
        values.extend(_split_param_values(entry))
    return values


# -----------------------------------------------------------------------
# Lookup results provided by translate service
# -----------------------------------------------------------------------
LOOKUP_RESULTS: Dict[str, Any] = {}


def _load_lookup_results() -> None:
    global LOOKUP_RESULTS
    LOOKUP_RESULTS = load_lookup_results()


_load_lookup_results()

# -----------------------------------------------------------------------
# Ejemplo de uso (solo referencia para el equipo funcional):
# """
# {
#   "set_var": {
#     "name": "DbtrAcct.Id.Othr.Id",
#     "scope": "global",
#     "value": {
#       "from_db_query": {
#         "alias": "pacs.008::DbtrAcct.Id.Othr.Id",
#         "lookup_message_type": "pacs.008",
#         "which_uetr_to_lookup": { "mtField": "121" },
#         "where_to_lookup_uetr": {
#           "message_type": "pacs.008",
#           "path_kind": "xml",
#           "field_path": ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.PmtId.UETR"
#         },
#         "field_to_extract": {
#           "message_type": "pacs.008",
#           "path_kind": "xml",
#           "field_path": ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.DbtrAcct.Id.Othr.Id"
#         },
#         "fallback_literal": ""
#       }
#     }
#   }
# }
# ""
#
# Este bloque no se ejecuta; sirve como guía rápida de cómo referenciar la
# búsqueda histórica desde la configuración del traductor.

# -----------------------------------------------------------------------
# Parser 103
# -----------------------------------------------------------------------
MT_FIELD_RE = re.compile(r":([0-9A-Z]{2,3}[A-Z]?):")


def parse_mt103(text: str) -> Dict[str, Any]:
    """Extrae campos del MT103. Devuelve {"blocks":{...}, "fields":{tag:value}}"""
    result: Dict[str, Any] = {"blocks": {}, "fields": {}}
    t = text.replace("\r\n", "\n").replace("\r", "\n")

    # Bloques {1:}{2:}{4:...-}
    for m in re.finditer(r"\{(\d):([^}]*)\}", t, re.DOTALL):
        result["blocks"][m.group(1)] = m.group(2)

    # Cuerpo (bloque 4)
    m4 = re.search(r"\{4:\s*(.*?)-\s*\}", t, re.DOTALL)
    payload = m4.group(1) if m4 else t

    fields: Dict[str, str] = {}
    tags = list(re.finditer(r":([0-9A-Z]{2,3}[A-Z]?):", payload))
    positions = [(m.start(), m.group(1)) for m in tags]
    positions.append((len(payload), None))
    for i in range(len(positions) - 1):
        start, tag = positions[i]
        end, _ = positions[i + 1]
        start_real = payload.rfind(":" + tag + ":", 0, start + 5)
        if start_real == -1:
            start_real = start
        value = payload[start_real + len(tag) + 2 : end].strip("\n")
        fields[tag] = value.strip()

    result["fields"] = fields
    return result


# -----------------------------------------------------------------------
# Motor de reglas (genérico, reutilizable)
# -----------------------------------------------------------------------
ValueSpec = Dict[str, Any]
CondSpec = Dict[str, Any]
RuleSpec = Dict[str, Any]
FieldsSpec = Dict[str, Any]


class FieldContext:
    def __init__(self):
        self.vars: Dict[str, Any] = {}


def _mt_value(mt_fields: Dict[str, str], tag: str) -> str:
    return str(mt_fields.get(tag, "") or "").strip()


# --- TRANSFORMS GENÉRICOS (no atados a ningún campo/tag) ----------------
def _fn_regex_extract(
    text_val: str, pattern: str, group: Union[int, List[int]] = 0, flags: int = 0
) -> Union[str, List[str]]:
    """
    Extrae por regex. Si 'group' es int → string; si es lista de ints → lista de strings.
    Retorna "" si no hay match.
    """
    if text_val is None:
        return "" if isinstance(group, int) else []
    m = re.search(pattern, str(text_val), flags)
    if not m:
        return "" if isinstance(group, int) else []
    if isinstance(group, list):
        out = []
        for g in group:
            try:
                out.append(m.group(g))
            except IndexError:
                out.append("")
        return out
    else:
        try:
            return m.group(group)
        except IndexError:
            return ""


def _fn_regex_replace(text_val: str, pattern: str, repl: str, flags: int = 0) -> str:
    if text_val is None:
        return ""
    try:
        return re.sub(pattern, repl, str(text_val), flags=flags)
    except Exception:
        return ""


def _fn_param_value(ctx, args):
    """
    param_value(name) -> str
    Retorna el valor del parámetro fijo 'name' desde ctx.params (o "" si no existe).
    """
    name = str(args.get("name", "") or "")
    return str((getattr(ctx, "params", {}) or {}).get(name, "") or "")


def _count_tag_in_block4(ctx: FieldContext, tag: str) -> int:
    """
    Cuenta ocurrencias de un tag MT específico dentro del bloque 4 crudo.
    Usa ctx.vars['block4'] que ya está cargado en build_fields_from_mt.
    """
    block4 = str(ctx.vars.get("block4", "") or "")
    # Busca ":71F:" o el tag que se indique
    try:
        return len(re.findall(rf":{re.escape(tag)}:", block4))
    except Exception:
        return 0


def _fn_repeat_occurrences(
    ctx: FieldContext, mt_fields: Dict[str, str], args: Dict[str, Any]
):
    """
    repeat_occurrences(mt_tag, emit) -> list
      - mt_tag: string del tag MT a contar (p.ej. '71F', '71G')
      - emit:   ValueSpec que se evaluará por cada ocurrencia
    """
    tag = str(args.get("mt_tag", "")).strip()
    emit_vs = args.get("emit", {"literal": ""})
    n = _count_tag_in_block4(ctx, tag) if tag else 0
    out = []
    for i in range(n):
        # Si tu emit necesita conocer el índice, puedes exponerlo vía ctx.vars temporalmente
        prev_idx = ctx.vars.get("_idx", None)
        ctx.vars["_idx"] = i
        try:
            out.append(_eval_value_mt(ctx, mt_fields, emit_vs))
        finally:
            if prev_idx is None:
                ctx.vars.pop("_idx", None)
            else:
                ctx.vars["_idx"] = prev_idx
    return out


def _fn_normalize_bic(raw: str) -> str:
    if raw is None:
        return ""

    # 1) limpiar y normalizar
    s = str(raw).strip().replace(" ", "")
    s = s.upper()

    # 2) validación alfanumérica
    if not s.isalnum():
        # valor inválido -> no generamos el nodo
        return ""

    # 3) completar si es BIC8
    if len(s) == 8:
        s = s + "XXX"
    elif len(s) == 11:
        # ok tal cual
        pass
    else:
        # longitud inválida -> no generamos el nodo
        return ""

    # 4) validación final: 11 alfanuméricos
    if len(s) != 11:
        return ""

    return s


def _fn_number_format(num_str: str, decimals: int = 2) -> str:
    """Normaliza string numérico (coma/punto) y formatea con n decimales."""
    if num_str is None or str(num_str).strip() == "":
        return ""
    s = str(num_str).strip().replace(",", ".")
    try:
        val = float(s)
        return f"{val:.{int(decimals)}f}"
    except Exception:
        return ""


def _fn_yymmdd_to_yyyy_mm_dd(yymmdd: str) -> str:
    """Convierte YYMMDD → YYYY-MM-DD (regla genérica de fecha corta)."""
    if not yymmdd or not re.match(r"^\d{6}$", yymmdd):
        return ""
    yy = int(yymmdd[:2])
    yyyy = 2000 + yy if yy < 70 else 1900 + yy
    return f"{yyyy}-{yymmdd[2:4]}-{yymmdd[4:6]}"


def _fn_iso_dt_13c(t_raw: str, base_date: str) -> str:
    """
    Convierte HHMM(+/-)HHMM + fecha base a YYYY-MM-DDThh:mm:ssZ.
    base_date puede ser:
      - 'YYMMDD...' (p.ej., derivado con regex de algún campo)
      - 'YYYY-MM-DD'
    Si no hay fecha base válida → retorna "" (para NO crear el nodo).
    """
    # 1) Fecha base
    dt_date: Optional[datetime] = None
    b = (base_date or "").strip()
    if re.match(r"^\d{6}", b):  # YYMMDD...
        yymmdd = b[:6]
        dt = _fn_yymmdd_to_yyyy_mm_dd(yymmdd)
        if dt:
            try:
                dt_date = datetime.strptime(dt, "%Y-%m-%d")
            except Exception:
                dt_date = None
    elif re.match(r"^\d{4}-\d{2}-\d{2}$", b):  # YYYY-MM-DD
        try:
            dt_date = datetime.strptime(b, "%Y-%m-%d")
        except Exception:
            dt_date = None
    if dt_date is None:
        return ""  # sin fecha base NO se crea el nodo

    # 2) Hora + desfase
    t = (t_raw or "").strip()
    m = re.match(r"^(\d{2})(\d{2})([+-])(\d{2})(\d{2})$", t)
    if not m:
        t_clean = "".join(ch for ch in t if ch.isdigit() or ch in "+-")
        m = re.match(r"^(\d{2})(\d{2})([+-])(\d{2})(\d{2})$", t_clean)
    if not m:
        return ""
    hh, mi, sign, oh, om = m.groups()
    hh = int(hh)
    mi = int(mi)
    off_minutes = int(oh) * 60 + int(om)
    if sign == "-":
        off_minutes = -off_minutes
    tz = timezone(timedelta(minutes=off_minutes))
    try:
        local_dt = dt_date.replace(
            hour=hh, minute=mi, second=0, microsecond=0, tzinfo=tz
        )
        utc_dt = local_dt.astimezone(timezone.utc)
        return utc_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return ""


def _fn_now_iso8601() -> str:
    """Devuelve fecha y hora actual en formato ISO8601 con zona -05:00 (Colombia)"""
    colombia_tz = timezone(timedelta(hours=-5))
    return datetime.now(colombia_tz).strftime("%Y-%m-%dT%H:%M:%S-05:00")


def _fn_sender_reference_timestamp_full() -> str:
    """Marca de tiempo extendida con año."""
    return datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")


def _fn_generate_sumid() -> str:
    """
    Genera un identificador único tipo SWIFT SUMID (16 caracteres hexadecimales)
    Ejemplo: 1709623FFFFD7E11
    """
    return "".join(random.choice("0123456789ABCDEF") for _ in range(16))


def _fn_expiry_datetime_plus2() -> str:
    """
    Devuelve la fecha y hora actuales +2 días en formato YYYYMMDDhhmmss (para ExpiryDateTime)
    Ejemplo: si hoy es 2025-10-30 21:03:49 -> devuelve 20251101210349
    """
    now_plus_2 = datetime.now(timezone.utc) + timedelta(days=2)
    return now_plus_2.strftime("%Y%m%d%H%M%S")


def _fn_sender_reference_timestamp() -> str:
    """Marca de tiempo para SenderReference: DDMMYYYYHHMMSS"""
    return datetime.now(timezone.utc).strftime("%d%m%Y%H%M%S")


def _fn_swift_ref_timestamp() -> str:
    """Timestamp para SWIFTRef: YYYY-MM-DDTHH:MM:SS.ssss.ffffffZ"""
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-2] + "Z"


def _fn_snl_ref_timestamp() -> str:
    """Timestamp para SNLRef: YYYY-MM-DDTHH:MM:SS.ssss.ffffffZ"""
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-2] + "Z"


def _fn_snf_input_time() -> str:
    """Timestamp para SnFInputTime: YYYY-MM-DDTHH:MM:SS"""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


def _fn_snf_delivery_time() -> str:
    """Timestamp para SnFDeliveryTime: YYYY-MM-DDTHH:MM:SSZ"""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _fn_dn_from_bicfi(bicfi: str) -> str:
    """
    Genera DN desde BICFI: ou=XXX,o=XXXXXXXX,o=swift
    Toma últimos 3 caracteres y primeros 8 en minúscula
    """
    if not bicfi or len(bicfi) < 8:
        return ""
    last3 = bicfi[-3:].lower()
    first8 = bicfi[:8].lower()
    return f"ou={last3},o={first8},o=swift"


def _eval_value_mt(ctx: FieldContext, mt_fields: Dict[str, str], vs: ValueSpec) -> Any:
    """Evalúa ValueSpec genéricamente."""
    if "mt" in vs:
        return _mt_value(mt_fields, str(vs["mt"]).strip())

    if "mtField" in vs:
        return _mt_value(mt_fields, str(vs["mtField"]).strip())

    if "literal" in vs:
        return str(vs["literal"])

    if "map" in vs:
        m = vs["map"]
        # 1. Evaluamos el input recursivamente
        input_val = _eval_value_mt(ctx, mt_fields, m["input"])
        # 2. Lookup en el diccionario
        mapping = m.get("map", {})
        return mapping.get(str(input_val), input_val)

    if "substr" in vs:
        conf = vs["substr"]
        s = _eval_value_mt(ctx, mt_fields, conf.get("value", {}))
        s = "" if s is None else str(s)
        start = int(conf.get("start", 0))
        ln = conf.get("len", None)
        if ln is None:
            return s[start:]
        return s[start : start + int(ln)]

    if "pad" in vs:
        conf = vs["pad"]
        inner = _eval_value_mt(ctx, mt_fields, conf.get("value", {}))
        n = int(conf.get("len", 0))
        fill = str(conf.get("fill", " "))
        s = "" if inner is None else str(inner)
        return s[:n] if len(s) >= n else s.ljust(n, fill)

    if "concat" in vs:
        parts = [str(_eval_value_mt(ctx, mt_fields, p)) for p in vs.get("concat", [])]
        return "".join(parts)

    if "from_db_query" in vs:
        conf = vs.get("from_db_query")
        resolved = evaluate_from_db_query(
            conf,
            evaluate=lambda nested: _eval_value_mt(ctx, mt_fields, nested),
            lookups=LOOKUP_RESULTS,
        )
        if isinstance(resolved, list):
            return resolved
        return "" if resolved is None else str(resolved)

    if "var" in vs:
        var_name = str(vs["var"])
        val = ctx.vars.get(var_name, "")
        if isinstance(val, (list, dict)):
            return val
        return "" if val is None else str(val)

    if "param" in vs:
        key = str(vs.get("param", "")).strip()
        if not key:
            return []
        scope = str(vs.get("scope", "system")).lower().strip()
        joiner = vs.get("join", None)

        src_df = valueDbUserAll if scope == "user" else valueDbSystemAll

        vals = get_list_from_values_db(src_df, key)  # -> List[str]

        if isinstance(joiner, str):
            return joiner.join(vals)

        return vals

    if "trim" in vs:
        conf = vs["trim"]
        inner = _eval_value_mt(ctx, mt_fields, conf.get("value", {}))
        s = "" if inner is None else str(inner)
        side = str(conf.get("side", "both")).lower()
        chars = conf.get("chars", None)
        if side == "left":
            return s.lstrip(chars)
        if side == "right":
            return s.rstrip(chars)
        return s.strip(chars)

    if "fn" in vs:
        fn = vs.get("fn")
        args = vs.get("args", {})

        if fn == "normalize_bic":
            raw = _eval_value_mt(ctx, mt_fields, args.get("value", {}))
            return _fn_normalize_bic(raw)

        if fn == "regex_extract":
            text_val = _eval_value_mt(ctx, mt_fields, args.get("text", {}))
            pattern = str(args.get("pattern", ""))
            group = args.get("group", 0)
            flags = int(args.get("flags", 0))
            return _fn_regex_extract(text_val, pattern, group, flags)

        if fn == "repeat_occurrences":
            return _fn_repeat_occurrences(ctx, mt_fields, args)

        if fn == "regex_replace":
            text_val = _eval_value_mt(ctx, mt_fields, args.get("text", {}))
            pattern = str(args.get("pattern", ""))
            repl = str(args.get("repl", ""))
            flags = int(args.get("flags", 0))
            return _fn_regex_replace(text_val, pattern, repl, flags)

        if fn == "param_value":
            return _fn_param_value(ctx, args)

        if fn == "number_format":
            num_str = _eval_value_mt(ctx, mt_fields, args.get("value", {}))
            decimals = int(args.get("decimals", 2))
            return _fn_number_format(num_str, decimals)

        if fn == "yymmdd_to_yyyy_mm_dd":
            yymmdd = _eval_value_mt(ctx, mt_fields, args.get("value", {}))
            return _fn_yymmdd_to_yyyy_mm_dd(yymmdd)

        if fn == "iso_dt_13c":
            t_raw = _eval_value_mt(ctx, mt_fields, args.get("time", {}))
            bdat = _eval_value_mt(ctx, mt_fields, args.get("date", {}))
            return _fn_iso_dt_13c(str(t_raw or ""), str(bdat or ""))

        if fn == "now_iso8601":
            return _fn_now_iso8601()

        if fn == "amount_normalize":
            raw = _eval_value_mt(ctx, mt_fields, args.get("value", {}))
            return _fn_amount_normalize(raw)

        if fn == "_fn_sender_reference_timestamp":
            return _fn_sender_reference_timestamp()

        if fn == "_fn_swift_ref_timestamp":
            return _fn_swift_ref_timestamp()

        if fn == "_fn_snl_ref_timestamp":
            return _fn_snl_ref_timestamp()

        if fn == "_fn_snf_input_time":
            return _fn_snf_input_time()

        if fn == "_fn_snf_delivery_time":
            return _fn_snf_delivery_time()

        if fn == "_fn_dn_from_bicfi":
            bicfi = _eval_value_mt(ctx, mt_fields, args.get("bicfi", {}))
            return _fn_dn_from_bicfi(str(bicfi))

        if fn == "_fn_sender_reference_timestamp_full":
            return _fn_sender_reference_timestamp_full()

        if fn == "_fn_generate_sumid":
            return _fn_generate_sumid()

        if fn == "_fn_expiry_datetime_plus2":
            return _fn_expiry_datetime_plus2()

        if fn == "build_saa_dn":
            bic_arg = _eval_value_mt(ctx, mt_fields, args.get("bicfi", {}))
            return build_saa_dn(bic_arg)

        if fn == "lower":
            text_arg = _eval_value_mt(ctx, mt_fields, args.get("text", {}))
            return str(text_arg).lower()

    return ""


def _fn_amount_normalize(raw: str) -> str:
    if raw is None:
        return ""
    s = "".join(ch for ch in str(raw).strip() if ch.isdigit() or ch in ",.")
    if not s:
        return ""

    has_comma = "," in s
    has_dot = "." in s

    def _assemble(int_part: str, dec_part: str) -> str:
        int_part = "".join(ch for ch in int_part if ch.isdigit())
        dec_part = "".join(ch for ch in dec_part if ch.isdigit())
        return f"{int_part}.{dec_part}"

    if has_comma and has_dot:
        last_comma = s.rfind(",")
        last_dot = s.rfind(".")
        dec_sep = "," if last_comma > last_dot else "."
        idx = s.rfind(dec_sep)
        int_part = s[:idx]
        dec_part = s[idx + 1 :]
        int_part = int_part.replace(",", "").replace(".", "")
        return _assemble(int_part, dec_part)

    if has_comma and not has_dot:
        parts = s.split(",")
        if len(parts) > 2:
            int_part = s.replace(",", "")
            return _assemble(int_part, "")
        left, right = parts[0], parts[1]
        if len(right) == 3 and right.isdigit():
            int_part = left + right  # p.ej. "1,234" -> "1234"
            return _assemble(int_part, "")
        int_part = left.replace(",", "")
        return _assemble(int_part, right)

    if has_dot and not has_comma:
        parts = s.split(".")
        if len(parts) > 2:
            int_part = s.replace(".", "")
            return _assemble(int_part, "")
        left, right = parts[0], parts[1]
        if len(right) == 3 and right.isdigit():
            int_part = left + right
            return _assemble(int_part, "")
        return _assemble(left, right)

    return _assemble(s, "")


def _eval_logic_or_condition_mt(
    ctx: FieldContext, mt_fields: Dict[str, str], obj: Any
) -> bool:
    """
    Permite que dentro de las listas de 'all'/'any' haya:
      - condiciones simples (con 'op', 'left', 'right')
      - o subgrupos lógicos que también tengan 'all' / 'any'
    """
    # Si es un grupo lógico (tiene 'all' o 'any'), evaluarlo recursivamente
    if isinstance(obj, dict) and ("all" in obj or "any" in obj):
        return _eval_logic_mt(ctx, mt_fields, obj)

    # Si no, es una condición normal
    return _eval_condition_mt(ctx, mt_fields, obj)


def _eval_condition_mt(
    ctx: FieldContext, mt_fields: Dict[str, str], cond: CondSpec
) -> bool:
    op = cond.get("op")
    left = _eval_value_mt(ctx, mt_fields, cond.get("left", {}))
    rv = cond.get("right", {})
    right = _eval_value_mt(ctx, mt_fields, rv) if isinstance(rv, dict) else rv

    ls = str(left).strip()

    # Igual / distinto
    if op == "=":
        rs = str(right).strip()
        return ls == rs
    if op == "!=":
        rs = str(right).strip()
        return ls != rs

    # in / not in
    if op in ("in", "not in"):
        # right puede ser lista o string separado por comas
        if isinstance(right, (list, tuple, set)):
            options = [str(x).strip() for x in right]
        else:
            options = [s.strip() for s in str(right or "").split(",") if s is not None]
        found = ls in options
        return found if op == "in" else (not found)

    return False


def _eval_logic_mt(
    ctx: FieldContext, mt_fields: Dict[str, str], when: Dict[str, Any]
) -> bool:
    all_ok = True
    any_ok = False

    if "all" in when:
        # Aquí usamos la función que soporta condiciones simples o subgrupos
        all_ok = all(
            _eval_logic_or_condition_mt(ctx, mt_fields, c) for c in when["all"]
        )

    if "any" in when:
        # Igual aquí
        any_ok = any(
            _eval_logic_or_condition_mt(ctx, mt_fields, c) for c in when["any"]
        )

    return (
        (all_ok and any_ok)
        if "all" in when and "any" in when
        else (any_ok if "any" in when else all_ok)
    )


def _field_leaf_and_attr(path: str) -> Tuple[Optional[str], Optional[str]]:
    if not path:
        return None, None
    leaf = path.rsplit(".", 1)[-1]
    attr = None
    if leaf.startswith("@"):
        attr = leaf[1:]
        leaf = ""
    if ".@" in path and leaf.startswith("@"):
        attr = leaf[1:]
        leaf = ""
    if "@" in leaf and not leaf.startswith("@"):
        leaf, attr = leaf.split("@", 1)
    if leaf.endswith("]") and "[" in leaf:
        leaf = leaf[: leaf.rfind("[")]
    leaf = leaf or None
    return leaf, attr


def _normalize_field_output(target_path: str, value: Any) -> Any:
    leaf, attr = _field_leaf_and_attr(target_path)
    if isinstance(value, list):
        normalized: List[Any] = []
        for item in value:
            norm = _normalize_field_output(target_path, item)
            if isinstance(norm, list):
                normalized.extend(norm)
            elif norm not in ("", None):
                normalized.append(norm)
        return normalized
    if isinstance(value, dict):
        if attr and "@attributes" in value:
            attrs = value["@attributes"]
            if isinstance(attrs, dict) and attr in attrs:
                return attrs[attr]
        if leaf and leaf in value:
            return _normalize_field_output(target_path, value[leaf])
        if "#text" in value and len(value) == 1:
            return value["#text"]
    return value


def _collect_then_mt(
    ctx: FieldContext,
    mt_fields: Dict[str, str],
    target_path: str,
    then_obj: Dict[str, Any],
) -> List[str]:
    out: List[str] = []
    if "value" in then_obj:
        v = _normalize_field_output(
            target_path, _eval_value_mt(ctx, mt_fields, then_obj["value"])
        )
        if isinstance(v, list):
            out.extend([str(x) for x in v if str(x).strip() != ""])
        else:
            if str(v).strip() != "":
                out.append(str(v))
    elif "lines" in then_obj:
        for vs in then_obj["lines"]:
            v = _normalize_field_output(target_path, _eval_value_mt(ctx, mt_fields, vs))
            if isinstance(v, list):
                out.extend([str(x) for x in v if str(x).strip() != ""])
            else:
                if str(v).strip() != "":
                    out.append(str(v))
    return out


def _apply_set_mt(
    ctx: FieldContext, mt_fields: Dict[str, str], set_list: List[Dict[str, Any]]
):
    """
    Formato traslate001:
    "set": [
        {"set_var": {"name": "has_20", "value": {"literal": "1"}, "scope": "global"}}
    ]
    """
    if not set_list:
        return
    for item in set_list:
        try:
            if "set_var" in item:
                sv = item["set_var"] or {}
                name = str(sv.get("name", "")).strip()
                scope = str(sv.get("scope", "global")).strip().lower()
                value_spec = sv.get("value")
                val = ""
                if value_spec is not None:
                    val = _eval_value_mt(ctx, mt_fields, value_spec)
                if name and scope == "global":
                    ctx.vars[name] = val
        except Exception:
            pass


def build_fields_from_mt(
    mt: Dict[str, Any], fields_spec: Dict[str, Any]
) -> Dict[str, List[str]]:
    mt_fields = mt.get("fields", {})
    result: Dict[str, List[str]] = {}
    ctx = FieldContext()

    blocks = mt.get("blocks", {}) or {}
    ctx.vars.update(
        {
            "block1": str(blocks.get("1", "") or ""),
            "block2": str(blocks.get("2", "") or ""),
            "block3": str(blocks.get("3", "") or ""),
            "block4": str(blocks.get("4", "") or ""),
        }
    )
    seed_lookup_vars(ctx.vars, LOOKUP_RESULTS)

    for target_path, spec in fields_spec.items():
        produced: List[str] = []
        if isinstance(spec, dict) and spec.get("mode") == "append":
            for rule in spec.get("rules", []):
                if _eval_logic_mt(ctx, mt_fields, rule.get("when", {})):
                    produced.extend(
                        _collect_then_mt(
                            ctx, mt_fields, target_path, rule.get("then", {})
                        )
                    )
                    _apply_set_mt(ctx, mt_fields, rule.get("set", []))
        elif isinstance(spec, dict) and spec.get("mode") == "set":
            for rule in spec.get("rules", []):
                if _eval_logic_mt(ctx, mt_fields, rule.get("when", {})):
                    produced = _collect_then_mt(
                        ctx, mt_fields, target_path, rule.get("then", {})
                    )
                    _apply_set_mt(ctx, mt_fields, rule.get("set", []))
                    break
        elif isinstance(spec, list):
            for rule in spec:
                if _eval_logic_mt(ctx, mt_fields, rule.get("when", {})):
                    produced = _collect_then_mt(
                        ctx, mt_fields, target_path, rule.get("then", {})
                    )
                    _apply_set_mt(ctx, mt_fields, rule.get("set", []))
                    break
        result[target_path] = produced
    return result


def print_fields(fields: Dict[str, List[str]]):
    if DEBUG:
        print("FIELDS:")
        for path, lines in fields.items():
            if not lines:
                print(f"  {path}: <empty>")
                continue
            for i, val in enumerate(lines, 1):
                print(f"  {path}[{i}]: {val}")


def build_saa_dn(bicfi: str) -> str:
    """
    Construye el Distinguished Name (DN) SAA a partir de un BICFI.
    Ejemplo: 'BREPCOBBXXX' -> 'ou=xxx,o=brepcobb,o=swift'
    """
    if not bicfi:
        return "ou=unk,o=unknown,o=swift"
    bicfi = str(bicfi).strip().upper()
    # Normaliza longitud a 11 (BIC + branch)
    if len(bicfi) < 8:
        bicfi = bicfi.ljust(8, "X")
    ou = bicfi[-3:].lower()
    o = bicfi[:8].lower()
    return f"ou={ou},o={o},o=swift"


# -----------------------------------------------------------------------
# Namespaces  creación on-demand optimizada
# -----------------------------------------------------------------------
NS = {
    "Saa": "urn:swift:saa:xsd:saa.2.0",
    "SwSec": "urn:swift:snl:ns.SwSec",
    "SwGbl": "urn:swift:snl:ns.SwGbl",
    "SwInt": "urn:swift:snl:ns.SwInt",
    "Sw": "urn:swift:snl:ns.Sw",
    "head": "urn:iso:std:iso:20022:tech:xsd:head.001.001.02",
    "pacs": "urn:iso:std:iso:20022:tech:xsd:pacs.008.001.08",
    "xsi": "http://www.w3.org/2001/XMLSchema-instance",
}
ET.register_namespace("Saa", NS["Saa"])
ET.register_namespace("Sw", NS["Sw"])
ET.register_namespace("SwInt", NS["SwInt"])
ET.register_namespace("SwGbl", NS["SwGbl"])
ET.register_namespace("SwSec", NS["SwSec"])


NS_MAP = {"Header": "Saa", "Body": "Saa"}


def _el_ns(parent, ns_key, name, text=None):
    """Crea elemento con namespace"""
    # Para head y pacs usamos prefijo literal para evitar ns1/ns2
    if ns_key == "head":
        q = f"head:{name}"
    elif ns_key == "pacs":
        q = f"pacs:{name}"
    elif ns_key in NS:
        q = f"{{{NS[ns_key]}}}{name}"
    else:
        q = f"{ns_key}:{name}"
    e = ET.SubElement(parent, q)
    if text:
        e.text = str(text)
    return e


def create_empty_envelope() -> ET.Element:
    """Crea estructura DataPDU completa según estándar"""
    root = ET.Element(f"{{{NS['Saa']}}}DataPDU")

    # Agregar namespaces en el orden específico requerido
    root.set("xmlns:Sw", NS["Sw"])
    root.set("xmlns:SwInt", NS["SwInt"])
    root.set("xmlns:SwGbl", NS["SwGbl"])
    root.set("xmlns:SwSec", NS["SwSec"])
    # Revision (fijo)
    _el_ns(root, "Saa", "Revision", "2.0.14")

    # NO crear Header y Body vacíos - se crearán cuando sean necesarios
    # Esto evita duplicados

    return root


def _localname(tag: str) -> str:
    """Extrae nombre local"""
    return tag.split("}", 1)[1] if "{" in tag else tag.split(":", 1)[-1]


def _find_or_create(parent, ns, name):
    """Busca o crea elemento hijo - SIEMPRE reutiliza si existe"""
    # Manejar arrays: Signature[0], Signature[1], etc.
    if "[" in name and "]" in name:
        base_name = name.split("[")[0]
        index = int(name.split("[")[1].split("]")[0])

        # Manejar namespaces con prefijo
        if ns == "head":
            tag = f"head:{base_name}"
        elif ns == "pacs":
            tag = f"pacs:{base_name}"
        elif ns in NS:
            tag = f"{{{NS[ns]}}}{base_name}"
        else:
            tag = f"{ns}:{base_name}"

        # Buscar todos los elementos con ese tag
        children = [c for c in parent if c.tag == tag]

        # Si no hay suficientes, crear los faltantes
        while len(children) <= index:
            child = _el_ns(parent, ns, base_name)
            children.append(child)

        return children[index]
    else:
        # Manejar namespaces con prefijo
        if ns == "head":
            tag = f"head:{name}"
        elif ns == "pacs":
            tag = f"pacs:{name}"
        elif ns in NS:
            tag = f"{{{NS[ns]}}}{name}"
        else:
            tag = f"{ns}:{name}"

        # BUSCAR RECURSIVAMENTE en todos los hijos (no solo directos)
        def find_deep(elem, target_tag):
            """Busca un elemento por tag en el árbol"""
            for child in elem:
                if child.tag == target_tag:
                    return child
                # Buscar en profundidad
                found = find_deep(child, target_tag)
                if found is not None:
                    return found
            return None

        # Primero buscar en hijos directos
        child = next((c for c in parent if c.tag == tag), None)

        # Si no se encuentra, buscar en profundidad (para reutilizar Message, etc.)
        if not child and name in [
            "Message",
            "Sender",
            "Receiver",
            "InterfaceInfo",
            "NetworkInfo",
            "SecurityInfo",
            "SWIFTNetSecurityInfo",
        ]:
            child = find_deep(parent, tag)

        if not child:
            child = _el_ns(parent, ns, name)
        return child


def _get_namespace_for_element(name: str, parent_ns: str) -> str:
    """Determina el namespace correcto para un elemento"""
    # Elementos específicos de SwSec
    if name in [
        "Signature",
        "SignedInfo",
        "SignatureValue",
        "KeyInfo",
        "Manifest",
        "SignDN",
        "CertPolicyId",
    ]:
        return "SwSec"
    # Elementos específicos de Sw
    if name in ["DigestValue", "DigestRef", "Object", "RND"]:
        return "Sw"
    # Elementos específicos de SwInt
    if name in ["ValResult"]:
        return "SwInt"
    # Por defecto, mantener el namespace del padre
    return parent_ns


def _ensure_path_with_ns(root: ET.Element, dotted_path: str) -> ET.Element:
    """Crea/encuentra ruta XML con namespaces - REUTILIZA contenedores existentes"""
    parts = [p for p in dotted_path[1:].split(".") if p]  # Quita '.' inicial

    # Determinar namespace base
    if parts[0] in ["Header", "Body"]:
        # SIEMPRE reutilizar el Header o Body existente (creado en create_empty_envelope)
        container = None
        for c in root:
            if _localname(c.tag) == parts[0]:
                container = c
                break

        if container is None:
            container = _el_ns(root, "Saa", parts[0])

        current = container
        parts = parts[1:]

        # Determinar namespace del siguiente nivel
        if parts and parts[0] in NS_MAP:
            ns = NS_MAP[parts[0]]
            current = _find_or_create(current, ns, parts[0])
            if not current.get(f"xmlns:{ns}"):
                current.set(f"xmlns:{ns}", NS[ns])
            parts = parts[1:]
        elif parts:
            # Elementos dentro de Header (Message, etc.) usan namespace Saa
            ns = "Saa"
    else:
        # Compatibilidad con paths antiguos (.AppHdr, .Document)
        ns = NS_MAP.get(parts[0], "pacs")
        # SIEMPRE reutilizar el Body existente
        body = None
        for c in root:
            if _localname(c.tag) == "Body":
                body = c
                break

        if not body:
            # Si no existe (no debería pasar), crear uno
            body = _el_ns(root, "Saa", "Body")

        current = body

        # Manejar AppHdr y Document con sus namespaces específicos
        if parts[0] == "AppHdr":
            # Buscar AppHdr existente
            node = None
            for c in list(current):
                if c.tag == "head:AppHdr":
                    node = c
                    break
            if node is None:
                node = _el_ns(current, "head", "AppHdr")
                node.set("xmlns:head", NS["head"])  # declaración local del prefijo
            current = node
            ns = "head"
            parts = parts[1:]
        elif parts[0] == "Document":
            # Buscar Document existente
            node = None
            for c in list(current):
                if c.tag == "pacs:Document":
                    node = c
                    break
            if node is None:
                node = _el_ns(current, "pacs", "Document")
                node.set("xmlns:pacs", NS["pacs"])  # declaración local del prefijo
            current = node
            ns = "pacs"
            parts = parts[1:]
        elif parts[0] in NS_MAP:
            current = _find_or_create(current, ns, parts[0])
            if not current.get(f"xmlns:{ns}"):
                current.set(f"xmlns:{ns}", NS[ns])
            parts = parts[1:]

    # Resto de niveles - detectar namespace automáticamente
    for name in parts:
        # Manejar arrays [0], [1], etc.
        base_name = name.split("[")[0] if "[" in name else name
        ns = _get_namespace_for_element(base_name, ns)
        current = _find_or_create(current, ns, name)

    return current


def _extract_paths_preorder(el: ET.Element, base: str, out: List[str]):
    """Extrae rutas recursivamente"""
    path = f"{base}.{_localname(el.tag)}" if base else f".{_localname(el.tag)}"
    out.append(path)
    for child in el:
        _extract_paths_preorder(child, path, out)


def _order_from_template(acc_xml_path: Path) -> List[str]:
    """Extrae orden de plantilla"""
    try:
        root = ET.parse(str(acc_xml_path)).getroot()
        order = []
        for section in ["AppHdr", "Document"]:
            node = next((c for c in root if _localname(c.tag) == section), None)
            if node:
                _extract_paths_preorder(node, "", order)
        return order
    except:
        return []


def _gen_paths(base, paths):
    return [f"{base}.{p}" for p in paths]


UNIVERSAL_PACS008_ORDER = [
    # Header.Message
    *_gen_paths(
        ".Header.Message",
        ["SenderReference", "MessageIdentifier", "Format", "SubFormat"],
    ),
    *_gen_paths(".Header.Message.Sender", ["DN", "FullName.X1", "FullName.X2"]),
    *_gen_paths(".Header.Message.Receiver", ["DN", "FullName.X1", "FullName.X2"]),
    *_gen_paths(
        ".Header.Message.InterfaceInfo",
        ["UserReference", "MessageCreator", "MessageContext", "MessageNature", "Sumid"],
    ),
    *_gen_paths(
        ".Header.Message.NetworkInfo",
        ["Priority", "IsPossibleDuplicate", "Service", "Network", "SessionNr", "SeqNr"],
    ),
    *_gen_paths(
        ".Header.Message.NetworkInfo.TransactionData", ["TransactionDataResult"]
    ),
    *_gen_paths(
        ".Header.Message.NetworkInfo", ["TranslatedResult", "TranslationResultDetails"]
    ),
    *_gen_paths(
        ".Header.Message.NetworkInfo.SWIFTNetNetworkInfo",
        [
            "RequestType",
            "RequestSubtype",
            "SWIFTRef",
            "SNLRef",
            "Reference",
            "IsPossibleDuplicateResponse",
            "SnFQueueName",
            "SnFInputTime",
            "SnFDeliveryTime",
        ],
    ),
    *_gen_paths(
        ".Header.Message.NetworkInfo.SWIFTNetNetworkInfo.ValidationDescriptor",
        ["ValResult"],
    ),
    # SecurityInfo - Firma 1
    *_gen_paths(".Header.Message.SecurityInfo", ["TransactionDataSecurityResult"]),
    *_gen_paths(
        ".Header.Message.SecurityInfo.SWIFTNetSecurityInfo", ["SignatureResult"]
    ),
    *_gen_paths(
        ".Header.Message.SecurityInfo.SWIFTNetSecurityInfo.SignatureValue.Signature[0].SignedInfo.Reference",
        ["DigestValue"],
    ),
    *_gen_paths(
        ".Header.Message.SecurityInfo.SWIFTNetSecurityInfo.SignatureValue.Signature[0]",
        ["SignatureValue"],
    ),
    *_gen_paths(
        ".Header.Message.SecurityInfo.SWIFTNetSecurityInfo.SignatureValue.Signature[0].KeyInfo",
        ["SignDN", "CertPolicyId"],
    ),
    *_gen_paths(
        ".Header.Message.SecurityInfo.SWIFTNetSecurityInfo.SignatureValue.Signature[0].Manifest.Reference[0]",
        ["DigestRef", "DigestValue"],
    ),
    *_gen_paths(
        ".Header.Message.SecurityInfo.SWIFTNetSecurityInfo.SignatureValue.Signature[0].Manifest.Reference[1]",
        ["DigestRef", "DigestValue"],
    ),
    *_gen_paths(
        ".Header.Message.SecurityInfo.SWIFTNetSecurityInfo.SignatureValue.Signature[0].Manifest.Reference[2]",
        ["DigestRef", "DigestValue"],
    ),
    *_gen_paths(
        ".Header.Message.SecurityInfo.SWIFTNetSecurityInfo.SignatureValue.Signature[0].Object",
        ["RND"],
    ),
    # SecurityInfo - Firma 2
    *_gen_paths(
        ".Header.Message.SecurityInfo.SWIFTNetSecurityInfo.SignatureValue.Signature[1].SignedInfo.Reference",
        ["DigestValue"],
    ),
    *_gen_paths(
        ".Header.Message.SecurityInfo.SWIFTNetSecurityInfo.SignatureValue.Signature[1]",
        ["SignatureValue"],
    ),
    *_gen_paths(
        ".Header.Message.SecurityInfo.SWIFTNetSecurityInfo.SignatureValue.Signature[1].KeyInfo",
        ["SignDN", "CertPolicyId"],
    ),
    *_gen_paths(
        ".Header.Message.SecurityInfo.SWIFTNetSecurityInfo.SignatureValue.Signature[1].Manifest.Reference[0]",
        ["DigestRef", "DigestValue"],
    ),
    *_gen_paths(
        ".Header.Message.SecurityInfo.SWIFTNetSecurityInfo.SignatureValue.Signature[1].Manifest.Reference[1]",
        ["DigestRef", "DigestValue"],
    ),
    *_gen_paths(
        ".Header.Message.SecurityInfo.SWIFTNetSecurityInfo.SignatureValue.Signature[1].Manifest.Reference[2]",
        ["DigestRef", "DigestValue"],
    ),
    *_gen_paths(
        ".Header.Message.SecurityInfo.SWIFTNetSecurityInfo.SignatureValue.Signature[1].Manifest.Reference[3]",
        ["DigestRef", "DigestValue"],
    ),
    *_gen_paths(
        ".Header.Message.SecurityInfo.SWIFTNetSecurityInfo.SignatureValue.Signature[1].Object",
        ["RND"],
    ),
    ".Header.Message.ExpiryDateTime",
    ".Body.AppHdr.Fr.FIId.FinInstnId.BICFI",
    ".Body.AppHdr.To.FIId.FinInstnId.BICFI",
    ".Body.AppHdr.BizMsgIdr",
    ".Body.AppHdr.MsgDefIdr",
    ".Body.AppHdr.BizSvc",
    ".Body.AppHdr.CreDt",
    # GrpHdr
    *_gen_paths(
        ".Document.FIToFICstmrCdtTrf.GrpHdr",
        [
            "MsgId",
            "CreDtTm",
            "NbOfTxs",
            "SttlmInf.SttlmMtd",
            "SttlmInf.SttlmAcct.Id.Othr.Id",
            "SttlmInf.SttlmAcct.Id.IBAN",
        ],
    ),
    # Payment ID & Settlement
    *_gen_paths(
        ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf",
        [
            "PmtId.InstrId",
            "PmtId.EndToEndId",
            "PmtId.UETR",
            "IntrBkSttlmAmt.@Ccy",
            "IntrBkSttlmAmt",
            "IntrBkSttlmDt",
            "InstdAmt.@Ccy",
            "InstdAmt",
            "XchgRate",
            "ChrgBr",
            "ChrgsInf.Amt.@Ccy",
            "ChrgsInf.Amt",
            "ClrSysMmbId.MmbId",
            "FinInstnId.Nm",
            "PmtTpInf.ClrChanlPmtTpInf.SvcLvl.Cd",
            "PmtTpInf.CtgyPurp.Cd",
        ],
    ),
    # Agents genéricos: InstgAgt / InstdAgt
    *[
        f".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.{ag}.FinInstnId.{fld}"
        for ag in ["InstgAgt", "InstdAgt"]
        for fld in ["BICFI", "Nm", "PstlAdr.AdrLine"]
    ],
    *_gen_paths(
        ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.IntrmyAgt1.FinInstnId",
        [
            "ClrSysMmbId.ClrSysId.Cd",
            "ClrSysMmbId.MmbId",
            "BICFI",
            "Nm",
            "PstlAdr.AdrLine",
            "PstlAdr.StrtNm",
            "PstlAdr.TwnNm",
            "PstlAdr.Ctry",
        ],
    ),
    # Debtor
    *_gen_paths(
        ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.Dbtr",
        [
            "Nm",
            "Id.OrgId.AnyBIC",
            "PstlAdr.Dept",
            "PstlAdr.StrtNm",
            "PstlAdr.TwnNm",
            "PstlAdr.Ctry",
            "PstlAdr.AdrLine",
        ],
    ),
    *_gen_paths(
        ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.DbtrAcct.Id",
        ["IBAN", "Othr.Id", "Othr.SchmeNm.Cd"],
    ),
    # Debtor Agent
    *_gen_paths(
        ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.DbtrAgt.FinInstnId",
        [
            "BICFI",
            "Nm",
            "ClrSysMmbId.ClrSysId.Cd",
            "ClrSysMmbId.MmbId",
            "PstlAdr.AdrLine",
        ],
    ),
    *_gen_paths(
        ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.DbtrAgtAcct.Id",
        ["IBAN", "Othr.Id", "Othr.SchmeNm.Cd"],
    ),
    # Creditor Agent & Creditor
    *_gen_paths(
        ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.CdtrAgt.FinInstnId",
        [
            "BICFI",
            "ClrSysMmbId.ClrSysId.Cd",
            "ClrSysMmbId.MmbId",
            "Nm",
            "PstlAdr.StrtNm",
            "PstlAdr.TwnNm",
            "PstlAdr.Ctry",
            "PstlAdr.PstCd",
            "PstlAdr.AdrLine",
        ],
    ),
    *_gen_paths(
        ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.CdtrAgtAcct.Id",
        ["IBAN", "Othr.Id", "Othr.SchmeNm.Cd"],
    ),
    *_gen_paths(
        ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.Cdtr",
        [
            "Nm",
            "PstlAdr.StrtNm",
            "PstlAdr.PstCd",
            "PstlAdr.TwnNm",
            "PstlAdr.Ctry",
            "PstlAdr.AdrLine",
        ],
    ),
    *_gen_paths(".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.Cdtr.Id.OrgId", ["AnyBIC"]),
    *_gen_paths(
        ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.CdtrAcct.Id", ["IBAN", "Othr.Id"]
    ),
    # Instructions & Settlement Time
    *_gen_paths(
        ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf",
        [
            "InstrForCdtrAgt.Cd",
            "InstrForCdtrAgt.InstrInf",
            "InstrForNxtAgt[0].InstrInf",
            "InstrForNxtAgt[1].InstrInf",
            "InstrForNxtAgt[2].InstrInf",
            "InstrForNxtAgt[3].InstrInf",
            "InstrForNxtAgt[4].InstrInf",
            "Purp.Cd",
            "Purp.Prtry",
            "SttlmTmIndctn.DbtDtTm",
            "SttlmTmIndctn.CdtDtTm",
            "SttlmTmReq.CLSTm",
            "SttlmTmReq.TillTm",
            "SttlmTmReq.FrTm",
            "SttlmTmReq.RjctTm",
        ],
    ),
    *_gen_paths(
        ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.RmtInf",
        ["Strd.RfrdDocInf.Tp.CdOrPrtry.Cd"],
    ),
]


# Índices precalculados
UNIVERSAL_POS = {p: i for i, p in enumerate(UNIVERSAL_PACS008_ORDER)}
ALLOWED_ELEMENTS = frozenset(UNIVERSAL_PACS008_ORDER)  # frozenset es más rápido


def validate_quality_structure(fields: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Filtra campos según estructura permitida"""

    def is_allowed(path):
        base = path.split(".@")[0] if ".@" in path else path
        # Permitir paths directos
        if base in ALLOWED_ELEMENTS:
            return True
        # Permitir .AppHdr aunque el orden tenga .Body.AppHdr (compatibilidad)
        if base.startswith(".AppHdr"):
            body_path = ".Body" + base
            if body_path in ALLOWED_ELEMENTS:
                return True
        # Permitir .Document aunque el orden tenga .Body.Document (compatibilidad)
        if base.startswith(".Document"):
            return True
        return False

    return {
        path: values
        for path, values in fields.items()
        if is_allowed(path) or dbg(f"ELIMINADO: {path}") or False
    }


def apply_fields_to_xml(
    envelope_root: ET.Element,
    fields: Dict[str, List[str]],
    *,
    template_order=None,
    prefix_priority=None,
):
    """Aplica campos al XML respetando orden jerárquico"""
    fields = validate_quality_structure(fields)

    def sort_key(p):
        base = p.split(".@")[0] if ".@" in p else p

        # Buscar posición en UNIVERSAL_POS
        pos = UNIVERSAL_POS.get(p)
        if pos is None:
            pos = UNIVERSAL_POS.get(base)

        # Si no se encuentra y empieza con .AppHdr, buscar con .Body.AppHdr
        if pos is None and base.startswith(".AppHdr"):
            body_path = ".Body" + base
            pos = UNIVERSAL_POS.get(body_path)

        # Si aún no se encuentra, usar 999999
        if pos is None:
            pos = 999999

        return (
            base.startswith(".Document"),  # AppHdr primero (False=0)
            pos,
            base.count("."),
            base,
        )

    # Agrupar atributos con sus elementos base
    processed_paths = set()

    for path in sorted(fields, key=sort_key):
        if path in processed_paths or not (values := fields.get(path)):
            continue

        if ".@" in path:
            # Es un atributo - procesar junto con su elemento base
            base_path, attr = path.rsplit(".@", 1)

            # Verificar si hay valores para el elemento base
            base_values = fields.get(base_path, [])

            if base_values:
                # Crear elemento con atributo y texto
                for i, base_val in enumerate(base_values):
                    leaf = _ensure_path_with_ns(envelope_root, base_path)
                    # Establecer atributo
                    attr_val = values[i] if i < len(values) else values[-1]
                    leaf.set(attr, str(attr_val))
                    # Establecer texto
                    leaf.text = str(base_val)
                processed_paths.add(base_path)
            else:
                # Solo atributo sin texto
                leaf = _ensure_path_with_ns(envelope_root, base_path)
                for val in values:
                    leaf.set(attr, str(val))
            processed_paths.add(path)
        else:
            # Es un elemento - verificar si ya fue procesado con atributo
            if path not in processed_paths:
                # Verificar si tiene atributos pendientes
                has_attrs = any(p.startswith(path + ".@") for p in fields.keys())

                if not has_attrs:
                    # No tiene atributos, procesar normalmente
                    for val in values:
                        _ensure_path_with_ns(envelope_root, path).text = str(val)
                    processed_paths.add(path)


# -----------------------------------------------------------------------
# ESPECIFICACIONES (solo lo que se haya definido: nada extra)
# -----------------------------------------------------------------------
fields_spec: FieldsSpec = {
    # Part 1 - Application DATAPDU Header
    # =============================== -->
    ".Header.Message.SenderReference": {
        "mode": "append",
        "rules": [
            {
                "when": {
                    "all": [
                        {
                            "op": "!=",
                            "left": {
                                "trim": {
                                    "value": {
                                        "substr": {
                                            "value": {"var": "block1"},
                                            "start": 3,
                                            "len": 11,
                                        }
                                    }
                                }
                            },
                            "right": "",
                        }
                    ]
                },
                "then": {
                    "value": {
                        "concat": [
                            {"literal": "I"},
                            {
                                "substr": {
                                    "value": {"var": "block1"},
                                    "start": 3,
                                    "len": 11,
                                }
                            },
                            {"literal": "008$"},
                            {"fn": "_fn_sender_reference_timestamp"},
                        ]
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_Header.Message.SenderReference",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            }
        ],
    },
    ".Header.Message.MessageIdentifier": {
        "mode": "append",
        "rules": [
            {
                "then": {"value": {"literal": "pacs.008.001.08"}},
                "set": [
                    {
                        "set_var": {
                            "name": "has_Message.MessageIdentifier",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            }
        ],
    },
    ".Header.Message.Format": {
        "mode": "append",
        "rules": [
            {
                "then": {"value": {"literal": "MX"}},
                "set": [
                    {
                        "set_var": {
                            "name": "has_Message.Format",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            }
        ],
    },
    ".Header.Message.SubFormat": {
        "mode": "append",
        "rules": [
            # --- Caso I = Input ---
            {
                "when": {
                    "all": [
                        {
                            "op": "=",
                            "left": {
                                "substr": {
                                    "value": {"var": "block2"},
                                    "start": 0,
                                    "len": 1,
                                }
                            },
                            "right": "I",
                        }
                    ]
                },
                "then": {"value": {"literal": "Input"}},
                "set": [
                    {
                        "set_var": {
                            "name": "has_Header.Message.SubFormat",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            },
            {
                "when": {
                    "all": [
                        {
                            "op": "=",
                            "left": {
                                "substr": {
                                    "value": {"var": "block2"},
                                    "start": 0,
                                    "len": 1,
                                }
                            },
                            "right": "O",
                        }
                    ]
                },
                "then": {"value": {"literal": "Output"}},
                "set": [
                    {
                        "set_var": {
                            "name": "has_Header.Message.SubFormat",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            },
        ],
    },
    ".Header.Message.Sender.DN": {
        "mode": "append",
        "rules": [
            {
                "when": {"exists": "block1"},
                "then": {
                    "value": {
                        "fn": "build_saa_dn",
                        "args": {
                            "bicfi": {
                                "concat": [
                                    {
                                        "substr": {
                                            "value": {"var": "block1"},
                                            "start": 3,
                                            "len": 8,
                                        }
                                    },
                                    {
                                        "substr": {
                                            "value": {"var": "block1"},
                                            "start": 12,
                                            "len": 3,
                                        }
                                    },
                                ]
                            }
                        },
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_Message.Sender.DN",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            }
        ],
    },
    # --------FullName-------
    ".Header.Message.Sender.FullName.X1": {
        "mode": "append",
        "rules": [
            {
                "when": {
                    "all": [
                        {
                            "op": "!=",
                            "left": {
                                "trim": {
                                    "value": {
                                        "substr": {
                                            "value": {"var": "block1"},
                                            "start": 3,
                                            "len": 8,
                                        }
                                    }
                                }
                            },
                            "right": "",
                        }
                    ]
                },
                "then": {
                    "value": {
                        "concat": [
                            {
                                "substr": {
                                    "value": {"var": "block1"},
                                    "start": 3,
                                    "len": 8,
                                }
                            },
                            {
                                "substr": {
                                    "value": {"var": "block1"},
                                    "start": 12,
                                    "len": 3,
                                }
                            },
                        ]
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_Header.Message.Sender.FullName.X1",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            }
        ],
    },
    ".Header.Message.Sender.FullName.X2": {
        "mode": "append",
        "rules": [
            {
                "when": {
                    "all": [
                        {
                            "op": "!=",
                            "left": {
                                "trim": {
                                    "value": {
                                        "substr": {
                                            "value": {"var": "block1"},
                                            "start": 12,
                                            "len": 3,
                                        }
                                    }
                                }
                            },
                            "right": "",
                        }
                    ]
                },
                "then": {
                    "value": {
                        "fn": "lower",
                        "args": {
                            "text": {
                                "substr": {
                                    "value": {"var": "block1"},
                                    "start": 12,
                                    "len": 3,
                                }
                            }
                        },
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_Header.Message.Sender.FullName.X2",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            }
        ],
    },
    ".Header.Message.Receiver.DN": {
        "mode": "append",
        "rules": [
            {
                "when": {"exists": "block2"},
                "then": {
                    "value": {
                        "fn": "build_saa_dn",
                        "args": {
                            "bicfi": {
                                "concat": [
                                    {
                                        "substr": {
                                            "value": {"var": "block2"},
                                            "start": 4,
                                            "len": 8,
                                        }
                                    },
                                    {
                                        "substr": {
                                            "value": {"var": "block2"},
                                            "start": 13,
                                            "len": 3,
                                        }
                                    },
                                ]
                            }
                        },
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_Message.Receiver.DN",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            }
        ],
    },
    # ------FullName---------
    ".Header.Message.Receiver.FullName.X1": {
        "mode": "set",
        "rules": [
            {
                "when": {
                    "all": [
                        {
                            "op": "!=",
                            "left": {
                                "trim": {
                                    "value": {
                                        "substr": {
                                            "value": {"var": "block2"},
                                            "start": 4,
                                            "len": 8,
                                        }
                                    }
                                }
                            },
                            "right": "",
                        }
                    ]
                },
                "then": {
                    "value": {
                        "concat": [
                            {
                                "substr": {
                                    "value": {"var": "block2"},
                                    "start": 4,
                                    "len": 8,
                                }
                            },
                            {
                                "substr": {
                                    "value": {"var": "block2"},
                                    "start": 12,
                                    "len": 3,
                                }
                            },
                        ]
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_Header.Message.Sender.FullName.X1",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            }
        ],
    },
    ".Header.Message.Receiver.FullName.X2": {
        "mode": "append",
        "rules": [
            {
                "when": {
                    "all": [
                        {
                            "op": "!=",
                            "left": {
                                "trim": {
                                    "value": {
                                        "substr": {
                                            "value": {"var": "block2"},
                                            "start": 13,
                                            "len": 3,
                                        }
                                    }
                                }
                            },
                            "right": "",
                        }
                    ]
                },
                "then": {
                    "value": {
                        "fn": "lower",
                        "args": {
                            "text": {
                                "substr": {
                                    "value": {"var": "block2"},
                                    "start": 13,
                                    "len": 3,
                                }
                            }
                        },
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_Header.Message.Receiver.FullName.X2",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            }
        ],
    },
    # -------InterfaceInfo---------
    ".Header.Message.InterfaceInfo.MessageCreator": {
        "mode": "append",
        "rules": [
            {
                "then": {"value": {"literal": "Messenger"}},
                "set": [
                    {
                        "set_var": {
                            "name": "has_Message.InterfaceInfo.MessageCreator",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            }
        ],
    },
    ".Header.Message.InterfaceInfo.MessageContext": {
        "mode": "append",
        "rules": [
            {
                "then": {"value": {"literal": "Original"}},
                "set": [
                    {
                        "set_var": {
                            "name": "has_Message.InterfaceInfo.MessageContext",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            }
        ],
    },
    ".Header.Message.InterfaceInfo.MessageNature": {
        "mode": "append",
        "rules": [
            {
                "then": {"value": {"literal": "Financial"}},
                "set": [
                    {
                        "set_var": {
                            "name": "has_Message.InterfaceInfo.MessageNature",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            }
        ],
    },
    ".Header.Message.InterfaceInfo.Sumid": {
        "mode": "append",
        "rules": [
            {
                "then": {"value": {"fn": "_fn_generate_sumid"}},
                "set": [
                    {
                        "set_var": {
                            "name": "has_Header.Message.InterfaceInfo.Sumid",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            }
        ],
    },
    # -----NetworkInfo----
    ".Header.Message.NetworkInfo.Priority": {
        "mode": "append",
        "rules": [
            # --- Caso N = Normal ---
            {
                "when": {
                    "all": [
                        {
                            "op": "=",
                            "left": {
                                "substr": {
                                    "value": {"var": "block2"},
                                    "start": 16,
                                    "len": 1,
                                }
                            },
                            "right": "N",
                        }
                    ]
                },
                "then": {"value": {"literal": "Normal"}},
                "set": [
                    {
                        "set_var": {
                            "name": "has_Header.Message.NetworkInfo.Priority",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            },
            # --- Caso U = Urgent ---
            {
                "when": {
                    "all": [
                        {
                            "op": "=",
                            "left": {
                                "substr": {
                                    "value": {"var": "block2"},
                                    "start": 16,
                                    "len": 1,
                                }
                            },
                            "right": "U",
                        }
                    ]
                },
                "then": {"value": {"literal": "Urgent"}},
                "set": [
                    {
                        "set_var": {
                            "name": "has_Header.Message.NetworkInfo.Priority",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            },
        ],
    },
    ".Header.Message.NetworkInfo.IsPossibleDuplicate": {
        "mode": "append",
        "rules": [
            {
                "then": {"value": {"literal": "false"}},
                "set": [
                    {
                        "set_var": {
                            "name": "has_Message.NetworkInfo.IsPossibleDuplicate",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            }
        ],
    },
    ".Header.Message.NetworkInfo.Service": {
        "mode": "append",
        "rules": [
            {
                "then": {"value": {"param": "SWIFT_FINPLUS", "scope": "user"}},
                "set": [
                    {
                        "set_var": {
                            "name": "has_Message.NetworkInfo.Service",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            }
        ],
    },
    # ----Campos Eliminados------
    # ---.Eliminado.Header.Message.NetworkInfo.Network
    # ---.Header.Message.NetworkInfo.SessionNr
    # ---.Header.Message.NetworkInfo.SeqNr
    # ---.Header.Message.NetworkInfo.TransactionData.TransactionDataResult
    # ---.Header.Message.NetworkInfo.TranslatedResult
    # ---.Header.Message.NetworkInfo.TranslationResultDetails
    # ---SWIFTNetNetworkInfo-----
    ".Header.Message.NetworkInfo.SWIFTNetNetworkInfo.RequestType": {
        "mode": "append",
        "rules": [
            {
                "then": {"value": {"literal": "pacs.008.001.08"}},
                "set": [
                    {
                        "set_var": {
                            "name": "has_Message.NetworkInfo.SWIFTNetNetworkInfo.RequestType",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            }
        ],
    },
    ".Header.Message.NetworkInfo.SWIFTNetNetworkInfo.RequestSubtype": {
        "mode": "append",
        "rules": [
            {
                "then": {"value": {"literal": "swift.cbprplus.03"}},
                "set": [
                    {
                        "set_var": {
                            "name": "has_Message.NetworkInfo.SWIFTNetNetworkInfo.RequestSubtype",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            }
        ],
    },
    ".Header.Message.NetworkInfo.SWIFTNetNetworkInfo.IsPossibleDuplicateResponse": {
        "mode": "set",
        "rules": [
            {"when": {"exists": "block4"}, "then": {"value": {"literal": "false"}}}
        ],
    },
    ".Header.Message.Receiver.Sumid": {
        "mode": "set",
        "rules": [
            {
                "when": {"exists": "block4"},
                "then": {"value": {"literal": "89B61DC27B3534D4"}},
            }
        ],
    },
    ".Header.Message.InterfaceInfo.UserReference": {
        "mode": "set",
        "rules": [{"when": {"mt": "20"}, "then": {"value": {"mt": "20"}}}],
    },
    # ------Campos Eliminados------------
    # ------.Header.Message.NetworkInfo.SWIFTNetNetworkInfo.SWIFTRef
    # ------.Header.Message.NetworkInfo.SWIFTNetNetworkInfo.SNLRef
    ".Header.Message.NetworkInfo.SWIFTNetNetworkInfo.Reference": {
        "mode": "append",
        "rules": [
            {
                "when": {
                    "all": [
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"var": "block3"},
                                    "pattern": "(?:\\{121:|:121:)([^\\r\\n}\\s]+)",
                                    "group": 1,
                                },
                            },
                            "right": "",
                        }
                    ]
                },
                "then": {
                    "value": {
                        "concat": [
                            {
                                "fn": "regex_replace",
                                "args": {
                                    "text": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"var": "block3"},
                                            "pattern": "(?:\\{121:|:121:)([^\\r\\n}\\s]+)",
                                            "group": 1,
                                        },
                                    },
                                    "pattern": "-",
                                    "repl": "",
                                },
                            }
                        ]
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_Header.Message.SWIFTNetNetworkInfo.Reference",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            }
        ],
    },
    # -----------Campos Eliminados-------------
    # -----.Header.Message.NetworkInfo.SWIFTNetNetworkInfo.SnFQueueName
    # -----.Header.Message.NetworkInfo.SWIFTNetNetworkInfo.SnFInputTime
    # -----.Header.Message.NetworkInfo.SWIFTNetNetworkInfo.SnFDeliveryTime
    # -----.Header.Message.NetworkInfo.SWIFTNetNetworkInfo.ValidationDescriptor.ValResult
    # ------Bloque SecurityInfo eliminado
    # ---- Expiry Date Time ----
    # -".Header.Message.ExpiryDateTime": {
    # - "mode": "append",
    # - "rules": [
    # -  {
    # -   "then": {"value": {"fn": "_fn_expiry_datetime_plus2", "args": []}},
    # -"set": [
    # - {
    # -  "set_var": {
    # -   "name": "has_Header.Message.ExpiryDateTime",
    # -"value": {"literal": "1"},
    # -"scope": "global",
    # -}
    # -}
    # -],
    # -}
    # -],
    # -},
    # -----------------------------Fin Bloque DataPDU------------------------------------
    # Part 2 - Application Header
    # =============================== -->
    ".AppHdr.Fr.FIId.FinInstnId.BICFI": {
        "mode": "append",
        "rules": [
            {
                "when": {
                    "all": [
                        {
                            "op": "!=",
                            "left": {
                                "trim": {
                                    "value": {
                                        "substr": {
                                            "value": {"var": "block1"},
                                            "start": 3,
                                            "len": 11,
                                        }
                                    }
                                }
                            },
                            "right": "",
                        }
                    ]
                },
                "then": {
                    "value": {
                        "concat": [
                            {
                                "substr": {
                                    "value": {"var": "block1"},
                                    "start": 3,
                                    "len": 8,
                                }
                            },
                            {"literal": "XXX"},
                        ]
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_AppHdr.Fr.FIId.FinInstnId.BICFI",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            }
        ],
    },
    ".AppHdr.To.FIId.FinInstnId.BICFI": {
        "mode": "append",
        "rules": [
            {
                "when": {
                    "all": [
                        {
                            "op": "!=",
                            "left": {
                                "trim": {
                                    "value": {
                                        "substr": {
                                            "value": {"var": "block2"},
                                            "start": 4,
                                            "len": 11,
                                        }
                                    }
                                }
                            },
                            "right": "",
                        }
                    ]
                },
                "then": {
                    "value": {
                        "concat": [
                            {
                                "substr": {
                                    "value": {"var": "block2"},
                                    "start": 4,
                                    "len": 8,
                                }
                            },
                            {
                                "substr": {
                                    "value": {"var": "block2"},
                                    "start": 13,
                                    "len": 3,
                                }
                            },
                        ]
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_AppHdr.To.FIId.FinInstnId.BICFI",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            }
        ],
    },
    ".AppHdr.BizMsgIdr": {
        "mode": "append",
        "rules": [
            {
                "when": {"all": [{"op": "!=", "left": {"mt": "20"}, "right": ""}]},
                "then": {"value": {"mt": "20"}},
                "set": [
                    {
                        "set_var": {
                            "name": "has_AppHdr.BizMsgIdr",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            }
        ],
    },
    ".AppHdr.MsgDefIdr": {
        "mode": "append",
        "rules": [
            {
                "then": {"value": {"literal": "pacs.008.001.08"}},
                "set": [
                    {
                        "set_var": {
                            "name": "has_AppHdr.MsgDefIdr",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            }
        ],
    },
    ".AppHdr.BizSvc": {
        "mode": "append",
        "rules": [
            {
                "then": {"value": {"literal": "swift.cbprplus.03"}},
                "set": [
                    {
                        "set_var": {
                            "name": "has_AppHdr.BizSvc",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            }
        ],
    },
    ".AppHdr.CreDt": {
        "mode": "append",
        "rules": [
            {
                "then": {"value": {"fn": "now_iso8601"}},
                "set": [
                    {
                        "set_var": {
                            "name": "has_AppHdr.CreDt",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            }
        ],
    },
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.PmtId.UETR": {
        "mode": "append",
        "rules": [
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"var": "block3"}, "right": ""},
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"var": "block3"},
                                    "pattern": "(?:\\{121:|:121:)([^\\r\\n}\\s]+)",
                                    "group": 1,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "fn": "regex_extract",
                        "args": {
                            "text": {"var": "block3"},
                            "pattern": "(?:\\{121:|:121:)([^\\r\\n}\\s]+)",
                            "group": 1,
                        },
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_PmtId.UETR",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            }
        ],
    },
    ".Document.FIToFICstmrCdtTrf.GrpHdr.MsgId": {
        "mode": "append",
        "rules": [
            {
                "when": {"all": [{"op": "!=", "left": {"mt": "20"}, "right": ""}]},
                "then": {"value": {"mt": "20"}},
                "set": [
                    {
                        "set_var": {
                            "name": "has_GrpHdr.MsgId",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            }
        ],
    },
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.PmtId.InstrId": {
        "mode": "append",
        "rules": [
            {
                "when": {"all": [{"op": "!=", "left": {"mt": "20"}, "right": ""}]},
                "then": {"value": {"mt": "20"}},
                "set": [
                    {
                        "set_var": {
                            "name": "has_PmtId.InstrId",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            }
        ],
    },
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.SttlmTmIndctn.DbtDtTm": {
        "mode": "append",
        "rules": [
            {
                "when": {
                    "all": [
                        {
                            "op": "=",
                            "left": {
                                "substr": {"value": {"mt": "13C"}, "start": 0, "len": 9}
                            },
                            "right": "/SNDTIME/",
                        },
                        {"op": "!=", "left": {"mt": "13C"}, "right": ""},
                    ]
                },
                "set": [
                    {
                        "set_var": {
                            "name": "BaseDate",
                            "scope": "global",
                            "value": {"literal": "0001-01-01"},
                        }
                    }
                ],
                "then": {
                    "value": {
                        "fn": "iso_dt_13c",
                        "args": {
                            "time": {"substr": {"value": {"mt": "13C"}, "start": 9}},
                            "date": {"var": "BaseDate"},
                        },
                    }
                },
            }
        ],
    },
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.SttlmTmIndctn.CdtDtTm": {
        "mode": "append",
        "rules": [
            {
                "when": {
                    "all": [
                        {
                            "op": "=",
                            "left": {
                                "substr": {"value": {"mt": "13C"}, "start": 0, "len": 9}
                            },
                            "right": "/RNCTIME/",
                        },
                        {"op": "!=", "left": {"mt": "13C"}, "right": ""},
                    ]
                },
                "set": [
                    {
                        "set_var": {
                            "name": "BaseDate",
                            "scope": "global",
                            "value": {"literal": "0001-01-01"},
                        }
                    }
                ],
                "then": {
                    "value": {
                        "fn": "iso_dt_13c",
                        "args": {
                            "time": {"substr": {"value": {"mt": "13C"}, "start": 9}},
                            "date": {"var": "BaseDate"},
                        },
                    }
                },
            }
        ],
    },
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.SttlmTmReq.CLSTm": {
        "mode": "append",
        "rules": [
            {
                "when": {
                    "all": [
                        {
                            "op": "=",
                            "left": {
                                "substr": {"value": {"mt": "13C"}, "start": 0, "len": 9}
                            },
                            "right": "/CLSTIME/",
                        },
                        {"op": "!=", "left": {"mt": "13C"}, "right": ""},
                    ]
                },
                "set": [
                    {
                        "set_var": {
                            "name": "BaseDate",
                            "scope": "global",
                            "value": {"literal": "0001-01-01"},
                        }
                    }
                ],
                "then": {
                    "value": {
                        "fn": "iso_dt_13c",
                        "args": {
                            "time": {"substr": {"value": {"mt": "13C"}, "start": 9}},
                            "date": {"var": "BaseDate"},
                        },
                    }
                },
            }
        ],
    },
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.SttlmTmReq.TillTm": {
        "mode": "append",
        "rules": [
            {
                "when": {
                    "all": [
                        {
                            "op": "=",
                            "left": {
                                "substr": {"value": {"mt": "13C"}, "start": 0, "len": 9}
                            },
                            "right": "/TILTIME/",
                        },
                        {"op": "!=", "left": {"mt": "13C"}, "right": ""},
                    ]
                },
                "set": [
                    {
                        "set_var": {
                            "name": "BaseDate",
                            "scope": "global",
                            "value": {"literal": "0001-01-01"},
                        }
                    }
                ],
                "then": {
                    "value": {
                        "fn": "iso_dt_13c",
                        "args": {
                            "time": {"substr": {"value": {"mt": "13C"}, "start": 9}},
                            "date": {"var": "BaseDate"},
                        },
                    }
                },
            }
        ],
    },
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.SttlmTmReq.FrTm": {
        "mode": "append",
        "rules": [
            {
                "when": {
                    "all": [
                        {
                            "op": "=",
                            "left": {
                                "substr": {"value": {"mt": "13C"}, "start": 0, "len": 9}
                            },
                            "right": "/FROTIME/",
                        },
                        {"op": "!=", "left": {"mt": "13C"}, "right": ""},
                    ]
                },
                "set": [
                    {
                        "set_var": {
                            "name": "BaseDate",
                            "scope": "global",
                            "value": {"literal": "0001-01-01"},
                        }
                    }
                ],
                "then": {
                    "value": {
                        "fn": "iso_dt_13c",
                        "args": {
                            "time": {"substr": {"value": {"mt": "13C"}, "start": 9}},
                            "date": {"var": "BaseDate"},
                        },
                    }
                },
            }
        ],
    },
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.SttlmTmReq.RjctTm": {
        "mode": "append",
        "rules": [
            {
                "when": {
                    "all": [
                        {
                            "op": "=",
                            "left": {
                                "substr": {"value": {"mt": "13C"}, "start": 0, "len": 9}
                            },
                            "right": "/REJTIME/",
                        },
                        {"op": "!=", "left": {"mt": "13C"}, "right": ""},
                    ]
                },
                "set": [
                    {
                        "set_var": {
                            "name": "BaseDate",
                            "scope": "global",
                            "value": {"literal": "0001-01-01"},
                        }
                    }
                ],
                "then": {
                    "value": {
                        "fn": "iso_dt_13c",
                        "args": {
                            "time": {"substr": {"value": {"mt": "13C"}, "start": 9}},
                            "date": {"var": "BaseDate"},
                        },
                    }
                },
            }
        ],
    },
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.InstrForCdtrAgt.Cd": {
        "mode": "append",
        "rules": [
            {
                "when": {
                    "all": [
                        {
                            "op": "in",
                            "left": {
                                "substr": {"value": {"mt": "23E"}, "start": 0, "len": 4}
                            },
                            "right": "CHQB,HOLD,PHOB,TELB",
                        }
                    ]
                },
                "then": {
                    "value": {"substr": {"value": {"mt": "23E"}, "start": 0, "len": 4}}
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_InstrForCdtrAgt.Cd",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            }
        ],
    },
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.InstrForCdtrAgt.InstrInf": {
        "mode": "append",
        "rules": [
            {
                "when": {
                    "all": [
                        {
                            "op": "in",
                            "left": {
                                "substr": {"value": {"mt": "23E"}, "start": 0, "len": 4}
                            },
                            "right": "HOLD,PHOB,CHQB,TELB",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "trim": {
                                    "value": {
                                        "substr": {
                                            "value": {"mt": "23E"},
                                            "start": 6,
                                            "len": 1,
                                        }
                                    }
                                }
                            },
                            "right": "",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "trim": {
                                    "value": {
                                        "substr": {
                                            "value": {"mt": "72"},
                                            "start": 1,
                                            "len": 1,
                                        }
                                    }
                                }
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "concat": [
                            {
                                "trim": {
                                    "value": {
                                        "substr": {
                                            "value": {"mt": "23E"},
                                            "start": 6,
                                            "len": 20,
                                        }
                                    }
                                }
                            },
                            {
                                "trim": {
                                    "value": {
                                        "substr": {
                                            "value": {"mt": "72"},
                                            "start": 0,
                                            "len": 35,
                                        }
                                    }
                                }
                            },
                            {
                                "trim": {
                                    "value": {
                                        "substr": {
                                            "value": {"mt": "72"},
                                            "start": 35,
                                            "len": 35,
                                        }
                                    }
                                }
                            },
                            {
                                "trim": {
                                    "value": {
                                        "substr": {
                                            "value": {"mt": "72"},
                                            "start": 70,
                                            "len": 35,
                                        }
                                    }
                                }
                            },
                            {
                                "trim": {
                                    "value": {
                                        "substr": {
                                            "value": {"mt": "72"},
                                            "start": 105,
                                            "len": 65,
                                        }
                                    }
                                }
                            },
                            {
                                "trim": {
                                    "value": {
                                        "substr": {
                                            "value": {"mt": "72"},
                                            "start": 140,
                                            "len": 35,
                                        }
                                    }
                                }
                            },
                            {
                                "trim": {
                                    "value": {
                                        "substr": {
                                            "value": {"mt": "72"},
                                            "start": 175,
                                            "len": 35,
                                        }
                                    }
                                }
                            },
                        ]
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_InstrForCdtrAgt.InstrInf",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            }
        ],
    },
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.InstgAgt.FinInstnId.BICFI": {
        "mode": "append",
        "rules": [
            {
                "when": {
                    "all": [
                        {
                            "op": "!=",
                            "left": {
                                "trim": {
                                    "value": {
                                        "substr": {
                                            "value": {"var": "block1"},
                                            "start": 3,
                                            "len": 11,
                                        }
                                    }
                                }
                            },
                            "right": "",
                        }
                    ]
                },
                "then": {
                    "value": {
                        "concat": [
                            {
                                "substr": {
                                    "value": {"var": "block1"},
                                    "start": 3,
                                    "len": 8,
                                }
                            },
                            {"literal": "XXX"},
                        ]
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_AppHdr.Fr.FIId.FinInstnId.BICFI",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            }
        ],
    },
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.InstdAgt.FinInstnId.BICFI": {
        "mode": "append",
        "rules": [
            {
                "when": {
                    "all": [
                        {
                            "op": "!=",
                            "left": {
                                "trim": {
                                    "value": {
                                        "substr": {
                                            "value": {"var": "block2"},
                                            "start": 4,
                                            "len": 11,
                                        }
                                    }
                                }
                            },
                            "right": "",
                        }
                    ]
                },
                "then": {
                    "value": {
                        "concat": [
                            {
                                "substr": {
                                    "value": {"var": "block2"},
                                    "start": 4,
                                    "len": 8,
                                }
                            },
                            {
                                "substr": {
                                    "value": {"var": "block2"},
                                    "start": 13,
                                    "len": 3,
                                }
                            },
                        ]
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_AppHdr.To.FIId.FinInstnId.BICFI",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            }
        ],
    },
    # ---- Campo 50K
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.DbtrAcct.Id.Othr.Id": {
        "mode": "set",
        "rules": [
            {
                "set": {
                    "set_var": {
                        "name": "DbtrAcct.Id.Othr.Id",
                        "scope": "global",
                        "value": {
                            "from_db_query": {
                                "alias": "pacs.008::DbtrAcct.Id.Othr.Id",
                                "lookup_message_type": "pacs.008",
                                "which_uetr_to_lookup": {"mtField": "121"},
                                "where_to_lookup_uetr": {
                                    "message_type": "pacs.008",
                                    "path_kind": "xml",
                                    "field_path": ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.PmtId.UETR",
                                },
                                "field_to_extract": {
                                    "message_type": "pacs.008",
                                    "path_kind": "xml",
                                    "field_path": ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.DbtrAcct.Id.Othr.Id",
                                },
                                "fallback_literal": "",
                            }
                        },
                    }
                },
                "when": {
                    "all": [
                        {
                            "op": "in",
                            "left": {
                                "substr": {"value": {"mt": "20"}, "start": 0, "len": 3}
                            },
                            "right": ["662", "663", "651"],
                        },
                        {
                            "op": "!=",
                            "left": {"var": "pacs.008::DbtrAcct.Id.Othr.Id"},
                            "right": "",
                        },
                    ],
                    "any": [
                        {"op": "!=", "left": {"mt": "50F"}, "right": ""},
                        {"op": "!=", "left": {"mt": "50K"}, "right": ""},
                    ],
                },
                "then": {"value": {"var": "pacs.008::DbtrAcct.Id.Othr.Id"}},
            },
            {
                "when": {"all": [{"op": "!=", "left": {"mt": "50K"}, "right": ""}]},
                "then": {
                    "value": {
                        "fn": "regex_extract",
                        "args": {
                            "text": {"mt": "50K"},
                            "pattern": "^/?([A-Za-z0-9]+)",
                            "group": 1,
                        },
                    }
                },
            },
            {
                "when": {"all": [{"op": "!=", "left": {"mt": "50F"}, "right": ""}]},
                "then": {
                    "value": {
                        "fn": "regex_extract",
                        "args": {
                            "text": {"mt": "50F"},
                            "pattern": "^/?([A-Za-z0-9]+)",
                            "group": 1,
                        },
                    }
                },
            },
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "50A"}, "right": ""},
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "50A"},
                                    "pattern": "^\\s*/\\s*([^\\r\\n]+)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            }
                        }
                    }
                },
            },
        ],
    },
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.PmtTpInf.CtgyPurp.Cd": {
        "mode": "append",
        "rules": [
            {
                "when": {
                    "all": [
                        {
                            "op": "in",
                            "left": {
                                "substr": {"value": {"mt": "23E"}, "start": 0, "len": 4}
                            },
                            "right": "CORT,INTC",
                        }
                    ]
                },
                "then": {
                    "value": {"substr": {"value": {"mt": "23E"}, "start": 0, "len": 4}}
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_CtgyPurp.Cd",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            }
        ],
    },
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.PmtTpInf.SvcLvl.Cd": {
        "mode": "append",
        "rules": [
            {
                "when": {
                    "all": [
                        {
                            "op": "in",
                            "left": {
                                "substr": {"value": {"mt": "23E"}, "start": 0, "len": 4}
                            },
                            "right": "SDVA",
                        }
                    ]
                },
                "then": {
                    "value": {"substr": {"value": {"mt": "23E"}, "start": 0, "len": 4}}
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_SvcLvl.Cd",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            }
        ],
    },
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.IntrBkSttlmDt": {
        "mode": "append",
        "rules": [
            {
                "when": {"all": [{"op": "!=", "left": {"mt": "32A"}, "right": ""}]},
                "then": {
                    "value": {
                        "fn": "yymmdd_to_yyyy_mm_dd",
                        "args": {
                            "value": {
                                "substr": {"value": {"mt": "32A"}, "start": 0, "len": 6}
                            }
                        },
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_CdtTrfTxInf.IntrBkSttlmDt",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            }
        ],
    },
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.IntrBkSttlmAmt.@Ccy": {
        "mode": "append",
        "rules": [
            {
                "when": {"all": [{"op": "!=", "left": {"mt": "32A"}, "right": ""}]},
                "then": {
                    "value": {"substr": {"value": {"mt": "32A"}, "start": 6, "len": 3}}
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_CdtTrfTxInf.IntrBkSttlmAmt.@Ccy",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            }
        ],
    },
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.IntrBkSttlmAmt": {
        "mode": "append",
        "rules": [
            {
                "when": {"all": [{"op": "!=", "left": {"mt": "32A"}, "right": ""}]},
                "then": {
                    "value": {
                        "fn": "amount_normalize",
                        "args": {
                            "value": {
                                "substr": {
                                    "value": {"mt": "32A"},
                                    "start": 9,
                                    "len": 15,
                                }
                            }
                        },
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_CdtTrfTxInf.IntrBkSttlmAmt",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            }
        ],
    },
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.InstdAmt.@Ccy": {
        "mode": "append",
        "rules": [
            {
                "when": {"all": [{"op": "!=", "left": {"mt": "33B"}, "right": ""}]},
                "then": {
                    "value": {"substr": {"value": {"mt": "33B"}, "start": 0, "len": 3}}
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_CdtTrfTxInf.InstdAmt.@Ccy",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            }
        ],
    },
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.InstdAmt": {
        "mode": "append",
        "rules": [
            {
                "when": {"all": [{"op": "!=", "left": {"mt": "33B"}, "right": ""}]},
                "then": {
                    "value": {
                        "fn": "amount_normalize",
                        "args": {
                            "value": {
                                "substr": {
                                    "value": {"mt": "33B"},
                                    "start": 3,
                                    "len": 15,
                                }
                            }
                        },
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_CdtTrfTxInf.InstdAmt",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            }
        ],
    },
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.XchgRate": {
        "mode": "append",
        "rules": [
            {
                "when": {"all": [{"op": "!=", "left": {"mt": "36"}, "right": ""}]},
                "then": {
                    "value": {
                        "fn": "amount_normalize",
                        "args": {
                            "value": {
                                "substr": {"value": {"mt": "36"}, "start": 1, "len": 15}
                            }
                        },
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_CdtTrfTxInf.XchgRate",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            }
        ],
    },
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.Dbtr.Id.OrgId.AnyBIC": {
        "mode": "append",
        "rules": [
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "50A"}, "right": ""},
                        {
                            "op": "!=",
                            "left": {
                                "trim": {
                                    "value": {
                                        "substr": {
                                            "value": {"mt": "50A"},
                                            "start": 0,
                                            "len": 11,
                                        }
                                    }
                                }
                            },
                            # "right": {"param": "BICFI", "scope": "user"},
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {"trim": {"value": {"substr": {"value": {"mt": "50A"}}}}}
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_OrgId.AnyBIC",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            },
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "50A"}, "right": ""},
                        {
                            "op": "=",
                            "left": {
                                "substr": {"value": {"mt": "50A"}, "start": 0, "len": 1}
                            },
                            "right": "/",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "trim": {
                                    "value": {
                                        "substr": {
                                            "value": {"mt": "50A"},
                                            "start": 1,
                                            "len": 11,
                                        }
                                    }
                                }
                            },
                            # "right": {"param": "BICFI", "scope": "user"},
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {"substr": {"value": {"mt": "50A"}, "start": 1}}
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_OrgId.AnyBIC",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            },
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "50A"}, "right": ""},
                        {
                            "op": "!=",
                            "left": {
                                "trim": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "50A"},
                                            "pattern": "^(?:[^\r\n]*\r?\n){1}([^\r\n]*)",
                                            "group": 1,
                                        },
                                    }
                                }
                            },
                            # "right": {"param": "BICFI", "scope": "user"},
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "50A"},
                                    "pattern": "^(?:[^\r\n]*\r?\n){1}([^\r\n]*)",
                                    "group": 1,
                                },
                            }
                        }
                    }
                },
            },
        ],
    },
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.DbtrAgt.FinInstnId.BICFI": {
        "mode": "append",
        "rules": [
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "52A"}, "right": ""},
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "52A"},
                                    "pattern": "^(?:[^\\n]*\\n)([^\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            },
                            "right": "",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "52A"},
                                            "pattern": "^(?:[^\\n]*\\n)([^\\n]*)",
                                            "group": 1,
                                            "flags": 8,
                                        },
                                    },
                                    "start": 0,
                                    "len": 8,
                                }
                            },
                            # "right": {"param": "BICFI", "scope": "user"},
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "52A"},
                                    "pattern": "^(?:[^\\n]*\\n)([^\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            }
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_DbtrAgt.FinInstnId.BICFI",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            },
            {
                "when": {"all": [{"op": "!=", "left": {"var": "block1"}, "right": ""}]},
                "then": {
                    "value": {
                        "concat": [
                            {
                                "substr": {
                                    "value": {"var": "block1"},
                                    "start": 3,
                                    "len": 8,
                                }
                            },
                            {
                                "substr": {
                                    "value": {"var": "block1"},
                                    "start": 12,
                                    "len": 3,
                                }
                            },
                        ]
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_DbtrAgt.FinInstnId.BICFI",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            },
        ],
    },
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.DbtrAgtAcct.Id.Othr.Id": {
        "mode": "append",
        "rules": [
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "52A"}, "right": ""},
                        {
                            "op": "=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "52A"},
                                            "pattern": "^([^\\n\\r]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 4,
                                }
                            },
                            "right": "//CH",
                        },
                        {
                            "op": "=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {
                                        "trim": {
                                            "value": {
                                                "substr": {
                                                    "value": {
                                                        "fn": "regex_extract",
                                                        "args": {
                                                            "text": {"mt": "52A"},
                                                            "pattern": "^([^\\n\\r]*)",
                                                            "group": 1,
                                                        },
                                                    },
                                                    "start": 4,
                                                }
                                            }
                                        }
                                    },
                                    "pattern": "^[A-Z]{2}[0-9]{2}[A-Za-z0-9]{1,30}$",
                                    "group": 0,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "52A"},
                                            "pattern": "^([^\\n\\r]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 4,
                                }
                            }
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_DbtrAgtAcct.Id.Othr.Id",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            },
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "53A"}, "right": ""},
                        {
                            "op": "=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "53A"},
                                            "pattern": "^([^\\n\\r]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 2,
                                }
                            },
                            "right": "//",
                        },
                        {
                            "op": "=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {
                                        "trim": {
                                            "value": {
                                                "substr": {
                                                    "value": {
                                                        "fn": "regex_extract",
                                                        "args": {
                                                            "text": {"mt": "53A"},
                                                            "pattern": "^([^\\n\\r]*)",
                                                            "group": 1,
                                                        },
                                                    },
                                                    "start": 2,
                                                }
                                            }
                                        }
                                    },
                                    "pattern": "^[A-Z]{2}[0-9]{2}[A-Za-z0-9]{1,30}$",
                                    "group": 0,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "53A"},
                                            "pattern": "^([^\\n\\r]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 2,
                                }
                            }
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_DbtrAgtAcct.Id.Othr.Id",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            },
        ],
    },
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.DbtrAgtAcct.Id.IBAN": {
        "mode": "append",
        "rules": [
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "52A"}, "right": ""},
                        {
                            "op": "=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "52A"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 4,
                                }
                            },
                            "right": "//CH",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {
                                        "trim": {
                                            "value": {
                                                "substr": {
                                                    "value": {
                                                        "fn": "regex_extract",
                                                        "args": {
                                                            "text": {"mt": "52A"},
                                                            "pattern": "^([^\\r\\n]*)",
                                                            "group": 1,
                                                        },
                                                    },
                                                    "start": 4,
                                                }
                                            }
                                        }
                                    },
                                    "pattern": "^[A-Z]{2}[0-9]{2}[A-Za-z0-9]{1,30}$",
                                    "group": 0,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "52A"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 4,
                                }
                            }
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_DbtrAgtAcct.Id.IBAN",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            },
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "52A"}, "right": ""},
                        {"op": "!=", "left": {"mt": "53A"}, "right": ""},
                        {
                            "op": "=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "52A"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 2,
                                }
                            },
                            "right": "//",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "52A"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 4,
                                }
                            },
                            "right": "//CH",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {
                                        "trim": {
                                            "value": {
                                                "substr": {
                                                    "value": {
                                                        "fn": "regex_extract",
                                                        "args": {
                                                            "text": {"mt": "53A"},
                                                            "pattern": "^([^\\r\\n]*)",
                                                            "group": 1,
                                                        },
                                                    },
                                                    "start": 2,
                                                }
                                            }
                                        }
                                    },
                                    "pattern": "^[A-Z]{2}[0-9]{2}[A-Za-z0-9]{1,30}$",
                                    "group": 0,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "53A"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 2,
                                }
                            }
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_DbtrAgtAcct.Id.IBAN",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            },
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "53A"}, "right": ""},
                        {"op": "!=", "left": {"mt": "52A"}, "right": ""},
                        {
                            "op": "=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "53A"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 4,
                                }
                            },
                            "right": "//CH",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {
                                        "trim": {
                                            "value": {
                                                "substr": {
                                                    "value": {
                                                        "fn": "regex_extract",
                                                        "args": {
                                                            "text": {"mt": "52A"},
                                                            "pattern": "^([^\\r\\n]*)",
                                                            "group": 1,
                                                        },
                                                    },
                                                    "start": 4,
                                                }
                                            }
                                        }
                                    },
                                    "pattern": "^[A-Z]{2}[0-9]{2}[A-Za-z0-9]{1,30}$",
                                    "group": 0,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "52A"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 4,
                                }
                            }
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_DbtrAgtAcct.Id.IBAN",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            },
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "53A"}, "right": ""},
                        {
                            "op": "=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "53A"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 2,
                                }
                            },
                            "right": "//",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "53A"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 4,
                                }
                            },
                            "right": "//CH",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {
                                        "trim": {
                                            "value": {
                                                "substr": {
                                                    "value": {
                                                        "fn": "regex_extract",
                                                        "args": {
                                                            "text": {"mt": "53A"},
                                                            "pattern": "^([^\\r\\n]*)",
                                                            "group": 1,
                                                        },
                                                    },
                                                    "start": 2,
                                                }
                                            }
                                        }
                                    },
                                    "pattern": "^[A-Z]{2}[0-9]{2}[A-Za-z0-9]{1,30}$",
                                    "group": 0,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "53A"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 2,
                                }
                            }
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_DbtrAgtAcct.Id.IBAN",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            },
        ],
    },
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.DbtrAgtAcct.Id.Othr.SchmeNm.Cd": {
        "mode": "append",
        "rules": [
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "52A"}, "right": ""},
                        {
                            "op": "=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "52A"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 4,
                                }
                            },
                            "right": "//CH",
                        },
                    ]
                },
                "then": {"value": {"literal": "CUID"}},
                "set": [
                    {
                        "set_var": {
                            "name": "has_DbtrAgtAcct.Id.Othr.SchmeNm.Cd",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            }
        ],
    },
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.DbtrAgt.FinInstnId.Nm": {
        "mode": "append",
        "rules": [
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "52D"}, "right": ""},
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "52D"},
                                    "pattern": "^(?:[^\\r\\n]*\\n)([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "52D"},
                                    "pattern": "^(?:[^\\r\\n]*\\n)([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            }
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_DbtrAgt.FinInstnId.Nm",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            }
        ],
    },
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.DbtrAgt.FinInstnId.PstlAdr.AdrLine": {
        "mode": "append",
        "rules": [
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "52D"}, "right": ""},
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "52D"},
                                    "pattern": "^(?:[^\\r\\n]*\\n){2}([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "52D"},
                                    "pattern": "^(?:[^\\r\\n]*\\n){2}([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            }
                        }
                    }
                },
            },
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "52D"}, "right": ""},
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "52D"},
                                    "pattern": "^(?:[^\\r\\n]*\\n){3}([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "52D"},
                                    "pattern": "^(?:[^\\r\\n]*\\n){3}([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            }
                        }
                    }
                },
            },
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "52D"}, "right": ""},
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "52D"},
                                    "pattern": "^(?:[^\\r\\n]*\\n){4}([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "52D"},
                                    "pattern": "^(?:[^\\r\\n]*\\n){4}([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            }
                        }
                    }
                },
            },
        ],
    },
    ".Document.FIToFICstmrCdtTrf.GrpHdr.SttlmInf.InstgRmbrsmntAgt.FinInstnId.ClrSysMmbId.ClrSysId.Cd": {
        "mode": "append",
        "rules": [
            # Regla 53A Codigos ----------------
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "53A"}, "right": ""},
                        {
                            "op": "=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "53A"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 2,
                                }
                            },
                            "right": "//",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "53A"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 4,
                                }
                            },
                            "right": "//CH",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "53A"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 4,
                                }
                            },
                            "right": "//FW",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "map": {
                            "input": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "53A"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 2,
                                    "len": 2,
                                }
                            },
                            "map": {
                                "AT": "ATBLZ",
                                "AU": "AUBSB",
                                "BL": "DEBLZ",
                                "CC": "CACPA",
                                "CN": "CNAPS",
                                "ES": "ESNCC",
                                "CP": "USPID",
                                "GR": "GRBIC",
                                "HK": "HKNCC",
                                "IE": "IENCC",
                                "IN": "INFSC",
                                "IT": "ITNCC",
                                "PL": "PLKNR",
                                "PT": "PTNCC",
                                "RU": "RUCBC",
                                "SC": "GBDSC",
                                "SW": "CHSIC",
                                "NZ": "NZNCC",
                                "ZA": "ZANCC",
                            },
                        }
                    }
                },
            },
            # --------------------------------------------------
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "53B"}, "right": ""},
                        {
                            "op": "=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "53B"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 2,
                                }
                            },
                            "right": "//",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "53B"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 4,
                                }
                            },
                            "right": "//CH",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "53B"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 4,
                                }
                            },
                            "right": "//FW",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "53B"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 2,
                                }
                            }
                        }
                    }
                },
            },
            # Regla 53D Codigos ----------------
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "53D"}, "right": ""},
                        {
                            "op": "=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "53D"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 2,
                                }
                            },
                            "right": "//",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "53D"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 4,
                                }
                            },
                            "right": "//CH",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "53D"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 4,
                                }
                            },
                            "right": "//FW",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "map": {
                            "input": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "53D"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 2,
                                    "len": 2,
                                }
                            },
                            "map": {
                                "AT": "ATBLZ",
                                "AU": "AUBSB",
                                "BL": "DEBLZ",
                                "CC": "CACPA",
                                "CN": "CNAPS",
                                "ES": "ESNCC",
                                "CP": "USPID",
                                "GR": "GRBIC",
                                "HK": "HKNCC",
                                "IE": "IENCC",
                                "IN": "INFSC",
                                "IT": "ITNCC",
                                "PL": "PLKNR",
                                "PT": "PTNCC",
                                "RU": "RUCBC",
                                "SC": "GBDSC",
                                "SW": "CHSIC",
                                "NZ": "NZNCC",
                                "ZA": "ZANCC",
                            },
                        }
                    }
                },
            },
            # --------------------------------------------------
            # Regla 54A Codigos ----------------
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "54A"}, "right": ""},
                        {
                            "op": "=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "54A"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 2,
                                }
                            },
                            "right": "//",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "54A"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 4,
                                }
                            },
                            "right": "//CH",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "54A"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 4,
                                }
                            },
                            "right": "//FW",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "map": {
                            "input": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "54A"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 2,
                                    "len": 2,
                                }
                            },
                            "map": {
                                "AT": "ATBLZ",
                                "AU": "AUBSB",
                                "BL": "DEBLZ",
                                "CC": "CACPA",
                                "CN": "CNAPS",
                                "ES": "ESNCC",
                                "CP": "USPID",
                                "GR": "GRBIC",
                                "HK": "HKNCC",
                                "IE": "IENCC",
                                "IN": "INFSC",
                                "IT": "ITNCC",
                                "PL": "PLKNR",
                                "PT": "PTNCC",
                                "RU": "RUCBC",
                                "SC": "GBDSC",
                                "SW": "CHSIC",
                                "NZ": "NZNCC",
                                "ZA": "ZANCC",
                            },
                        }
                    }
                },
            },
            # --------------------------------------------------
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "54B"}, "right": ""},
                        {
                            "op": "=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "54B"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 2,
                                }
                            },
                            "right": "//",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "54B"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 4,
                                }
                            },
                            "right": "//CH",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "54B"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 4,
                                }
                            },
                            "right": "//FW",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "54B"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 2,
                                }
                            }
                        }
                    }
                },
            },
            # Regla 54D Codigos ----------------
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "54D"}, "right": ""},
                        {
                            "op": "=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "54D"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 2,
                                }
                            },
                            "right": "//",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "54D"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 4,
                                }
                            },
                            "right": "//CH",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "54D"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 4,
                                }
                            },
                            "right": "//FW",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "map": {
                            "input": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "54D"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 2,
                                    "len": 2,
                                }
                            },
                            "map": {
                                "AT": "ATBLZ",
                                "AU": "AUBSB",
                                "BL": "DEBLZ",
                                "CC": "CACPA",
                                "CN": "CNAPS",
                                "ES": "ESNCC",
                                "CP": "USPID",
                                "GR": "GRBIC",
                                "HK": "HKNCC",
                                "IE": "IENCC",
                                "IN": "INFSC",
                                "IT": "ITNCC",
                                "PL": "PLKNR",
                                "PT": "PTNCC",
                                "RU": "RUCBC",
                                "SC": "GBDSC",
                                "SW": "CHSIC",
                                "NZ": "NZNCC",
                                "ZA": "ZANCC",
                                "FR": "FRST",
                                "DE": "DEUT",
                            },
                        }
                    }
                },
            },
            # --------------------------------------------------
        ],
    },
    ".Document.FIToFICstmrCdtTrf.GrpHdr.SttlmInf.InstgRmbrsmntAgt.FinInstnId.ClrSysMmbId.MmbId": {
        "mode": "append",
        "rules": [
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "53A"}, "right": ""},
                        {
                            "op": "=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "53A"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 2,
                                }
                            },
                            "right": "//",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "53A"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 4,
                                }
                            },
                            "right": "//CH",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "53A"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 4,
                                }
                            },
                            "right": "//FW",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "53A"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 4,
                                }
                            }
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_InstgRmbrsmntAgt.FinInstnId.ClrSysMmbId.MmbId",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            },
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "53B"}, "right": ""},
                        {
                            "op": "=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "53B"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 2,
                                }
                            },
                            "right": "//",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "53B"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 4,
                                }
                            },
                            "right": "//CH",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "53B"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 4,
                                }
                            },
                            "right": "//FW",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "53B"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 2,
                                }
                            }
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_InstgRmbrsmntAgt.FinInstnId.ClrSysMmbId.MmbId",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            },
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "53D"}, "right": ""},
                        {
                            "op": "=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "53D"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 2,
                                }
                            },
                            "right": "//",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "53D"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 4,
                                }
                            },
                            "right": "//CH",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "53D"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 4,
                                }
                            },
                            "right": "//FW",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "53D"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 4,
                                }
                            }
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_InstgRmbrsmntAgt.FinInstnId.ClrSysMmbId.MmbId",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            },
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "54A"}, "right": ""},
                        {
                            "op": "=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "54A"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 2,
                                }
                            },
                            "right": "//",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "54A"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 4,
                                }
                            },
                            "right": "//CH",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "54A"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 4,
                                }
                            },
                            "right": "//FW",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "54A"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 2,
                                }
                            }
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_InstgRmbrsmntAgt.FinInstnId.ClrSysMmbId.MmbId",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            },
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "54B"}, "right": ""},
                        {
                            "op": "=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "54B"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 2,
                                }
                            },
                            "right": "//",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "54B"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 4,
                                }
                            },
                            "right": "//CH",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "54B"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 4,
                                }
                            },
                            "right": "//FW",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "54B"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 2,
                                }
                            }
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_InstgRmbrsmntAgt.FinInstnId.ClrSysMmbId.MmbId",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            },
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "54D"}, "right": ""},
                        {
                            "op": "=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "54D"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 2,
                                }
                            },
                            "right": "//",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "54D"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 4,
                                }
                            },
                            "right": "//CH",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "54D"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 4,
                                }
                            },
                            "right": "//FW",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "54D"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 4,
                                }
                            }
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_InstgRmbrsmntAgt.FinInstnId.ClrSysMmbId.MmbId",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            },
        ],
    },
    ".Document.FIToFICstmrCdtTrf.GrpHdr.SttlmInf.InstgRmbrsmntAgt.FinInstnId.BICFI": {
        "mode": "append",
        "rules": [
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "53A"}, "right": ""},
                        {
                            "op": "!=",
                            "left": {
                                "trim": {
                                    "value": {
                                        "substr": {
                                            "value": {
                                                "fn": "regex_extract",
                                                "args": {
                                                    "text": {"mt": "53A"},
                                                    "pattern": "^(?:[^\\r\\n]*\\n)([^\\r\\n]*)",
                                                    "group": 1,
                                                    "flags": 8,
                                                },
                                            },
                                            "start": 0,
                                            "len": 8,
                                        }
                                    }
                                }
                            },
                            # "right": {"param": "BICFI", "scope": "user"},
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "pad": {
                            "value": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "53A"},
                                            "pattern": "^(?:[^\\r\\n]*\\n)([^\\r\\n]*)",
                                            "group": 1,
                                            "flags": 8,
                                        },
                                    },
                                    "start": 0,
                                    "len": 8,
                                }
                            },
                            "len": 11,
                            "fill": "X",
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_InstgRmbrsmntAgt.FinInstnId.BICFI",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            },
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "54A"}, "right": ""},
                        {
                            "op": "!=",
                            "left": {
                                "trim": {
                                    "value": {
                                        "substr": {
                                            "value": {
                                                "fn": "regex_extract",
                                                "args": {
                                                    "text": {"mt": "54A"},
                                                    "pattern": "^(?:[^\\r\\n]*\\n)([^\\r\\n]*)",
                                                    "group": 1,
                                                    "flags": 8,
                                                },
                                            },
                                            "start": 0,
                                            "len": 8,
                                        }
                                    }
                                }
                            },
                            # "right": {"param": "BICFI", "scope": "user"},
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "pad": {
                            "value": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "54A"},
                                            "pattern": "^(?:[^\\r\\n]*\\n)([^\\r\\n]*)",
                                            "group": 1,
                                            "flags": 8,
                                        },
                                    },
                                    "start": 0,
                                    "len": 8,
                                }
                            },
                            "len": 11,
                            "fill": "X",
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_InstgRmbrsmntAgt.FinInstnId.BICFI",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            },
        ],
    },
    ".Document.FIToFICstmrCdtTrf.GrpHdr.SttlmInf.SttlmAcct.Id.Othr.Id": {
        "mode": "set",
        "rules": [
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "53B"}, "right": ""},
                        {
                            "op": "=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "53B"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 4,
                                }
                            },
                            "right": "//CH",
                        },
                        {
                            "op": "=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {
                                        "trim": {
                                            "value": {
                                                "substr": {
                                                    "value": {
                                                        "fn": "regex_extract",
                                                        "args": {
                                                            "text": {"mt": "53B"},
                                                            "pattern": "^([^\\r\\n]*)",
                                                            "group": 1,
                                                        },
                                                    },
                                                    "start": 4,
                                                }
                                            }
                                        }
                                    },
                                    "pattern": "^[A-Z]{2}[0-9]{2}[A-Za-z0-9]{1,30}$",
                                    "group": 0,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "53B"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 4,
                                }
                            }
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_SttlmAcct.Id.Othr.Id",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            },
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "53B"}, "right": ""},
                        {
                            "op": "=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "53B"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 2,
                                }
                            },
                            "right": "//",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "53B"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 4,
                                }
                            },
                            "right": "//CH",
                        },
                        {
                            "op": "=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {
                                        "trim": {
                                            "value": {
                                                "substr": {
                                                    "value": {
                                                        "fn": "regex_extract",
                                                        "args": {
                                                            "text": {"mt": "53B"},
                                                            "pattern": "^([^\\r\\n]*)",
                                                            "group": 1,
                                                        },
                                                    },
                                                    "start": 2,
                                                }
                                            }
                                        }
                                    },
                                    "pattern": "^[A-Z]{2}[0-9]{2}[A-Za-z0-9]{1,30}$",
                                    "group": 0,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "53B"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 2,
                                }
                            }
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_SttlmAcct.Id.Othr.Id",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            },
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "53D"}, "right": ""},
                        {
                            "op": "=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "53D"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 4,
                                }
                            },
                            "right": "//CH",
                        },
                        {
                            "op": "=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {
                                        "trim": {
                                            "value": {
                                                "substr": {
                                                    "value": {
                                                        "fn": "regex_extract",
                                                        "args": {
                                                            "text": {"mt": "53D"},
                                                            "pattern": "^([^\\r\\n]*)",
                                                            "group": 1,
                                                        },
                                                    },
                                                    "start": 4,
                                                }
                                            }
                                        }
                                    },
                                    "pattern": "^[A-Z]{2}[0-9]{2}[A-Za-z0-9]{1,30}$",
                                    "group": 0,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "53D"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 4,
                                }
                            }
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_SttlmAcct.Id.Othr.Id",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            },
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "53D"}, "right": ""},
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "53D"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 2,
                                }
                            },
                            "right": "//",
                        },
                        {
                            "op": "=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {
                                        "trim": {
                                            "value": {
                                                "substr": {
                                                    "value": {
                                                        "fn": "regex_extract",
                                                        "args": {
                                                            "text": {"mt": "53D"},
                                                            "pattern": "^([^\\r\\n]*)",
                                                            "group": 1,
                                                        },
                                                    },
                                                    "start": 1,
                                                }
                                            }
                                        }
                                    },
                                    "pattern": "^[A-Z]{2}[0-9]{2}[A-Za-z0-9]{1,30}$",
                                    "group": 0,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "53D"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 1,
                                }
                            }
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_SttlmAcct.Id.Othr.Id",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            },
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "53D"}, "right": ""},
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "53D"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 2,
                                }
                            },
                            "right": "//",
                        },
                        {
                            "op": "=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {
                                        "trim": {
                                            "value": {
                                                "substr": {
                                                    "value": {
                                                        "fn": "regex_extract",
                                                        "args": {
                                                            "text": {"mt": "53D"},
                                                            "pattern": "^([^\\r\\n]*)",
                                                            "group": 1,
                                                        },
                                                    },
                                                    "start": 1,
                                                }
                                            }
                                        }
                                    },
                                    "pattern": "^[A-Z]{2}\\d{2}[A-Z0-9]{10,30}$",
                                    "group": 0,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "53D"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 1,
                                }
                            }
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_SttlmAcct.Id.Othr.Id",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            },
        ],
    },
    ".Document.FIToFICstmrCdtTrf.GrpHdr.SttlmInf.SttlmAcct.Id.IBAN": {
        "mode": "append",
        "rules": [
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "53B"}, "right": ""},
                        {
                            "op": "=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "53B"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 4,
                                }
                            },
                            "right": "//CH",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {
                                        "trim": {
                                            "value": {
                                                "substr": {
                                                    "value": {
                                                        "fn": "regex_extract",
                                                        "args": {
                                                            "text": {"mt": "53B"},
                                                            "pattern": "^([^\\r\\n]*)",
                                                            "group": 1,
                                                        },
                                                    },
                                                    "start": 4,
                                                }
                                            }
                                        }
                                    },
                                    "pattern": "^[A-Z]{2}[0-9]{2}[A-Za-z0-9]{1,30}$",
                                    "group": 0,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "53B"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 4,
                                }
                            }
                        }
                    }
                },
            },
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "53B"}, "right": ""},
                        {
                            "op": "=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "53B"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 2,
                                }
                            },
                            "right": "//",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "53B"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 4,
                                }
                            },
                            "right": "//CH",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {
                                        "trim": {
                                            "value": {
                                                "substr": {
                                                    "value": {
                                                        "fn": "regex_extract",
                                                        "args": {
                                                            "text": {"mt": "53B"},
                                                            "pattern": "^([^\\r\\n]*)",
                                                            "group": 1,
                                                        },
                                                    },
                                                    "start": 2,
                                                }
                                            }
                                        }
                                    },
                                    "pattern": "^[A-Z]{2}[0-9]{2}[A-Za-z0-9]{1,30}$",
                                    "group": 0,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "53B"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 2,
                                }
                            }
                        }
                    }
                },
            },
        ],
    },
    ".Document.FIToFICstmrCdtTrf.GrpHdr.SttlmInf.InstgRmbrsmntAgt.FinInstnId.Nm": {
        "mode": "append",
        "rules": [
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "53D"}, "right": ""},
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "53D"},
                                    "pattern": "^(?:[^\\r\\n]*\\n)([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            },
                            "right": "",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "53D"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 1,
                                }
                            },
                            "right": "/",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "53D"},
                                    "pattern": "^(?:[^\\r\\n]*\\n)([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            }
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_InstgRmbrsmntAgt.FinInstnId.Nm",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            },
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "54D"}, "right": ""},
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "54D"},
                                    "pattern": "^(?:[^\\r\\n]*\\n)([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "54D"},
                                    "pattern": "^(?:[^\\r\\n]*\\n)([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            }
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_InstgRmbrsmntAgt.FinInstnId.Nm",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            },
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "55D"}, "right": ""},
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "55D"},
                                    "pattern": "^(?:[^\\r\\n]*\\n)([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "55D"},
                                    "pattern": "^(?:[^\\r\\n]*\\n)([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            }
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_InstgRmbrsmntAgt.FinInstnId.Nm",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            },
        ],
    },
    ".Document.FIToFICstmrCdtTrf.GrpHdr.SttlmInf.InstgRmbrsmntAgt.FinInstnId.PstlAdr.AdrLine": {
        "mode": "append",
        "rules": [
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "53D"}, "right": ""},
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "53D"},
                                    "pattern": "^(?:[^\\r\\n]*\\n){2}([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            },
                            "right": "",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "53D"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 1,
                                }
                            },
                            "right": "/",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "53D"},
                                    "pattern": "^(?:[^\\r\\n]*\\n){2}([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            }
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_InstgRmbrsmntAgt.PstlAdr.AdrLine",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            },
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "54D"}, "right": ""},
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "54D"},
                                    "pattern": "^(?:[^\\r\\n]*\\n){2}([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "54D"},
                                    "pattern": "^(?:[^\\r\\n]*\\n){2}([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            }
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_InstgRmbrsmntAgt.PstlAdr.AdrLine",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            },
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "55D"}, "right": ""},
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "55D"},
                                    "pattern": "^(?:[^\\r\\n]*\\n){2}([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "55D"},
                                    "pattern": "^(?:[^\\r\\n]*\\n){2}([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            }
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_InstgRmbrsmntAgt.PstlAdr.AdrLine",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            },
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "53D"}, "right": ""},
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "53D"},
                                    "pattern": "^(?:[^\\r\\n]*\\n){3}([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            },
                            "right": "",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "53D"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 1,
                                }
                            },
                            "right": "/",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "53D"},
                                    "pattern": "^(?:[^\\r\\n]*\\n){3}([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            }
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_InstgRmbrsmntAgt.PstlAdr.AdrLine",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            },
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "54D"}, "right": ""},
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "54D"},
                                    "pattern": "^(?:[^\\r\\n]*\\n){3}([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "54D"},
                                    "pattern": "^(?:[^\\r\\n]*\\n){3}([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            }
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_InstgRmbrsmntAgt.PstlAdr.AdrLine",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            },
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "55D"}, "right": ""},
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "55D"},
                                    "pattern": "^(?:[^\\r\\n]*\\n){3}([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "55D"},
                                    "pattern": "^(?:[^\\r\\n]*\\n){3}([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            }
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_InstgRmbrsmntAgt.PstlAdr.AdrLine",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            },
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "53D"}, "right": ""},
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "53D"},
                                    "pattern": "^(?:[^\\r\\n]*\\n){4}([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            },
                            "right": "",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "53D"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 1,
                                }
                            },
                            "right": "/",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "53D"},
                                    "pattern": "^(?:[^\\r\\n]*\\n){4}([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            }
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_InstgRmbrsmntAgt.PstlAdr.AdrLine",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            },
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "54D"}, "right": ""},
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "54D"},
                                    "pattern": "^(?:[^\\r\\n]*\\n){4}([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "54D"},
                                    "pattern": "^(?:[^\\r\\n]*\\n){4}([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            }
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_InstgRmbrsmntAgt.PstlAdr.AdrLine",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            },
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "55D"}, "right": ""},
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "55D"},
                                    "pattern": "^(?:[^\\r\\n]*\\n){4}([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "55D"},
                                    "pattern": "^(?:[^\\r\\n]*\\n){4}([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            }
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_InstgRmbrsmntAgt.PstlAdr.AdrLine",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            },
        ],
    },
    ".Document.FIToFICstmrCdtTrf.GrpHdr.SttlmInf.InstdRmbrsmntAgtAcct.Id.Othr.Id": {
        "mode": "append",
        "rules": [
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "53D"}, "right": ""},
                        {
                            "op": "=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "53D"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 4,
                                }
                            },
                            "right": "//CH",
                        },
                        {
                            "op": "=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {
                                        "trim": {
                                            "value": {
                                                "substr": {
                                                    "value": {
                                                        "fn": "regex_extract",
                                                        "args": {
                                                            "text": {"mt": "53D"},
                                                            "pattern": "^([^\\r\\n]*)",
                                                            "group": 1,
                                                        },
                                                    },
                                                    "start": 4,
                                                }
                                            }
                                        }
                                    },
                                    "pattern": "^[A-Z]{2}[0-9]{2}[A-Za-z0-9]{1,30}$",
                                    "group": 0,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "53D"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 4,
                                }
                            }
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_InstdRmbrsmntAgtAcct.Id.Othr.Id",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            },
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "53D"}, "right": ""},
                        {
                            "op": "=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "53D"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 2,
                                }
                            },
                            "right": "//",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "53D"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 4,
                                }
                            },
                            "right": "//CH",
                        },
                        {
                            "op": "=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {
                                        "trim": {
                                            "value": {
                                                "substr": {
                                                    "value": {
                                                        "fn": "regex_extract",
                                                        "args": {
                                                            "text": {"mt": "53D"},
                                                            "pattern": "^([^\\r\\n]*)",
                                                            "group": 1,
                                                        },
                                                    },
                                                    "start": 2,
                                                }
                                            }
                                        }
                                    },
                                    "pattern": "^[A-Z]{2}[0-9]{2}[A-Za-z0-9]{1,30}$",
                                    "group": 0,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "53D"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 2,
                                }
                            }
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_InstdRmbrsmntAgtAcct.Id.Othr.Id",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            },
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "54A"}, "right": ""},
                        {
                            "op": "=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "54A"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 4,
                                }
                            },
                            "right": "//CH",
                        },
                        {
                            "op": "=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {
                                        "trim": {
                                            "value": {
                                                "substr": {
                                                    "value": {
                                                        "fn": "regex_extract",
                                                        "args": {
                                                            "text": {"mt": "54A"},
                                                            "pattern": "^([^\\r\\n]*)",
                                                            "group": 1,
                                                        },
                                                    },
                                                    "start": 4,
                                                }
                                            }
                                        }
                                    },
                                    "pattern": "^[A-Z]{2}[0-9]{2}[A-Za-z0-9]{1,30}$",
                                    "group": 0,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "53D"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 4,
                                }
                            }
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_InstdRmbrsmntAgtAcct.Id.Othr.Id",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            },
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "54A"}, "right": ""},
                        {
                            "op": "=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "54A"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 2,
                                }
                            },
                            "right": "//",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "54A"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 4,
                                }
                            },
                            "right": "//CH",
                        },
                        {
                            "op": "=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {
                                        "trim": {
                                            "value": {
                                                "substr": {
                                                    "value": {
                                                        "fn": "regex_extract",
                                                        "args": {
                                                            "text": {"mt": "54A"},
                                                            "pattern": "^([^\\r\\n]*)",
                                                            "group": 1,
                                                        },
                                                    },
                                                    "start": 2,
                                                }
                                            }
                                        }
                                    },
                                    "pattern": "^[A-Z]{2}[0-9]{2}[A-Za-z0-9]{1,30}$",
                                    "group": 0,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "53D"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 2,
                                }
                            }
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_InstdRmbrsmntAgtAcct.Id.Othr.Id",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            },
        ],
    },
    ".Document.FIToFICstmrCdtTrf.GrpHdr.SttlmInf.InstdRmbrsmntAgtAcct.Id.IBAN": {
        "mode": "append",
        "rules": [
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "53D"}, "right": ""},
                        {
                            "op": "=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "53D"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 4,
                                }
                            },
                            "right": "//CH",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {
                                        "trim": {
                                            "value": {
                                                "substr": {
                                                    "value": {
                                                        "fn": "regex_extract",
                                                        "args": {
                                                            "text": {"mt": "53D"},
                                                            "pattern": "^([^\\r\\n]*)",
                                                            "group": 1,
                                                        },
                                                    },
                                                    "start": 4,
                                                }
                                            }
                                        }
                                    },
                                    "pattern": "^[A-Z]{2}[0-9]{2}[A-Za-z0-9]{1,30}$",
                                    "group": 0,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "53D"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 4,
                                }
                            }
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_InstdRmbrsmntAgtAcct.Id.IBAN",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            },
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "53D"}, "right": ""},
                        {
                            "op": "=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "53D"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 2,
                                }
                            },
                            "right": "//",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "53D"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 4,
                                }
                            },
                            "right": "//CH",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {
                                        "trim": {
                                            "value": {
                                                "substr": {
                                                    "value": {
                                                        "fn": "regex_extract",
                                                        "args": {
                                                            "text": {"mt": "53D"},
                                                            "pattern": "^([^\\r\\n]*)",
                                                            "group": 1,
                                                        },
                                                    },
                                                    "start": 2,
                                                }
                                            }
                                        }
                                    },
                                    "pattern": "^[A-Z]{2}[0-9]{2}[A-Za-z0-9]{1,30}$",
                                    "group": 0,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "53D"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 2,
                                }
                            }
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_InstdRmbrsmntAgtAcct.Id.IBAN",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            },
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "54A"}, "right": ""},
                        {
                            "op": "=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "54A"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 4,
                                }
                            },
                            "right": "//CH",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {
                                        "trim": {
                                            "value": {
                                                "substr": {
                                                    "value": {
                                                        "fn": "regex_extract",
                                                        "args": {
                                                            "text": {"mt": "54A"},
                                                            "pattern": "^([^\\r\\n]*)",
                                                            "group": 1,
                                                        },
                                                    },
                                                    "start": 4,
                                                }
                                            }
                                        }
                                    },
                                    "pattern": "^[A-Z]{2}[0-9]{2}[A-Za-z0-9]{1,30}$",
                                    "group": 0,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "54A"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 4,
                                }
                            }
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_InstdRmbrsmntAgtAcct.Id.IBAN",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            },
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "54A"}, "right": ""},
                        {
                            "op": "=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "54A"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 2,
                                }
                            },
                            "right": "//",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "54A"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 4,
                                }
                            },
                            "right": "//CH",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {
                                        "trim": {
                                            "value": {
                                                "substr": {
                                                    "value": {
                                                        "fn": "regex_extract",
                                                        "args": {
                                                            "text": {"mt": "54A"},
                                                            "pattern": "^([^\\r\\n]*)",
                                                            "group": 1,
                                                        },
                                                    },
                                                    "start": 2,
                                                }
                                            }
                                        }
                                    },
                                    "pattern": "^[A-Z]{2}[0-9]{2}[A-Za-z0-9]{1,30}$",
                                    "group": 0,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "54A"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 2,
                                }
                            }
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_InstdRmbrsmntAgtAcct.Id.IBAN",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            },
        ],
    },
    ".Document.FIToFICstmrCdtTrf.GrpHdr.SttlmInf.InstdRmbrsmntAgt.FinInstnId.PstlAdr.AdrLine": {
        "mode": "append",
        "rules": [
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "54B"}, "right": ""},
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "54B"},
                                    "pattern": "^(?:[^\\r\\n]*\\n){2}([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "54B"},
                                    "pattern": "^(?:[^\\r\\n]*\\n){2}([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            }
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_InstdRmbrsmntAgt.PstlAdr.AdrLine",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            },
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "54B"}, "right": ""},
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "54B"},
                                    "pattern": "^(?:[^\\r\\n]*\\n){3}([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "54B"},
                                    "pattern": "^(?:[^\\r\\n]*\\n){3}([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            }
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_InstdRmbrsmntAgt.PstlAdr.AdrLine",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            },
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "54B"}, "right": ""},
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "54B"},
                                    "pattern": "^(?:[^\\r\\n]*\\n){4}([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "54B"},
                                    "pattern": "^(?:[^\\r\\n]*\\n){4}([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            }
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_InstdRmbrsmntAgt.PstlAdr.AdrLine",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            },
        ],
    },
    ".Document.FIToFICstmrCdtTrf.GrpHdr.SttlmInf.InstgRmbrsmntAgtAcct.Id.IBAN": {
        "mode": "append",
        "rules": [
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "54B"}, "right": ""},
                        {
                            "op": "=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "54B"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                            "flags": 8,
                                        },
                                    },
                                    "start": 0,
                                    "len": 4,
                                }
                            },
                            "right": "//CH",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {
                                        "trim": {
                                            "value": {
                                                "substr": {
                                                    "value": {
                                                        "fn": "regex_extract",
                                                        "args": {
                                                            "text": {"mt": "54B"},
                                                            "pattern": "^([^\\r\\n]*)",
                                                            "group": 1,
                                                            "flags": 8,
                                                        },
                                                    },
                                                    "start": 4,
                                                }
                                            }
                                        }
                                    },
                                    "pattern": "^[A-Z]{2}[0-9]{2}[A-Za-z0-9]{1,30}$",
                                    "group": 0,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "54B"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                            "flags": 8,
                                        },
                                    },
                                    "start": 4,
                                }
                            }
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_InstgRmbrsmntAgtAcct.Id.IBAN",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            },
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "54B"}, "right": ""},
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "54B"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                            "flags": 8,
                                        },
                                    },
                                    "start": 0,
                                    "len": 2,
                                }
                            },
                            "right": "//",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {
                                        "trim": {
                                            "value": {
                                                "fn": "regex_extract",
                                                "args": {
                                                    "text": {"mt": "54B"},
                                                    "pattern": "^([^\\r\\n]*)",
                                                    "group": 1,
                                                    "flags": 8,
                                                },
                                            }
                                        }
                                    },
                                    "pattern": "^[A-Z]{2}[0-9]{2}[A-Za-z0-9]{1,30}$",
                                    "group": 0,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "54B"},
                                    "pattern": "^([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            }
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_InstgRmbrsmntAgtAcct.Id.IBAN",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            },
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "54D"}, "right": ""},
                        {
                            "op": "=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "54D"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                            "flags": 8,
                                        },
                                    },
                                    "start": 0,
                                    "len": 4,
                                }
                            },
                            "right": "//CH",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {
                                        "trim": {
                                            "value": {
                                                "substr": {
                                                    "value": {
                                                        "fn": "regex_extract",
                                                        "args": {
                                                            "text": {"mt": "54D"},
                                                            "pattern": "^([^\\r\\n]*)",
                                                            "group": 1,
                                                            "flags": 8,
                                                        },
                                                    },
                                                    "start": 4,
                                                }
                                            }
                                        }
                                    },
                                    "pattern": "^[A-Z]{2}[0-9]{2}[A-Za-z0-9]{1,30}$",
                                    "group": 0,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "54D"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                            "flags": 8,
                                        },
                                    },
                                    "start": 4,
                                }
                            }
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_InstgRmbrsmntAgtAcct.Id.IBAN",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            },
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "54D"}, "right": ""},
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "54D"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                            "flags": 8,
                                        },
                                    },
                                    "start": 0,
                                    "len": 2,
                                }
                            },
                            "right": "//",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {
                                        "trim": {
                                            "value": {
                                                "fn": "regex_extract",
                                                "args": {
                                                    "text": {"mt": "54D"},
                                                    "pattern": "^([^\\r\\n]*)",
                                                    "group": 1,
                                                    "flags": 8,
                                                },
                                            }
                                        }
                                    },
                                    "pattern": "^[A-Z]{2}[0-9]{2}[A-Za-z0-9]{1,30}$",
                                    "group": 0,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "54D"},
                                    "pattern": "^([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            }
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_InstgRmbrsmntAgtAcct.Id.IBAN",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            },
        ],
    },
    ".Document.FIToFICstmrCdtTrf.GrpHdr.SttlmInf.InstgRmbrsmntAgtAcct.Id.Othr.Id": {
        "mode": "append",
        "rules": [
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "54B"}, "right": ""},
                        {
                            "op": "=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "54B"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                            "flags": 8,
                                        },
                                    },
                                    "start": 0,
                                    "len": 4,
                                }
                            },
                            "right": "//CH",
                        },
                        {
                            "op": "=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {
                                        "trim": {
                                            "value": {
                                                "substr": {
                                                    "value": {
                                                        "fn": "regex_extract",
                                                        "args": {
                                                            "text": {"mt": "54B"},
                                                            "pattern": "^([^\\r\\n]*)",
                                                            "group": 1,
                                                            "flags": 8,
                                                        },
                                                    },
                                                    "start": 4,
                                                }
                                            }
                                        }
                                    },
                                    "pattern": "^[A-Z]{2}[0-9]{2}[A-Za-z0-9]{1,30}$",
                                    "group": 0,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "54B"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                            "flags": 8,
                                        },
                                    },
                                    "start": 4,
                                }
                            }
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_InstgRmbrsmntAgtAcct.Id.Othr.Id",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            },
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "54B"}, "right": ""},
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "54B"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                            "flags": 8,
                                        },
                                    },
                                    "start": 0,
                                    "len": 2,
                                }
                            },
                            "right": "//",
                        },
                        {
                            "op": "=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {
                                        "trim": {
                                            "value": {
                                                "fn": "regex_extract",
                                                "args": {
                                                    "text": {"mt": "54B"},
                                                    "pattern": "^([^\\r\\n]*)",
                                                    "group": 1,
                                                    "flags": 8,
                                                },
                                            }
                                        }
                                    },
                                    "pattern": "^[A-Z]{2}[0-9]{2}[A-Za-z0-9]{1,30}$",
                                    "group": 0,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "54B"},
                                    "pattern": "^([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            }
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_InstgRmbrsmntAgtAcct.Id.Othr.Id",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            },
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "54D"}, "right": ""},
                        {
                            "op": "=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "54D"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                            "flags": 8,
                                        },
                                    },
                                    "start": 0,
                                    "len": 4,
                                }
                            },
                            "right": "//CH",
                        },
                        {
                            "op": "=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {
                                        "trim": {
                                            "value": {
                                                "substr": {
                                                    "value": {
                                                        "fn": "regex_extract",
                                                        "args": {
                                                            "text": {"mt": "54D"},
                                                            "pattern": "^([^\\r\\n]*)",
                                                            "group": 1,
                                                            "flags": 8,
                                                        },
                                                    },
                                                    "start": 4,
                                                }
                                            }
                                        }
                                    },
                                    "pattern": "^[A-Z]{2}[0-9]{2}[A-Za-z0-9]{1,30}$",
                                    "group": 0,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "54D"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                            "flags": 8,
                                        },
                                    },
                                    "start": 4,
                                }
                            }
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_InstgRmbrsmntAgtAcct.Id.Othr.Id",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            },
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "54D"}, "right": ""},
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "54D"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                            "flags": 8,
                                        },
                                    },
                                    "start": 0,
                                    "len": 2,
                                }
                            },
                            "right": "//",
                        },
                        {
                            "op": "=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {
                                        "trim": {
                                            "value": {
                                                "fn": "regex_extract",
                                                "args": {
                                                    "text": {"mt": "54D"},
                                                    "pattern": "^([^\\r\\n]*)",
                                                    "group": 1,
                                                    "flags": 8,
                                                },
                                            }
                                        }
                                    },
                                    "pattern": "^[A-Z]{2}[0-9]{2}[A-Za-z0-9]{1,30}$",
                                    "group": 0,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "54D"},
                                    "pattern": "^([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            }
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_InstgRmbrsmntAgtAcct.Id.Othr.Id",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            },
        ],
    },
    ".Document.FIToFICstmrCdtTrf.GrpHdr.SttlmInf.InstgRmbrsmntAgtAcct.Id.Othr.SchmeNm.Cd": {
        "mode": "append",
        "rules": [
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "54B"}, "right": ""},
                        {
                            "op": "=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "54B"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                            "flags": 8,
                                        },
                                    },
                                    "start": 0,
                                    "len": 4,
                                }
                            },
                            "right": "//CH",
                        },
                        {
                            "op": "=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {
                                        "trim": {
                                            "value": {
                                                "substr": {
                                                    "value": {
                                                        "fn": "regex_extract",
                                                        "args": {
                                                            "text": {"mt": "54B"},
                                                            "pattern": "^([^\\r\\n]*)",
                                                            "group": 1,
                                                            "flags": 8,
                                                        },
                                                    },
                                                    "start": 4,
                                                }
                                            }
                                        }
                                    },
                                    "pattern": "^[A-Z]{2}[0-9]{2}[A-Za-z0-9]{1,30}$",
                                    "group": 0,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {"value": {"literal": "CUID"}},
                "set": [
                    {
                        "set_var": {
                            "name": "has_InstgRmbrsmntAgtAcct.Id.Othr.SchmeNm.Cd",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            },
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "54D"}, "right": ""},
                        {
                            "op": "=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "54D"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                            "flags": 8,
                                        },
                                    },
                                    "start": 0,
                                    "len": 4,
                                }
                            },
                            "right": "//CH",
                        },
                        {
                            "op": "=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {
                                        "trim": {
                                            "value": {
                                                "substr": {
                                                    "value": {
                                                        "fn": "regex_extract",
                                                        "args": {
                                                            "text": {"mt": "54D"},
                                                            "pattern": "^([^\\r\\n]*)",
                                                            "group": 1,
                                                            "flags": 8,
                                                        },
                                                    },
                                                    "start": 4,
                                                }
                                            }
                                        }
                                    },
                                    "pattern": "^[A-Z]{2}[0-9]{2}[A-Za-z0-9]{1,30}$",
                                    "group": 0,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {"value": {"literal": "CUID"}},
                "set": [
                    {
                        "set_var": {
                            "name": "has_InstgRmbrsmntAgtAcct.Id.Othr.SchmeNm.Cd",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            },
        ],
    },
    ".Document.FIToFICstmrCdtTrf.GrpHdr.SttlmInf.ThrdRmbrsmntAgt.FinInstnId.ClrSysMmbId.ClrSysId.Cd": {
        "mode": "append",
        "rules": [
            # Reglas Codigos 55A ------------------
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "55A"}, "right": ""},
                        {
                            "op": "=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "55A"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 2,
                                }
                            },
                            "right": "//",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "55A"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 4,
                                }
                            },
                            "right": "//CH",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "map": {
                            "input": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "55A"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 2,
                                    "len": 2,
                                }
                            },
                            "map": {"RT": "RTGS", "FW ": "FEDWIRE"},
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_ThrdRmbrsmntAgt.FinInstnId.ClrSysMmbId.ClrSysId.Cd",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            },
            # -----------------------------------------
            # Reglas Codigos 55B ------------------
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "55B"}, "right": ""},
                        {
                            "op": "=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "55B"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 2,
                                }
                            },
                            "right": "//",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "55B"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 4,
                                }
                            },
                            "right": "//CH",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "map": {
                            "input": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "55B"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 2,
                                    "len": 2,
                                }
                            },
                            "map": {
                                "AT": "ATBLZ",
                                "AU": "AUBSB",
                                "BL": "DEBLZ",
                                "CC": "CACPA",
                                "CN": "CNAPS",
                                "ES": "ESNCC",
                                "FW": "USABA",
                                "CP": "USPID",
                                "GR": "GRBIC",
                                "HK": "HKNCC",
                                "IE": "IENCC",
                                "IN": "INFSC",
                                "IT": "ITNCC",
                                "PL": "PLKNR",
                                "PT": "PTNCC",
                                "RU": "RUCBC",
                                "SC": "GBDSC",
                                "SW": "CHSIC",
                                "NZ": "NZNCC",
                                "ZA": "ZANCC",
                            },
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_ThrdRmbrsmntAgt.FinInstnId.ClrSysMmbId.ClrSysId.Cd",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            },
            # -----------------------------------------
            # Reglas Codigos 55D ------------------
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "55D"}, "right": ""},
                        {
                            "op": "=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "55D"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 2,
                                }
                            },
                            "right": "//",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "55D"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 4,
                                }
                            },
                            "right": "//CH",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "map": {
                            "input": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "55D"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 2,
                                    "len": 2,
                                }
                            },
                            "map": {
                                "AT": "ATBLZ",
                                "AU": "AUBSB",
                                "BL": "DEBLZ",
                                "CC": "CACPA",
                                "CN": "CNAPS",
                                "ES": "ESNCC",
                                "FW": "USABA",
                                "CP": "USPID",
                                "GR": "GRBIC",
                                "HK": "HKNCC",
                                "IE": "IENCC",
                                "IN": "INFSC",
                                "IT": "ITNCC",
                                "PL": "PLKNR",
                                "PT": "PTNCC",
                                "RU": "RUCBC",
                                "SC": "GBDSC",
                                "SW": "CHSIC",
                                "NZ": "NZNCC",
                                "ZA": "ZANCC",
                            },
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_ThrdRmbrsmntAgt.FinInstnId.ClrSysMmbId.ClrSysId.Cd",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            },
        ],
    },
    ".Document.FIToFICstmrCdtTrf.GrpHdr.SttlmInf.ThrdRmbrsmntAgt.FinInstnId.ClrSysMmbId.MmbId": {
        "mode": "append",
        "rules": [
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "55A"}, "right": ""},
                        {
                            "op": "=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "55A"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 2,
                                }
                            },
                            "right": "//",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "55A"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 4,
                                }
                            },
                            "right": "//CH",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "55A"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 4,
                                }
                            }
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_ThrdRmbrsmntAgt.FinInstnId.ClrSysMmbId.ClrSysId.Cd",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            },
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "55B"}, "right": ""},
                        {
                            "op": "=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "55B"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 2,
                                }
                            },
                            "right": "//",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "55B"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 4,
                                }
                            },
                            "right": "//CH",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "55B"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 4,
                                }
                            }
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_ThrdRmbrsmntAgt.FinInstnId.ClrSysMmbId.ClrSysId.Cd",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            },
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "55D"}, "right": ""},
                        {
                            "op": "=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "55D"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 2,
                                }
                            },
                            "right": "//",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "55D"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 4,
                                }
                            },
                            "right": "//CH",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "55D"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 4,
                                }
                            }
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_ThrdRmbrsmntAgt.FinInstnId.ClrSysMmbId.ClrSysId.Cd",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            },
        ],
    },
    ".Document.FIToFICstmrCdtTrf.GrpHdr.SttlmInf.ThrdRmbrsmntAgt.FinInstnId.BICFI": {
        "mode": "append",
        "rules": [
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "55A"}, "right": ""},
                        {
                            "op": " !=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "55A"},
                                            "pattern": "^(?:[^\\r\\n]*\\n)([^\\r\\n]*)",
                                            "group": 1,
                                            "flags": 8,
                                        },
                                    },
                                    "start": 0,
                                    "len": 8,
                                }
                            },
                            # "right": {"param": "BICFI", "scope": "user"},
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "pad": {
                            "value": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "55A"},
                                            "pattern": "^(?:[^\\r\\n]*\\n)([^\\r\\n]*)",
                                            "group": 1,
                                            "flags": 8,
                                        },
                                    },
                                    "start": 0,
                                    "len": 8,
                                }
                            },
                            "len": 11,
                            "fill": "X",
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_ThrdRmbrsmntAgt.FinInstnId.BICFI",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            }
        ],
    },
    ".Document.FIToFICstmrCdtTrf.GrpHdr.SttlmInf.ThrdRmbrsmntAgtAcct.Id.IBAN": {
        "mode": "append",
        "rules": [
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "55A"}, "right": ""},
                        {
                            "op": "=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "55A"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 4,
                                }
                            },
                            "right": "//CH",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {
                                        "trim": {
                                            "value": {
                                                "substr": {
                                                    "value": {
                                                        "fn": "regex_extract",
                                                        "args": {
                                                            "text": {"mt": "55A"},
                                                            "pattern": "^([^\\r\\n]*)",
                                                            "group": 1,
                                                        },
                                                    },
                                                    "start": 4,
                                                }
                                            }
                                        }
                                    },
                                    "pattern": "^[A-Z]{2}[0-9]{2}[A-Za-z0-9]{1,30}$",
                                    "group": 0,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "55A"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 4,
                                }
                            }
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_ThrdRmbrsmntAgtAcct.Id.IBAN",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            },
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "55A"}, "right": ""},
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "55A"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 2,
                                }
                            },
                            "right": "//",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {
                                        "trim": {
                                            "value": {
                                                "fn": "regex_extract",
                                                "args": {
                                                    "text": {"mt": "55A"},
                                                    "pattern": "^([^\\r\\n]*)",
                                                    "group": 1,
                                                },
                                            }
                                        }
                                    },
                                    "pattern": "^[A-Z]{2}[0-9]{2}[A-Za-z0-9]{1,30}$",
                                    "group": 0,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "55A"},
                                    "pattern": "^([^\\r\\n]*)",
                                    "group": 1,
                                },
                            }
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_ThrdRmbrsmntAgtAcct.Id.IBAN",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            },
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "55B"}, "right": ""},
                        {
                            "op": "=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "55B"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 4,
                                }
                            },
                            "right": "//CH",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {
                                        "trim": {
                                            "value": {
                                                "substr": {
                                                    "value": {
                                                        "fn": "regex_extract",
                                                        "args": {
                                                            "text": {"mt": "55B"},
                                                            "pattern": "^([^\\r\\n]*)",
                                                            "group": 1,
                                                        },
                                                    },
                                                    "start": 4,
                                                }
                                            }
                                        }
                                    },
                                    "pattern": "^[A-Z]{2}[0-9]{2}[A-Za-z0-9]{1,30}$",
                                    "group": 0,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "55B"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 4,
                                }
                            }
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_ThrdRmbrsmntAgtAcct.Id.IBAN",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            },
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "55B"}, "right": ""},
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "55B"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 2,
                                }
                            },
                            "right": "//",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {
                                        "trim": {
                                            "value": {
                                                "fn": "regex_extract",
                                                "args": {
                                                    "text": {"mt": "55B"},
                                                    "pattern": "^([^\\r\\n]*)",
                                                    "group": 1,
                                                },
                                            }
                                        }
                                    },
                                    "pattern": "^[A-Z]{2}[0-9]{2}[A-Za-z0-9]{1,30}$",
                                    "group": 0,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "55B"},
                                    "pattern": "^([^\\r\\n]*)",
                                    "group": 1,
                                },
                            }
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_ThrdRmbrsmntAgtAcct.Id.IBAN",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            },
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "55D"}, "right": ""},
                        {
                            "op": "=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "55D"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 4,
                                }
                            },
                            "right": "//CH",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {
                                        "trim": {
                                            "value": {
                                                "substr": {
                                                    "value": {
                                                        "fn": "regex_extract",
                                                        "args": {
                                                            "text": {"mt": "55D"},
                                                            "pattern": "^([^\\r\\n]*)",
                                                            "group": 1,
                                                        },
                                                    },
                                                    "start": 4,
                                                }
                                            }
                                        }
                                    },
                                    "pattern": "^[A-Z]{2}[0-9]{2}[A-Za-z0-9]{1,30}$",
                                    "group": 0,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "55D"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 4,
                                }
                            }
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_ThrdRmbrsmntAgtAcct.Id.IBAN",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            },
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "55D"}, "right": ""},
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "55D"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 2,
                                }
                            },
                            "right": "//",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {
                                        "trim": {
                                            "value": {
                                                "fn": "regex_extract",
                                                "args": {
                                                    "text": {"mt": "55D"},
                                                    "pattern": "^([^\\r\\n]*)",
                                                    "group": 1,
                                                },
                                            }
                                        }
                                    },
                                    "pattern": "^[A-Z]{2}[0-9]{2}[A-Za-z0-9]{1,30}$",
                                    "group": 0,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "55D"},
                                    "pattern": "^([^\\r\\n]*)",
                                    "group": 1,
                                },
                            }
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_ThrdRmbrsmntAgtAcct.Id.IBAN",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            },
        ],
    },
    ".Document.FIToFICstmrCdtTrf.GrpHdr.SttlmInf.ThrdRmbrsmntAgtAcct.Id.Othr.Id": {
        "mode": "append",
        "rules": [
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "55A"}, "right": ""},
                        {
                            "op": "=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "55A"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 4,
                                }
                            },
                            "right": "//CH",
                        },
                        {
                            "op": "=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {
                                        "trim": {
                                            "value": {
                                                "substr": {
                                                    "value": {
                                                        "fn": "regex_extract",
                                                        "args": {
                                                            "text": {"mt": "55A"},
                                                            "pattern": "^([^\\r\\n]*)",
                                                            "group": 1,
                                                        },
                                                    },
                                                    "start": 4,
                                                }
                                            }
                                        }
                                    },
                                    "pattern": "^[A-Z]{2}[0-9]{2}[A-Za-z0-9]{1,30}$",
                                    "group": 0,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "55A"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 4,
                                }
                            }
                        }
                    }
                },
            },
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "55A"}, "right": ""},
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "55A"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 2,
                                }
                            },
                            "right": "//",
                        },
                        {
                            "op": "=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {
                                        "trim": {
                                            "value": {
                                                "fn": "regex_extract",
                                                "args": {
                                                    "text": {"mt": "55A"},
                                                    "pattern": "^([^\\r\\n]*)",
                                                    "group": 1,
                                                },
                                            }
                                        }
                                    },
                                    "pattern": "^[A-Z]{2}[0-9]{2}[A-Za-z0-9]{1,30}$",
                                    "group": 0,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "55A"},
                                    "pattern": "^([^\\r\\n]*)",
                                    "group": 1,
                                },
                            }
                        }
                    }
                },
            },
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "55B"}, "right": ""},
                        {
                            "op": "=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "55B"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 4,
                                }
                            },
                            "right": "//CH",
                        },
                        {
                            "op": "=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {
                                        "trim": {
                                            "value": {
                                                "substr": {
                                                    "value": {
                                                        "fn": "regex_extract",
                                                        "args": {
                                                            "text": {"mt": "55B"},
                                                            "pattern": "^([^\\r\\n]*)",
                                                            "group": 1,
                                                        },
                                                    },
                                                    "start": 4,
                                                }
                                            }
                                        }
                                    },
                                    "pattern": "^[A-Z]{2}[0-9]{2}[A-Za-z0-9]{1,30}$",
                                    "group": 0,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "55B"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 4,
                                }
                            }
                        }
                    }
                },
            },
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "55B"}, "right": ""},
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "55B"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 2,
                                }
                            },
                            "right": "//",
                        },
                        {
                            "op": "=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {
                                        "trim": {
                                            "value": {
                                                "fn": "regex_extract",
                                                "args": {
                                                    "text": {"mt": "55B"},
                                                    "pattern": "^([^\\r\\n]*)",
                                                    "group": 1,
                                                },
                                            }
                                        }
                                    },
                                    "pattern": "^[A-Z]{2}[0-9]{2}[A-Za-z0-9]{1,30}$",
                                    "group": 0,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "55B"},
                                    "pattern": "^([^\\r\\n]*)",
                                    "group": 1,
                                },
                            }
                        }
                    }
                },
            },
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "55D"}, "right": ""},
                        {
                            "op": "=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "55D"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 4,
                                }
                            },
                            "right": "//CH",
                        },
                        {
                            "op": "=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {
                                        "trim": {
                                            "value": {
                                                "substr": {
                                                    "value": {
                                                        "fn": "regex_extract",
                                                        "args": {
                                                            "text": {"mt": "55D"},
                                                            "pattern": "^([^\\r\\n]*)",
                                                            "group": 1,
                                                        },
                                                    },
                                                    "start": 4,
                                                }
                                            }
                                        }
                                    },
                                    "pattern": "^[A-Z]{2}[0-9]{2}[A-Za-z0-9]{1,30}$",
                                    "group": 0,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "55D"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 4,
                                }
                            }
                        }
                    }
                },
            },
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "55D"}, "right": ""},
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "55D"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 2,
                                }
                            },
                            "right": "//",
                        },
                        {
                            "op": "=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {
                                        "trim": {
                                            "value": {
                                                "fn": "regex_extract",
                                                "args": {
                                                    "text": {"mt": "55D"},
                                                    "pattern": "^([^\\r\\n]*)",
                                                    "group": 1,
                                                },
                                            }
                                        }
                                    },
                                    "pattern": "^[A-Z]{2}[0-9]{2}[A-Za-z0-9]{1,30}$",
                                    "group": 0,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "55D"},
                                    "pattern": "^([^\\r\\n]*)",
                                    "group": 1,
                                },
                            }
                        }
                    }
                },
            },
        ],
    },
    ".Document.FIToFICstmrCdtTrf.GrpHdr.SttlmInf.ThrdRmbrsmntAgtAcct.Id.Othr.SchmeNm.Cd": {
        "mode": "set",
        "rules": [
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "55A"}, "right": ""},
                        {
                            "op": "=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "55A"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 4,
                                }
                            },
                            "right": "//CH",
                        },
                        {
                            "op": "=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {
                                        "trim": {
                                            "value": {
                                                "substr": {
                                                    "value": {
                                                        "fn": "regex_extract",
                                                        "args": {
                                                            "text": {"mt": "55A"},
                                                            "pattern": "^([^\\r\\n]*)",
                                                            "group": 1,
                                                        },
                                                    },
                                                    "start": 4,
                                                }
                                            }
                                        }
                                    },
                                    "pattern": "^[A-Z]{2}[0-9]{2}[A-Za-z0-9]{1,30}$",
                                    "group": 0,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {"value": "CUID"},
            },
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "55B"}, "right": ""},
                        {
                            "op": "=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "55B"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 4,
                                }
                            },
                            "right": "//CH",
                        },
                        {
                            "op": "=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {
                                        "trim": {
                                            "value": {
                                                "substr": {
                                                    "value": {
                                                        "fn": "regex_extract",
                                                        "args": {
                                                            "text": {"mt": "55B"},
                                                            "pattern": "^([^\\r\\n]*)",
                                                            "group": 1,
                                                        },
                                                    },
                                                    "start": 4,
                                                }
                                            }
                                        }
                                    },
                                    "pattern": "^[A-Z]{2}[0-9]{2}[A-Za-z0-9]{1,30}$",
                                    "group": 0,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {"value": "CUID"},
            },
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "55D"}, "right": ""},
                        {
                            "op": "=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "55D"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 4,
                                }
                            },
                            "right": "//CH",
                        },
                        {
                            "op": "=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {
                                        "trim": {
                                            "value": {
                                                "substr": {
                                                    "value": {
                                                        "fn": "regex_extract",
                                                        "args": {
                                                            "text": {"mt": "55D"},
                                                            "pattern": "^([^\\r\\n]*)",
                                                            "group": 1,
                                                        },
                                                    },
                                                    "start": 4,
                                                }
                                            }
                                        }
                                    },
                                    "pattern": "^[A-Z]{2}[0-9]{2}[A-Za-z0-9]{1,30}$",
                                    "group": 0,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {"value": "CUID"},
            },
        ],
    },
    ".Document.FIToFICstmrCdtTrf.GrpHdr.SttlmInf.ThrdRmbrsmntAgt.FinInstnId.Nm": {
        "mode": "set",
        "rules": [
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "55B"}, "right": ""},
                        {
                            "op": "!=",
                            "left": {
                                "trim": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "55B"},
                                            "pattern": "^[^\\r\\n]*\\r?\\n([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    }
                                }
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "55B"},
                                    "pattern": "^[^\\r\\n]*\\r?\\n([^\\r\\n]*)",
                                    "group": 1,
                                },
                            }
                        }
                    }
                },
            },
            {"when": {"all": []}, "then": {"value": "NOTPROVIDED"}},
        ],
    },
    ".Document.FIToFICstmrCdtTrf.GrpHdr.SttlmInf.ThrdRmbrsmntAgt.FinInstnId.PstlAdr.AdrLine": {
        "mode": "set",
        "rules": [
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "55B"}, "right": ""},
                        {
                            "op": "=",
                            "left": {
                                "trim": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "55B"},
                                            "pattern": "^(?:[^\\r\\n]*\\r?\\n){2}([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    }
                                }
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {"value": ["NOTPROVIDED"]},
            },
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "55B"}, "right": ""},
                        {
                            "op": "!=",
                            "left": {
                                "trim": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "55B"},
                                            "pattern": "^(?:[^\\r\\n]*\\r?\\n){2}([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    }
                                }
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": [
                        {
                            "trim": {
                                "value": {
                                    "fn": "regex_extract",
                                    "args": {
                                        "text": {"mt": "55B"},
                                        "pattern": "^(?:[^\\r\\n]*\\r?\\n){2}([^\\r\\n]*)",
                                        "group": 1,
                                    },
                                }
                            }
                        },
                        {
                            "trim": {
                                "value": {
                                    "fn": "regex_extract",
                                    "args": {
                                        "text": {"mt": "55B"},
                                        "pattern": "^(?:[^\\r\\n]*\\r?\\n){3}([^\\r\\n]*)",
                                        "group": 1,
                                    },
                                }
                            }
                        },
                        {
                            "trim": {
                                "value": {
                                    "fn": "regex_extract",
                                    "args": {
                                        "text": {"mt": "55B"},
                                        "pattern": "^(?:[^\\r\\n]*\\r?\\n){4}([^\\r\\n]*)",
                                        "group": 1,
                                    },
                                }
                            }
                        },
                    ]
                },
            },
        ],
    },
    # ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.PmtTpInf.ClrChanl": {
    #     "mode": "set",
    #     "rules": [
    #         {
    #             "when": {
    #                 "all": [
    #                     {"op": "!=", "left": {"mt": "56A"}, "right": ""},
    #                     {
    #                         "op": "=",
    #                         "left": {
    #                             "substr": {
    #                                 "value": {
    #                                     "fn": "regex_extract",
    #                                     "args": {
    #                                         "text": {"mt": "56A"},
    #                                         "pattern": "^([^\\r\\n]*)",
    #                                         "group": 1,
    #                                     },
    #                                 },
    #                                 "start": 0,
    #                                 "len": 4,
    #                             }
    #                         },
    #                         "right": "//RT",
    #                     },
    #                 ]
    #             },
    #             "then": {"value": {"literal": "RTGS"}},
    #         },
    #         {
    #             "when": {
    #                 "all": [
    #                     {"op": "!=", "left": {"mt": "56A"}, "right": ""},
    #                     {
    #                         "op": "=",
    #                         "left": {
    #                             "substr": {
    #                                 "value": {
    #                                     "fn": "regex_extract",
    #                                     "args": {
    #                                         "text": {"mt": "56A"},
    #                                         "pattern": "^([^\\r\\n]*)",
    #                                         "group": 1,
    #                                     },
    #                                 },
    #                                 "start": 0,
    #                                 "len": 4,
    #                             }
    #                         },
    #                         "right": "//FW",
    #                     },
    #                 ]
    #             },
    #             "then": {"value": {"literal": "RTGS"}},
    #         },
    #         # --- Nueva del 56C
    #         {
    #             "when": {
    #                 "all": [
    #                     {"op": "!=", "left": {"mt": "56C"}, "right": ""},
    #                     {
    #                         "op": "=",
    #                         "left": {
    #                             "substr": {
    #                                 "value": {
    #                                     "fn": "regex_extract",
    #                                     "args": {
    #                                         "text": {"mt": "56C"},
    #                                         "pattern": "^([^\\r\\n]*)",
    #                                         "group": 1,
    #                                     },
    #                                 },
    #                                 "start": 0,
    #                                 "len": 4,
    #                             }
    #                         },
    #                         "right": "//RT",
    #                     },
    #                 ]
    #             },
    #             "then": {"value": {"literal": "RTGS"}},
    #         },
    #         {
    #             "when": {
    #                 "all": [
    #                     {"op": "!=", "left": {"mt": "56D"}, "right": ""},
    #                     {
    #                         "op": "=",
    #                         "left": {
    #                             "substr": {
    #                                 "value": {
    #                                     "fn": "regex_extract",
    #                                     "args": {
    #                                         "text": {"mt": "56D"},
    #                                         "pattern": "^([^\\r\\n]*)",
    #                                         "group": 1,
    #                                     },
    #                                 },
    #                                 "start": 0,
    #                                 "len": 4,
    #                             }
    #                         },
    #                         "right": "//RT",
    #                     },
    #                 ]
    #             },
    #             "then": {"value": {"literal": "RTGS"}},
    #         },
    #         {
    #             "when": {
    #                 "all": [
    #                     {"op": "!=", "left": {"mt": "56D"}, "right": ""},
    #                     {
    #                         "op": "=",
    #                         "left": {
    #                             "substr": {
    #                                 "value": {
    #                                     "fn": "regex_extract",
    #                                     "args": {
    #                                         "text": {"mt": "56D"},
    #                                         "pattern": "^([^\\r\\n]*)",
    #                                         "group": 1,
    #                                     },
    #                                 },
    #                                 "start": 0,
    #                                 "len": 4,
    #                             }
    #                         },
    #                         "right": "//FW",
    #                     },
    #                 ]
    #             },
    #             "then": {"value": {"literal": "RTGS"}},
    #         },
    #         {
    #             "when": {
    #                 "all": [
    #                     {"op": "!=", "left": {"mt": "57A"}, "right": ""},
    #                     {
    #                         "op": "=",
    #                         "left": {
    #                             "substr": {
    #                                 "value": {
    #                                     "fn": "regex_extract",
    #                                     "args": {
    #                                         "text": {"mt": "57A"},
    #                                         "pattern": "^([^\\r\\n]*)",
    #                                         "group": 1,
    #                                     },
    #                                 },
    #                                 "start": 0,
    #                                 "len": 4,
    #                             }
    #                         },
    #                         "right": "//RT",
    #                     },
    #                 ]
    #             },
    #             "then": {"value": {"literal": "RTGS"}},
    #         },
    #         {
    #             "when": {
    #                 "all": [
    #                     {"op": "!=", "left": {"mt": "57A"}, "right": ""},
    #                     {
    #                         "op": "=",
    #                         "left": {
    #                             "substr": {
    #                                 "value": {
    #                                     "fn": "regex_extract",
    #                                     "args": {
    #                                         "text": {"mt": "57A"},
    #                                         "pattern": "^([^\\r\\n]*)",
    #                                         "group": 1,
    #                                     },
    #                                 },
    #                                 "start": 0,
    #                                 "len": 4,
    #                             }
    #                         },
    #                         "right": "//FW",
    #                     },
    #                 ]
    #             },
    #             "then": {"value": {"literal": "RTGS"}},
    #         },
    #         {
    #             "when": {
    #                 "all": [
    #                     {"op": "!=", "left": {"mt": "57C"}, "right": ""},
    #                     {
    #                         "op": "=",
    #                         "left": {
    #                             "substr": {
    #                                 "value": {
    #                                     "fn": "regex_extract",
    #                                     "args": {
    #                                         "text": {"mt": "57C"},
    #                                         "pattern": "^([^\\r\\n]*)",
    #                                         "group": 1,
    #                                     },
    #                                 },
    #                                 "start": 0,
    #                                 "len": 4,
    #                             }
    #                         },
    #                         "right": "//RT",
    #                     },
    #                 ]
    #             },
    #             "then": {"value": {"literal": "RTGS"}},
    #         },
    #         {
    #             "when": {
    #                 "all": [
    #                     {"op": "!=", "left": {"mt": "57C"}, "right": ""},
    #                     {
    #                         "op": "=",
    #                         "left": {
    #                             "substr": {
    #                                 "value": {
    #                                     "fn": "regex_extract",
    #                                     "args": {
    #                                         "text": {"mt": "57C"},
    #                                         "pattern": "^([^\\r\\n]*)",
    #                                         "group": 1,
    #                                     },
    #                                 },
    #                                 "start": 0,
    #                                 "len": 4,
    #                             }
    #                         },
    #                         "right": "//FW",
    #                     },
    #                 ]
    #             },
    #             "then": {"value": {"literal": "RTGS"}},
    #         },
    #         {
    #             "when": {
    #                 "all": [
    #                     {"op": "!=", "left": {"mt": "57D"}, "right": ""},
    #                     {
    #                         "op": "=",
    #                         "left": {
    #                             "substr": {
    #                                 "value": {
    #                                     "fn": "regex_extract",
    #                                     "args": {
    #                                         "text": {"mt": "57D"},
    #                                         "pattern": "^([^\\r\\n]*)",
    #                                         "group": 1,
    #                                     },
    #                                 },
    #                                 "start": 0,
    #                                 "len": 4,
    #                             }
    #                         },
    #                         "right": "//RT",
    #                     },
    #                 ]
    #             },
    #             "then": {"value": {"literal": "RTGS"}},
    #         },
    #         {
    #             "when": {
    #                 "all": [
    #                     {"op": "!=", "left": {"mt": "57D"}, "right": ""},
    #                     {
    #                         "op": "=",
    #                         "left": {
    #                             "substr": {
    #                                 "value": {
    #                                     "fn": "regex_extract",
    #                                     "args": {
    #                                         "text": {"mt": "57D"},
    #                                         "pattern": "^([^\\r\\n]*)",
    #                                         "group": 1,
    #                                     },
    #                                 },
    #                                 "start": 0,
    #                                 "len": 4,
    #                             }
    #                         },
    #                         "right": "//FW",
    #                     },
    #                 ]
    #             },
    #             "then": {"value": {"literal": "RTGS"}},
    #         },
    #     ],
    # },
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.IntrmyAgt1.FinInstnId.ClrSysMmbId.ClrSysId.Cd": {
        "mode": "set",
        "rules": [
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "56A"}, "right": ""},
                        {
                            "op": "=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "56A"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 2,
                                }
                            },
                            "right": "//",
                        },
                        {
                            "op": "not_in",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "56A"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 4,
                                }
                            },
                            "right": ["//CH", "//FW", "//RT"],
                        },
                    ]
                },
                "then": {
                    "value": {
                        "substr": {
                            "value": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "56A"},
                                    "pattern": "^([^\\r\\n]*)",
                                    "group": 1,
                                },
                            },
                            "start": 2,
                            "len": 2,
                        }
                    }
                },
            },
            # -----------------------
            # {
            #     "when": {
            #         "all": [
            #             {"op": "!=", "left": {"mt": "56C"}, "right": ""},
            #             {
            #                 "op": "=",
            #                 "left": {
            #                     "substr": {
            #                         "value": {
            #                             "fn": "regex_extract",
            #                             "args": {
            #                                 "text": {"mt": "56C"},
            #                                 "pattern": "^([^\\r\\n]*)",
            #                                 "group": 1,
            #                             },
            #                         },
            #                         "start": 0,
            #                         "len": 4,
            #                     }
            #                 },
            #                 "right": "//CH",
            #             },
            #         ]
            #     },
            #     "then": {"value": {"literal": "CHSIC"}},
            # },
            # {
            #     "when": {
            #         "all": [
            #             {"op": "!=", "left": {"mt": "56C"}, "right": ""},
            #             {
            #                 "op": "=",
            #                 "left": {
            #                     "substr": {
            #                         "value": {
            #                             "fn": "regex_extract",
            #                             "args": {
            #                                 "text": {"mt": "56C"},
            #                                 "pattern": "^([^\\r\\n]*)",
            #                                 "group": 1,
            #                             },
            #                         },
            #                         "start": 0,
            #                         "len": 4,
            #                     }
            #                 },
            #                 "right": "//FW",
            #             },
            #         ]
            #     },
            #     "then": {"value": {"literal": "USABA"}},
            # },
            # ---------------------------------
            # Reglas del 56C
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "56C"}, "right": ""},
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "56C"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 4,
                                }
                            },
                            "right": "//CH",
                        },
                        {
                            "op": "=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "56C"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 4,
                                }
                            },
                            "right": "//FW",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "map": {
                            "input": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "56C"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 2,
                                    "len": 2,
                                }
                            },
                            "map": {
                                "FW": "USABA",
                            },
                        }
                    }
                },
            },
            # --------------------------------
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "56D"}, "right": ""},
                        {
                            "op": "=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "56D"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 2,
                                }
                            },
                            "right": "//",
                        },
                        {
                            "op": "not_in",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "56D"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 4,
                                }
                            },
                            "right": ["//CH", "//FW", "//RT"],
                        },
                    ]
                },
                "then": {
                    "value": {
                        "substr": {
                            "value": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "56D"},
                                    "pattern": "^([^\\r\\n]*)",
                                    "group": 1,
                                },
                            },
                            "start": 2,
                            "len": 2,
                        }
                    }
                },
            },
        ],
    },
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.IntrmyAgt1.FinInstnId.ClrSysMmbId.MmbId": {
        "mode": "set",
        "rules": [
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "56A"}, "right": ""},
                        {
                            "op": "=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "56A"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 2,
                                }
                            },
                            "right": "//",
                        },
                        {
                            "op": "not in",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "56A"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 4,
                                }
                            },
                            "right": ["//CH", "//FW", "//RT"],
                        },
                    ]
                },
                "then": {
                    "value": {
                        "substr": {
                            "value": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "56A"},
                                    "pattern": "^([^\\r\\n]*)",
                                    "group": 1,
                                },
                            },
                            "start": 2,
                            "len": 2,
                        }
                    }
                },
            },
            {
                "when": {
                    "all": [{"op": "!=", "left": {"mt": "56C"}, "right": ""}],
                    "any": [
                        {
                            "op": "=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "56C"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 4,
                                }
                            },
                            "right": "//CH",
                        },
                        {
                            "op": "=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "56C"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 4,
                                }
                            },
                            "right": "//FW",
                        },
                    ],
                },
                "then": {
                    "value": {
                        "substr": {
                            "value": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "56C"},
                                    "pattern": "^([^\\r\\n]*)",
                                    "group": 1,
                                },
                            },
                            "start": 4,
                            "len": 34,
                        }
                    }
                },
            },
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "56D"}, "right": ""},
                        {
                            "op": "=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "56D"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 2,
                                }
                            },
                            "right": "//",
                        },
                        {
                            "op": "not in",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "56D"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 4,
                                }
                            },
                            "right": ["//CH", "//FW", "//RT"],
                        },
                    ]
                },
                "then": {
                    "value": {
                        "substr": {
                            "value": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "56D"},
                                    "pattern": "^([^\\r\\n]*)",
                                    "group": 1,
                                },
                            },
                            "start": 2,
                            "len": 2,
                        }
                    }
                },
            },
        ],
    },
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.IntrmyAgt1Acct.Id.IBAN": {
        "mode": "set",
        "rules": [{"when": {"all": []}, "then": {"value": {"literal": ""}}}],
    },
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.IntrmyAgt1Acct.Id.Othr.Id": {
        "mode": "set",
        "rules": [{"when": {"all": []}, "then": {"value": {"literal": ""}}}],
    },
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.IntrmyAgt1Acct.Id.Othr.SchmeNm.Cd": {
        "mode": "set",
        "rules": [{"when": {"all": []}, "then": {"value": {"literal": ""}}}],
    },
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.IntrmyAgt1Acct.Id.Othr.Id": {
        "mode": "set",
        "rules": [{"when": {"all": []}, "then": {"value": {"literal": ""}}}],
    },
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.IntrmyAgt1Acct.Id.Othr.SchmeNm.Cd": {
        "mode": "set",
        "rules": [{"when": {"all": []}, "then": {"value": {"literal": ""}}}],
    },
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.IntrmyAgt1.FinInstnId.PstlAdr.AdrLine": {
        "mode": "append",
        "rules": [
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "56C"}, "right": ""},
                        {
                            "op": "=",
                            "left": {
                                "trim": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "56C"},
                                            "pattern": "^(?:[^\\r\\n]*\\r?\\n){2}([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    }
                                }
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {"value": {"literal": "NOTPROVIDED"}},
            },
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "56C"}, "right": ""},
                        {
                            "op": "!=",
                            "left": {
                                "trim": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "56C"},
                                            "pattern": "^(?:[^\\r\\n]*\\r?\\n){2}([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    }
                                }
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "56C"},
                                    "pattern": "^(?:[^\\r\\n]*\\r?\\n){2}([^\\r\\n]*)",
                                    "group": 1,
                                },
                            }
                        }
                    }
                },
            },
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "56C"}, "right": ""},
                        {
                            "op": "!=",
                            "left": {
                                "trim": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "56C"},
                                            "pattern": "^(?:[^\\r\\n]*\\r?\\n){3}([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    }
                                }
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "56C"},
                                    "pattern": "^(?:[^\\r\\n]*\\r?\\n){3}([^\\r\\n]*)",
                                    "group": 1,
                                },
                            }
                        }
                    }
                },
            },
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "56C"}, "right": ""},
                        {
                            "op": "!=",
                            "left": {
                                "trim": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "56C"},
                                            "pattern": "^(?:[^\\r\\n]*\\r?\\n){4}([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    }
                                }
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "56C"},
                                    "pattern": "^(?:[^\\r\\n]*\\r?\\n){4}([^\\r\\n]*)",
                                    "group": 1,
                                },
                            }
                        }
                    }
                },
            },
        ],
    },
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.CdtrAgt.FinInstnId.ClrSysMmbId.ClrSysId.Cd": {
        "mode": "set",
        "rules": [
            # {
            #     "when": {
            #         "all": [
            #             {"op": "!=", "left": {"mt": "57D"}, "right": ""},
            #             {
            #                 "op": "=",
            #                 "left": {
            #                     "substr": {
            #                         "value": {
            #                             "fn": "regex_extract",
            #                             "args": {
            #                                 "text": {"mt": "57D"},
            #                                 "pattern": "^([^\\r\\n]*)",
            #                                 "group": 1,
            #                             },
            #                         },
            #                         "start": 0,
            #                         "len": 4,
            #                     }
            #                 },
            #                 "right": "//FW",
            #             },
            #         ]
            #     },
            #     "then": {"value": {"literal": "USABA"}},
            # },
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "57D"}, "right": ""},
                        {
                            "op": "=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "57D"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 2,
                                }
                            },
                            "right": "//",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "57D"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 4,
                                }
                            },
                            "right": "//CH",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "map": {
                            "input": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "57D"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 2,
                                    "len": 2,
                                }
                            },
                            "map": {
                                "AT": "ATBLZ",
                                "AU": "AUBSB",
                                "BL": "DEBLZ",
                                "CC": "CACPA",
                                "CN": "CNAPS",
                                "DE": "DEBLZ",
                                "ES": "ESNCC",
                                "FW": "USABA",
                                "CP": "USPID",
                                "GR": "GRBIC",
                                "HK": "HKNCC",
                                "IE": "IENCC",
                                "IN": "INFSC",
                                "IT": "ITNCC",
                                "PL": "PLKNR",
                                "PT": "PTNCC",
                                "RU": "RUCBC",
                                "SC": "GBDSC",
                                "SW": "CHSIC",
                                "NZ": "NZNCC",
                                "ZA": "ZANCC",
                            },
                        }
                    }
                },
            },
            # Regla 57A--------------
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "57A"}, "right": ""},
                        {
                            "op": "=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "57A"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 2,
                                }
                            },
                            "right": "//",
                        },
                        {
                            "op": "not in",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "57A"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 4,
                                }
                            },
                            "right": ["//CH", "//FW", "//RT"],
                        },
                    ]
                },
                "then": {
                    "value": {
                        "map": {
                            "input": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "57A"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 2,
                                    "len": 2,
                                }
                            },
                            "map": {
                                "AT": "ATBLZ",
                                "AU": "AUBSB",
                                "BL": "DEBLZ",
                                "CC": "CACPA",
                                "CN": "CNAPS",
                                "DE": "DEBLZ",
                                "ES": "ESNCC",
                                "CP": "USPID",
                                "GR": "GRBIC",
                                "HK": "HKNCC",
                                "IE": "IENCC",
                                "IN": "INFSC",
                                "IT": "ITNCC",
                                "PL": "PLKNR",
                                "PT": "PTNCC",
                                "RU": "RUCBC",
                                "SC": "GBDSC",
                                "SW": "CHSIC",
                                "NZ": "NZNCC",
                                "ZA": "ZANCC",
                            },
                        }
                    }
                },
            },
            # ----------------------------------
            # Regla 57C--------------
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "57C"}, "right": ""},
                        {
                            "op": "=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "57C"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 2,
                                }
                            },
                            "right": "//",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "57C"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 4,
                                }
                            },
                            "right": "//CH",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "map": {
                            "input": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "57C"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 2,
                                    "len": 2,
                                }
                            },
                            "map": {
                                "SC": "GBDSC",
                                "DE": "DEBLZ",
                                "CP": "USPID",
                                "AU": "AUBSB",
                                "FW": "USABA",
                                "AT": "ATBLZ",
                                "BL": "DEBLZ",
                                "CC": "CACPA",
                                "CN": "CNAPS",
                                "ES": "ESNCC",
                                "GR": "GRBIC",
                                "HK": "HKNCC",
                                "IE": "IENCC",
                                "IN": "INFSC",
                                "IT": "ITNCC",
                                "PL": "PLKNR",
                                "PT": "PTNCC",
                                "RU": "RUCBC",
                                "SW": "CHSIC",
                                "NZ": "NZNCC",
                                "ZA": "ZANCC",
                            },
                        }
                    }
                },
            },
            # ----------------------------------
            # Regla 57B-------------
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "57B"}, "right": ""},
                        {
                            "op": "=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "57B"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 2,
                                }
                            },
                            "right": "//",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "57B"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 4,
                                }
                            },
                            "right": "//CH",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "map": {
                            "input": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "57B"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 2,
                                    "len": 2,
                                }
                            },
                            "map": {
                                "SC": "GBDSC",
                                "DE": "DEBLZ",
                                "CP": "USPID",
                                "AU": "AUBSB",
                                "FW": "USABA",
                                "AT": "ATBLZ",
                                "BL": "DEBLZ",
                                "CC": "CACPA",
                                "CN": "CNAPS",
                                "ES": "ESNCC",
                                "GR": "GRBIC",
                                "HK": "HKNCC",
                                "IE": "IENCC",
                                "IN": "INFSC",
                                "IT": "ITNCC",
                                "PL": "PLKNR",
                                "PT": "PTNCC",
                                "RU": "RUCBC",
                                "SW": "CHSIC",
                                "NZ": "NZNCC",
                                "ZA": "ZANCC",
                            },
                        }
                    }
                },
            },
            # ----------------------------------
        ],
    },
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.CdtrAgt.FinInstnId.ClrSysMmbId.MmbId": {
        "mode": "set",
        "rules": [
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "57A"}, "right": ""},
                        {
                            "op": "=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "trim": {
                                            "value": {
                                                "fn": "regex_extract",
                                                "args": {
                                                    "text": {"mt": "57A"},
                                                    "pattern": "^([^\\r\\n]*)",
                                                    "group": 1,
                                                },
                                            }
                                        }
                                    },
                                    "start": 0,
                                    "len": 2,
                                }
                            },
                            "right": "//",
                        },
                        {
                            "op": "not in",
                            "left": {
                                "substr": {
                                    "value": {
                                        "trim": {
                                            "value": {
                                                "fn": "regex_extract",
                                                "args": {
                                                    "text": {"mt": "57A"},
                                                    "pattern": "^([^\\r\\n]*)",
                                                    "group": 1,
                                                },
                                            }
                                        }
                                    },
                                    "start": 0,
                                    "len": 4,
                                }
                            },
                            "right": ["//CH", "//FW", "//RT"],
                        },
                        {
                            "op": "!=",
                            "left": {
                                "trim": {
                                    "value": {
                                        "substr": {
                                            "value": {
                                                "trim": {
                                                    "value": {
                                                        "fn": "regex_extract",
                                                        "args": {
                                                            "text": {"mt": "57A"},
                                                            "pattern": "^([^\\r\\n]*)",
                                                            "group": 1,
                                                        },
                                                    }
                                                }
                                            },
                                            "start": 4,
                                            "len": 30,
                                        }
                                    }
                                }
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "substr": {
                                    "value": {
                                        "trim": {
                                            "value": {
                                                "fn": "regex_extract",
                                                "args": {
                                                    "text": {"mt": "57A"},
                                                    "pattern": "^([^\\r\\n]*)",
                                                    "group": 1,
                                                },
                                            }
                                        }
                                    },
                                    "start": 4,
                                    "len": 30,
                                }
                            }
                        }
                    }
                },
            },
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "57C"}, "right": ""},
                        {
                            "op": "=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "57C"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 2,
                                }
                            },
                            "right": "//",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "57C"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 4,
                                }
                            },
                            "right": ["//CH", "//FW"],
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "57C"},
                                    "pattern": ".{4}(.+)",
                                    "group": 1,
                                },
                            }
                        }
                    }
                },
            },
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "57D"}, "right": ""},
                        {
                            "op": "in",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "57D"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 2,
                                }
                            },
                            "right": "//",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "57D"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 4,
                                    "len": 4,
                                }
                            },
                            "right": "//CH",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "57D"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 4,
                                }
                            }
                        },
                    }
                },
            },
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "57B"}, "right": ""},
                        {
                            "op": "=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "57B"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 2,
                                }
                            },
                            "right": "//",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "57B"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 4,
                                }
                            },
                            "right": "//CH",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "57B"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 4,
                                }
                            }
                        },
                    }
                },
            },
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "57A"}, "right": ""},
                        {
                            "op": "=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "57A"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 2,
                                }
                            },
                            "right": "//",
                        },
                        {
                            "op": "not in",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "57A"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 4,
                                }
                            },
                            "right": ["//CH", "//FW", "//RT"],
                        },
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "57A"},
                                    "pattern": "^.{4}(.+)$",
                                    "group": 1,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "57A"},
                                    "pattern": "^.{4}(.+)$",
                                    "group": 1,
                                },
                            }
                        }
                    }
                },
            },
        ],
    },
    # -------Campo 57A-----------
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.CdtrAgt.FinInstnId.BICFI": {
        "mode": "set",
        "rules": [
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "57A"}, "right": ""},
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"trim": {"value": {"mt": "57A"}}},
                                    "pattern": "([A-Z0-9]{8}(?:[A-Z0-9]{3})?)\\s*$",
                                    "group": 1,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"trim": {"value": {"mt": "57A"}}},
                                    "pattern": "([A-Z0-9]{8}(?:[A-Z0-9]{3})?)\\s*$",
                                    "group": 1,
                                },
                            }
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_CdtrAgtFinInstnIdBICFI",
                            "scope": "global",
                            "value": {"literal": "1"},
                        }
                    }
                ],
            }
        ],
    },
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.CdtrAgtAcct.Id.IBAN": {
        "mode": "set",
        "rules": [
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "57A"}, "right": ""},
                        {
                            "op": "=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "57A"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 4,
                                }
                            },
                            "right": "//CH",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {
                                        "trim": {
                                            "value": {
                                                "substr": {
                                                    "value": {
                                                        "fn": "regex_extract",
                                                        "args": {
                                                            "text": {"mt": "57A"},
                                                            "pattern": "^([^\\r\\n]*)",
                                                            "group": 1,
                                                        },
                                                    },
                                                    "start": 4,
                                                }
                                            }
                                        }
                                    },
                                    "pattern": "^[A-Z]{2}[0-9]{2}[A-Za-z0-9]{1,30}$",
                                    "group": 0,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "57A"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 4,
                                }
                            }
                        }
                    }
                },
            },
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "57A"}, "right": ""},
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "57A"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 2,
                                }
                            },
                            "right": "//",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {
                                        "trim": {
                                            "value": {
                                                "fn": "regex_extract",
                                                "args": {
                                                    "text": {"mt": "57A"},
                                                    "pattern": "^([^\\r\\n]*)",
                                                    "group": 1,
                                                },
                                            }
                                        }
                                    },
                                    "pattern": "^[A-Z]{2}[0-9]{2}[A-Za-z0-9]{1,30}$",
                                    "group": 0,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "57A"},
                                    "pattern": "^([^\\r\\n]*)",
                                    "group": 1,
                                },
                            }
                        }
                    }
                },
            },
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "57B"}, "right": ""},
                        {
                            "op": "=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "57B"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 4,
                                }
                            },
                            "right": "//CH",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {
                                        "trim": {
                                            "value": {
                                                "substr": {
                                                    "value": {
                                                        "fn": "regex_extract",
                                                        "args": {
                                                            "text": {"mt": "57B"},
                                                            "pattern": "^([^\\r\\n]*)",
                                                            "group": 1,
                                                        },
                                                    },
                                                    "start": 4,
                                                }
                                            }
                                        }
                                    },
                                    "pattern": "^[A-Z]{2}[0-9]{2}[A-Za-z0-9]{1,30}$",
                                    "group": 0,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "57B"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 4,
                                }
                            }
                        }
                    }
                },
            },
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "57B"}, "right": ""},
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "57B"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 2,
                                }
                            },
                            "right": "//",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {
                                        "trim": {
                                            "value": {
                                                "fn": "regex_extract",
                                                "args": {
                                                    "text": {"mt": "57B"},
                                                    "pattern": "^([^\\r\\n]*)",
                                                    "group": 1,
                                                },
                                            }
                                        }
                                    },
                                    "pattern": "^[A-Z]{2}[0-9]{2}[A-Za-z0-9]{1,30}$",
                                    "group": 0,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "57B"},
                                    "pattern": "^([^\\r\\n]*)",
                                    "group": 1,
                                },
                            }
                        }
                    }
                },
            },
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "57C"}, "right": ""},
                        {
                            "op": "=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "57C"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 4,
                                }
                            },
                            "right": "//CH",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {
                                        "trim": {
                                            "value": {
                                                "substr": {
                                                    "value": {
                                                        "fn": "regex_extract",
                                                        "args": {
                                                            "text": {"mt": "57C"},
                                                            "pattern": "^([^\\r\\n]*)",
                                                            "group": 1,
                                                        },
                                                    },
                                                    "start": 4,
                                                }
                                            }
                                        }
                                    },
                                    "pattern": "^[A-Z]{2}[0-9]{2}[A-Za-z0-9]{1,30}$",
                                    "group": 0,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "57C"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 4,
                                }
                            }
                        }
                    }
                },
            },
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "57C"}, "right": ""},
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "57C"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 2,
                                }
                            },
                            "right": "//",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {
                                        "trim": {
                                            "value": {
                                                "fn": "regex_extract",
                                                "args": {
                                                    "text": {"mt": "57C"},
                                                    "pattern": "^([^\\r\\n]*)",
                                                    "group": 1,
                                                },
                                            }
                                        }
                                    },
                                    "pattern": "^[A-Z]{2}[0-9]{2}[A-Za-z0-9]{1,30}$",
                                    "group": 0,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "57C"},
                                    "pattern": "^([^\\r\\n]*)",
                                    "group": 1,
                                },
                            }
                        }
                    }
                },
            },
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "57D"}, "right": ""},
                        {
                            "op": "=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "57D"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 4,
                                }
                            },
                            "right": "//CH",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {
                                        "trim": {
                                            "value": {
                                                "substr": {
                                                    "value": {
                                                        "fn": "regex_extract",
                                                        "args": {
                                                            "text": {"mt": "57D"},
                                                            "pattern": "^([^\\r\\n]*)",
                                                            "group": 1,
                                                        },
                                                    },
                                                    "start": 4,
                                                }
                                            }
                                        }
                                    },
                                    "pattern": "^[A-Z]{2}[0-9]{2}[A-Za-z0-9]{1,30}$",
                                    "group": 0,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "57D"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 4,
                                }
                            }
                        }
                    }
                },
            },
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "57D"}, "right": ""},
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "57D"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 2,
                                }
                            },
                            "right": "//",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {
                                        "trim": {
                                            "value": {
                                                "fn": "regex_extract",
                                                "args": {
                                                    "text": {"mt": "57D"},
                                                    "pattern": "^([^\\r\\n]*)",
                                                    "group": 1,
                                                },
                                            }
                                        }
                                    },
                                    "pattern": "^[A-Z]{2}[0-9]{2}[A-Za-z0-9]{1,30}$",
                                    "group": 0,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "57D"},
                                    "pattern": "^([^\\r\\n]*)",
                                    "group": 1,
                                },
                            }
                        }
                    }
                },
            },
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "59A"}, "right": ""},
                        {
                            "op": "=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "59A"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 4,
                                }
                            },
                            "right": "//CH",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {
                                        "trim": {
                                            "value": {
                                                "substr": {
                                                    "value": {
                                                        "fn": "regex_extract",
                                                        "args": {
                                                            "text": {"mt": "59A"},
                                                            "pattern": "^([^\\r\\n]*)",
                                                            "group": 1,
                                                        },
                                                    },
                                                    "start": 4,
                                                }
                                            }
                                        }
                                    },
                                    "pattern": "^[A-Z]{2}[0-9]{2}[A-Za-z0-9]{1,30}$",
                                    "group": 0,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "59A"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 4,
                                }
                            }
                        }
                    }
                },
            },
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "59A"}, "right": ""},
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "59A"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 2,
                                }
                            },
                            "right": "//",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {
                                        "trim": {
                                            "value": {
                                                "fn": "regex_extract",
                                                "args": {
                                                    "text": {"mt": "59A"},
                                                    "pattern": "^([^\\r\\n]*)",
                                                    "group": 1,
                                                },
                                            }
                                        }
                                    },
                                    "pattern": "^[A-Z]{2}[0-9]{2}[A-Za-z0-9]{1,30}$",
                                    "group": 0,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "59A"},
                                    "pattern": "^([^\\r\\n]*)",
                                    "group": 1,
                                },
                            }
                        }
                    }
                },
            },
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "59F"}, "right": ""},
                        {
                            "op": "=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "59F"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 4,
                                }
                            },
                            "right": "//CH",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {
                                        "trim": {
                                            "value": {
                                                "substr": {
                                                    "value": {
                                                        "fn": "regex_extract",
                                                        "args": {
                                                            "text": {"mt": "59F"},
                                                            "pattern": "^([^\\r\\n]*)",
                                                            "group": 1,
                                                        },
                                                    },
                                                    "start": 4,
                                                }
                                            }
                                        }
                                    },
                                    "pattern": "^[A-Z]{2}[0-9]{2}[A-Za-z0-9]{1,30}$",
                                    "group": 0,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "59F"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 4,
                                }
                            }
                        }
                    }
                },
            },
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "59F"}, "right": ""},
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "59F"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 2,
                                }
                            },
                            "right": "//",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {
                                        "trim": {
                                            "value": {
                                                "fn": "regex_extract",
                                                "args": {
                                                    "text": {"mt": "59F"},
                                                    "pattern": "^([^\\r\\n]*)",
                                                    "group": 1,
                                                },
                                            }
                                        }
                                    },
                                    "pattern": "^[A-Z]{2}[0-9]{2}[A-Za-z0-9]{1,30}$",
                                    "group": 0,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "59F"},
                                    "pattern": "^([^\\r\\n]*)",
                                    "group": 1,
                                },
                            }
                        }
                    }
                },
            },
        ],
    },
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.CdtrAgtAcct.Id.Othr.Id": {
        "mode": "set",
        "rules": [
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "57A"}, "right": ""},
                        {
                            "op": "=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "57A"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 4,
                                }
                            },
                            "right": "//CH",
                        },
                        {
                            "op": "=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {
                                        "trim": {
                                            "value": {
                                                "substr": {
                                                    "value": {
                                                        "fn": "regex_extract",
                                                        "args": {
                                                            "text": {"mt": "57A"},
                                                            "pattern": "^([^\\r\\n]*)",
                                                            "group": 1,
                                                        },
                                                    },
                                                    "start": 4,
                                                }
                                            }
                                        }
                                    },
                                    "pattern": "^[A-Z]{2}[0-9]{2}[A-Za-z0-9]{1,30}$",
                                    "group": 0,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "fn": "regex_extract",
                        "args": {
                            "text": {
                                "trim": {
                                    "value": {
                                        "substr": {
                                            "value": {
                                                "fn": "regex_extract",
                                                "args": {
                                                    "text": {"mt": "57A"},
                                                    "pattern": "^([^\\r\\n]*)",
                                                    "group": 1,
                                                },
                                            },
                                            "start": 4,
                                        }
                                    }
                                }
                            },
                            "pattern": "^/?(.+)$",
                            "group": 1,
                        },
                    }
                },
            },
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "57A"}, "right": ""},
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "57A"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 2,
                                }
                            },
                            "right": "//",
                        },
                        {
                            "op": "=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {
                                        "trim": {
                                            "value": {
                                                "fn": "regex_extract",
                                                "args": {
                                                    "text": {"mt": "57A"},
                                                    "pattern": "^([^\\r\\n]*)",
                                                    "group": 1,
                                                },
                                            }
                                        }
                                    },
                                    "pattern": "^[A-Z]{2}[0-9]{2}[A-Za-z0-9]{1,30}$",
                                    "group": 0,
                                },
                            },
                            "right": "",
                        },
                        {
                            "op": "=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {
                                        "trim": {
                                            "value": {
                                                "fn": "regex_extract",
                                                "args": {
                                                    "text": {"mt": "57A"},
                                                    "pattern": "^([^\\r\\n]*)",
                                                    "group": 1,
                                                },
                                            }
                                        }
                                    },
                                    "pattern": "^[A-Z0-9]{11}$",
                                    "group": 0,
                                },
                            },
                            "right": "",
                        },
                        {
                            "op": "=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {
                                        "trim": {
                                            "value": {
                                                "fn": "regex_extract",
                                                "args": {
                                                    "text": {"mt": "57A"},
                                                    "pattern": "^([^\\r\\n]*)",
                                                    "group": 1,
                                                },
                                            }
                                        }
                                    },
                                    "pattern": "^[A-Z0-9]{8}$",
                                    "group": 0,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "fn": "regex_extract",
                        "args": {
                            "text": {
                                "trim": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "57A"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    }
                                }
                            },
                            "pattern": "^/?(.+)$",
                            "group": 1,
                        },
                    }
                },
            },
        ],
    },
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.CdtrAgtAcct.Id.Othr.SchmeNm.Cd": {
        "mode": "set",
        "rules": [
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "57A"}, "right": ""},
                        {
                            "op": "=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "57A"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 4,
                                }
                            },
                            "right": "//CH",
                        },
                        {
                            "op": "=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {
                                        "trim": {
                                            "value": {
                                                "substr": {
                                                    "value": {
                                                        "fn": "regex_extract",
                                                        "args": {
                                                            "text": {"mt": "57A"},
                                                            "pattern": "^([^\\r\\n]*)",
                                                            "group": 1,
                                                        },
                                                    },
                                                    "start": 4,
                                                }
                                            }
                                        }
                                    },
                                    "pattern": "^[A-Z]{2}[0-9]{2}[A-Za-z0-9]{1,30}$",
                                    "group": 0,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {"value": "CUID"},
            },
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "57B"}, "right": ""},
                        {
                            "op": "=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "57B"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 4,
                                }
                            },
                            "right": "//CH",
                        },
                        {
                            "op": "=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {
                                        "trim": {
                                            "value": {
                                                "substr": {
                                                    "value": {
                                                        "fn": "regex_extract",
                                                        "args": {
                                                            "text": {"mt": "57B"},
                                                            "pattern": "^([^\\r\\n]*)",
                                                            "group": 1,
                                                        },
                                                    },
                                                    "start": 4,
                                                }
                                            }
                                        }
                                    },
                                    "pattern": "^[A-Z]{2}[0-9]{2}[A-Za-z0-9]{1,30}$",
                                    "group": 0,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {"value": "CUID"},
            },
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "57D"}, "right": ""},
                        {
                            "op": "=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "57D"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 4,
                                }
                            },
                            "right": "//CH",
                        },
                        {
                            "op": "=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {
                                        "trim": {
                                            "value": {
                                                "substr": {
                                                    "value": {
                                                        "fn": "regex_extract",
                                                        "args": {
                                                            "text": {"mt": "57D"},
                                                            "pattern": "^([^\\r\\n]*)",
                                                            "group": 1,
                                                        },
                                                    },
                                                    "start": 4,
                                                }
                                            }
                                        }
                                    },
                                    "pattern": "^[A-Z]{2}[0-9]{2}[A-Za-z0-9]{1,30}$",
                                    "group": 0,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {"value": "CUID"},
            },
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "59A"}, "right": ""},
                        {
                            "op": "=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "59A"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 4,
                                }
                            },
                            "right": "//CH",
                        },
                        {
                            "op": "=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {
                                        "trim": {
                                            "value": {
                                                "substr": {
                                                    "value": {
                                                        "fn": "regex_extract",
                                                        "args": {
                                                            "text": {"mt": "59A"},
                                                            "pattern": "^([^\\r\\n]*)",
                                                            "group": 1,
                                                        },
                                                    },
                                                    "start": 4,
                                                }
                                            }
                                        }
                                    },
                                    "pattern": "^[A-Z]{2}[0-9]{2}[A-Za-z0-9]{1,30}$",
                                    "group": 0,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {"value": "CUID"},
            },
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "59F"}, "right": ""},
                        {
                            "op": "=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "59F"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 4,
                                }
                            },
                            "right": "//CH",
                        },
                        {
                            "op": "=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {
                                        "trim": {
                                            "value": {
                                                "substr": {
                                                    "value": {
                                                        "fn": "regex_extract",
                                                        "args": {
                                                            "text": {"mt": "59F"},
                                                            "pattern": "^([^\\r\\n]*)",
                                                            "group": 1,
                                                        },
                                                    },
                                                    "start": 4,
                                                }
                                            }
                                        }
                                    },
                                    "pattern": "^[A-Z]{2}[0-9]{2}[A-Za-z0-9]{1,30}$",
                                    "group": 0,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {"value": "CUID"},
            },
        ],
    },
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.Cdtr.Nm": {
        "mode": "set",
        "rules": [
            {
                "set": [
                    {
                        "set_var": {
                            "name": "pacs.008::Cdtr.Nm",
                            "scope": "global",
                            "value": {
                                "from_db_query": {
                                    "alias": "pacs.008::Cdtr.Nm",
                                    "lookup_message_type": "pacs.008",
                                    "which_uetr_to_lookup": {"mtField": "121"},
                                    "where_to_lookup_uetr": {
                                        "message_type": "pacs.008",
                                        "path_kind": "xml",
                                        "field_path": ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.PmtId.UETR",
                                    },
                                    "field_to_extract": {
                                        "message_type": "pacs.008",
                                        "path_kind": "xml",
                                        "field_path": ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.Cdtr.Nm",
                                    },
                                    "fallback_literal": "",
                                }
                            },
                        }
                    }
                ],
                "when": {
                    "all": [
                        {
                            "op": "in",
                            "left": {
                                "substr": {"value": {"mt": "20"}, "start": 0, "len": 3}
                            },
                            "right": ["662", "663", "651"],
                        },
                        {"op": "!=", "left": {"var": "pacs.008::Cdtr.Nm"}, "right": ""},
                    ],
                    "any": [
                        {"op": "!=", "left": {"mt": "59"}, "right": ""},
                        {"op": "!=", "left": {"mt": "59F"}, "right": ""},
                    ],
                },
                "then": {"value": {"var": "pacs.008::Cdtr.Nm"}},
            },
            {
                "when": {
                    "all": [
                        {
                            "op": "=",
                            "left": {"var": "has_CdtrPstlAdrAdrLine"},
                            "right": "",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {
                                        "substr": {
                                            "value": {"mt": "20"},
                                            "start": 0,
                                            "len": 3,
                                        }
                                    },
                                    "pattern": "^(7[0-9]{2})$",
                                    "group": 1,
                                },
                            },
                            "right": "",
                        },
                        {
                            "op": "=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "32A"},
                                    "pattern": "^.{6}([A-Z]{3})",
                                    "group": 1,
                                },
                            },
                            "right": "CAD",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "59"},
                                    "pattern": "^(?:[^\\r\\n]*\\r?\\n){1}([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "59"},
                                            "pattern": "^(?:[^\\r\\n]*\\r?\\n){1}([^\\r\\n]*)",
                                            "group": 1,
                                            "flags": 8,
                                        },
                                    },
                                    "start": 0,
                                    "len": 35,
                                }
                            }
                        }
                    }
                },
            },
            {
                "when": {"all": [{"op": "!=", "left": {"mt": "59F"}, "right": ""}]},
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "59F"},
                                    "pattern": "^(?:\\/[^\\r\\n]*\\r?\\n)?(?:\\d\\/)?([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            }
                        }
                    }
                },
            },
            {
                "when": {"all": [{"op": "!=", "left": {"mt": "59"}, "right": ""}]},
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "59"},
                                    "pattern": "^(?:\\/[^\\r\\n]*\\r?\\n)?(?:\\d\\/)?([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            }
                        }
                    }
                },
            },
            {"when": {"all": []}, "then": {"value": "NOTPROVIDED"}},
        ],
    },
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.CdtrAcct.Id.IBAN": {
        "mode": "set",
        "rules": [
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "59"}, "right": ""},
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "59"},
                                    "pattern": "^/?([A-Za-z]{2}[A-Za-z0-9]+)",
                                    "group": 1,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "fn": "regex_extract",
                        "args": {
                            "text": {"mt": "59"},
                            "pattern": "^/?([A-Za-z0-9]+)",
                            "group": 1,
                        },
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_CdtrAcctIdIBAN",
                            "scope": "global",
                            "value": {"literal": "1"},
                        }
                    }
                ],
            },
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "59F"}, "right": ""},
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "59F"},
                                    "pattern": "^/?([A-Za-z]{2}[A-Za-z0-9]+)",
                                    "group": 1,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "fn": "regex_extract",
                        "args": {
                            "text": {"mt": "59F"},
                            "pattern": "^/?([A-Za-z0-9]+)",
                            "group": 1,
                        },
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_CdtrAcctIdIBAN",
                            "scope": "global",
                            "value": {"literal": "1"},
                        }
                    }
                ],
            },
        ],
    },
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.CdtrAcct.Id.Othr.Id": {
        "mode": "set",
        "rules": [
            {
                "set": [
                    {
                        "set_var": {
                            "name": "pacs.008::CdtrAcct.Id.Othr.Id",
                            "scope": "global",
                            "value": {
                                "from_db_query": {
                                    "alias": "pacs.008::CdtrAcct.Id.Othr.Id",
                                    "lookup_message_type": "pacs.008",
                                    "which_uetr_to_lookup": {"mtField": "121"},
                                    "where_to_lookup_uetr": {
                                        "message_type": "pacs.008",
                                        "path_kind": "xml",
                                        "field_path": ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.PmtId.UETR",
                                    },
                                    "field_to_extract": {
                                        "message_type": "pacs.008",
                                        "path_kind": "xml",
                                        "field_path": ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.CdtrAcct.Id.Othr.Id",
                                    },
                                    "fallback_literal": "",
                                }
                            },
                        }
                    }
                ],
                "when": {
                    "all": [
                        {
                            "op": "in",
                            "left": {
                                "substr": {"value": {"mt": "20"}, "start": 0, "len": 3}
                            },
                            "right": ["662", "663", "651"],
                        },
                        {
                            "op": "!=",
                            "left": {"var": "pacs.008::CdtrAcct.Id.Othr.Id"},
                            "right": "",
                        },
                    ],
                    "any": [
                        {"op": "!=", "left": {"mt": "59"}, "right": ""},
                        {"op": "!=", "left": {"mt": "59F"}, "right": ""},
                    ],
                },
                "then": {"value": {"var": "pacs.008::CdtrAcct.Id.Othr.Id"}},
            },
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "59"}, "right": ""},
                        {"op": "=", "left": {"var": "has_CdtrAcctIdIBAN"}, "right": ""},
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "59"},
                                    "pattern": "^\\s*/\\s*([^\\r\\n]+)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            }
                        }
                    }
                },
            },
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "59F"}, "right": ""},
                        {"op": "=", "left": {"var": "has_CdtrAcctIdIBAN"}, "right": ""},
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "59F"},
                                    "pattern": "^\\s*/\\s*([^\\r\\n]+)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            }
                        }
                    }
                },
            },
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "59A"}, "right": ""},
                        {"op": "=", "left": {"var": "has_CdtrAcctIdIBAN"}, "right": ""},
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "59A"},
                                    "pattern": "^\\s*/\\s*([^\\r\\n]+)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            }
                        }
                    }
                },
            },
        ],
    },
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.Dbtr.PstlAdr.Ctry": {
        "mode": "set",
        "rules": [
            # --------------------Comienza de reglas BREP -------------------------#
            {
                "set": [
                    {
                        "set_var": {
                            "name": "pacs.008::Dbtr.PstlAdr.Ctry",
                            "scope": "global",
                            "value": {
                                "from_db_query": {
                                    "alias": "pacs.008::Dbtr.PstlAdr.Ctry",
                                    "lookup_message_type": "pacs.008",
                                    "which_uetr_to_lookup": {"mtField": "121"},
                                    "where_to_lookup_uetr": {
                                        "message_type": "pacs.008",
                                        "path_kind": "xml",
                                        "field_path": ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.PmtId.UETR",
                                    },
                                    "field_to_extract": {
                                        "message_type": "pacs.008",
                                        "path_kind": "xml",
                                        "field_path": ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.Dbtr.PstlAdr.Ctry",
                                    },
                                    "fallback_literal": "",
                                }
                            },
                        }
                    },
                    {
                        "set_var": {
                            "name": "has_DbtrCtry",
                            "scope": "global",
                            "value": {"literal": "1"},
                        }
                    },
                ],
                "when": {
                    "all": [
                        {
                            "op": "in",
                            "left": {
                                "substr": {"value": {"mt": "20"}, "start": 0, "len": 3}
                            },
                            "right": ["662", "663", "651"],
                        },
                        {
                            "op": "!=",
                            "left": {"var": "pacs.008::Dbtr.PstlAdr.Ctry"},
                            "right": "",
                        },
                    ],
                    "any": [
                        {"op": "!=", "left": {"mt": "50F"}, "right": ""},
                        {"op": "!=", "left": {"mt": "50K"}, "right": ""},
                    ],
                },
                "then": {"value": {"var": "pacs.008::Dbtr.PstlAdr.Ctry"}},
            },
            {
                "when": {
                    "all": [
                        {
                            "op": "=",
                            "left": {"var": "has_DbtrPstlAdrAdrLine"},
                            "right": "",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "20"},
                                    "pattern": "^(7\\d{2})",
                                    "group": 1,
                                },
                            },
                            "right": "",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "50K"},
                                    "pattern": "^(?:[^\\r\\n]*\\r?\\n){4}[^,]*,\\s*([^\\r\\n]+)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "substr": {
                            "value": {
                                "trim": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "50K"},
                                            "pattern": "^(?:[^\\r\\n]*\\r?\\n){4}[^,]*,\\s*([^\\r\\n]+)",
                                            "group": 1,
                                            "flags": 8,
                                        },
                                    }
                                }
                            },
                            # -- "start": 0,
                            # --"len": 2,
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_DbtrCtry",
                            "scope": "global",
                            "value": {"literal": "1"},
                        }
                    },
                ],
            },
            {
                "when": {
                    "all": [
                        {
                            "op": "=",
                            "left": {"var": "has_DbtrPstlAdrAdrLine"},
                            "right": "",
                        },
                        {"op": "!=", "left": {"mt": "50K"}, "right": ""},
                        {
                            "op": "in",
                            "left": {
                                "substr": {"value": {"mt": "20"}, "start": 0, "len": 3}
                            },
                            "right": [
                                "500",
                                "506",
                                "510",
                                "513",
                                "520",
                                "523",
                                "526",
                                "527",
                                "540",
                                "543",
                                "544",
                                "545",
                                "546",
                                "547",
                                "501",
                                "502",
                                "511",
                                "512",
                                "521",
                                "522",
                                "524",
                                "525",
                                "541",
                                "544",
                            ],
                        },
                    ]
                },
                "then": {"value": {"literal": "CO"}},
                "set": [
                    {
                        "set_var": {
                            "name": "has_DbtrCtry",
                            "scope": "global",
                            "value": {"literal": "1"},
                        }
                    },
                ],
            },
        ],
    },
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.Dbtr.PstlAdr.TwnNm": {
        "mode": "append",
        "rules": [
            # --------------------Comienzo de reglas BREP -------------------------#
            {
                "set": [
                    {
                        "set_var": {
                            "name": "pacs.008::Dbtr.PstlAdr.TwnNm",
                            "scope": "global",
                            "value": {
                                "from_db_query": {
                                    "alias": "pacs.008::Dbtr.PstlAdr.TwnNm",
                                    "lookup_message_type": "pacs.008",
                                    "which_uetr_to_lookup": {"mtField": "121"},
                                    "where_to_lookup_uetr": {
                                        "message_type": "pacs.008",
                                        "path_kind": "xml",
                                        "field_path": ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.PmtId.UETR",
                                    },
                                    "field_to_extract": {
                                        "message_type": "pacs.008",
                                        "path_kind": "xml",
                                        "field_path": ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.Dbtr.PstlAdr.TwnNm",
                                    },
                                    "fallback_literal": "",
                                }
                            },
                        }
                    },
                    {
                        "set_var": {
                            "name": "has_DbtrTwnNm",
                            "scope": "global",
                            "value": {"literal": "1"},
                        }
                    },
                ],
                "when": {
                    "all": [
                        {
                            "op": "in",
                            "left": {
                                "substr": {"value": {"mt": "20"}, "start": 0, "len": 3}
                            },
                            "right": ["662", "663", "651"],
                        },
                        {
                            "op": "!=",
                            "left": {"var": "pacs.008::Dbtr.PstlAdr.TwnNm"},
                            "right": "",
                        },
                    ]
                },
                "then": {"value": {"var": "pacs.008::Dbtr.PstlAdr.TwnNm"}},
            },
            {
                "when": {
                    "all": [
                        {
                            "op": "!=",
                            "left": {"var": "has_DbtrCtry"},
                            "right": "",
                        },
                        {
                            "op": "=",
                            "left": {"var": "has_DbtrPstlAdrAdrLine"},
                            "right": "",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {
                                        "substr": {
                                            "value": {
                                                "fn": "regex_extract",
                                                "args": {
                                                    "text": {"mt": "20"},
                                                    "pattern": "^([^\\r\\n]*)",
                                                    "group": 1,
                                                },
                                            },
                                            "start": 0,
                                            "len": 3,
                                        }
                                    },
                                    "pattern": "^(7[0-9]{2})$",
                                    "group": 1,
                                },
                            },
                            "right": "",
                        },
                        {"op": "!=", "left": {"mt": "50K"}, "right": ""},
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "50K"},
                                    "pattern": "^(?:[^\\r\\n]*\\r?\\n){4}([^,\\r\\n]+)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "50K"},
                                    "pattern": "^(?:[^\\r\\n]*\\r?\\n){4}([^,\\r\\n]+)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            }
                        }
                    }
                },
            },
            {
                "when": {
                    "all": [
                        {
                            "op": "!=",
                            "left": {"var": "has_DbtrCtry"},
                            "right": "",
                        },
                        {
                            "op": "=",
                            "left": {"var": "has_DbtrPstlAdrAdrLine"},
                            "right": "",
                        },
                        {
                            "op": "in",
                            "left": {
                                "substr": {"value": {"mt": "20"}, "start": 0, "len": 3}
                            },
                            "right": [
                                "500",
                                "506",
                                "510",
                                "513",
                                "520",
                                "523",
                                "526",
                                "527",
                                "540",
                                "543",
                                "544",
                                "545",
                                "546",
                                "547",
                                "501",
                                "502",
                                "511",
                                "512",
                                "521",
                                "522",
                                "524",
                                "525",
                                "541",
                                "571",
                            ],
                        },
                        {"op": "!=", "left": {"mt": "50K"}, "right": ""},
                    ]
                },
                "then": {"value": {"literal": "BOGOTA D.C."}},
            },
            {
                "when": {
                    "all": [
                        {
                            "op": "!=",
                            "left": {"var": "has_DbtrCtry"},
                            "right": "",
                        },
                        {
                            "op": "=",
                            "left": {"var": "has_DbtrPstlAdrAdrLine"},
                            "right": "",
                        },
                        {
                            "op": "in",
                            "left": {
                                "substr": {"value": {"mt": "20"}, "start": 0, "len": 3}
                            },
                            "right": [
                                "500",
                                "506",
                                "510",
                                "513",
                                "520",
                                "523",
                                "526",
                                "527",
                                "540",
                                "543",
                                "544",
                                "545",
                                "546",
                                "547",
                                "501",
                                "502",
                                "511",
                                "512",
                                "521",
                                "522",
                                "524",
                                "525",
                                "541",
                            ],
                        },
                        {"op": "!=", "left": {"mt": "50F"}, "right": ""},
                    ]
                },
                "then": {"value": {"literal": "BOGOTA D.C."}},
            },
            # --------------------Fin de reglas BREP -------------------------#
        ],
    },
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.Dbtr.PstlAdr.StrtNm": {
        "mode": "set",
        "rules": [
            # --------------------Comienzo de reglas BREP -------------------------#
            {
                "set": [
                    {
                        "set_var": {
                            "name": "pacs.008::Dbtr.PstlAdr.StrtNm",
                            "scope": "global",
                            "value": {
                                "from_db_query": {
                                    "alias": "pacs.008::Dbtr.PstlAdr.StrtNm",
                                    "lookup_message_type": "pacs.008",
                                    "which_uetr_to_lookup": {"mtField": "121"},
                                    "where_to_lookup_uetr": {
                                        "message_type": "pacs.008",
                                        "path_kind": "xml",
                                        "field_path": ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.PmtId.UETR",
                                    },
                                    "field_to_extract": {
                                        "message_type": "pacs.008",
                                        "path_kind": "xml",
                                        "field_path": ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.Dbtr.PstlAdr.StrtNm",
                                    },
                                    "fallback_literal": "",
                                }
                            },
                        }
                    },
                    {
                        "set_var": {
                            "name": "has_DbtrStrtNm",
                            "scope": "global",
                            "value": {"literal": "1"},
                        }
                    },
                ],
                "when": {
                    "all": [
                        {
                            "op": "!=",
                            "left": {"var": "has_DbtrCtry"},
                            "right": "",
                        },
                        {
                            "op": "in",
                            "left": {
                                "substr": {"value": {"mt": "20"}, "start": 0, "len": 3}
                            },
                            "right": ["662", "663", "651"],
                        },
                        {
                            "op": "!=",
                            "left": {"var": "pacs.008::Dbtr.PstlAdr.StrtNm"},
                            "right": "",
                        },
                    ],
                    "any": [
                        {"op": "!=", "left": {"mt": "50F"}, "right": ""},
                        {"op": "!=", "left": {"mt": "50K"}, "right": ""},
                    ],
                },
                "then": {
                    "value": {
                        "substr": {
                            "value": {"var": "pacs.008::Dbtr.PstlAdr.StrtNm"},
                            "start": 0,
                            "len": 70,
                        }
                    }
                },
            },
            {
                "when": {
                    "all": [
                        {
                            "op": "!=",
                            "left": {"var": "has_DbtrCtry"},
                            "right": "",
                        },
                        {
                            "op": "=",
                            "left": {"var": "has_DbtrPstlAdrAdrLine"},
                            "right": "",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "20"},
                                    "pattern": "^(7\\d{9})",
                                    "group": 1,
                                },
                            },
                            "right": "",
                        },
                        {"op": "!=", "left": {"mt": "50K"}, "right": ""},
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "50K"},
                                    "pattern": "^(?:[^\\r\\n]*\\r?\\n){3}[^,]*,\\s*([^\\r\\n]+)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "substr": {
                            "value": {
                                "trim": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "50K"},
                                            "pattern": "^(?:[^\\r\\n]*\\r?\\n){3}[^,]*,\\s*([^\\r\\n]+)",
                                            "group": 1,
                                            "flags": 8,
                                        },
                                    }
                                }
                            },
                            "start": 0,
                            "len": 70,
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_DbtrStrtNm",
                            "scope": "global",
                            "value": {"literal": "1"},
                        }
                    }
                ],
            },
            {
                "when": {
                    "all": [
                        {
                            "op": "!=",
                            "left": {"var": "has_DbtrCtry"},
                            "right": "",
                        },
                        {
                            "op": "=",
                            "left": {"var": "has_DbtrPstlAdrAdrLine"},
                            "right": "",
                        },
                        {
                            "op": "in",
                            "left": {
                                "substr": {
                                    "value": {"mt": "20"},
                                    "start": 0,
                                    "len": 3,
                                }
                            },
                            "right": [
                                "500",
                                "506",
                                "510",
                                "513",
                                "520",
                                "523",
                                "526",
                                "527",
                                "540",
                                "543",
                                "544",
                                "545",
                                "546",
                                "547",
                                "501",
                                "502",
                                "511",
                                "512",
                                "521",
                                "522",
                                "524",
                                "525",
                                "541",
                                "571",
                            ],
                        },
                        {"op": "!=", "left": {"mt": "50K"}, "right": ""},
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "50K"},
                                    "pattern": "^(?:[^\\r\\n]*\\r?\\n){1}([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "substr": {
                            "value": {
                                "trim": {
                                    "value": {
                                        "fn": "regex_replace",
                                        "args": {
                                            "text": {
                                                "fn": "regex_replace",
                                                "args": {
                                                    "text": {
                                                        "fn": "regex_replace",
                                                        "args": {
                                                            "text": {
                                                                "fn": "regex_extract",
                                                                "args": {
                                                                    "text": {
                                                                        "mt": "50K"
                                                                    },
                                                                    "pattern": "^(?:[^\\r\\n]*[\\r\\n]+){1}((?:[^\\r\\n]+(?:[\\r\\n]+|$)){1,4})",
                                                                    "group": 1,
                                                                    "flags": 8,
                                                                },
                                                            },
                                                            "pattern": "(?m)^\\d\\/",
                                                            "repl": "",
                                                            "flags": 8,
                                                        },
                                                    },
                                                    "pattern": "/",
                                                    "repl": " ",
                                                    "flags": 8,
                                                },
                                            },
                                            "pattern": "[\\r\\n]+",
                                            "repl": " ",
                                            "flags": 8,
                                        },
                                    }
                                }
                            },
                            "start": 0,
                            "len": 70,
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_DbtrStrtNm",
                            "scope": "global",
                            "value": {"literal": "1"},
                        }
                    }
                ],
            },
            {
                "when": {
                    "all": [
                        {
                            "op": "!=",
                            "left": {"var": "has_DbtrCtry"},
                            "right": "",
                        },
                        {
                            "op": "=",
                            "left": {"var": "has_DbtrPstlAdrAdrLine"},
                            "right": "",
                        },
                        {"op": "!=", "left": {"mt": "50F"}, "right": ""},
                        {
                            "op": "in",
                            "left": {
                                "substr": {"value": {"mt": "20"}, "start": 0, "len": 3}
                            },
                            "right": [
                                "500",
                                "506",
                                "510",
                                "513",
                                "520",
                                "523",
                                "526",
                                "527",
                                "540",
                                "543",
                                "544",
                                "545",
                                "546",
                                "547",
                                "501",
                                "502",
                                "511",
                                "512",
                                "521",
                                "522",
                                "524",
                                "525",
                                "541",
                            ],
                        },
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "50F"},
                                    "pattern": "^(?:[^\\r\\n]*\\r?\\n){1}([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "substr": {
                            "value": {
                                "trim": {
                                    "value": {
                                        "fn": "regex_replace",
                                        "args": {
                                            "text": {
                                                "fn": "regex_replace",
                                                "args": {
                                                    "text": {
                                                        "fn": "regex_replace",
                                                        "args": {
                                                            "text": {
                                                                "fn": "regex_extract",
                                                                "args": {
                                                                    "text": {
                                                                        "mt": "50F"
                                                                    },
                                                                    "pattern": "^(?:[^\\r\\n]*\\r?\\n){1}((?:[^\\r\\n]+(?:\\r?\\n|$)){1,3})",
                                                                    "group": 1,
                                                                    "flags": 8,
                                                                },
                                                            },
                                                            "pattern": "(?m)^\\d\\/",
                                                            "repl": "",
                                                            "flags": 8,
                                                        },
                                                    },
                                                    "pattern": "/",
                                                    "repl": " ",
                                                    "flags": 8,
                                                },
                                            },
                                            "pattern": "\\s*(?:\\r?\\n)+\\s*",
                                            "repl": " ",
                                            "flags": 8,
                                        },
                                    }
                                }
                            },
                            "start": 0,
                            "len": 70,
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_DbtrStrtNm",
                            "scope": "global",
                            "value": {"literal": "1"},
                        }
                    }
                ],
            },
            # --------------------Fin de reglas BREP -------------------------#
        ],
    },
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.Dbtr.PstlAdr.Dept": {
        "mode": "set",
        "rules": [
            # Regla PREVIA (662, 663, 651) - Primera prioridad
            {
                "set": [
                    {
                        "set_var": {
                            "name": "pacs.008::Dbtr.PstlAdr.Dept",
                            "scope": "global",
                            "value": {
                                "from_db_query": {
                                    "alias": "pacs.008::Dbtr.PstlAdr.Dept",
                                    "lookup_message_type": "pacs.008",
                                    "which_uetr_to_lookup": {"mtField": "121"},
                                    "where_to_lookup_uetr": {
                                        "message_type": "pacs.008",
                                        "path_kind": "xml",
                                        "field_path": ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.PmtId.UETR",
                                    },
                                    "field_to_extract": {
                                        "message_type": "pacs.008",
                                        "path_kind": "xml",
                                        "field_path": ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.Dbtr.PstlAdr.Dept",
                                    },
                                    "fallback_literal": "",
                                }
                            },
                        }
                    },
                    {
                        "set_var": {
                            "name": "has_DbtrDept",
                            "scope": "global",
                            "value": {"literal": "1"},
                        }
                    },
                ],
                "when": {
                    "all": [
                        {
                            "op": "!=",
                            "left": {"var": "has_DbtrCtry"},
                            "right": "",
                        },
                        {
                            "op": "in",
                            "left": {
                                "substr": {"value": {"mt": "20"}, "start": 0, "len": 3}
                            },
                            "right": ["662", "663", "651"],
                        },
                        {
                            "op": "!=",
                            "left": {"var": "pacs.008::Dbtr.PstlAdr.Dept"},
                            "right": "",
                        },
                    ],
                    "any": [
                        {"op": "!=", "left": {"mt": "50F"}, "right": ""},
                        {"op": "!=", "left": {"mt": "50K"}, "right": ""},
                    ],
                },
                "then": {"value": {"var": "pacs.008::Dbtr.PstlAdr.Dept"}},
            },
            {
                "set": [
                    {
                        "set_var": {
                            "name": "pacs.008::Dbtr.PstlAdr.Dept",
                            "scope": "global",
                            "value": {
                                "from_db_query": {
                                    "alias": "pacs.008::Dbtr.PstlAdr.Dept",
                                    "lookup_message_type": "pacs.008",
                                    "which_uetr_to_lookup": {"mtField": "121"},
                                    "where_to_lookup_uetr": {
                                        "message_type": "pacs.008",
                                        "path_kind": "xml",
                                        "field_path": ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.PmtId.UETR",
                                    },
                                    "field_to_extract": {
                                        "message_type": "pacs.008",
                                        "path_kind": "xml",
                                        "field_path": ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.Dbtr.PstlAdr.Dept",
                                    },
                                    "fallback_literal": "",
                                }
                            },
                        }
                    },
                    {
                        "set_var": {
                            "name": "has_DbtrDept",
                            "scope": "global",
                            "value": {"literal": "1"},
                        }
                    },
                ],
                "when": {
                    "all": [
                        {
                            "op": "!=",
                            "left": {"var": "has_DbtrCtry"},
                            "right": "",
                        },
                        {
                            "op": "in",
                            "left": {
                                "substr": {"value": {"mt": "20"}, "start": 0, "len": 3}
                            },
                            "right": ["662", "663", "651"],
                        },
                        {
                            "op": "!=",
                            "left": {"var": "pacs.008::Dbtr.PstlAdr.Dept"},
                            "right": "",
                        },
                    ],
                    "any": [
                        {"op": "!=", "left": {"mt": "59F"}, "right": ""},
                        {"op": "!=", "left": {"mt": "59"}, "right": ""},
                    ],
                },
                "then": {"value": {"var": "pacs.008::Dbtr.PstlAdr.Dept"}},
            },
            # Regla BREP (700-799) - Segunda prioridad
            {
                "when": {
                    "all": [
                        {
                            "op": "!=",
                            "left": {"var": "has_DbtrCtry"},
                            "right": "",
                        },
                        {
                            "op": "=",
                            "left": {"var": "has_DbtrPstlAdrAdrLine"},
                            "right": "",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {
                                        "substr": {
                                            "value": {"mt": "20"},
                                            "start": 0,
                                            "len": 3,
                                        }
                                    },
                                    "pattern": "^(7[0-9]{2})$",
                                    "group": 1,
                                },
                            },
                            "right": "",
                        },
                        {"op": "!=", "left": {"mt": "50K"}, "right": ""},
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "50K"},
                                    "pattern": "^(?:[^\\r\\n]*\\r?\\n){1}([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "50K"},
                                    "pattern": "^(?:[^\\r\\n]*\\r?\\n){1}([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            }
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_DbtrDept",
                            "scope": "global",
                            "value": {"literal": "1"},
                        }
                    }
                ],
            },
        ],
    },
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.Dbtr.PstlAdr.AdrLine": {
        "mode": "append",
        "rules": [
            {
                "set": [
                    {
                        "set_var": {
                            "name": "pacs.008::Dbtr.PstlAdr.AdrLine",
                            "scope": "global",
                            "value": {
                                "from_db_query": {
                                    "alias": "pacs.008::Dbtr.PstlAdr.AdrLine",
                                    "lookup_message_type": "pacs.008",
                                    "which_uetr_to_lookup": {"mtField": "121"},
                                    "where_to_lookup_uetr": {
                                        "message_type": "pacs.008",
                                        "path_kind": "xml",
                                        "field_path": ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.PmtId.UETR",
                                    },
                                    "field_to_extract": {
                                        "message_type": "pacs.008",
                                        "path_kind": "xml",
                                        "field_path": ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.Dbtr.PstlAdr.AdrLine",
                                    },
                                    "fallback_literal": "",
                                }
                            },
                        }
                    },
                    {
                        "set_var": {
                            "name": "has_DbtrPstlAdrAdrLine",
                            "scope": "global",
                            "value": {"literal": "1"},
                        }
                    },
                    {
                        "set_var": {
                            "name": "has_DbtrPstlAdrAdrLineMatch",
                            "scope": "global",
                            "value": {"literal": "1"},
                        }
                    },
                ],
                "when": {
                    "all": [
                        {
                            "op": "!=",
                            "left": {"var": "pacs.008::Dbtr.PstlAdr.AdrLine"},
                            "right": "",
                        },
                        {
                            "op": "in",
                            "left": {
                                "substr": {"value": {"mt": "20"}, "start": 0, "len": 3}
                            },
                            "right": ["662", "663", "651"],
                        },
                    ]
                },
                "then": {"value": {"var": "pacs.008::Dbtr.PstlAdr.AdrLine"}},
            },
            # -- Reglas 2
            {
                "when": {
                    "all": [
                        {
                            "op": "=",
                            "left": {"var": "has_DbtrCtry"},
                            "right": "",
                        },
                        {
                            "op": "=",
                            "left": {"var": "has_DbtrPstlAdrAdrLineMatch"},
                            "right": "",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "20"},
                                    "pattern": "^(7\\d{9})",
                                    "group": 1,
                                },
                            },
                            "right": "",
                        },
                        {"op": "!=", "left": {"mt": "50K"}, "right": ""},
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "50K"},
                                    "pattern": "^(?:[^\\r\\n]*\\r?\\n){2}([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            },
                            "right": "",
                        },
                    ],
                    "any": [
                        {
                            "all": [
                                {
                                    "op": "=",
                                    "left": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "50K"},
                                            "pattern": "^(?:[^\\r\\n]*\\r?\\n){3}([^\\r\\n]*)",
                                            "group": 1,
                                            "flags": 8,
                                        },
                                    },
                                    "right": "",
                                }
                            ]
                        },
                        {
                            "all": [
                                {
                                    "op": "=",
                                    "left": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "50K"},
                                            "pattern": "^(?:[^\\r\\n]*\\r?\\n){4}([^\\r\\n]*)",
                                            "group": 1,
                                            "flags": 8,
                                        },
                                    },
                                    "right": "",
                                }
                            ]
                        },
                        {
                            "all": [
                                {
                                    "op": "=",
                                    "left": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "50K"},
                                            "pattern": "^(?:[^\r\n]*\r?\n){4}([^,\r\n]*,[^\r\n]*)",
                                            "group": 1,
                                            "flags": 8,
                                        },
                                    },
                                    "right": "",
                                }
                            ]
                        },
                    ],
                },
                "then": {
                    "value": {
                        "substr": {
                            "value": {
                                "trim": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "50K"},
                                            "pattern": "^(?:[^\\r\\n]*\\r?\\n){2}([^\\r\\n]*)",
                                            "group": 1,
                                            "flags": 8,
                                        },
                                    }
                                }
                            },
                            "start": 0,
                            "len": 70,
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_DbtrPstlAdrAdrLine",
                            "scope": "global",
                            "value": {"literal": "1"},
                        }
                    }
                ],
            },
            # -- Reglas 3
            {
                "when": {
                    "all": [
                        {
                            "op": "=",
                            "left": {"var": "has_DbtrCtry"},
                            "right": "",
                        },
                        {
                            "op": "=",
                            "left": {"var": "has_DbtrPstlAdrAdrLineMatch"},
                            "right": "",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "20"},
                                    "pattern": "^(7\\d{9})",
                                    "group": 1,
                                },
                            },
                            "right": "",
                        },
                        {"op": "!=", "left": {"mt": "50K"}, "right": ""},
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "50K"},
                                    "pattern": "^(?:[^\\r\\n]*\\r?\\n){3}([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            },
                            "right": "",
                        },
                    ],
                    "any": [
                        {
                            "all": [
                                {
                                    "op": "=",
                                    "left": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "50K"},
                                            "pattern": "^(?:[^\\r\\n]*\\r?\\n){3}([^\\r\\n]*)",
                                            "group": 1,
                                            "flags": 8,
                                        },
                                    },
                                    "right": "",
                                }
                            ]
                        },
                        {
                            "all": [
                                {
                                    "op": "=",
                                    "left": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "50K"},
                                            "pattern": "^(?:[^\\r\\n]*\\r?\\n){4}([^\\r\\n]*)",
                                            "group": 1,
                                            "flags": 8,
                                        },
                                    },
                                    "right": "",
                                }
                            ]
                        },
                        {
                            "all": [
                                {
                                    "op": "=",
                                    "left": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "50K"},
                                            "pattern": "^(?:[^\r\n]*\r?\n){4}([^,\r\n]*,[^\r\n]*)",
                                            "group": 1,
                                            "flags": 8,
                                        },
                                    },
                                    "right": "",
                                }
                            ]
                        },
                    ],
                },
                "then": {
                    "value": {
                        "substr": {
                            "value": {
                                "trim": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "50K"},
                                            "pattern": "^(?:[^\\r\\n]*\\r?\\n){3}([^\\r\\n]*)",
                                            "group": 1,
                                            "flags": 8,
                                        },
                                    }
                                }
                            },
                            "start": 0,
                            "len": 70,
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_DbtrPstlAdrAdrLine",
                            "scope": "global",
                            "value": {"literal": "1"},
                        }
                    }
                ],
            },
            # -- Reglas 4
            {
                "when": {
                    "all": [
                        {
                            "op": "=",
                            "left": {"var": "has_DbtrCtry"},
                            "right": "",
                        },
                        {
                            "op": "=",
                            "left": {"var": "has_DbtrPstlAdrAdrLineMatch"},
                            "right": "",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "20"},
                                    "pattern": "^(7\\d{9})",
                                    "group": 1,
                                },
                            },
                            "right": "",
                        },
                        {"op": "!=", "left": {"mt": "50K"}, "right": ""},
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "50K"},
                                    "pattern": "^(?:[^\\r\\n]*\\r?\\n){4}([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            },
                            "right": "",
                        },
                    ],
                    "any": [
                        {
                            "all": [
                                {
                                    "op": "=",
                                    "left": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "50K"},
                                            "pattern": "^(?:[^\\r\\n]*\\r?\\n){3}([^\\r\\n]*)",
                                            "group": 1,
                                            "flags": 8,
                                        },
                                    },
                                    "right": "",
                                }
                            ]
                        },
                        {
                            "all": [
                                {
                                    "op": "=",
                                    "left": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "50K"},
                                            "pattern": "^(?:[^\\r\\n]*\\r?\\n){4}([^\\r\\n]*)",
                                            "group": 1,
                                            "flags": 8,
                                        },
                                    },
                                    "right": "",
                                }
                            ]
                        },
                        {
                            "all": [
                                {
                                    "op": "=",
                                    "left": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "50K"},
                                            "pattern": "^(?:[^\r\n]*\r?\n){4}([^,\r\n]*,[^\r\n]*)",
                                            "group": 1,
                                            "flags": 8,
                                        },
                                    },
                                    "right": "",
                                }
                            ]
                        },
                    ],
                },
                "then": {
                    "value": {
                        "substr": {
                            "value": {
                                "trim": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "50K"},
                                            "pattern": "^(?:[^\\r\\n]*\\r?\\n){4}([^\\r\\n]*)",
                                            "group": 1,
                                            "flags": 8,
                                        },
                                    }
                                }
                            },
                            "start": 0,
                            "len": 70,
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_DbtrPstlAdrAdrLine",
                            "scope": "global",
                            "value": {"literal": "1"},
                        }
                    }
                ],
            },
            # -- Regla 5
            {
                "when": {
                    "all": [
                        {
                            "op": "=",
                            "left": {"var": "has_DbtrCtry"},
                            "right": "",
                        },
                        {
                            "op": "=",
                            "left": {"var": "has_DbtrPstlAdrAdrLineMatch"},
                            "right": "",
                        },
                        {
                            "op": "in",
                            "left": {
                                "substr": {"value": {"mt": "20"}, "start": 0, "len": 3}
                            },
                            "right": [
                                "500",
                                "506",
                                "510",
                                "513",
                                "520",
                                "523",
                                "526",
                                "527",
                                "540",
                                "543",
                                "544",
                                "545",
                                "546",
                                "547",
                                "501",
                                "502",
                                "511",
                                "512",
                                "521",
                                "522",
                                "524",
                                "525",
                                "541",
                                "571",
                            ],
                        },
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "50K"},
                                    "pattern": "^(?:[^\\r\\n]*\\r?\\n){2}([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            },
                            "right": "",
                        },
                        {
                            "op": "=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "50K"},
                                    "pattern": "^(?:[^\\r\\n]*\\r?\\n){1}([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            },
                            "right": "",
                        },
                        {"op": "=", "left": {"mt": "50F"}, "right": ""},
                    ]
                },
                "then": {
                    "value": {
                        "substr": {
                            "value": {
                                "trim": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "50K"},
                                            "pattern": "^(?:[^\\r\\n]*\\r?\\n){2}([^\\r\\n]*)",
                                            "group": 1,
                                            "flags": 8,
                                        },
                                    }
                                }
                            },
                            "start": 0,
                            "len": 70,
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_DbtrPstlAdrAdrLine",
                            "scope": "global",
                            "value": {"literal": "1"},
                        }
                    }
                ],
            },
            # -- Regla 6
            {
                "when": {
                    "all": [
                        {
                            "op": "=",
                            "left": {"var": "has_DbtrCtry"},
                            "right": "",
                        },
                        {
                            "op": "=",
                            "left": {"var": "has_DbtrPstlAdrAdrLineMatch"},
                            "right": "",
                        },
                        {
                            "op": "in",
                            "left": {
                                "substr": {"value": {"mt": "20"}, "start": 0, "len": 3}
                            },
                            "right": [
                                "500",
                                "506",
                                "510",
                                "513",
                                "520",
                                "523",
                                "526",
                                "527",
                                "540",
                                "543",
                                "544",
                                "545",
                                "546",
                                "547",
                                "501",
                                "502",
                                "511",
                                "512",
                                "521",
                                "522",
                                "524",
                                "525",
                                "541",
                                "571",
                            ],
                        },
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "50K"},
                                    "pattern": "^(?:[^\\r\\n]*\\r?\\n){3}([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            },
                            "right": "",
                        },
                        {
                            "op": "=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "50K"},
                                    "pattern": "^(?:[^\\r\\n]*\\r?\\n){1}([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            },
                            "right": "",
                        },
                        {"op": "=", "left": {"mt": "50F"}, "right": ""},
                    ]
                },
                "then": {
                    "value": {
                        "substr": {
                            "value": {
                                "trim": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "50K"},
                                            "pattern": "^(?:[^\\r\\n]*\\r?\\n){3}([^\\r\\n]*)",
                                            "group": 1,
                                            "flags": 8,
                                        },
                                    }
                                }
                            },
                            "start": 0,
                            "len": 70,
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_DbtrPstlAdrAdrLine",
                            "scope": "global",
                            "value": {"literal": "1"},
                        }
                    }
                ],
            },
            # -- Regla 7
            {
                "when": {
                    "all": [
                        {
                            "op": "=",
                            "left": {"var": "has_DbtrCtry"},
                            "right": "",
                        },
                        {
                            "op": "=",
                            "left": {"var": "has_DbtrPstlAdrAdrLineMatch"},
                            "right": "",
                        },
                        {
                            "op": "in",
                            "left": {
                                "substr": {"value": {"mt": "20"}, "start": 0, "len": 3}
                            },
                            "right": [
                                "500",
                                "506",
                                "510",
                                "513",
                                "520",
                                "523",
                                "526",
                                "527",
                                "540",
                                "543",
                                "544",
                                "545",
                                "546",
                                "547",
                                "501",
                                "502",
                                "511",
                                "512",
                                "521",
                                "522",
                                "524",
                                "525",
                                "541",
                                "571",
                            ],
                        },
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "50K"},
                                    "pattern": "^(?:[^\\r\\n]*\\r?\\n){4}([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            },
                            "right": "",
                        },
                        {
                            "op": "=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "50K"},
                                    "pattern": "^(?:[^\\r\\n]*\\r?\\n){1}([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            },
                            "right": "",
                        },
                        {"op": "=", "left": {"mt": "50F"}, "right": ""},
                    ]
                },
                "then": {
                    "value": {
                        "substr": {
                            "value": {
                                "trim": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "50K"},
                                            "pattern": "^(?:[^\\r\\n]*\\r?\\n){4}([^\\r\\n]*)",
                                            "group": 1,
                                            "flags": 8,
                                        },
                                    }
                                }
                            },
                            "start": 0,
                            "len": 70,
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_DbtrPstlAdrAdrLine",
                            "scope": "global",
                            "value": {"literal": "1"},
                        }
                    }
                ],
            },
            # -- Regla 8
            {
                "when": {
                    "all": [
                        {
                            "op": "=",
                            "left": {"var": "has_DbtrCtry"},
                            "right": "",
                        },
                        {
                            "op": "=",
                            "left": {"var": "has_DbtrPstlAdrAdrLineMatch"},
                            "right": "",
                        },
                        {
                            "op": "in",
                            "left": {
                                "substr": {"value": {"mt": "20"}, "start": 0, "len": 3}
                            },
                            "right": [
                                "500",
                                "506",
                                "510",
                                "513",
                                "520",
                                "523",
                                "526",
                                "527",
                                "540",
                                "543",
                                "544",
                                "545",
                                "546",
                                "547",
                                "501",
                                "502",
                                "511",
                                "512",
                                "521",
                                "522",
                                "524",
                                "525",
                                "541",
                                "571",
                            ],
                        },
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "50K"},
                                    "pattern": "^(?:[^\\r\\n]*\\r?\\n){2}([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            },
                            "right": "",
                        },
                        {"op": "!=", "left": {"mt": "50F"}, "right": ""},
                        {
                            "op": "=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "50F"},
                                    "pattern": "^(?:[^\\r\\n]*\\r?\\n){1}([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            },
                            "right": "",
                        },
                        {"op": "=", "left": {"mt": "50K"}, "right": ""},
                    ]
                },
                "then": {
                    "value": {
                        "substr": {
                            "value": {
                                "trim": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "50K"},
                                            "pattern": "^(?:[^\\r\\n]*\\r?\\n){2}([^\\r\\n]*)",
                                            "group": 1,
                                            "flags": 8,
                                        },
                                    }
                                }
                            },
                            "start": 0,
                            "len": 70,
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_DbtrPstlAdrAdrLine",
                            "scope": "global",
                            "value": {"literal": "1"},
                        }
                    }
                ],
            },
            # -- Regla 9
            {
                "when": {
                    "all": [
                        {
                            "op": "=",
                            "left": {"var": "has_DbtrCtry"},
                            "right": "",
                        },
                        {
                            "op": "=",
                            "left": {"var": "has_DbtrPstlAdrAdrLineMatch"},
                            "right": "",
                        },
                        {
                            "op": "in",
                            "left": {
                                "substr": {"value": {"mt": "20"}, "start": 0, "len": 3}
                            },
                            "right": [
                                "500",
                                "506",
                                "510",
                                "513",
                                "520",
                                "523",
                                "526",
                                "527",
                                "540",
                                "543",
                                "544",
                                "545",
                                "546",
                                "547",
                                "501",
                                "502",
                                "511",
                                "512",
                                "521",
                                "522",
                                "524",
                                "525",
                                "541",
                                "571",
                            ],
                        },
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "50K"},
                                    "pattern": "^(?:[^\\r\\n]*\\r?\\n){3}([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            },
                            "right": "",
                        },
                        {"op": "!=", "left": {"mt": "50F"}, "right": ""},
                        {
                            "op": "=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "50F"},
                                    "pattern": "^(?:[^\\r\\n]*\\r?\\n){1}([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            },
                            "right": "",
                        },
                        {"op": "=", "left": {"mt": "50K"}, "right": ""},
                    ]
                },
                "then": {
                    "value": {
                        "substr": {
                            "value": {
                                "trim": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "50K"},
                                            "pattern": "^(?:[^\\r\\n]*\\r?\\n){3}([^\\r\\n]*)",
                                            "group": 1,
                                            "flags": 8,
                                        },
                                    }
                                }
                            },
                            "start": 0,
                            "len": 70,
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_DbtrPstlAdrAdrLine",
                            "scope": "global",
                            "value": {"literal": "1"},
                        }
                    }
                ],
            },
            # -- Regla 10
            {
                "when": {
                    "all": [
                        {
                            "op": "=",
                            "left": {"var": "has_DbtrCtry"},
                            "right": "",
                        },
                        {
                            "op": "=",
                            "left": {"var": "has_DbtrPstlAdrAdrLineMatch"},
                            "right": "",
                        },
                        {
                            "op": "in",
                            "left": {
                                "substr": {"value": {"mt": "20"}, "start": 0, "len": 3}
                            },
                            "right": [
                                "500",
                                "506",
                                "510",
                                "513",
                                "520",
                                "523",
                                "526",
                                "527",
                                "540",
                                "543",
                                "544",
                                "545",
                                "546",
                                "547",
                                "501",
                                "502",
                                "511",
                                "512",
                                "521",
                                "522",
                                "524",
                                "525",
                                "541",
                                "571",
                            ],
                        },
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "50K"},
                                    "pattern": "^(?:[^\\r\\n]*\\r?\\n){4}([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            },
                            "right": "",
                        },
                        {"op": "!=", "left": {"mt": "50F"}, "right": ""},
                        {
                            "op": "=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "50F"},
                                    "pattern": "^(?:[^\\r\\n]*\\r?\\n){1}([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            },
                            "right": "",
                        },
                        {"op": "=", "left": {"mt": "50K"}, "right": ""},
                    ]
                },
                "then": {
                    "value": {
                        "substr": {
                            "value": {
                                "trim": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "50K"},
                                            "pattern": "^(?:[^\\r\\n]*\\r?\\n){4}([^\\r\\n]*)",
                                            "group": 1,
                                            "flags": 8,
                                        },
                                    }
                                }
                            },
                            "start": 0,
                            "len": 70,
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_DbtrPstlAdrAdrLine",
                            "scope": "global",
                            "value": {"literal": "1"},
                        }
                    }
                ],
            },
            # -- Regla 11
            {
                "when": {
                    "all": [
                        {
                            "op": "=",
                            "left": {"var": "has_DbtrCtry"},
                            "right": "",
                        },
                        {
                            "op": "=",
                            "left": {"var": "has_DbtrPstlAdrAdrLineMatch"},
                            "right": "",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "20"},
                                    "pattern": "^(6\\d{9})",
                                    "group": 1,
                                },
                            },
                            "right": "",
                        },
                        {"op": "!=", "left": {"mt": "50K"}, "right": ""},
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "50K"},
                                    "pattern": "^(?:[^\\r\\n]*\\r?\\n){2}([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "substr": {
                            "value": {
                                "trim": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "50K"},
                                            "pattern": "^(?:[^\\r\\n]*\\r?\\n){2}([^\\r\\n]*)",
                                            "group": 1,
                                            "flags": 8,
                                        },
                                    }
                                }
                            },
                            "start": 0,
                            "len": 70,
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_DbtrPstlAdrAdrLine",
                            "scope": "global",
                            "value": {"literal": "1"},
                        }
                    }
                ],
            },
            # -- Regla 12
            {
                "when": {
                    "all": [
                        {
                            "op": "=",
                            "left": {"var": "has_DbtrCtry"},
                            "right": "",
                        },
                        {
                            "op": "=",
                            "left": {"var": "has_DbtrPstlAdrAdrLineMatch"},
                            "right": "",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "20"},
                                    "pattern": "^(6\\d{9})",
                                    "group": 1,
                                },
                            },
                            "right": "",
                        },
                        {"op": "!=", "left": {"mt": "50K"}, "right": ""},
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "50K"},
                                    "pattern": "^(?:[^\\r\\n]*\\r?\\n){3}([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "substr": {
                            "value": {
                                "trim": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "50K"},
                                            "pattern": "^(?:[^\\r\\n]*\\r?\\n){3}([^\\r\\n]*)",
                                            "group": 1,
                                            "flags": 8,
                                        },
                                    }
                                }
                            },
                            "start": 0,
                            "len": 70,
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_DbtrPstlAdrAdrLine",
                            "scope": "global",
                            "value": {"literal": "1"},
                        }
                    }
                ],
            },
            # -- Regla 13
            {
                "when": {
                    "all": [
                        {
                            "op": "=",
                            "left": {"var": "has_DbtrCtry"},
                            "right": "",
                        },
                        {
                            "op": "=",
                            "left": {"var": "has_DbtrPstlAdrAdrLineMatch"},
                            "right": "",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "20"},
                                    "pattern": "^(6\\d{9})",
                                    "group": 1,
                                },
                            },
                            "right": "",
                        },
                        {"op": "!=", "left": {"mt": "50K"}, "right": ""},
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "50K"},
                                    "pattern": "^(?:[^\\r\\n]*\\r?\\n){4}([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "substr": {
                            "value": {
                                "trim": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "50K"},
                                            "pattern": "^(?:[^\\r\\n]*\\r?\\n){4}([^\\r\\n]*)",
                                            "group": 1,
                                            "flags": 8,
                                        },
                                    }
                                }
                            },
                            "start": 0,
                            "len": 70,
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_DbtrPstlAdrAdrLine",
                            "scope": "global",
                            "value": {"literal": "1"},
                        }
                    }
                ],
            },
            # Reglas 14
            {
                "when": {
                    "all": [
                        {
                            "op": "=",
                            "left": {"var": "has_DbtrCtry"},
                            "right": "",
                        },
                        {
                            "op": "=",
                            "left": {"var": "has_DbtrPstlAdrAdrLineMatch"},
                            "right": "",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "20"},
                                    "pattern": "^(6\\d{9})",
                                    "group": 1,
                                },
                            },
                            "right": "",
                        },
                        {"op": "=", "left": {"mt": "50K"}, "right": ""},
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "50F"},
                                    "pattern": "^(?:[^\\r\\n]*\\r?\\n){2}([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "substr": {
                            "value": {
                                "trim": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "50F"},
                                            "pattern": "(?:^|\\r?\\n)2\\/([^\\r\\n]*)",
                                            "group": 1,
                                            "flags": 8,
                                        },
                                    }
                                }
                            },
                            "start": 0,
                            "len": 70,
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_DbtrPstlAdrAdrLine",
                            "scope": "global",
                            "value": {"literal": "1"},
                        }
                    }
                ],
            },
            # Reglas 15
            {
                "when": {
                    "all": [
                        {
                            "op": "=",
                            "left": {"var": "has_DbtrCtry"},
                            "right": "",
                        },
                        {
                            "op": "=",
                            "left": {"var": "has_DbtrPstlAdrAdrLineMatch"},
                            "right": "",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "20"},
                                    "pattern": "^(6\\d{9})",
                                    "group": 1,
                                },
                            },
                            "right": "",
                        },
                        {"op": "=", "left": {"mt": "50K"}, "right": ""},
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "50F"},
                                    "pattern": "^(?:[^\\r\\n]*\\r?\\n){3}([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "substr": {
                            "value": {
                                "trim": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "50F"},
                                            "pattern": "(?:^|\\r?\\n)3\\/([^\\r\\n]*)",
                                            "group": 1,
                                            "flags": 8,
                                        },
                                    }
                                }
                            },
                            "start": 0,
                            "len": 70,
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_DbtrPstlAdrAdrLine",
                            "scope": "global",
                            "value": {"literal": "1"},
                        }
                    }
                ],
            },
            # Reglas 16
            {
                "when": {
                    "all": [
                        {
                            "op": "=",
                            "left": {"var": "has_DbtrCtry"},
                            "right": "",
                        },
                        {
                            "op": "=",
                            "left": {"var": "has_DbtrPstlAdrAdrLineMatch"},
                            "right": "",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "20"},
                                    "pattern": "^(6\\d{9})",
                                    "group": 1,
                                },
                            },
                            "right": "",
                        },
                        {"op": "=", "left": {"mt": "50K"}, "right": ""},
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "50F"},
                                    "pattern": "^(?:[^\\r\\n]*\\r?\\n){4}([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "substr": {
                            "value": {
                                "trim": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "50F"},
                                            "pattern": "(?:^|\\r?\\n)4\\/([^\\r\\n]*)",
                                            "group": 1,
                                            "flags": 8,
                                        },
                                    }
                                }
                            },
                            "start": 0,
                            "len": 70,
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_DbtrPstlAdrAdrLine",
                            "scope": "global",
                            "value": {"literal": "1"},
                        }
                    }
                ],
            },
        ],
    },
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.Cdtr.PstlAdr.Ctry": {
        "mode": "set",
        "rules": [
            {
                "set": [
                    {
                        "set_var": {
                            "name": "pacs.008::Cdtr.PstlAdr.Ctry",
                            "scope": "global",
                            "value": {
                                "from_db_query": {
                                    "alias": "pacs.008::Cdtr.PstlAdr.Ctry",
                                    "lookup_message_type": "pacs.008",
                                    "which_uetr_to_lookup": {"mtField": "121"},
                                    "where_to_lookup_uetr": {
                                        "message_type": "pacs.008",
                                        "path_kind": "xml",
                                        "field_path": ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.PmtId.UETR",
                                    },
                                    "field_to_extract": {
                                        "message_type": "pacs.008",
                                        "path_kind": "xml",
                                        "field_path": ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.Cdtr.PstlAdr.Ctry",
                                    },
                                    "fallback_literal": "",
                                }
                            },
                        }
                    },
                    {
                        "set_var": {
                            "name": "has_CdtrCtry",
                            "scope": "global",
                            "value": {"literal": "1"},
                        }
                    },
                ],
                "when": {
                    "all": [
                        {
                            "op": "in",
                            "left": {
                                "substr": {"value": {"mt": "20"}, "start": 0, "len": 3}
                            },
                            "right": ["662", "663", "651"],
                        },
                        {
                            "op": "!=",
                            "left": {"var": "pacs.008::Cdtr.PstlAdr.Ctry"},
                            "right": "",
                        },
                    ],
                },
                "then": {"value": {"var": "pacs.008::Cdtr.PstlAdr.Ctry"}},
            },
            {
                "when": {
                    "all": [
                        {
                            "op": "=",
                            "left": {"var": "has_CdtrPstlAdrAdrLine"},
                            "right": "",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {
                                        "substr": {
                                            "value": {"mt": "20"},
                                            "start": 0,
                                            "len": 3,
                                        }
                                    },
                                    "pattern": "^(7[0-9]{2})$",
                                    "group": 1,
                                },
                            },
                            "right": "",
                        },
                        {"op": "!=", "left": {"mt": "59"}, "right": ""},
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "59"},
                                    "pattern": "^(?:[^\\r\\n]*\\r?\\n){4}([A-Z]{2})",
                                    "group": 1,
                                    "flags": 8,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "59"},
                                    "pattern": "^(?:[^\\r\\n]*\\r?\\n){4}([A-Z]{2})",
                                    "group": 1,
                                    "flags": 8,
                                },
                            }
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_CdtrCtry",
                            "scope": "global",
                            "value": {"literal": "1"},
                        }
                    },
                ],
            },
            {
                "when": {
                    "all": [
                        {
                            "op": "=",
                            "left": {"var": "has_CdtrPstlAdrAdrLine"},
                            "right": "",
                        },
                        {"op": "!=", "left": {"mt": "59"}, "right": ""},
                        {
                            "op": "in",
                            "left": {
                                "substr": {"value": {"mt": "20"}, "start": 0, "len": 3}
                            },
                            "right": [
                                "500",
                                "506",
                                "510",
                                "513",
                                "520",
                                "523",
                                "526",
                                "527",
                                "540",
                                "543",
                                "544",
                                "545",
                                "546",
                                "547",
                                "501",
                                "502",
                                "511",
                                "512",
                                "521",
                                "522",
                                "524",
                                "525",
                                "541",
                            ],
                        },
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "59"},
                                    "pattern": "^(?:[^\\r\\n]*\\r?\\n){4}([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "59"},
                                    "pattern": "^(?:[^\\r\\n]*\\r?\\n){4}([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            }
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_CdtrCtry",
                            "scope": "global",
                            "value": {"literal": "1"},
                        }
                    },
                ],
            },
        ],
    },
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.Cdtr.PstlAdr.TwnNm": {
        "mode": "set",
        "rules": [
            {
                "set": [
                    {
                        "set_var": {
                            "name": "pacs.008::Cdtr.PstlAdr.TwnNm",
                            "scope": "global",
                            "value": {
                                "from_db_query": {
                                    "alias": "pacs.008::Cdtr.PstlAdr.TwnNm",
                                    "lookup_message_type": "pacs.008",
                                    "which_uetr_to_lookup": {"mtField": "121"},
                                    "where_to_lookup_uetr": {
                                        "message_type": "pacs.008",
                                        "path_kind": "xml",
                                        "field_path": ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.PmtId.UETR",
                                    },
                                    "field_to_extract": {
                                        "message_type": "pacs.008",
                                        "path_kind": "xml",
                                        "field_path": ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.Cdtr.PstlAdr.TwnNm",
                                    },
                                    "fallback_literal": "",
                                }
                            },
                        }
                    },
                    {
                        "set_var": {
                            "name": "has_CdtrTwnNm",
                            "scope": "global",
                            "value": {"literal": "1"},
                        }
                    },
                ],
                "when": {
                    "all": [
                        {
                            "op": "in",
                            "left": {
                                "substr": {"value": {"mt": "20"}, "start": 0, "len": 3}
                            },
                            "right": ["662", "663", "651"],
                        },
                        {
                            "op": "!=",
                            "left": {"var": "pacs.008::Cdtr.PstlAdr.TwnNm"},
                            "right": "",
                        },
                    ],
                },
                "then": {"value": {"var": "pacs.008::Cdtr.PstlAdr.TwnNm"}},
            },
            {
                "when": {
                    "all": [
                        {
                            "op": "!=",
                            "left": {"var": "has_CdtrCtry"},
                            "right": "",
                        },
                        {
                            "op": "=",
                            "left": {"var": "has_CdtrPstlAdrAdrLine"},
                            "right": "",
                        },
                        {"op": "!=", "left": {"mt": "59"}, "right": ""},
                        {
                            "op": "in",
                            "left": {
                                "substr": {
                                    "value": {"mt": "20"},
                                    "start": 0,
                                    "len": 3,
                                }
                            },
                            "right": [
                                "500",
                                "506",
                                "510",
                                "513",
                                "520",
                                "523",
                                "526",
                                "527",
                                "540",
                                "543",
                                "544",
                                "545",
                                "546",
                                "547",
                                "501",
                                "502",
                                "511",
                                "512",
                                "521",
                                "522",
                                "524",
                                "525",
                                "541",
                            ],
                        },
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "59"},
                                    "pattern": "^(?:[^\r\n]*\r?\n){3}([^\r\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "59"},
                                    "pattern": "^(?:[^\r\n]*\r?\n){3}([^\r\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            }
                        }
                    }
                },
            },
            {
                "when": {
                    "all": [
                        {
                            "op": "!=",
                            "left": {"var": "has_CdtrCtry"},
                            "right": "",
                        },
                        {
                            "op": "=",
                            "left": {"var": "has_CdtrPstlAdrAdrLine"},
                            "right": "",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {
                                        "substr": {
                                            "value": {"mt": "20"},
                                            "start": 0,
                                            "len": 3,
                                        }
                                    },
                                    "pattern": "^(7[0-9]{2})$",
                                    "group": 1,
                                },
                            },
                            "right": "",
                        },
                        {"op": "!=", "left": {"mt": "59"}, "right": ""},
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "59"},
                                    "pattern": "^(?:[^\r\n]*\r?\n){3}([^\r\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "59"},
                                    "pattern": "^(?:[^\r\n]*\r?\n){3}([^\r\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            }
                        }
                    }
                },
            },
            # --------------------Fin de reglas BREP -------------------------#
        ],
    },
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.Cdtr.PstlAdr.StrtNm": {
        "mode": "set",
        "rules": [
            {
                "set": [
                    {
                        "set_var": {
                            "name": "pacs.008::Cdtr.PstlAdr.StrtNm",
                            "scope": "global",
                            "value": {
                                "from_db_query": {
                                    "alias": "pacs.008::Cdtr.PstlAdr.StrtNm",
                                    "lookup_message_type": "pacs.008",
                                    "which_uetr_to_lookup": {"mtField": "121"},
                                    "where_to_lookup_uetr": {
                                        "message_type": "pacs.008",
                                        "path_kind": "xml",
                                        "field_path": ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.PmtId.UETR",
                                    },
                                    "field_to_extract": {
                                        "message_type": "pacs.008",
                                        "path_kind": "xml",
                                        "field_path": ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.Cdtr.PstlAdr.StrtNm",
                                    },
                                    "fallback_literal": "",
                                }
                            },
                        }
                    },
                    {
                        "set_var": {
                            "name": "has_CdtrStrtNm",
                            "scope": "global",
                            "value": {"literal": "1"},
                        }
                    },
                ],
                "when": {
                    "all": [
                        {
                            "op": "in",
                            "left": {
                                "substr": {"value": {"mt": "20"}, "start": 0, "len": 3}
                            },
                            "right": ["662", "663", "651"],
                        },
                        {
                            "op": "!=",
                            "left": {"var": "pacs.008::Cdtr.PstlAdr.StrtNm"},
                            "right": "",
                        },
                    ]
                },
                "then": {"value": {"var": "pacs.008::Cdtr.PstlAdr.StrtNm"}},
            },
            {
                "when": {
                    "all": [
                        {
                            "op": "!=",
                            "left": {"var": "has_CdtrCtry"},
                            "right": "",
                        },
                        {
                            "op": "=",
                            "left": {"var": "has_CdtrPstlAdrAdrLine"},
                            "right": "",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {
                                        "substr": {
                                            "value": {"mt": "20"},
                                            "start": 0,
                                            "len": 3,
                                        }
                                    },
                                    "pattern": "^(7[0-9]{2})$",
                                    "group": 1,
                                },
                            },
                            "right": "",
                        },
                        {
                            "op": "=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "32A"},
                                    "pattern": "^.{6}([A-Z]{3})",
                                    "group": 1,
                                },
                            },
                            "right": "CAD",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "59"},
                                    "pattern": "^(?:[^\\r\\n]*\\r?\\n){2}([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "59"},
                                            "pattern": "^(?:[^\\r\\n]*\\r?\\n){2}([^\\r\\n]*)",
                                            "group": 1,
                                            "flags": 8,
                                        },
                                    },
                                    "start": 7,
                                    "len": 70,
                                }
                            }
                        }
                    }
                },
            },
            {
                "when": {
                    "all": [
                        {
                            "op": "!=",
                            "left": {"var": "has_CdtrCtry"},
                            "right": "",
                        },
                        {
                            "op": "=",
                            "left": {"var": "has_CdtrPstlAdrAdrLine"},
                            "right": "",
                        },
                        {
                            "op": "in",
                            "left": {
                                "substr": {"value": {"mt": "20"}, "start": 0, "len": 3}
                            },
                            "right": [
                                "500",
                                "506",
                                "510",
                                "513",
                                "520",
                                "523",
                                "526",
                                "527",
                                "540",
                                "543",
                                "544",
                                "545",
                                "546",
                                "547",
                                "501",
                                "502",
                                "511",
                                "512",
                                "521",
                                "522",
                                "524",
                                "525",
                                "541",
                                "542",
                            ],
                        },
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "59"},
                                    "pattern": "^(?:[^\\r\\n]*\\r?\\n){2}([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "59"},
                                    "pattern": "^(?:[^\\r\\n]*\\r?\\n){2}([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            },
                        },
                        "start": 0,
                        "len": 70,
                    },
                },
            },
            {
                "when": {
                    "all": [
                        {
                            "op": "!=",
                            "left": {"var": "has_CdtrCtry"},
                            "right": "",
                        },
                        {
                            "op": "=",
                            "left": {"var": "has_CdtrPstlAdrAdrLine"},
                            "right": "",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {
                                        "substr": {
                                            "value": {"mt": "20"},
                                            "start": 0,
                                            "len": 3,
                                        }
                                    },
                                    "pattern": "^(7(0[1-9]|[1-9][0-9]))",
                                    "group": 1,
                                },
                            },
                            "right": "",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "59"},
                                    "pattern": "^(?:[^\\r\\n]*\\r?\\n){2}([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "59"},
                                    "pattern": "^(?:[^\\r\\n]*\\r?\\n){2}([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            }
                        },
                        "start": 0,
                        "len": 70,
                    },
                },
            },
        ],
    },
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.Cdtr.PstlAdr.AdrLine": {
        "mode": "append",
        "rules": [
            {
                "set": [
                    {
                        "set_var": {
                            "name": "pacs.008::Cdtr.PstlAdr.AdrLine",
                            "scope": "global",
                            "value": {
                                "from_db_query": {
                                    "alias": "pacs.008::Cdtr.PstlAdr.AdrLine",
                                    "lookup_message_type": "pacs.008",
                                    "which_uetr_to_lookup": {"mtField": "121"},
                                    "where_to_lookup_uetr": {
                                        "message_type": "pacs.008",
                                        "path_kind": "xml",
                                        "field_path": ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.PmtId.UETR",
                                    },
                                    "field_to_extract": {
                                        "message_type": "pacs.008",
                                        "path_kind": "xml",
                                        "field_path": ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.Cdtr.PstlAdr.AdrLine",
                                    },
                                    "fallback_literal": "",
                                }
                            },
                        }
                    },
                    {
                        "set_var": {
                            "name": "has_CdtrPstlAdrAdrLinePrevio",
                            "scope": "global",
                            "value": {"literal": "1"},
                        }
                    },
                    {
                        "set_var": {
                            "name": "has_CdtrPstlAdrAdrLineMatch",
                            "scope": "global",
                            "value": {"literal": "1"},
                        }
                    },
                ],
                "when": {
                    "all": [
                        {
                            "op": "in",
                            "left": {
                                "substr": {"value": {"mt": "20"}, "start": 0, "len": 3}
                            },
                            "right": ["662", "663", "651"],
                        },
                        {
                            "op": "!=",
                            "left": {"var": "pacs.008::Cdtr.PstlAdr.AdrLine"},
                            "right": "",
                        },
                    ]
                },
                "then": {"value": {"var": "pacs.008::Cdtr.PstlAdr.AdrLine"}},
            },
            {
                "when": {
                    "all": [
                        {
                            "op": "=",
                            "left": {"var": "has_CdtrCtry"},
                            "right": "",
                        },
                        {
                            "op": "=",
                            "left": {"var": "has_CdtrPstlAdrAdrLineMatch"},
                            "right": "",
                        },
                        {
                            "op": "=",
                            "left": {"var": "has_CdtrPstlAdrAdrLinePrevio"},
                            "right": "",
                        },
                        {
                            "op": "in",
                            "left": {
                                "substr": {"value": {"mt": "20"}, "start": 0, "len": 3}
                            },
                            "right": ["662", "663", "651"],
                        },
                        {
                            "op": "!=",
                            "left": {"var": "pacs.008::Cdtr.PstlAdr.AdrLine"},
                            "right": "",
                        },
                    ]
                },
                "then": {"value": {"var": "pacs.008::Cdtr.PstlAdr.AdrLine"}},
            },
            # --Reglas 2
            {
                "when": {
                    "all": [
                        {
                            "op": "=",
                            "left": {"var": "has_CdtrCtry"},
                            "right": "",
                        },
                        {
                            "op": "=",
                            "left": {"var": "has_CdtrPstlAdrAdrLineMatch"},
                            "right": "",
                        },
                        {
                            "op": "=",
                            "left": {"var": "has_CdtrPstlAdrAdrLinePrevio"},
                            "right": "",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "20"},
                                    "pattern": "^(7\\d{9})",
                                    "group": 1,
                                },
                            },
                            "right": "",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "59"},
                                    "pattern": "^(?:[^\\r\\n]*\\r?\\n){2}([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            },
                            "right": "",
                        },
                    ],
                    "any": [
                        {
                            "all": [
                                {
                                    "op": "!=",
                                    "left": {
                                        "substr": {
                                            "value": {"mt": "32A"},
                                            "start": 6,
                                            "len": 3,
                                        }
                                    },
                                    "right": "CAD",
                                }
                            ]
                        },
                        {
                            "all": [
                                {
                                    "op": "=",
                                    "left": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "59"},
                                            "pattern": "^(?:[^\\r\\n]*\\r?\\n){3}([^\\r\\n]*)",
                                            "group": 1,
                                            "flags": 8,
                                        },
                                    },
                                    "right": "",
                                }
                            ]
                        },
                        {
                            "all": [
                                {
                                    "op": "=",
                                    "left": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "59"},
                                            "pattern": "^(?:[^\\r\\n]*\\r?\\n){4}([^\\r\\n]*)",
                                            "group": 1,
                                            "flags": 8,
                                        },
                                    },
                                    "right": "",
                                }
                            ]
                        },
                    ],
                },
                "then": {
                    "value": {
                        "substr": {
                            "value": {
                                "trim": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "59"},
                                            "pattern": "^(?:[^\\r\\n]*\\r?\\n){2}([^\\r\\n]*)",
                                            "group": 1,
                                            "flags": 8,
                                        },
                                    }
                                }
                            },
                            "start": 0,
                            "len": 70,
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_CdtrPstlAdrAdrLine",
                            "scope": "global",
                            "value": {"literal": "1"},
                        }
                    }
                ],
            },
            # --Reglas 3
            {
                "when": {
                    "all": [
                        {
                            "op": "=",
                            "left": {"var": "has_CdtrCtry"},
                            "right": "",
                        },
                        {
                            "op": "=",
                            "left": {"var": "has_CdtrPstlAdrAdrLineMatch"},
                            "right": "",
                        },
                        {
                            "op": "=",
                            "left": {"var": "has_CdtrPstlAdrAdrLinePrevio"},
                            "right": "",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "20"},
                                    "pattern": "^(7\\d{9})",
                                    "group": 1,
                                },
                            },
                            "right": "",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "59"},
                                    "pattern": "^(?:[^\\r\\n]*\\r?\\n){3}([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            },
                            "right": "",
                        },
                    ],
                    "any": [
                        {
                            "all": [
                                {
                                    "op": "!=",
                                    "left": {
                                        "substr": {
                                            "value": {"mt": "32A"},
                                            "start": 6,
                                            "len": 3,
                                        }
                                    },
                                    "right": "CAD",
                                }
                            ]
                        },
                        {
                            "all": [
                                {
                                    "op": "=",
                                    "left": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "59"},
                                            "pattern": "^(?:[^\\r\\n]*\\r?\\n){3}([^\\r\\n]*)",
                                            "group": 1,
                                            "flags": 8,
                                        },
                                    },
                                    "right": "",
                                }
                            ]
                        },
                        {
                            "all": [
                                {
                                    "op": "=",
                                    "left": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "59"},
                                            "pattern": "^(?:[^\\r\\n]*\\r?\\n){4}([^\\r\\n]*)",
                                            "group": 1,
                                            "flags": 8,
                                        },
                                    },
                                    "right": "",
                                }
                            ]
                        },
                    ],
                },
                "then": {
                    "value": {
                        "substr": {
                            "value": {
                                "trim": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "59"},
                                            "pattern": "^(?:[^\\r\\n]*\\r?\\n){3}([^\\r\\n]*)",
                                            "group": 1,
                                            "flags": 8,
                                        },
                                    }
                                }
                            },
                            "start": 0,
                            "len": 70,
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_CdtrPstlAdrAdrLine",
                            "scope": "global",
                            "value": {"literal": "1"},
                        }
                    }
                ],
            },
            # --Reglas 4
            {
                "when": {
                    "all": [
                        {
                            "op": "=",
                            "left": {"var": "has_CdtrCtry"},
                            "right": "",
                        },
                        {
                            "op": "=",
                            "left": {"var": "has_CdtrPstlAdrAdrLineMatch"},
                            "right": "",
                        },
                        {
                            "op": "=",
                            "left": {"var": "has_CdtrPstlAdrAdrLinePrevio"},
                            "right": "",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "20"},
                                    "pattern": "^(7\\d{9})",
                                    "group": 1,
                                },
                            },
                            "right": "",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "59"},
                                    "pattern": "^(?:[^\\r\\n]*\\r?\\n){4}([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            },
                            "right": "",
                        },
                    ],
                    "any": [
                        {
                            "all": [
                                {
                                    "op": "!=",
                                    "left": {
                                        "substr": {
                                            "value": {"mt": "32A"},
                                            "start": 6,
                                            "len": 3,
                                        }
                                    },
                                    "right": "CAD",
                                }
                            ]
                        },
                        {
                            "all": [
                                {
                                    "op": "=",
                                    "left": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "59"},
                                            "pattern": "^(?:[^\\r\\n]*\\r?\\n){3}([^\\r\\n]*)",
                                            "group": 1,
                                            "flags": 8,
                                        },
                                    },
                                    "right": "",
                                }
                            ]
                        },
                        {
                            "all": [
                                {
                                    "op": "=",
                                    "left": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "59"},
                                            "pattern": "^(?:[^\\r\\n]*\\r?\\n){4}([^\\r\\n]*)",
                                            "group": 1,
                                            "flags": 8,
                                        },
                                    },
                                    "right": "",
                                }
                            ]
                        },
                    ],
                },
                "then": {
                    "value": {
                        "substr": {
                            "value": {
                                "trim": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "59"},
                                            "pattern": "^(?:[^\\r\\n]*\\r?\\n){4}([^\\r\\n]*)",
                                            "group": 1,
                                            "flags": 8,
                                        },
                                    }
                                }
                            },
                            "start": 0,
                            "len": 70,
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_CdtrPstlAdrAdrLine",
                            "scope": "global",
                            "value": {"literal": "1"},
                        }
                    }
                ],
            },
            # -- Reglas 5
            {
                "when": {
                    "all": [
                        {
                            "op": "=",
                            "left": {"var": "has_CdtrCtry"},
                            "right": "",
                        },
                        {
                            "op": "=",
                            "left": {"var": "has_CdtrPstlAdrAdrLineMatch"},
                            "right": "",
                        },
                        {
                            "op": "=",
                            "left": {"var": "has_CdtrPstlAdrAdrLinePrevio"},
                            "right": "",
                        },
                        {
                            "op": "in",
                            "left": {
                                "substr": {"value": {"mt": "20"}, "start": 0, "len": 3}
                            },
                            "right": [
                                "500",
                                "506",
                                "510",
                                "513",
                                "520",
                                "523",
                                "526",
                                "527",
                                "540",
                                "543",
                                "544",
                                "545",
                                "546",
                                "547",
                                "501",
                                "502",
                                "511",
                                "512",
                                "521",
                                "522",
                                "524",
                                "525",
                                "541",
                                "571",
                            ],
                        },
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "59"},
                                    "pattern": "^(?:[^\\r\\n]*\\r?\\n){2}([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            },
                            "right": "",
                        },
                    ],
                    "any": [
                        {
                            "all": [
                                {
                                    "op": "=",
                                    "left": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "59"},
                                            "pattern": "^(?:[^\\r\\n]*\\r?\\n){3}([^\\r\\n]*)",
                                            "group": 1,
                                            "flags": 8,
                                        },
                                    },
                                    "right": "",
                                }
                            ]
                        },
                        {
                            "all": [
                                {
                                    "op": "=",
                                    "left": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "59"},
                                            "pattern": "^(?:[^\\r\\n]*\\r?\\n){4}([^\\r\\n]*)",
                                            "group": 1,
                                            "flags": 8,
                                        },
                                    },
                                    "right": "",
                                }
                            ]
                        },
                    ],
                },
                "then": {
                    "value": {
                        "substr": {
                            "value": {
                                "trim": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "59"},
                                            "pattern": "^(?:[^\\r\\n]*\\r?\\n){2}([^\\r\\n]*)",
                                            "group": 1,
                                            "flags": 8,
                                        },
                                    }
                                }
                            },
                            "start": 0,
                            "len": 70,
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_CdtrPstlAdrAdrLine",
                            "scope": "global",
                            "value": {"literal": "1"},
                        }
                    }
                ],
            },
            # -- Reglas 6
            {
                "when": {
                    "all": [
                        {
                            "op": "=",
                            "left": {"var": "has_CdtrCtry"},
                            "right": "",
                        },
                        {
                            "op": "=",
                            "left": {"var": "has_CdtrPstlAdrAdrLineMatch"},
                            "right": "",
                        },
                        {
                            "op": "=",
                            "left": {"var": "has_CdtrPstlAdrAdrLinePrevio"},
                            "right": "",
                        },
                        {
                            "op": "in",
                            "left": {
                                "substr": {"value": {"mt": "20"}, "start": 0, "len": 3}
                            },
                            "right": [
                                "500",
                                "506",
                                "510",
                                "513",
                                "520",
                                "523",
                                "526",
                                "527",
                                "540",
                                "543",
                                "544",
                                "545",
                                "546",
                                "547",
                                "501",
                                "502",
                                "511",
                                "512",
                                "521",
                                "522",
                                "524",
                                "525",
                                "541",
                                "571",
                            ],
                        },
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "59"},
                                    "pattern": "^(?:[^\\r\\n]*\\r?\\n){3}([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            },
                            "right": "",
                        },
                    ],
                    "any": [
                        {
                            "all": [
                                {
                                    "op": "=",
                                    "left": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "59"},
                                            "pattern": "^(?:[^\\r\\n]*\\r?\\n){3}([^\\r\\n]*)",
                                            "group": 1,
                                            "flags": 8,
                                        },
                                    },
                                    "right": "",
                                }
                            ]
                        },
                        {
                            "all": [
                                {
                                    "op": "=",
                                    "left": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "59"},
                                            "pattern": "^(?:[^\\r\\n]*\\r?\\n){4}([^\\r\\n]*)",
                                            "group": 1,
                                            "flags": 8,
                                        },
                                    },
                                    "right": "",
                                }
                            ]
                        },
                    ],
                },
                "then": {
                    "value": {
                        "substr": {
                            "value": {
                                "trim": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "59"},
                                            "pattern": "^(?:[^\\r\\n]*\\r?\\n){3}([^\\r\\n]*)",
                                            "group": 1,
                                            "flags": 8,
                                        },
                                    }
                                }
                            },
                            "start": 0,
                            "len": 70,
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_CdtrPstlAdrAdrLine",
                            "scope": "global",
                            "value": {"literal": "1"},
                        }
                    }
                ],
            },
            # -- Reglas 7
            {
                "when": {
                    "all": [
                        {
                            "op": "=",
                            "left": {"var": "has_CdtrCtry"},
                            "right": "",
                        },
                        {
                            "op": "=",
                            "left": {"var": "has_CdtrPstlAdrAdrLineMatch"},
                            "right": "",
                        },
                        {
                            "op": "=",
                            "left": {"var": "has_CdtrPstlAdrAdrLinePrevio"},
                            "right": "",
                        },
                        {
                            "op": "in",
                            "left": {
                                "substr": {"value": {"mt": "20"}, "start": 0, "len": 3}
                            },
                            "right": [
                                "500",
                                "506",
                                "510",
                                "513",
                                "520",
                                "523",
                                "526",
                                "527",
                                "540",
                                "543",
                                "544",
                                "545",
                                "546",
                                "547",
                                "501",
                                "502",
                                "511",
                                "512",
                                "521",
                                "522",
                                "524",
                                "525",
                                "541",
                                "571",
                            ],
                        },
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "59"},
                                    "pattern": "^(?:[^\\r\\n]*\\r?\\n){4}([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            },
                            "right": "",
                        },
                    ],
                    "any": [
                        {
                            "all": [
                                {
                                    "op": "=",
                                    "left": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "59"},
                                            "pattern": "^(?:[^\\r\\n]*\\r?\\n){3}([^\\r\\n]*)",
                                            "group": 1,
                                            "flags": 8,
                                        },
                                    },
                                    "right": "",
                                }
                            ]
                        },
                        {
                            "all": [
                                {
                                    "op": "=",
                                    "left": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "59"},
                                            "pattern": "^(?:[^\\r\\n]*\\r?\\n){4}([^\\r\\n]*)",
                                            "group": 1,
                                            "flags": 8,
                                        },
                                    },
                                    "right": "",
                                }
                            ]
                        },
                    ],
                },
                "then": {
                    "value": {
                        "substr": {
                            "value": {
                                "trim": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "59"},
                                            "pattern": "^(?:[^\\r\\n]*\\r?\\n){4}([^\\r\\n]*)",
                                            "group": 1,
                                            "flags": 8,
                                        },
                                    }
                                }
                            },
                            "start": 0,
                            "len": 70,
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_CdtrPstlAdrAdrLine",
                            "scope": "global",
                            "value": {"literal": "1"},
                        }
                    }
                ],
            },
            # -- Regla 8
            {
                "when": {
                    "all": [
                        {
                            "op": "=",
                            "left": {"var": "has_CdtrCtry"},
                            "right": "",
                        },
                        {
                            "op": "=",
                            "left": {"var": "has_CdtrPstlAdrAdrLineMatch"},
                            "right": "",
                        },
                        {
                            "op": "=",
                            "left": {"var": "has_CdtrPstlAdrAdrLinePrevio"},
                            "right": "",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "20"},
                                    "pattern": "^(6\\d{9})",
                                    "group": 1,
                                },
                            },
                            "right": "",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "59"},
                                    "pattern": "^(?:[^\\r\\n]*\\r?\\n){2}([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "substr": {
                            "value": {
                                "trim": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "59"},
                                            "pattern": "^(?:[^\\r\\n]*\\r?\\n){2}([^\\r\\n]*)",
                                            "group": 1,
                                            "flags": 8,
                                        },
                                    }
                                }
                            },
                            "start": 0,
                            "len": 70,
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_CdtrPstlAdrAdrLine",
                            "scope": "global",
                            "value": {"literal": "1"},
                        }
                    }
                ],
            },
            # -- Regla 9
            {
                "when": {
                    "all": [
                        {
                            "op": "=",
                            "left": {"var": "has_CdtrCtry"},
                            "right": "",
                        },
                        {
                            "op": "=",
                            "left": {"var": "has_CdtrPstlAdrAdrLineMatch"},
                            "right": "",
                        },
                        {
                            "op": "=",
                            "left": {"var": "has_CdtrPstlAdrAdrLinePrevio"},
                            "right": "",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "20"},
                                    "pattern": "^(6\\d{9})",
                                    "group": 1,
                                },
                            },
                            "right": "",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "59"},
                                    "pattern": "^(?:[^\\r\\n]*\\r?\\n){3}([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "substr": {
                            "value": {
                                "trim": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "59"},
                                            "pattern": "^(?:[^\\r\\n]*\\r?\\n){3}([^\\r\\n]*)",
                                            "group": 1,
                                            "flags": 8,
                                        },
                                    }
                                }
                            },
                            "start": 0,
                            "len": 70,
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_CdtrPstlAdrAdrLine",
                            "scope": "global",
                            "value": {"literal": "1"},
                        }
                    }
                ],
            },
            # -- Regla 10
            {
                "when": {
                    "all": [
                        {
                            "op": "=",
                            "left": {"var": "has_CdtrCtry"},
                            "right": "",
                        },
                        {
                            "op": "=",
                            "left": {"var": "has_CdtrPstlAdrAdrLineMatch"},
                            "right": "",
                        },
                        {
                            "op": "=",
                            "left": {"var": "has_CdtrPstlAdrAdrLinePrevio"},
                            "right": "",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "20"},
                                    "pattern": "^(6\\d{9})",
                                    "group": 1,
                                },
                            },
                            "right": "",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "59"},
                                    "pattern": "^(?:[^\\r\\n]*\\r?\\n){4}([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "substr": {
                            "value": {
                                "trim": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "59"},
                                            "pattern": "^(?:[^\\r\\n]*\\r?\\n){4}([^\\r\\n]*)",
                                            "group": 1,
                                            "flags": 8,
                                        },
                                    }
                                }
                            },
                            "start": 0,
                            "len": 70,
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_CdtrPstlAdrAdrLine",
                            "scope": "global",
                            "value": {"literal": "1"},
                        }
                    }
                ],
            },
            # -- Regla 11
            {
                "when": {
                    "all": [
                        {
                            "op": "=",
                            "left": {"var": "has_CdtrCtry"},
                            "right": "",
                        },
                        {
                            "op": "=",
                            "left": {"var": "has_CdtrPstlAdrAdrLineMatch"},
                            "right": "",
                        },
                        {
                            "op": "=",
                            "left": {"var": "has_CdtrPstlAdrAdrLinePrevio"},
                            "right": "",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "20"},
                                    "pattern": "^(6\\d{9})",
                                    "group": 1,
                                },
                            },
                            "right": "",
                        },
                        {"op": "=", "left": {"mt": "59"}, "right": ""},
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "59F"},
                                    "pattern": "^(?:[^\\r\\n]*\\r?\\n){2}\\d/([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "substr": {
                            "value": {
                                "trim": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "59F"},
                                            "pattern": "^(?:[^\\r\\n]*\\r?\\n){2}\\d/([^\\r\\n]*)",
                                            "group": 1,
                                            "flags": 8,
                                        },
                                    }
                                }
                            },
                            "start": 0,
                            "len": 70,
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_CdtrPstlAdrAdrLine",
                            "scope": "global",
                            "value": {"literal": "1"},
                        }
                    }
                ],
            },
            # -- Regla 12
            {
                "when": {
                    "all": [
                        {
                            "op": "=",
                            "left": {"var": "has_CdtrCtry"},
                            "right": "",
                        },
                        {
                            "op": "=",
                            "left": {"var": "has_CdtrPstlAdrAdrLineMatch"},
                            "right": "",
                        },
                        {
                            "op": "=",
                            "left": {"var": "has_CdtrPstlAdrAdrLinePrevio"},
                            "right": "",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "20"},
                                    "pattern": "^(6\\d{9})",
                                    "group": 1,
                                },
                            },
                            "right": "",
                        },
                        {"op": "=", "left": {"mt": "59"}, "right": ""},
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "59F"},
                                    "pattern": "^(?:[^\\r\\n]*\\r?\\n){3}([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "substr": {
                            "value": {
                                "trim": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "59F"},
                                            "pattern": "^(?:[^\\r\\n]*\\r?\\n){3}\\d/([^\\r\\n]*)",
                                            "group": 1,
                                            "flags": 8,
                                        },
                                    }
                                }
                            },
                            "start": 0,
                            "len": 70,
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_CdtrPstlAdrAdrLine",
                            "scope": "global",
                            "value": {"literal": "1"},
                        }
                    }
                ],
            },
            # -- Regla 13
            {
                "when": {
                    "all": [
                        {
                            "op": "=",
                            "left": {"var": "has_CdtrCtry"},
                            "right": "",
                        },
                        {
                            "op": "=",
                            "left": {"var": "has_CdtrPstlAdrAdrLineMatch"},
                            "right": "",
                        },
                        {
                            "op": "=",
                            "left": {"var": "has_CdtrPstlAdrAdrLinePrevio"},
                            "right": "",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "20"},
                                    "pattern": "^(6\\d{9})",
                                    "group": 1,
                                },
                            },
                            "right": "",
                        },
                        {"op": "=", "left": {"mt": "59"}, "right": ""},
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "59F"},
                                    "pattern": "^(?:[^\\r\\n]*\\r?\\n){4}([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "substr": {
                            "value": {
                                "trim": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "59F"},
                                            "pattern": "^(?:[^\\r\\n]*\\r?\\n){4}\\d/([^\\r\\n]*)",
                                            "group": 1,
                                            "flags": 8,
                                        },
                                    }
                                }
                            },
                            "start": 0,
                            "len": 70,
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_CdtrPstlAdrAdrLine",
                            "scope": "global",
                            "value": {"literal": "1"},
                        }
                    }
                ],
            },
        ],
    },
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.Cdtr.Id.OrgId.AnyBIC": {
        "mode": "set",
        "rules": [
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "59A"}, "right": ""},
                        {
                            "op": "!=",
                            "left": {
                                "trim": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "59A"},
                                            "pattern": "^(?:[^\r\n]*\r?\n){1}([^\r\n]*)",
                                            "group": 1,
                                        },
                                    }
                                }
                            },
                            # "right": {"param": "BICFI", "scope": "user"},
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "59A"},
                                    "pattern": "^(?:[^\r\n]*\r?\n){1}([^\r\n]*)",
                                    "group": 1,
                                },
                            }
                        }
                    }
                },
            }
        ],
    },
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.PmtId.EndToEndId": {
        "mode": "set",
        "rules": [
            {
                "set": [
                    {
                        "set_var": {
                            "name": "pacs.008::PmtId.EndToEndId",
                            "scope": "global",
                            "value": {
                                "from_db_query": {
                                    "alias": "pacs.008::PmtId.EndToEndId",
                                    "lookup_message_type": "pacs.008",
                                    "which_uetr_to_lookup": {"mtField": "121"},
                                    "where_to_lookup_uetr": {
                                        "message_type": "pacs.008",
                                        "path_kind": "xml",
                                        "field_path": ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.PmtId.UETR",
                                    },
                                    "field_to_extract": {
                                        "message_type": "pacs.008",
                                        "path_kind": "xml",
                                        "field_path": ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.PmtId.EndToEndId",
                                    },
                                    "fallback_literal": "",
                                }
                            },
                        }
                    }
                ],
                "when": {
                    "all": [
                        {
                            "op": "!=",
                            "left": {"var": "pacs.008::PmtId.EndToEndId"},
                            "right": "",
                        },
                        {
                            "op": "!=",
                            "left": {"var": "pacs.008::PmtId.EndToEndId"},
                            "right": "NOTPROVIDED",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {
                                        "substr": {
                                            "value": {"mt": "20"},
                                            "start": 0,
                                            "len": 3,
                                        }
                                    },
                                    "pattern": "^(6[0-9]{2})$",
                                    "group": 0,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {"value": {"var": "pacs.008::PmtId.EndToEndId"}},
            },
            {
                "set": [
                    {
                        "set_var": {
                            "name": "pacs.008::PmtId.EndToEndId",
                            "scope": "global",
                            "value": {
                                "from_db_query": {
                                    "alias": "pacs.008::PmtId.EndToEndId",
                                    "lookup_message_type": "pacs.008",
                                    "which_uetr_to_lookup": {"mtField": "121"},
                                    "where_to_lookup_uetr": {
                                        "message_type": "pacs.008",
                                        "path_kind": "xml",
                                        "field_path": ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.PmtId.UETR",
                                    },
                                    "field_to_extract": {
                                        "message_type": "pacs.008",
                                        "path_kind": "xml",
                                        "field_path": ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.PmtId.EndToEndId",
                                    },
                                    "fallback_literal": "",
                                }
                            },
                        }
                    }
                ],
                "when": {
                    "all": [
                        {
                            "op": "!=",
                            "left": {"var": "pacs.008::PmtId.EndToEndId"},
                            "right": "",
                        },
                        {
                            "op": "=",
                            "left": {"var": "pacs.008::PmtId.EndToEndId"},
                            "right": "NOTPROVIDED",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {
                                        "substr": {
                                            "value": {"mt": "20"},
                                            "start": 0,
                                            "len": 3,
                                        }
                                    },
                                    "pattern": "^(6[0-9]{2})$",
                                    "group": 0,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {"value": {"mt": "20"}},
            },
            {
                "when": {
                    "all": [{"op": "!=", "left": {"mt": "20"}, "right": ""}],
                    "any": [
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "20"},
                                    "pattern": "^(7\\d{9})",
                                    "group": 1,
                                },
                            },
                            "right": "",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "20"},
                                    "pattern": "^(5\\d{9})",
                                    "group": 1,
                                },
                            },
                            "right": "",
                        },
                    ],
                },
                "then": {"value": {"mt": "20"}},
            },
            {"when": {"all": []}, "then": {"value": {"literal": "NOTPROVIDED"}}},
        ],
    },
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.RmtInf.Ustrd": {
        "mode": "set",
        "rules": [
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "70"}, "right": ""},
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "70"},
                                    "pattern": "^([^\\r\\n]*)",
                                    "group": 1,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "fn": "regex_replace",
                        "args": {
                            "text": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "70"},
                                    "pattern": "^(?:[^\\r\\n]*\\r?\\n){0}([^\\r\\n]*)",
                                    "group": 1,
                                },
                            },
                            "pattern": "\\r?\\n+",
                            "repl": "",
                        },
                    }
                },
            }
        ],
    },
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.ChrgBr": {
        "mode": "set",
        "rules": [
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "71A"}, "right": ""},
                        {
                            "op": "=",
                            "left": {
                                "trim": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "71A"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    }
                                }
                            },
                            "right": "BEN",
                        },
                    ]
                },
                "then": {"value": {"literal": "CRED"}},
            },
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "71A"}, "right": ""},
                        {
                            "op": "=",
                            "left": {
                                "trim": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "71A"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    }
                                }
                            },
                            "right": "OUR",
                        },
                    ]
                },
                "then": {"value": {"literal": "DEBT"}},
            },
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "71A"}, "right": ""},
                        {
                            "op": "=",
                            "left": {
                                "trim": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "71A"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    }
                                }
                            },
                            "right": "SHA",
                        },
                    ]
                },
                "then": {"value": {"literal": "SHAR"}},
            },
        ],
    },
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.ChrgsInf.Amt.@Ccy": {
        "mode": "append",
        "rules": [
            {
                "when": {"all": [{"op": "!=", "left": {"mt": "71F"}, "right": ""}]},
                "then": {
                    "value": {"substr": {"value": {"mt": "71F"}, "start": 0, "len": 3}}
                },
                "set": [
                    {
                        "name": "has_CdtTrfTxInf.ChrgsInf.Amt.@Ccy",
                        "value": {"literal": "1"},
                        "scope": "global",
                    }
                ],
            }
        ],
    },
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.ChrgsInf.Amt": {
        "mode": "append",
        "rules": [
            {
                "when": {"all": [{"op": "!=", "left": {"mt": "71F"}, "right": ""}]},
                "then": {
                    "value": {
                        "fn": "amount_normalize",
                        "args": {
                            "value": {
                                "substr": {
                                    "value": {"mt": "71F"},
                                    "start": 3,
                                    "len": 15,
                                }
                            }
                        },
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_CdtTrfTxInf.ChrgsInf.Amt",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            }
        ],
    },
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.ChrgsInf.Agt.FinInstnId.Nm": {
        "mode": "append",
        "rules": [
            {
                "when": {"all": [{"op": "!=", "left": {"mt": "71F"}, "right": ""}]},
                "then": {
                    "value": {
                        "fn": "repeat_occurrences",
                        "args": {"mt_tag": "71F", "emit": {"literal": "NOTPROVIDED"}},
                    }
                },
            },
            {
                "when": {"all": [{"op": "!=", "left": {"mt": "71G"}, "right": ""}]},
                "then": {
                    "value": {
                        "fn": "repeat_occurrences",
                        "args": {"mt_tag": "71G", "emit": {"literal": "NOTPROVIDED"}},
                    }
                },
            },
        ],
    },
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.ChrgsInf.Agt.FinInstnId.PstlAdr.AdrLine": {
        "mode": "append",
        "rules": [
            {
                "when": {"all": [{"op": "!=", "left": {"mt": "71F"}, "right": ""}]},
                "then": {
                    "value": {
                        "fn": "repeat_occurrences",
                        "args": {"mt_tag": "71F", "emit": {"literal": "NOTPROVIDED"}},
                    }
                },
            },
            {
                "when": {"all": [{"op": "!=", "left": {"mt": "71G"}, "right": ""}]},
                "then": {
                    "value": {
                        "fn": "repeat_occurrences",
                        "args": {"mt_tag": "71G", "emit": {"literal": "NOTPROVIDED"}},
                    }
                },
            },
        ],
    },
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.PrvsInstgAgt1.FinInstnId.BICFI": {
        "mode": "set",
        "rules": [
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "72A"}, "right": ""},
                        {
                            "op": "=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "72A"},
                                            "pattern": "^[^\\r\\n]*\\r?\\n([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 3,
                                }
                            },
                            "right": "INS",
                        },
                        {
                            "op": "in",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "72A"},
                                            "pattern": "^[^\\r\\n]*\\r?\\n([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "pattern": "^[^:]*:[^:]*:([A-Za-z0-9]{8})",
                                    "group": 1,
                                },
                            },
                            "right": {"list": "BICFI"},
                        },
                    ]
                },
                "then": {
                    "value": {
                        "fn": "concat",
                        "args": [
                            {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "72A"},
                                            "pattern": "^[^\\r\\n]*\\r?\\n([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "pattern": "^[^:]*:[^:]*:([A-Za-z0-9]{8})",
                                    "group": 1,
                                },
                            },
                            "XXX",
                        ],
                    }
                },
            },
            {"when": {"all": []}, "then": {"value": "NOTPROVIDED"}},
        ],
    },
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.PrvsInstgAgt2.FinInstnId.BICFI": {
        "mode": "set",
        "rules": [
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "72A"}, "right": ""},
                        {
                            "op": "=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "72A"},
                                            "pattern": "^[^\\r\\n]*\\r?\\n([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 3,
                                }
                            },
                            "right": "INS",
                        },
                        {
                            "op": "in",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "72A"},
                                            "pattern": "^[^\\r\\n]*\\r?\\n([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "pattern": "^(?:[^:]*:){2}([A-Za-z0-9]{8})",
                                    "group": 1,
                                },
                            },
                            "right": {"list": "BICFI"},
                        },
                    ]
                },
                "then": {
                    "value": {
                        "fn": "concat",
                        "args": [
                            {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "72A"},
                                            "pattern": "^[^\\r\\n]*\\r?\\n([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "pattern": "^(?:[^:]*:){2}([A-Za-z0-9]{8})",
                                    "group": 1,
                                },
                            },
                            "XXX",
                        ],
                    }
                },
            },
            {"when": {"all": []}, "then": {"value": "NOTPROVIDED"}},
        ],
    },
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.PrvsInstgAgt3.FinInstnId.BICFI": {
        "mode": "set",
        "rules": [
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "72A"}, "right": ""},
                        {
                            "op": "=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "72A"},
                                            "pattern": "^[^\\r\\n]*\\r?\\n([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 3,
                                }
                            },
                            "right": "INS",
                        },
                        {
                            "op": "in",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "72A"},
                                            "pattern": "^[^\\r\\n]*\\r?\\n([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "pattern": "^(?:[^:]*:){3}([A-Za-z0-9]{8})",
                                    "group": 1,
                                },
                            },
                            "right": {"list": "BICFI"},
                        },
                    ]
                },
                "then": {
                    "value": {
                        "fn": "concat",
                        "args": [
                            {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "72A"},
                                            "pattern": "^[^\\r\\n]*\\r?\\n([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "pattern": "^(?:[^:]*:){3}([A-Za-z0-9]{8})",
                                    "group": 1,
                                },
                            },
                            "XXX",
                        ],
                    }
                },
            },
            {"when": {"all": []}, "then": {"value": "NOTPROVIDED"}},
        ],
    },
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.InstrForCdtrAgt.InstrInf": {
        "mode": "append",
        "rules": [
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "72"}, "right": ""},
                        {
                            "op": "=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "72"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 5,
                                }
                            },
                            "right": "/ACC/",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "substr": {
                            "value": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "72"},
                                    "pattern": "^([^\\r\\n]*)",
                                    "group": 1,
                                },
                            },
                            "start": 4,
                            "len": 35,
                        }
                    }
                },
            },
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "72"}, "right": ""},
                        {
                            "op": "=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "72"},
                                            "pattern": "^[^\\r\\n]*\\r?\\n([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 5,
                                }
                            },
                            "right": "/ACC/",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "substr": {
                            "value": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "72"},
                                    "pattern": "^[^\\r\\n]*\\r?\\n([^\\r\\n]*)",
                                    "group": 1,
                                },
                            },
                            "start": 4,
                            "len": 35,
                        }
                    }
                },
            },
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "72"}, "right": ""},
                        {
                            "op": "=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "72"},
                                            "pattern": "^(?:[^\\r\\n]*\\r?\\n){2}([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 5,
                                }
                            },
                            "right": "/ACC/",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "substr": {
                            "value": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "72"},
                                    "pattern": "^(?:[^\\r\\n]*\\r?\\n){2}([^\\r\\n]*)",
                                    "group": 1,
                                },
                            },
                            "start": 4,
                            "len": 35,
                        }
                    }
                },
            },
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "72"}, "right": ""},
                        {
                            "op": "=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "72"},
                                            "pattern": "^(?:[^\\r\\n]*\\r?\\n){3}([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 5,
                                }
                            },
                            "right": "/ACC/",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "substr": {
                            "value": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "72"},
                                    "pattern": "^(?:[^\\r\\n]*\\r?\\n){3}([^\\r\\n]*)",
                                    "group": 1,
                                },
                            },
                            "start": 4,
                            "len": 35,
                        }
                    }
                },
            },
        ],
    },
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.InstrForNxtAgt[0].InstrInf": {
        "mode": "set",
        "rules": [
            {
                "when": {
                    "all": [
                        {
                            "op": "in",
                            "left": {
                                "substr": {"value": {"mt": "20"}, "start": 0, "len": 3}
                            },
                            "right": "662, 663, 651",
                        },
                    ]
                },
                "then": {"value": {"literal": ""}},
            },
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "72"}, "right": ""},
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "72"},
                                            "pattern": "^([^\r\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 5,
                                }
                            },
                            "right": "/ACC/",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "72"},
                                            "pattern": "^([^\r\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 5,
                                }
                            },
                            "right": "/INS/",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "72"},
                                            "pattern": "^([^\r\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 6,
                                }
                            },
                            "right": "/INTA/",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "72"},
                                            "pattern": "^([^\r\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 8,
                                }
                            },
                            "right": "/SVCLVL/",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "72"},
                                            "pattern": "^([^\r\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 8,
                                }
                            },
                            "right": "/LOCINS/",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "72"},
                                            "pattern": "^([^\r\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 9,
                                }
                            },
                            "right": "/CATPURP/",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "substr": {
                            "value": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "72"},
                                    "pattern": "^([^\r\n]*)",
                                    "group": 1,
                                },
                            },
                            "start": 0,
                            "len": 35,
                        }
                    }
                },
            },
        ],
    },
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.InstrForNxtAgt[1].InstrInf": {
        "mode": "set",
        "rules": [
            {
                "when": {
                    "all": [
                        {
                            "op": "in",
                            "left": {
                                "substr": {"value": {"mt": "20"}, "start": 0, "len": 3}
                            },
                            "right": "662, 663, 651",
                        },
                    ]
                },
                "then": {"value": {"literal": ""}},
            },
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "72"}, "right": ""},
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "72"},
                                            "pattern": "^[^\r\n]*\r?\n([^\r\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 5,
                                }
                            },
                            "right": "/ACC/",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "72"},
                                            "pattern": "^[^\r\n]*\r?\n([^\r\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 5,
                                }
                            },
                            "right": "/INS/",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "72"},
                                            "pattern": "^[^\r\n]*\r?\n([^\r\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 6,
                                }
                            },
                            "right": "/INTA/",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "72"},
                                            "pattern": "^[^\r\n]*\r?\n([^\r\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 8,
                                }
                            },
                            "right": "/SVCLVL/",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "72"},
                                            "pattern": "^[^\r\n]*\r?\n([^\r\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 8,
                                }
                            },
                            "right": "/LOCINS/",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "72"},
                                            "pattern": "^[^\r\n]*\r?\n([^\r\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 9,
                                }
                            },
                            "right": "/CATPURP/",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "substr": {
                            "value": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "72"},
                                    "pattern": "^[^\r\n]*\r?\n(?:\/\/)?([^\r\n]*)",
                                    "group": 1,
                                },
                            },
                            "start": 0,
                            "len": 35,
                        }
                    }
                },
            },
        ],
    },
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.InstrForNxtAgt[2].InstrInf": {
        "mode": "set",
        "rules": [
            {
                "when": {
                    "all": [
                        {
                            "op": "in",
                            "left": {
                                "substr": {"value": {"mt": "20"}, "start": 0, "len": 3}
                            },
                            "right": "662, 663, 651",
                        },
                    ]
                },
                "then": {"value": {"literal": ""}},
            },
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "72"}, "right": ""},
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "72"},
                                            "pattern": "^(?:[^\r\n]*\r?\n){2}([^\r\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 5,
                                }
                            },
                            "right": "/ACC/",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "72"},
                                            "pattern": "^(?:[^\r\n]*\r?\n){2}([^\r\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 5,
                                }
                            },
                            "right": "/INS/",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "72"},
                                            "pattern": "^(?:[^\r\n]*\r?\n){2}([^\r\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 6,
                                }
                            },
                            "right": "/INTA/",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "72"},
                                            "pattern": "^(?:[^\r\n]*\r?\n){2}([^\r\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 8,
                                }
                            },
                            "right": "/SVCLVL/",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "72"},
                                            "pattern": "^(?:[^\r\n]*\r?\n){2}([^\r\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 8,
                                }
                            },
                            "right": "/LOCINS/",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "72"},
                                            "pattern": "^(?:[^\r\n]*\r?\n){2}([^\r\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 9,
                                }
                            },
                            "right": "/CATPURP/",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "substr": {
                            "value": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "72"},
                                    "pattern": "^(?:[^\r\n]*\r?\n){2}\s*(?:\/\/)?\s*([^\r\n]*)",
                                    "group": 1,
                                },
                            },
                            "start": 0,
                            "len": 35,
                        }
                    }
                },
            },
        ],
    },
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.InstrForNxtAgt[3].InstrInf": {
        "mode": "set",
        "rules": [
            {
                "when": {
                    "all": [
                        {
                            "op": "in",
                            "left": {
                                "substr": {"value": {"mt": "20"}, "start": 0, "len": 3}
                            },
                            "right": "662, 663, 651",
                        },
                    ]
                },
                "then": {"value": {"literal": ""}},
            },
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "72"}, "right": ""},
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "72"},
                                            "pattern": "^(?:[^\r\n]*\r?\n){3}([^\r\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 5,
                                }
                            },
                            "right": "/ACC/",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "72"},
                                            "pattern": "^(?:[^\r\n]*\r?\n){3}([^\r\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 5,
                                }
                            },
                            "right": "/INS/",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "72"},
                                            "pattern": "^(?:[^\r\n]*\r?\n){3}([^\r\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 6,
                                }
                            },
                            "right": "/INTA/",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "72"},
                                            "pattern": "^(?:[^\r\n]*\r?\n){3}([^\r\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 8,
                                }
                            },
                            "right": "/SVCLVL/",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "72"},
                                            "pattern": "^(?:[^\r\n]*\r?\n){3}([^\r\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 8,
                                }
                            },
                            "right": "/LOCINS/",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "72"},
                                            "pattern": "^(?:[^\r\n]*\r?\n){3}([^\r\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 9,
                                }
                            },
                            "right": "/CATPURP/",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "substr": {
                            "value": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "72"},
                                    "pattern": "^(?:[^\r\n]*\r?\n){3}([^\r\n]*)",
                                    "group": 1,
                                },
                            },
                            "start": 0,
                            "len": 35,
                        }
                    }
                },
            },
        ],
    },
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.InstrForNxtAgt[4].InstrInf": {
        "mode": "set",
        "rules": [
            {
                "when": {
                    "all": [
                        {
                            "op": "in",
                            "left": {
                                "substr": {"value": {"mt": "20"}, "start": 0, "len": 3}
                            },
                            "right": "662, 663, 651",
                        },
                    ]
                },
                "then": {"value": {"literal": ""}},
            },
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "72"}, "right": ""},
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "72"},
                                            "pattern": "^(?:[^\r\n]*\r?\n){4}([^\r\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 5,
                                }
                            },
                            "right": "/ACC/",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "72"},
                                            "pattern": "^(?:[^\r\n]*\r?\n){4}([^\r\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 5,
                                }
                            },
                            "right": "/INS/",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "72"},
                                            "pattern": "^(?:[^\r\n]*\r?\n){4}([^\r\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 6,
                                }
                            },
                            "right": "/INTA/",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "72"},
                                            "pattern": "^(?:[^\r\n]*\r?\n){4}([^\r\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 8,
                                }
                            },
                            "right": "/SVCLVL/",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "72"},
                                            "pattern": "^(?:[^\r\n]*\r?\n){4}([^\r\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 8,
                                }
                            },
                            "right": "/LOCINS/",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "72"},
                                            "pattern": "^(?:[^\r\n]*\r?\n){4}([^\r\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 9,
                                }
                            },
                            "right": "/CATPURP/",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "substr": {
                            "value": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "72"},
                                    "pattern": "^(?:[^\r\n]*\r?\n){4}\s*(?:\/\/)?\s*([^\r\n]*)",
                                    "group": 1,
                                },
                            },
                            "start": 0,
                            "len": 35,
                        }
                    }
                },
            },
        ],
    },
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.Cdtr.CtryOfRes": {
        "mode": "set",
        "rules": [
            {
                "when": {"all": [{"op": "!=", "left": {"mt": "77B"}, "right": ""}]},
                "then": {
                    "value": {
                        "substr": {
                            "value": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "77B"},
                                    "pattern": "^([^\\r\\n]*)",
                                    "group": 1,
                                },
                            },
                            "start": 4,
                            "len": 2,
                        }
                    }
                },
            }
        ],
    },
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.Dbtr.CtryOfRes": {
        "mode": "set",
        "rules": [
            {
                "when": {"all": [{"op": "!=", "left": {"mt": "77B"}, "right": ""}]},
                "then": {
                    "value": {
                        "substr": {
                            "value": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "77B"},
                                    "pattern": "^([^\\r\\n]*)",
                                    "group": 1,
                                },
                            },
                            "start": 4,
                            "len": 2,
                        }
                    }
                },
            }
        ],
    },
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.IntrmyAgt1.FinInstnId.BICFI": {
        "mode": "set",
        "rules": [
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "56A"}, "right": ""},
                        {
                            "op": "!=",
                            "left": {
                                "trim": {
                                    "value": {
                                        "substr": {
                                            "value": {"mt": "56A"},
                                            "start": 0,
                                            "len": 11,
                                        }
                                    }
                                }
                            },
                            # "right": {"param": "BICFI", "scope": "user"},
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "fn": "normalize_bic",
                        "args": {
                            "value": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "56A"},
                                    "pattern": "([A-Z0-9]{8}([A-Z0-9]{3})?)",
                                    "group": 1,
                                },
                            }
                        },
                    }
                },
            },
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "72"}, "right": ""},
                        {
                            "op": "=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "72"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 6,
                                }
                            },
                            "right": "/INTA/",
                        },
                        {
                            "op": "in",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "72"},
                                    "pattern": "^/INTA/([A-Za-z0-9]{8})",
                                    "group": 1,
                                },
                            },
                            "right": {"list": "BICFI"},
                        },
                    ]
                },
                "then": {
                    "value": {
                        "fn": "concat",
                        "args": [
                            {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "72"},
                                    "pattern": "^/INTA/([A-Za-z0-9]{8})",
                                    "group": 1,
                                },
                            },
                            "XXX",
                        ],
                    }
                },
            },
        ],
    },
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.IntrmyAgt1.FinInstnId.Nm": {
        "mode": "append",
        "rules": [
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "56D"}, "right": ""},
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "56D"},
                                    "pattern": "^([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            },
                            "right": "",
                        },
                    ],
                    "any": [
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "20"},
                                    "pattern": "^(7\\d{2})",
                                    "group": 1,
                                },
                            },
                            "right": "",
                        },
                        {
                            "op": "in",
                            "left": {
                                "substr": {"value": {"mt": "20"}, "start": 0, "len": 3}
                            },
                            "right": [
                                "500",
                                "506",
                                "510",
                                "513",
                                "520",
                                "523",
                                "526",
                                "527",
                                "540",
                                "543",
                                "544",
                                "545",
                                "546",
                                "547",
                                "501",
                                "502",
                                "511",
                                "512",
                                "521",
                                "522",
                                "524",
                                "525",
                                "541",
                                "542",
                            ],
                        },
                    ],
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "56D"},
                                    "pattern": "^([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            }
                        }
                    }
                },
            },
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "56C"}, "right": ""},
                        {
                            "op": "=",
                            "left": {
                                "trim": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "56C"},
                                            "pattern": "^(?:[^\\r\\n]*\\r?\\n){1}([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    }
                                }
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {"value": {"literal": "NOTPROVIDED"}},
            },
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "56C"}, "right": ""},
                        {
                            "op": "!=",
                            "left": {
                                "trim": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "56C"},
                                            "pattern": "^(?:[^\\r\\n]*\\r?\\n){1}([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    }
                                }
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "56C"},
                                    "pattern": "^(?:[^\\r\\n]*\\r?\\n){1}([^\\r\\n]*)",
                                    "group": 1,
                                },
                            }
                        }
                    }
                },
            },
        ],
    },
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.IntrmyAgt1.FinInstnId.PstlAdr.StrtNm": {
        "mode": "set",
        "rules": [
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "56D"}, "right": ""},
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "56D"},
                                    "pattern": "^(?:[^\\r\\n]*\\r?\\n){1}([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            },
                            "right": "",
                        },
                    ],
                    "any": [
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "20"},
                                    "pattern": "^7\\d{2}",
                                    "group": 0,
                                },
                            },
                            "right": "",
                        },
                        {
                            "op": "in",
                            "left": {
                                "substr": {"value": {"mt": "20"}, "start": 0, "len": 3}
                            },
                            "right": [
                                "500",
                                "506",
                                "510",
                                "513",
                                "520",
                                "523",
                                "526",
                                "527",
                                "540",
                                "543",
                                "544",
                                "545",
                                "546",
                                "547",
                                "501",
                                "502",
                                "511",
                                "512",
                                "521",
                                "522",
                                "524",
                                "525",
                                "541",
                                "544",
                            ],
                        },
                    ],
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "56D"},
                                    "pattern": "^(?:[^\\r\\n]*\\r?\\n){1}([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            }
                        }
                    }
                },
            }
        ],
    },
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.IntrmyAgt1.FinInstnId.PstlAdr.TwnNm": {
        "mode": "append",
        "rules": [
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "56D"}, "right": ""},
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "56D"},
                                    "pattern": "^(?:[^\\r\\n]*\\n){2}([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            },
                            "right": "",
                        },
                    ],
                    "any": [
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "20"},
                                    "pattern": "^7\\d{2}",
                                    "group": 0,
                                },
                            },
                            "right": "",
                        },
                        {
                            "op": "in",
                            "left": {
                                "substr": {"value": {"mt": "20"}, "start": 0, "len": 3}
                            },
                            "right": [
                                "500",
                                "506",
                                "510",
                                "513",
                                "520",
                                "523",
                                "526",
                                "527",
                                "540",
                                "543",
                                "544",
                                "545",
                                "546",
                                "547",
                                "501",
                                "502",
                                "511",
                                "512",
                                "521",
                                "522",
                                "524",
                                "525",
                                "541",
                                "544",
                            ],
                        },
                    ],
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "56D"},
                                    "pattern": "^(?:[^\\r\\n]*\\n){2}([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            }
                        }
                    }
                },
            }
        ],
    },
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.IntrmyAgt1.FinInstnId.PstlAdr.Ctry": {
        "mode": "append",
        "rules": [
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "56D"}, "right": ""},
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "56D"},
                                    "pattern": "^(?:[^\\r\\n]*\\n){3}([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            },
                            "right": "",
                        },
                    ],
                    "any": [
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "20"},
                                    "pattern": "^7\\d{2}",
                                    "group": 0,
                                },
                            },
                            "right": "",
                        },
                        {
                            "op": "in",
                            "left": {
                                "substr": {"value": {"mt": "20"}, "start": 0, "len": 3}
                            },
                            "right": [
                                "500",
                                "506",
                                "510",
                                "513",
                                "520",
                                "523",
                                "526",
                                "527",
                                "540",
                                "543",
                                "544",
                                "545",
                                "546",
                                "547",
                                "501",
                                "502",
                                "511",
                                "512",
                                "521",
                                "522",
                                "524",
                                "525",
                                "541",
                                "544",
                            ],
                        },
                    ],
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "56D"},
                                    "pattern": "^(?:[^\\r\\n]*\\n){3}([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            }
                        }
                    }
                },
            }
        ],
    },
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.CdtrAgt.FinInstnId.PstlAdr.TwnNm": {
        "mode": "set",
        "rules": [
            {
                "when": {
                    "all": [
                        {
                            "op": "in",
                            "left": {
                                "substr": {"value": {"mt": "20"}, "start": 0, "len": 3}
                            },
                            "right": [
                                "500",
                                "506",
                                "510",
                                "513",
                                "520",
                                "523",
                                "526",
                                "527",
                                "540",
                                "543",
                                "544",
                                "545",
                                "546",
                                "547",
                                "501",
                                "502",
                                "511",
                                "512",
                                "521",
                                "522",
                                "524",
                                "525",
                                "541",
                            ],
                        },
                        {"op": "!=", "left": {"mt": "57D"}, "right": ""},
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "57D"},
                                    "pattern": "^(?:[^\\r\\n]*\\r?\\n){0}([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            },
                            "right": "",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "57D"},
                                    "pattern": "^(?:[^\\r\\n]*\\r?\\n){3}([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "57D"},
                                    "pattern": "^(?:[^\\r\\n]*\\r?\\n){3}([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            }
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_CdtrAgtTwnNm",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            },
        ],
    },
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.CdtrAgt.FinInstnId.PstlAdr.StrtNm": {
        "mode": "set",
        "rules": [
            {
                "when": {
                    "all": [
                        {
                            "op": "in",
                            "left": {
                                "substr": {"value": {"mt": "20"}, "start": 0, "len": 3}
                            },
                            "right": [
                                "500",
                                "506",
                                "510",
                                "513",
                                "520",
                                "523",
                                "526",
                                "527",
                                "540",
                                "543",
                                "544",
                                "545",
                                "546",
                                "547",
                                "501",
                                "502",
                                "511",
                                "512",
                                "521",
                                "522",
                                "524",
                                "525",
                                "541",
                            ],
                        },
                        {"op": "!=", "left": {"mt": "57D"}, "right": ""},
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "57D"},
                                    "pattern": "^(?:[^\\r\\n]*\\r?\\n){2}([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            },
                            "right": "",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "57D"},
                                    "pattern": "^(?:[^\\r\\n]*\\r?\\n){3}([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "57D"},
                                    "pattern": "^(?:[^\\r\\n]*\\r?\\n){2}([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            }
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_CdtrAgtStrtNm",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            },
            {
                "when": {
                    "all": [
                        {
                            "op": "in",
                            "left": {
                                "substr": {"value": {"mt": "20"}, "start": 0, "len": 3}
                            },
                            "right": [
                                "500",
                                "506",
                                "510",
                                "513",
                                "520",
                                "523",
                                "526",
                                "527",
                                "540",
                                "543",
                                "544",
                                "545",
                                "546",
                                "547",
                                "501",
                                "502",
                                "511",
                                "512",
                                "521",
                                "522",
                                "524",
                                "525",
                                "541",
                            ],
                        },
                        {"op": "!=", "left": {"mt": "57D"}, "right": ""},
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "57D"},
                                    "pattern": "^(?:[^\\r\\n]*\\r?\\n){3}([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            },
                            "right": "",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "57D"},
                                    "pattern": "^(?:[^\\r\\n]*\\r?\\n){1}([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "57D"},
                                    "pattern": "^(?:[^\\r\\n]*\\r?\\n){1}([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            }
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_CdtrAgtStrtNm",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            },
        ],
    },
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.CdtrAgt.FinInstnId.PstlAdr.Ctry": {
        "mode": "set",
        "rules": [
            {
                "when": {
                    "all": [
                        {
                            "op": "in",
                            "left": {
                                "substr": {"value": {"mt": "20"}, "start": 0, "len": 3}
                            },
                            "right": [
                                "500",
                                "506",
                                "510",
                                "513",
                                "520",
                                "523",
                                "526",
                                "527",
                                "540",
                                "543",
                                "544",
                                "545",
                                "546",
                                "547",
                                "501",
                                "502",
                                "511",
                                "512",
                                "521",
                                "522",
                                "524",
                                "525",
                                "541",
                            ],
                        },
                        {"op": "!=", "left": {"mt": "57D"}, "right": ""},
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "57D"},
                                    "pattern": "^(?:[^\\r\\n]*\\r?\\n){3}([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {"value": {"literal": "US"}},
                "set": [
                    {
                        "set_var": {
                            "name": "has_CdtrAgtCtry",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            }
        ],
    },
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.CdtrAgt.FinInstnId.Nm": {
        "mode": "set",
        "rules": [
            {
                "when": {
                    "all": [
                        {
                            "op": "=",
                            "left": {"var": "has_CdtrAgtFinInstnIdBICFI"},
                            "right": "",
                        },
                        {"op": "!=", "left": {"mt": "57D"}, "right": ""},
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "57D"},
                                    "pattern": "^(?:[^\\r\\n]*\\r?\\n){1}([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            },
                            "right": "",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "57D"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                            "flags": 8,
                                        },
                                    },
                                    "pattern": "^(//|[A-Z0-9]{3,4}\\d)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "57D"},
                                    "pattern": "^(?:[^\\r\\n]*\\r?\\n){1}([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            }
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_CdtrAgt.FinInstnId.Nm",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            },
            {
                "when": {
                    "all": [
                        {
                            "op": "=",
                            "left": {"var": "has_CdtrAgtFinInstnIdBICFI"},
                            "right": "",
                        },
                        {"op": "!=", "left": {"mt": "57D"}, "right": ""},
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "57D"},
                                    "pattern": "^([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            },
                            "right": "",
                        },
                        {
                            "op": "=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "57D"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                            "flags": 8,
                                        },
                                    },
                                    "pattern": "^(//|[A-Z0-9]{3,4}\\d)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "57D"},
                                    "pattern": "^([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            }
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_CdtrAgt.FinInstnId.Nm",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            },
            {
                "when": {
                    "all": [
                        {
                            "op": "=",
                            "left": {"var": "has_CdtrAgtFinInstnIdBICFI"},
                            "right": "",
                        },
                        {"op": "!=", "left": {"mt": "57D"}, "right": ""},
                        {
                            "op": "=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "57D"},
                                    "pattern": "^(?:[^\\r\\n]*\\r?\\n){1}([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {"value": {"literal": "NOTPROVIDED"}},
                "set": [
                    {
                        "set_var": {
                            "name": "has_CdtrAgt.FinInstnId.Nm",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            },
            {
                "when": {
                    "all": [
                        {
                            "op": "=",
                            "left": {"var": "has_CdtrAgtFinInstnIdBICFI"},
                            "right": "",
                        },
                        {"op": "!=", "left": {"mt": "57C"}, "right": ""},
                        {
                            "op": "=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "57C"},
                                    "pattern": "^(?:[^\\r\\n]*\\r?\\n){1}([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {"value": {"literal": "NOTPROVIDED"}},
                "set": [
                    {
                        "set_var": {
                            "name": "has_CdtrAgt.FinInstnId.Nm",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            },
        ],
    },
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.Cdtr.PstlAdr.PstCd": {
        "mode": "set",
        "rules": [
            {
                "when": {
                    "all": [
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {
                                        "substr": {
                                            "value": {"mt": "20"},
                                            "start": 0,
                                            "len": 3,
                                        }
                                    },
                                    "pattern": "^(7[0-9]{2})$",
                                    "group": 1,
                                },
                            },
                            "right": "",
                        },
                        {
                            "op": "=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "32A"},
                                    "pattern": "^.{6}([A-Z]{3})",
                                    "group": 1,
                                },
                            },
                            "right": "CAD",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "59"},
                                    "pattern": "^(?:[^\\r\\n]*\\r?\\n){2}([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "59"},
                                            "pattern": "^(?:[^\\r\\n]*\\r?\\n){2}([^\\r\\n]*)",
                                            "group": 1,
                                            "flags": 8,
                                        },
                                    },
                                    "start": 0,
                                    "len": 6,
                                }
                            }
                        }
                    }
                },
            }
        ],
    },
    ".Document.FIToFICstmrCdtTrf.GrpHdr.CreDtTm": {
        "mode": "append",
        "rules": [
            {
                "then": {"value": {"fn": "now_iso8601"}},
                "set": [
                    {
                        "set_var": {
                            "name": "has_GrpHdr.CreDtTm",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
            }
        ],
    },
    ".Document.FIToFICstmrCdtTrf.GrpHdr.NbOfTxs": {
        "mode": "set",
        "rules": [{"then": {"value": {"literal": "1"}}}],
    },
    ".Document.FIToFICstmrCdtTrf.GrpHdr.SttlmInf.SttlmMtd": {
        "mode": "set",
        "rules": [
            {
                "when": {
                    "all": [
                        {
                            "op": "in",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "20"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 3,
                                }
                            },
                            "right": [
                                "500",
                                "506",
                                "510",
                                "513",
                                "520",
                                "523",
                                "526",
                                "527",
                                "540",
                                "543",
                                "544",
                                "545",
                                "546",
                                "547",
                                "501",
                                "502",
                                "511",
                                "512",
                                "521",
                                "522",
                                "524",
                                "525",
                                "541",
                                "571",
                                "700",
                                "701",
                                "662",
                                "663",
                                "651",
                            ],
                        },
                        {
                            "op": "=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "20"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 3,
                                }
                            },
                            "right": "651",
                        },
                    ]
                },
                "then": {"value": {"literal": "INGA"}},
            },
            {
                "when": {
                    "all": [
                        {
                            "op": "in",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "20"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 3,
                                }
                            },
                            "right": [
                                "500",
                                "506",
                                "510",
                                "513",
                                "520",
                                "523",
                                "526",
                                "527",
                                "540",
                                "543",
                                "544",
                                "545",
                                "546",
                                "547",
                                "501",
                                "502",
                                "511",
                                "512",
                                "521",
                                "522",
                                "524",
                                "525",
                                "541",
                                "571",
                                "700",
                                "701",
                                "662",
                                "663",
                                "651",
                            ],
                        },
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "20"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    },
                                    "start": 0,
                                    "len": 3,
                                }
                            },
                            "right": "651",
                        },
                    ]
                },
                "then": {"value": {"literal": "INDA"}},
            },
            {"when": {"all": []}, "then": {"value": "NOTPROVIDED"}},
        ],
    },
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.Purp.Prtry": {
        "mode": "set",
        "rules": [
            {
                "when": {"all": [{"op": "!=", "left": {"mt": "26T"}, "right": ""}]},
                "then": {"value": {"trim": {"value": {"mt": "26T"}}}},
            },
            {
                "when": {
                    "all": [
                        {"op": "=", "left": {"mt": "26T"}, "right": ""},
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "20"},
                                    "pattern": "^(7\\d{2})",
                                    "group": 1,
                                },
                            },
                            "right": "",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "50K"},
                                    "pattern": "^(?:[^\\r\\n]*\\r?\\n){2}([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "50K"},
                                    "pattern": "^(?:[^\\r\\n]*\\r?\\n){2}([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            }
                        }
                    }
                },
            },
        ],
    },
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.Dbtr.Nm": {
        "mode": "set",
        "rules": [
            {
                "set": [
                    {
                        "set_var": {
                            "name": "pacs.008::Dbtr.Nm",
                            "scope": "global",
                            "value": {
                                "from_db_query": {
                                    "alias": "pacs.008::Dbtr.Nm",
                                    "lookup_message_type": "pacs.008",
                                    "which_uetr_to_lookup": {"mtField": "121"},
                                    "where_to_lookup_uetr": {
                                        "message_type": "pacs.008",
                                        "path_kind": "xml",
                                        "field_path": ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.PmtId.UETR",
                                    },
                                    "field_to_extract": {
                                        "message_type": "pacs.008",
                                        "path_kind": "xml",
                                        "field_path": ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.Dbtr.Nm",
                                    },
                                    "fallback_literal": "",
                                }
                            },
                        }
                    }
                ],
                "when": {
                    "all": [
                        {
                            "op": "in",
                            "left": {
                                "substr": {"value": {"mt": "20"}, "start": 0, "len": 3}
                            },
                            "right": ["662", "663", "651"],
                        },
                        {"op": "!=", "left": {"var": "pacs.008::Dbtr.Nm"}, "right": ""},
                    ],
                    "any": [
                        {"op": "!=", "left": {"mt": "50K"}, "right": ""},
                        {"op": "!=", "left": {"mt": "50F"}, "right": ""},
                    ],
                },
                "then": {"value": {"var": "pacs.008::Dbtr.Nm"}},
            },
            {
                "when": {
                    "all": [
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {
                                        "substr": {
                                            "value": {"mt": "20"},
                                            "start": 0,
                                            "len": 3,
                                        }
                                    },
                                    "pattern": "^(7[0-9]{2})$",
                                    "group": 1,
                                },
                            },
                            "right": "",
                        },
                        {"op": "!=", "left": {"mt": "50K"}, "right": ""},
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "50K"},
                                    "pattern": "^(?:[^\\r\\n]*\\r?\\n){3}([^,\\r\\n]+)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "50K"},
                                    "pattern": "^(?:[^\\r\\n]*\\r?\\n){3}([^,\\r\\n]+)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            }
                        }
                    }
                },
            },
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "50K"}, "right": ""},
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "50K"},
                                    "pattern": "(?:[^\\r\\n]*\\r?\\n)([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "50K"},
                                    "pattern": "(?:[^\\r\\n]*\\r?\\n)([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            }
                        }
                    }
                },
            },
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "50F"}, "right": ""},
                        {
                            "op": "!=",
                            "left": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "50F"},
                                    "pattern": "^(?:^|\\r?\\n)1\\/([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "50F"},
                                    "pattern": "(?:^|\\r?\\n)1\\/([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            }
                        }
                    }
                },
            },
            {"when": {"all": []}, "then": {"value": "NOTPROVIDED"}},
        ],
    },
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.CdtrAgt.FinInstnId.PstlAdr.AdrLine": {
        "mode": "append",
        "rules": [
            {
                "when": {
                    "all": [
                        {
                            "op": "=",
                            "left": {"var": "has_CdtrAgtFinInstnIdBICFI"},
                            "right": "",
                        },
                        {"op": "=", "left": {"var": "has_CdtrAgtStrtNm"}, "right": ""},
                        {"op": "=", "left": {"var": "has_CdtrAgtTwnNm"}, "right": ""},
                        {"op": "=", "left": {"var": "has_CdtrAgtCtry"}, "right": ""},
                        {"op": "!=", "left": {"mt": "57B"}, "right": ""},
                        {
                            "op": "=",
                            "left": {
                                "trim": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "57B"},
                                            "pattern": "^(?:[^\\r\\n]*\\r?\\n){2}([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    }
                                }
                            },
                            "right": "",
                        },
                        {
                            "op": "=",
                            "left": {
                                "trim": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "57B"},
                                            "pattern": "^(?:[^\\r\\n]*\\r?\\n){3}([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    }
                                }
                            },
                            "right": "",
                        },
                        {
                            "op": "=",
                            "left": {
                                "trim": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "57B"},
                                            "pattern": "^(?:[^\\r\\n]*\\r?\\n){4}([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    }
                                }
                            },
                            "right": "",
                        },
                    ],
                },
                "then": {"value": "NOTPROVIDED"},
            },
            {
                "when": {
                    "all": [
                        {
                            "op": "=",
                            "left": {"var": "has_CdtrAgtFinInstnIdBICFI"},
                            "right": "",
                        },
                        {"op": "=", "left": {"var": "has_CdtrAgtStrtNm"}, "right": ""},
                        {"op": "=", "left": {"var": "has_CdtrAgtTwnNm"}, "right": ""},
                        {"op": "=", "left": {"var": "has_CdtrAgtCtry"}, "right": ""},
                        {"op": "!=", "left": {"mt": "57B"}, "right": ""},
                        {
                            "op": "!=",
                            "left": {
                                "trim": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "57B"},
                                            "pattern": "^(?:[^\\r\\n]*\\r?\\n){2}([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    }
                                }
                            },
                            "right": "",
                        },
                    ],
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "57B"},
                                    "pattern": "^(?:[^\\r\\n]*\\r?\\n){2}([^\\r\\n]*)",
                                    "group": 1,
                                },
                            }
                        }
                    }
                },
            },
            {
                "when": {
                    "all": [
                        {
                            "op": "=",
                            "left": {"var": "has_CdtrAgtFinInstnIdBICFI"},
                            "right": "",
                        },
                        {"op": "=", "left": {"var": "has_CdtrAgtStrtNm"}, "right": ""},
                        {"op": "=", "left": {"var": "has_CdtrAgtTwnNm"}, "right": ""},
                        {"op": "=", "left": {"var": "has_CdtrAgtCtry"}, "right": ""},
                        {"op": "!=", "left": {"mt": "57B"}, "right": ""},
                        {
                            "op": "!=",
                            "left": {
                                "trim": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "57B"},
                                            "pattern": "^(?:[^\\r\\n]*\\r?\\n){3}([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    }
                                }
                            },
                            "right": "",
                        },
                    ],
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "57B"},
                                    "pattern": "^(?:[^\\r\\n]*\\r?\\n){3}([^\\r\\n]*)",
                                    "group": 1,
                                },
                            }
                        }
                    }
                },
            },
            {
                "when": {
                    "all": [
                        {
                            "op": "=",
                            "left": {"var": "has_CdtrAgtFinInstnIdBICFI"},
                            "right": "",
                        },
                        {"op": "=", "left": {"var": "has_CdtrAgtStrtNm"}, "right": ""},
                        {"op": "=", "left": {"var": "has_CdtrAgtTwnNm"}, "right": ""},
                        {"op": "=", "left": {"var": "has_CdtrAgtCtry"}, "right": ""},
                        {"op": "!=", "left": {"mt": "57B"}, "right": ""},
                        {
                            "op": "!=",
                            "left": {
                                "trim": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "57B"},
                                            "pattern": "^(?:[^\\r\\n]*\\r?\\n){4}([^\\r\\n]*)",
                                            "group": 1,
                                        },
                                    }
                                }
                            },
                            "right": "",
                        },
                    ],
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "57B"},
                                    "pattern": "^(?:[^\\r\\n]*\\r?\\n){4}([^\\r\\n]*)",
                                    "group": 1,
                                },
                            }
                        }
                    }
                },
            },
            # {
            #     "when": {
            #         "all": [
            #             {
            #                 "op": "=",
            #                 "left": {"var": "has_CdtrAgtFinInstnIdBICFI"},
            #                 "right": "",
            #             },
            #             {"op": "=", "left": {"var": "has_CdtrAgtStrtNm"}, "right": ""},
            #             {"op": "=", "left": {"var": "has_CdtrAgtTwnNm"}, "right": ""},
            #             {"op": "=", "left": {"var": "has_CdtrAgtCtry"}, "right": ""},
            #             {"op": "=", "left": {"mt": "57B"}, "right": ""},
            #             {"op": "!=", "left": {"mt": "57D"}, "right": ""},
            #             {
            #                 "op": "!=",
            #                 "left": {
            #                     "trim": {
            #                         "value": {
            #                             "fn": "regex_extract",
            #                             "args": {
            #                                 "text": {"mt": "57D"},
            #                                 "pattern": "^(?:[^\\r\\n]*(?:\\r\\n|\\r|\\n)){3}([^\\r\\n]*)",
            #                                 "group": 1,
            #                             },
            #                         }
            #                     }
            #                 },
            #                 "right": "",
            #             },
            #             {
            #                 "op": "!=",
            #                 "left": {
            #                     "fn": "lower",
            #                     "args": {
            #                         "text": {
            #                             "trim": {
            #                                 "value": {
            #                                     "fn": "regex_extract",
            #                                     "args": {
            #                                         "text": {"mt": "57D"},
            #                                         "pattern": "^(?:[^\\r\\n]*(?:\\r\\n|\\r|\\n)){3}([^\\r\\n]*)",
            #                                         "group": 1,
            #                                     },
            #                                 }
            #                             }
            #                         }
            #                     },
            #                 },
            #                 "right": "us",
            #             },
            #         ]
            #     },
            #     "then": {
            #         "value": {
            #             "trim": {
            #                 "value": {
            #                     "fn": "regex_extract",
            #                     "args": {
            #                         "text": {"mt": "57D"},
            #                         "pattern": "^(?:[^\\r\\n]*(?:\\r\\n|\\r|\\n)){3}([^\\r\\n]*)",
            #                         "group": 1,
            #                     },
            #                 }
            #             }
            #         }
            #     },
            # },
            {
                "when": {
                    "all": [
                        {
                            "op": "=",
                            "left": {"var": "has_CdtrAgtFinInstnIdBICFI"},
                            "right": "",
                        },
                        {"op": "=", "left": {"mt": "57D"}, "right": ""},
                    ]
                },
                "then": {"value": {"literal": "NOTPROVIDED"}},
            },
            {
                "when": {
                    "all": [
                        {
                            "op": "=",
                            "left": {"var": "has_CdtrAgtFinInstnIdBICFI"},
                            "right": "",
                        },
                        {"op": "=", "left": {"var": "has_CdtrAgtTwnNm"}, "right": ""},
                        {"op": "=", "left": {"var": "has_CdtrAgtCtry"}, "right": ""},
                        {"op": "!=", "left": {"mt": "57D"}, "right": ""},
                        {
                            "op": "!=",
                            "left": {
                                "trim": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "57D"},
                                            "pattern": "^(?:[^\\r\\n]*\\r?\\n){1}([^\\r\\n]*)",
                                            "group": 1,
                                            "flags": 8,
                                        },
                                    }
                                }
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "57D"},
                                    "pattern": "^(?:[^\\r\\n]*\\r?\\n){1}([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            }
                        }
                    }
                },
            },
            {
                "when": {
                    "all": [
                        {
                            "op": "=",
                            "left": {"var": "has_CdtrAgtFinInstnIdBICFI"},
                            "right": "",
                        },
                        {"op": "=", "left": {"var": "has_CdtrAgtTwnNm"}, "right": ""},
                        {"op": "=", "left": {"var": "has_CdtrAgtCtry"}, "right": ""},
                        {"op": "=", "left": {"var": "has_CdtrAgtStrtNm"}, "right": ""},
                        {"op": "!=", "left": {"mt": "57D"}, "right": ""},
                        {
                            "op": "!=",
                            "left": {
                                "trim": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "57D"},
                                            "pattern": "^(?:[^\\r\\n]*\\r?\\n){2}([^\\r\\n]*)",
                                            "group": 1,
                                            "flags": 8,
                                        },
                                    }
                                }
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "57D"},
                                    "pattern": "^(?:[^\\r\\n]*\\r?\\n){2}([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            }
                        }
                    }
                },
            },
            {
                "when": {
                    "all": [
                        {
                            "op": "=",
                            "left": {"var": "has_CdtrAgtFinInstnIdBICFI"},
                            "right": "",
                        },
                        {"op": "=", "left": {"var": "has_CdtrAgtTwnNm"}, "right": ""},
                        {"op": "=", "left": {"var": "has_CdtrAgtCtry"}, "right": ""},
                        {"op": "!=", "left": {"mt": "57D"}, "right": ""},
                        {
                            "op": "!=",
                            "left": {
                                "trim": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "57D"},
                                            "pattern": "^(?:[^\\r\\n]*\\r?\\n){3}([^\\r\\n]*)",
                                            "group": 1,
                                            "flags": 8,
                                        },
                                    }
                                }
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "trim": {
                            "value": {
                                "fn": "regex_extract",
                                "args": {
                                    "text": {"mt": "57D"},
                                    "pattern": "^(?:[^\\r\\n]*\\r?\\n){3}([^\\r\\n]*)",
                                    "group": 1,
                                    "flags": 8,
                                },
                            }
                        }
                    }
                },
            },
            {
                "when": {
                    "all": [
                        {
                            "op": "=",
                            "left": {"var": "has_CdtrAgtFinInstnIdBICFI"},
                            "right": "",
                        },
                        {"op": "=", "left": {"var": "has_CdtrAgtTwnNm"}, "right": ""},
                        {"op": "=", "left": {"var": "has_CdtrAgtCtry"}, "right": ""},
                        {"op": "!=", "left": {"mt": "57D"}, "right": ""},
                        {
                            "op": "!=",
                            "left": {
                                "trim": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "57D"},
                                            "pattern": "^([^\\r\\n]*)",
                                            "group": 1,
                                            "flags": 8,
                                        },
                                    }
                                }
                            },
                            "right": "",
                        },
                        {
                            "op": "=",
                            "left": {
                                "trim": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "57D"},
                                            "pattern": "^(?:[^\\r\\n]*\\r?\\n){1}([^\\r\\n]*)",
                                            "group": 1,
                                            "flags": 8,
                                        },
                                    }
                                }
                            },
                            "right": "",
                        },
                        {
                            "op": "=",
                            "left": {
                                "trim": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "57D"},
                                            "pattern": "^(?:[^\\r\\n]*\\r?\\n){2}([^\\r\\n]*)",
                                            "group": 1,
                                            "flags": 8,
                                        },
                                    }
                                }
                            },
                            "right": "",
                        },
                        {
                            "op": "=",
                            "left": {
                                "trim": {
                                    "value": {
                                        "fn": "regex_extract",
                                        "args": {
                                            "text": {"mt": "57D"},
                                            "pattern": "^(?:[^\\r\\n]*\\r?\\n){3}([^\\r\\n]*)",
                                            "group": 1,
                                            "flags": 8,
                                        },
                                    }
                                }
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {"value": {"literal": "NOTPROVIDED"}},
            },
        ],
    },
}


TEXTLIST_LONG = "long"
TEXTLIST_SHORT = "short"

_NOT_ALLOWED_LONG_RE = re.compile(
    r"[^0-9A-Za-z/\-\?:\(\)\.,'\+ !#\$%&\*=\^_`\{\|\}~\";<>@\[\]\\\]]"
)

_NOT_ALLOWED_SHORT_RE = re.compile(r"[^0-9A-Za-z/\-\?:\(\)\.,'\+ ]")


def _strip_accents_keep_letters(s: str) -> str:
    if not s:
        return s
    nfd = unicodedata.normalize("NFD", s)
    return "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn")


def sanitize_translate_text(value: Any, *, text_list: str = TEXTLIST_LONG) -> str:
    if value is None:
        return ""
    s = str(value)

    s = _strip_accents_keep_letters(s)

    if text_list == TEXTLIST_SHORT:
        return _NOT_ALLOWED_SHORT_RE.sub("", s)
    return _NOT_ALLOWED_LONG_RE.sub("", s)


def sanitize_translate_fields(
    fields: Dict[str, List[str]],
    *,
    default_text_list: str = TEXTLIST_LONG,
    per_path_text_list: Optional[Dict[str, str]] = None,
) -> Dict[str, List[str]]:

    if not fields:
        return fields

    per_path_text_list = per_path_text_list or {}
    out: Dict[str, List[str]] = {}

    for path, values in (fields or {}).items():
        if path in per_path_text_list:
            policy = per_path_text_list[path]
            cleaned_vals: List[str] = []
            for v in values or []:
                c = sanitize_translate_text(v, text_list=policy)
                if str(c).strip() == "":
                    continue
                cleaned_vals.append(c)
            if cleaned_vals:
                out[path] = cleaned_vals
        else:
            if values:
                out[path] = values

    for p in list(out.keys()):
        if ".@" in p:
            base, _attr = p.rsplit(".@", 1)
            if base not in out or not out.get(base):
                out.pop(p, None)

    return out


TEXTLIST_BY_PATH = {
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.InstrForCdtrAgt.InstrInf": "short",
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.Purp.Prtry": "short",
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.Dbtr.Nm": "short",
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.Dbtr.PstlAdr": "short",
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.Dbtr.PstlAdr.StrtNm": "short",
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.Dbtr.PstlAdr.TwnNm": "short",
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.Dbtr.PstlAdr.Ctry": "short",
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.Dbtr.PstlAdr.AdrLine": "short",
    ".Document.FIToFICstmrCdtTrf.GrpHdr.SttlmInf.InstdRmbrsmntAgt.FinInstnId.Nm": "short",
    ".Document.FIToFICstmrCdtTrf.GrpHdr.SttlmInf.InstdRmbrsmntAgt.FinInstnId.PstlAdr.AdrLine": "short",
    ".Document.FIToFICstmrCdtTrf.GrpHdr.SttlmInf.InstgRmbrsmntAgt.FinInstnId.Nm": "short",
    ".Document.FIToFICstmrCdtTrf.GrpHdr.SttlmInf.InstgRmbrsmntAgt.FinInstnId.PstlAdr.AdrLine": "short",
    ".Document.FIToFICstmrCdtTrf.GrpHdr.SttlmInf.ThrdRmbrsmntAgt.FinInstnId.ClrSysMmbId.ClrSysId.Cd": "short",
    ".Document.FIToFICstmrCdtTrf.GrpHdr.SttlmInf.ThrdRmbrsmntAgt.FinInstnId.ClrSysMmbId.MmbrId": "short",
    ".Document.FIToFICstmrCdtTrf.GrpHdr.SttlmInf.ThrdRmbrsmntAgt.FinInstnId.Nm": "short",
    ".Document.FIToFICstmrCdtTrf.GrpHdr.SttlmInf.ThrdRmbrsmntAgt.FinInstnId.PstlAdr.AdrLine": "short",
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.IntrmyAgt1.FinInstnId.ClrSysMmbId.ClrSysId.Cd": "short",
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.IntrmyAgt1.FinInstnId.ClrSysMmbId.MmbrId": "short",
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.IntrmyAgt1.FinInstnId.Nm": "short",
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.IntrmyAgt1.FinInstnId.PstlAdr.AdrLine": "short",
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.IntrmyAgt1.FinInstnId.PstlAdr.StrtNm": "short",
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.IntrmyAgt1.FinInstnId.PstlAdr.TwnNm": "short",
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.CdtrAgt.FinInstnId.Nm": "short",
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.CdtrAgt.FinInstnId.PstlAdr.AdrLine": "short",
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.CdtrAgt.FinInstnId.PstlAdr.StrtNm": "short",
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.CdtrAgt.FinInstnId.PstlAdr.TwnNm": "short",
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.CdtrAgt.FinInstnId.PstlAdr.Ctry": "short",
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.Cdtr.Nm": "short",
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.Cdtr.PstlAdr": "short",
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.Cdtr.PstlAdr.AdrLine": "short",
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.Cdtr.PstlAdr.StrtNm": "short",
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.Cdtr.PstlAdr.TwnNm": "short",
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.Cdtr.PstlAdr.Ctry": "short",
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.InstrForNxtAgt[0].InstrInf": "short",
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.InstrForNxtAgt[1].InstrInf": "short",
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.InstrForNxtAgt[2].InstrInf": "short",
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.InstrForNxtAgt[3].InstrInf": "short",
    ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.InstrForNxtAgt[4].InstrInf": "short",
    ".Document.FIToFICstmrCdtTrf.GrpHdr.SttlmInf.SttlmMtd": "short",
}


# -----------------------------------------------------------------------
# IO helpers
# -----------------------------------------------------------------------
def find_mt_files(entry_dir: str) -> List[Path]:
    p = Path(entry_dir)
    if not p.exists():
        return []
    files: List[Path] = []
    for ext in ("*.mt103", "*.txt", "*.mt"):
        files.extend(p.glob(ext))
    return files


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def write_xml(tree: ET.ElementTree, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        ET.indent(tree, space="  ")
    except Exception:
        pass
    body = ET.tostring(tree.getroot(), encoding="utf-8", xml_declaration=False)
    out_path.write_bytes(b'<?xml version="1.0" encoding="UTF-8"?>\n' + body)


# -----------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------
if __name__ == "__main__":
    # Ajustes sandbox opcionales
    sand_entry = Path("/mnt/data/entry")
    sand_dest = Path("/mnt/data/destiny")
    if sand_entry.exists():
        ubicationEntry = str(sand_entry)
        ubicationDestiny = str(sand_dest)

    flog(
        "Inicio traductor 002", f"entry={ubicationEntry}", f"destiny={ubicationDestiny}"
    )

    # Preferir un único archivo indicado por el servicio
    single_input = os.getenv("TRANSLATE_SINGLE_INPUT")
    expect_one = os.getenv("TRANSLATE_EXPECT_ONE", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    files: List[Path] = []
    if single_input:
        p = Path(single_input)
        if p.exists():
            files = [p]
            flog("Archivo indicado por servicio:", p)
        else:
            flog("Archivo indicado no existe:", p, "- se listarán por patrón.")
            files = find_mt_files(ubicationEntry)
    else:
        files = find_mt_files(ubicationEntry)

    if not files:
        dbg("No MT103 files found in", ubicationEntry)
        flog("Sin archivos de entrada en", ubicationEntry)
    else:
        flog(
            f"Archivos detectados ({len(files)}):",
            ", ".join(sorted(p.name for p in files)),
        )

        if expect_one and len(files) > 1 and not single_input:
            try:
                selected = max(files, key=lambda x: (x.stat().st_mtime, x.name))
            except Exception:
                selected = sorted(files)[-1]
            flog("Múltiples entradas detectadas; procesando solo:", selected.name)
            files = [selected]

    for f in files:
        try:
            flog("Procesando", f)
            txt = read_text(f)
            mt = parse_mt103(txt)

            # 1) Ejecuta reglas y muestra cada field
            fields = build_fields_from_mt(mt, fields_spec)
            # 1.1) Limpieza de caracteres no permitidos (antes de armar el XML)
            _default_list = (
                (os.getenv("TRANSLATE_TEXTLIST", "long") or "long").strip().lower()
            )
            if _default_list not in {TEXTLIST_LONG, TEXTLIST_SHORT}:
                _default_list = TEXTLIST_LONG
            fields = sanitize_translate_fields(
                fields,
                default_text_list=_default_list,
                per_path_text_list=globals().get("TEXTLIST_BY_PATH"),
            )

            print_fields(fields)

            # 2) Crea Envelope vacío y aplica SOLO los fields definidos
            root = create_empty_envelope()
            apply_fields_to_xml(root, fields)
            tree = ET.ElementTree(root)

            out_name = f.stem + "_pacs008.xml"
            out_path = Path(ubicationDestiny) / out_name
            write_xml(tree, out_path)
            flog("XML escrito en", out_path)

            xml_body = ET.tostring(
                tree.getroot(), encoding="utf-8", xml_declaration=False
            )
            print(
                (b'<?xml version="1.0" encoding="UTF-8"?>\n' + xml_body).decode("utf-8")
            )

        except Exception as e:
            print("ERROR with", f, ":", e, file=sys.stderr, flush=True)
