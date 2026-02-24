import os
import sys
import re
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime, timezone, timedelta
import pandas as pd
from sqlalchemy import create_engine, text

from translators._shared.lookups import (
    evaluate_from_db_query,
    load_lookup_results,
    seed_lookup_vars,
)

# === DEBUG ===
DEBUG = False


def dbg(*a):
    if DEBUG:
        print("[DBG]", *a, file=sys.stderr, flush=True)


# Activar logs de flujo (sin contaminar stdout) con TRANSLATE_LOG_FLOW=1
FLOW_LOG = os.getenv("TRANSLATE_LOG_FLOW", "0").lower() in {"1", "true", "yes", "on"}


def flog(*a):
    if FLOW_LOG or DEBUG:
        print("[T006]", *a, file=sys.stderr, flush=True)


# -----------------------------------------------------------------------
# Configuración de ubicaciones (ajusta según tu entorno)
# -----------------------------------------------------------------------
traslateId = "006"


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
# Params.json (ÚNICA fuente de parámetros)
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
    """Devuelve (system_df, user_df) desde params.json:
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

    def to_df(d: Dict[str, List[str]]) -> pd.DataFrame:
        if not d:
            return pd.DataFrame()
        max_len = max((len(v) for v in d.values()), default=0)
        fixed = {k: (v + [""] * (max_len - len(v))) for k, v in d.items()}
        return pd.DataFrame(fixed).fillna("")

    return to_df(sys_map), to_df(usr_map)


def _split_param_values(raw: Any) -> List[str]:
    """Normaliza listas provenientes de Params (DB o archivo) separadas por comas/saltos."""
    if raw is None:
        return []
    text = str(raw).strip()
    if not text:
        return []
    tokens = re.split(r"[,\r\n]+", text)
    return [tok.strip() for tok in tokens if tok and tok.strip()]


def _dict_to_df(data: Dict[str, List[str]]) -> pd.DataFrame:
    if not data:
        return pd.DataFrame()
    max_len = max((len(v) for v in data.values()), default=0)
    fixed = {k: (v + [""] * (max_len - len(v))) for k, v in data.items()}
    return pd.DataFrame(fixed).fillna("")


def read_params_from_db() -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
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

valueDbSystemAll = _sys_df if _sys_df is not None else pd.DataFrame()
valueDbUserAll = _usr_df if _usr_df is not None else pd.DataFrame()


def get_list_from_values_db(df: pd.DataFrame, key: str) -> List[str]:
    if df is None or df.empty or key not in df.columns:
        return []
    col = df[key].dropna().astype(str).tolist()
    values: List[str] = []
    for entry in col:
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
# Parser 299
# -----------------------------------------------------------------------
MT_FIELD_RE = re.compile(r":([0-9A-Z]{2,3}[A-Z]?):")


def parse_mt299(text: str) -> Dict[str, Any]:
    """Extrae campos del MT299. Devuelve {"blocks":{...}, "fields":{tag:value}}"""
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

    # --- Extra: exponer BICs y UETR (121) como pseudo-tags para reglas ---
    b1 = result["blocks"].get("1", "")
    b2 = result["blocks"].get("2", "")
    b3 = result["blocks"].get("3", "")
    import re as _re

    # Ejemplo bloque 1:
    # {1:F01BREPCOBBAXX0000000000}
    #  b1 = "F01BREPCOBBAXX0000000000"
    #  BIC8 deseado = BREPCOBB (posiciones 3..10)
    if b1:
        try:
            sender8 = b1[3:11]  # "BREPCOBB"
            fields["BIC_SENDER_8"] = sender8
        except Exception:
            pass

    # Ejemplo bloque 2:
    # {2:I299CAFEC0BBXXXN}
    #  b2 = "I299CAFEC0BBXXXN"
    #  BIC8 deseado = CAFEC0BB (posiciones 4..11)
    if b2:
        try:
            receiver8 = b2[4:12]  # "CAFEC0BB"
            fields["BIC_RECEIVER_8"] = receiver8
        except Exception:
            pass

    # Extraer UETR del bloque 3 {121:...}
    m121 = _re.search(r"121:([0-9a-fA-F-]{36})", b3 or "")
    if m121:
        fields["121"] = m121.group(1).strip()

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


def _fn_clean_79(text: str) -> str:
    """Limpia campo 79: quita saltos de línea y # / `."""
    if text is None:
        return ""
    s = str(text)
    # quitar saltos de línea
    s = s.replace("\r", "").replace("\n", "")
    # quitar caracteres especiales
    for ch in "#/`":
        s = s.replace(ch, "")
    return s


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


# === Función auxiliar para fecha/hora actual ===
def _fn_now_iso():
    """Devuelve fecha/hora actual en formato ISO 8601 compatible con el validador (YYYY-MM-DDThh:mm:ss+HH:MM)."""
    dt = datetime.now().astimezone()
    s = dt.strftime("%Y-%m-%dT%H:%M:%S%z")  
    return s[:-2] + ":" + s[-2:]   




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


def _eval_value_mt(ctx: FieldContext, mt_fields: Dict[str, str], vs: ValueSpec) -> Any:
    """Evalúa ValueSpec genéricamente."""
    if "mt" in vs:
        return _mt_value(mt_fields, str(vs["mt"]).strip())

    if "literal" in vs:
        return str(vs["literal"])

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
        return str(ctx.vars.get(str(vs["var"]), ""))

    # --- transforms genéricos independientes de 'fn' ---
    if "trim" in vs:
        conf = vs["trim"]
        inner = _eval_value_mt(ctx, mt_fields, conf.get("value", {}))
        s = "" if inner is None else str(inner)
        side = str(conf.get("side", "both")).lower()  # "left", "right", "both"
        chars = conf.get("chars", None)  # None = espacios en blanco
        if side == "left":
            return s.lstrip(chars)
        if side == "right":
            return s.rstrip(chars)
        return s.strip(chars)

    # --- transforms parametrizables por nombre ---
    if "fn" in vs:
        fn = vs.get("fn")
        args = vs.get("args", {})

        if fn == "regex_extract":
            text_val = _eval_value_mt(ctx, mt_fields, args.get("text", {}))
            pattern = str(args.get("pattern", ""))
            group = args.get("group", 0)
            flags = int(args.get("flags", 0))
            return _fn_regex_extract(text_val, pattern, group, flags)

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

        if fn == "amount_normalize":
            raw = _eval_value_mt(ctx, mt_fields, args.get("value", {}))
            return _fn_amount_normalize(raw)

        if fn == "clean_79":
            raw = _eval_value_mt(ctx, mt_fields, args.get("value", {}))
            return _fn_clean_79(raw)

    return ""


def _fn_amount_normalize(raw: str) -> str:
    # 1) Sanitiza
    if raw is None:
        return ""
    s = str(raw).strip()
    # Conservar sólo dígitos y separadores , .
    s = "".join(ch for ch in s if ch.isdigit() or ch in ",.")
    if s == "":
        return ""

    # 2) Normaliza separador a COMA (todo punto -> coma)
    s = s.replace(".", ",")

    # 3) Separa entero/decimal si existe coma
    if "," in s:
        entero, dec = s.split(",", 1)
    else:
        entero, dec = s, ""

    # 4) Quitar ceros a la izquierda del entero
    entero = entero.lstrip("0") or "0"

    # 5) Resultado: sin coma si no hay decimales
    return entero if dec == "" else f"{entero},{dec}"


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
        all_ok = all(_eval_condition_mt(ctx, mt_fields, c) for c in when["all"])
    if "any" in when:
        any_ok = any(_eval_condition_mt(ctx, mt_fields, c) for c in when["any"])
    return (
        (all_ok and any_ok)
        if "all" in when and "any" in when
        else (any_ok if "any" in when else all_ok)
    )


def _collect_then_mt(
    ctx: FieldContext, mt_fields: Dict[str, str], then_obj: Dict[str, Any]
) -> List[str]:
    out: List[str] = []
    if "value" in then_obj:
        v = _eval_value_mt(ctx, mt_fields, then_obj["value"])
        if isinstance(v, list):
            out.extend([str(x) for x in v if str(x).strip() != ""])
        else:
            if str(v).strip() != "":
                out.append(str(v))
    elif "lines" in then_obj:
        for vs in then_obj["lines"]:
            v = _eval_value_mt(ctx, mt_fields, vs)
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
                value_spec = sv.get("value", {})
                val = _eval_value_mt(ctx, mt_fields, value_spec)
                if name and scope == "global":
                    ctx.vars[name] = val
        except Exception:
            pass

    seed_lookup_vars(ctx.vars, LOOKUP_RESULTS)


def build_fields_from_mt(
    mt: Dict[str, Any], fields_spec: Dict[str, Any]
) -> Dict[str, List[str]]:
    mt_fields = mt.get("fields", {})
    result: Dict[str, List[str]] = {}
    ctx = FieldContext()
    for target_path, spec in fields_spec.items():
        produced: List[str] = []
        if isinstance(spec, dict) and spec.get("mode") == "append":
            for rule in spec.get("rules", []):
                if _eval_logic_mt(ctx, mt_fields, rule.get("when", {})):
                    produced.extend(
                        _collect_then_mt(ctx, mt_fields, rule.get("then", {}))
                    )
                    _apply_set_mt(ctx, mt_fields, rule.get("set", []))
        elif isinstance(spec, list):
            for rule in spec:
                if _eval_logic_mt(ctx, mt_fields, rule.get("when", {})):
                    produced = _collect_then_mt(ctx, mt_fields, rule.get("then", {}))
                    _apply_set_mt(ctx, mt_fields, rule.get("set", []))
                    break
        result[target_path] = produced
    return result


def print_fields(fields: Dict[str, List[str]]):
    print("FIELDS:")
    for path, lines in fields.items():
        if not lines:
            print(f"  {path}: <empty>")
            continue
        for i, val in enumerate(lines, 1):
            print(f"  {path}[{i}]: {val}")


# -----------------------------------------------------------------------
# Namespaces + creación on-demand por rutas con puntos
# -----------------------------------------------------------------------
NS = {
    "env": "urn:swift:xsd:envelope",
    "head": "urn:iso:std:iso:20022:tech:xsd:head.001.001.02",
    "pacs": "urn:iso:std:iso:20022:tech:xsd:pacs.002.001.10",
}
ET.register_namespace("", NS["env"])
ET.register_namespace("head", NS["head"])
ET.register_namespace("pacs", NS["pacs"])


def _el_ns(parent, ns_key, name, text=None):
    q = f"{{{NS[ns_key]}}}{name}"
    e = ET.SubElement(parent, q)
    if text is not None:
        e.text = str(text)
    return e


def create_empty_envelope() -> ET.Element:
    """Crea solo el root Envelope (sin AppHdr/Document)."""
    return ET.Element(f"{{{NS['env']}}}Envelope")


def _ensure_path_with_ns(root: ET.Element, dotted_path: str) -> ET.Element:
    """
    Crea nodos según una ruta con puntos.
    Reglas:
      - .AppHdr.*  => namespace 'head' bajo Envelope
      - .Document.* => namespace 'pacs' bajo Envelope
    Devuelve el elemento final (leaf).
    """
    assert dotted_path.startswith("."), "La ruta debe iniciar con '.'"
    parts = [p for p in dotted_path.split(".") if p]  # elimina vacío inicial

    current = root
    if parts[0] == "AppHdr":
        node = current.find(f"./{{{NS['head']}}}AppHdr")
        if node is None:
            node = _el_ns(current, "head", "AppHdr")
        current = node
        ns = "head"
    elif parts[0] == "Document":
        node = current.find(f"./{{{NS['pacs']}}}Document")
        if node is None:
            node = _el_ns(current, "pacs", "Document")
        current = node
        ns = "pacs"
    else:
        ns = "pacs"  # por defecto

    for name in parts[1:]:
        # hijo directo con ese nombre
        child = None
        for c in list(current):
            if c.tag == f"{{{NS[ns]}}}{name}":
                child = c
                break
        if child is None:
            child = _el_ns(current, ns, name)
        current = child

    return current


def apply_fields_to_xml(envelope_root: ET.Element, fields: Dict[str, List[str]]):
    """
    Inserta en el XML los campos producidos por fields_spec.
    - Si un path tiene varias líneas, crea múltiples elementos hermanos con el mismo nombre.
    - Si la lista está vacía, NO crea nada.
    """
    # Orden estable: AppHdr primero, luego Document
    for path in sorted(fields.keys(), key=lambda p: (not p.startswith(".AppHdr"), p)):
        values = fields.get(path, [])
        if not values:
            continue
        if not path.startswith("."):
            dbg("Ruta ignorada (no inicia con punto):", path)
            continue

        if ".@" in path:
            base_path, attr_name = path.rsplit(".@", 1)
            leaf = _ensure_path_with_ns(envelope_root, base_path)
            for val in values:
                leaf.set(attr_name, str(val))
            continue

        for val in values:
            leaf = _ensure_path_with_ns(envelope_root, path)
            leaf.text = str(val)


# -----------------------------------------------------------------------
# ESPECIFICACIONES (solo lo que se haya definido: nada extra)
# -----------------------------------------------------------------------

fields_spec = {
    ".AppHdr.CreDt": {
        "mode": "append",
        "rules": [
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "20"}, "right": ""},
                        {
                            "op": "=",
                            "left": {
                                "substr": {"value": {"mt": "20"}, "start": 0, "len": 3}
                            },
                            "right": "400",
                        },
                    ]
                },
                "then": {"value": {"literal": _fn_now_iso()}},
            }
        ],
    },
    ".Document.FIToFIPmtStsRpt.GrpHdr.CreDtTm": [
        {
            "when": {
                "all": [
                    {"op": "!=", "left": {"mt": "20"}, "right": ""},
                    {
                        "op": "=",
                        "left": {
                            "substr": {"value": {"mt": "20"}, "start": 0, "len": 3}
                        },
                        "right": "400",
                    },
                ]
            },
            "then": {"value": {"literal": _fn_now_iso()}},
        }
    ],
    # =========================
    # AppHdr
    # =========================
    ".AppHdr.BizMsgIdr": {
        "mode": "append",
        "rules": [
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "20"}, "right": ""},
                        {
                            "op": "=",
                            "left": {
                                "substr": {"value": {"mt": "20"}, "start": 0, "len": 3}
                            },
                            "right": "400",
                        },
                    ]
                },
                "then": {"value": {"mt": "20"}},
            }
        ],
    },
    ".AppHdr.MsgDefIdr": {
        "mode": "append",
        "rules": [
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "20"}, "right": ""},
                        {
                            "op": "=",
                            "left": {
                                "substr": {"value": {"mt": "20"}, "start": 0, "len": 3}
                            },
                            "right": "400",
                        },
                    ]
                },
                "then": {"value": {"literal": "pacs.002.001.10"}},
            }
        ],
    },
    ".AppHdr.BizSvc": {
        "mode": "append",
        "rules": [
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "20"}, "right": ""},
                        {
                            "op": "=",
                            "left": {
                                "substr": {"value": {"mt": "20"}, "start": 0, "len": 3}
                            },
                            "right": "400",
                        },
                    ]
                },
                "then": {"value": {"literal": "swift.cbprplus.03"}},
            }
        ],
    },
    ".AppHdr.Fr.FIId.FinInstnId.BICFI": {
        "mode": "append",
        "rules": [
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "20"}, "right": ""},
                        {
                            "op": "=",
                            "left": {
                                "substr": {"value": {"mt": "20"}, "start": 0, "len": 3}
                            },
                            "right": "400",
                        },
                        {"op": "!=", "left": {"mt": "BIC_SENDER_8"}, "right": ""},
                    ]
                },
                "then": {"value": {"mt": "BIC_SENDER_8"}},
            }
        ],
    },
    ".AppHdr.To.FIId.FinInstnId.BICFI": {
        "mode": "append",
        "rules": [
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "20"}, "right": ""},
                        {
                            "op": "=",
                            "left": {
                                "substr": {"value": {"mt": "20"}, "start": 0, "len": 3}
                            },
                            "right": "400",
                        },
                        {"op": "!=", "left": {"mt": "BIC_RECEIVER_8"}, "right": ""},
                    ]
                },
                "then": {"value": {"mt": "BIC_RECEIVER_8"}},
            }
        ],
    },
    # =========================
    # Document / FIToFIPmtStsRpt
    # =========================
    ".Document.FIToFIPmtStsRpt.GrpHdr.MsgId": {
        "mode": "append",
        "rules": [
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "20"}, "right": ""},
                        {
                            "op": "=",
                            "left": {
                                "substr": {"value": {"mt": "20"}, "start": 0, "len": 3}
                            },
                            "right": "400",
                        },
                    ]
                },
                "then": {"value": {"mt": "20"}},
            }
        ],
    },
    # GrpHdr.CreDtTm — optional (Excel sugiere fecha/hora de recepción del sistema).
    # Se creará SOLO si está disponible en un campo/param futuro (no implementado aquí).
    ".Document.FIToFIPmtStsRpt.TxInfAndSts.OrgnlGrpInf.OrgnlMsgId": {
        "mode": "append",
        "rules": [
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "20"}, "right": ""},
                        {
                            "op": "=",
                            "left": {
                                "substr": {"value": {"mt": "20"}, "start": 0, "len": 3}
                            },
                            "right": "400",
                        },
                        {"op": "!=", "left": {"mt": "21"}, "right": ""},
                    ]
                },
                "then": {"value": {"mt": "21"}},
            }
        ],
    },
    ".Document.FIToFIPmtStsRpt.TxInfAndSts.OrgnlGrpInf.OrgnlMsgNmId": [
        {
            "set": [
                {
                    "set_var": {
                        "name": "pacs.009::AppHdr.MsgDefIdr",
                        "scope": "global",
                        "value": {
                            "from_db_query": {
                                "alias": "pacs.009::AppHdr.MsgDefIdr",
                                "lookup_message_type": "pacs.009",
                                "which_msgid_to_lookup": {"mtField": "21"},
                                "where_to_lookup_msgid": {
                                    "message_type": "pacs.009",
                                    "path_kind": "xml",
                                    "field_path": ".Document.FICdtTrf.GrpHdr.MsgId",
                                },
                                "field_to_extract": {
                                    "message_type": "pacs.009",
                                    "path_kind": "xml",
                                    "field_path": ".AppHdr.MsgDefIdr",
                                },
                                "fallback_literal": "",
                            }
                        },
                    }
                }
            ],
            "when": {
                "all": [
                    {"op": "!=", "left": {"mt": "20"}, "right": ""},
                    {
                        "op": "=",
                        "left": {
                            "substr": {"value": {"mt": "20"}, "start": 0, "len": 3}
                        },
                        "right": "400",
                    },
                    {
                        "op": "!=",
                        "left": {"var": "pacs.009::AppHdr.MsgDefIdr"},
                        "right": "",
                    },
                ]
            },
            "then": {"value": {"var": "pacs.009::AppHdr.MsgDefIdr"}},
        }
    ],
    ".Document.FIToFIPmtStsRpt.TxInfAndSts.OrgnlEndToEndId": [
        {
            "set": [
                {
                    "set_var": {
                        "name": "pacs.009::Document.FICdtTrf.CdtTrfTxInf.PmtId.EndToEndId",
                        "scope": "global",
                        "value": {
                            "from_db_query": {
                                "alias": "pacs.009::Document.FICdtTrf.CdtTrfTxInf.PmtId.EndToEndId",
                                "lookup_message_type": "pacs.009",
                                "which_msgid_to_lookup": {"mtField": "21"},
                                "where_to_lookup_msgid": {
                                    "message_type": "pacs.009",
                                    "path_kind": "xml",
                                    "field_path": ".Document.FICdtTrf.GrpHdr.MsgId",
                                },
                                "field_to_extract": {
                                    "message_type": "pacs.009",
                                    "path_kind": "xml",
                                    "field_path": ".Document.FICdtTrf.CdtTrfTxInf.PmtId.EndToEndId",
                                },
                                "fallback_literal": "",
                            }
                        },
                    }
                }
            ],
            "when": {
                "all": [
                    {"op": "!=", "left": {"mt": "20"}, "right": ""},
                    {
                        "op": "=",
                        "left": {
                            "substr": {"value": {"mt": "20"}, "start": 0, "len": 3}
                        },
                        "right": "400",
                    },
                    {
                        "op": "!=",
                        "left": {
                            "var": "pacs.009::Document.FICdtTrf.CdtTrfTxInf.PmtId.EndToEndId"
                        },
                        "right": "",
                    },
                ]
            },
            "then": {
                "value": {
                    "var": "pacs.009::Document.FICdtTrf.CdtTrfTxInf.PmtId.EndToEndId"
                }
            },
        }
    ],
    ".Document.FIToFIPmtStsRpt.TxInfAndSts.OrgnlUETR": [
        {
            "set": [
                {
                    "set_var": {
                        "name": "pacs.009::Document.FIToFIPmtStsRpt.TxInfAndSts.OrgnlUETR",
                        "scope": "global",
                        "value": {
                            "from_db_query": {
                                "alias": "pacs.009::Document.FIToFIPmtStsRpt.TxInfAndSts.OrgnlUETR",
                                "lookup_message_type": "pacs.009",
                                "which_msgid_to_lookup": {"mtField": "21"},
                                "where_to_lookup_msgid": {
                                    "message_type": "pacs.009",
                                    "path_kind": "xml",
                                    "field_path": ".Document.FICdtTrf.GrpHdr.MsgId",
                                },
                                "field_to_extract": {
                                    "message_type": "pacs.009",
                                    "path_kind": "xml",
                                    "field_path": ".Document.FIToFIPmtStsRpt.TxInfAndSts.OrgnlUETR",
                                },
                                "fallback_literal": "",
                            }
                        },
                    }
                }
            ],
            "when": {
                "all": [
                    {"op": "!=", "left": {"mt": "20"}, "right": ""},
                    {
                        "op": "=",
                        "left": {
                            "substr": {"value": {"mt": "20"}, "start": 0, "len": 3}
                        },
                        "right": "400",
                    },
                    {
                        "op": "!=",
                        "left": {
                            "var": "pacs.009::Document.FIToFIPmtStsRpt.TxInfAndSts.OrgnlUETR"
                        },
                        "right": "",
                    },
                ]
            },
            "then": {
                "value": {
                    "var": "pacs.009::Document.FIToFIPmtStsRpt.TxInfAndSts.OrgnlUETR"
                }
            },
        }
    ],
    ".Document.FIToFIPmtStsRpt.TxInfAndSts.OrgnlGrpInf.OrgnlCreDtTm": [
        {
            "set": [
                {
                    "set_var": {
                        "name": "pacs.009::Document.FIToFIPmtStsRpt.TxInfAndSts.OrgnlGrpInf.OrgnlCreDtTm",
                        "scope": "global",
                        "value": {
                            "from_db_query": {
                                "alias": "pacs.009::Document.FIToFIPmtStsRpt.TxInfAndSts.OrgnlGrpInf.OrgnlCreDtTm",
                                "lookup_message_type": "pacs.009",
                                "which_msgid_to_lookup": {"mtField": "21"},
                                "where_to_lookup_msgid": {
                                    "message_type": "pacs.009",
                                    "path_kind": "xml",
                                    "field_path": ".Document.FICdtTrf.GrpHdr.MsgId",
                                },
                                "field_to_extract": {
                                    "message_type": "pacs.009",
                                    "path_kind": "xml",
                                    "field_path": ".Document.FIToFIPmtStsRpt.TxInfAndSts.OrgnlGrpInf.OrgnlCreDtTm",
                                },
                                "fallback_literal": "",
                            }
                        },
                    }
                }
            ],
            "when": {
                "all": [
                    {"op": "!=", "left": {"mt": "20"}, "right": ""},
                    {
                        "op": "=",
                        "left": {
                            "substr": {"value": {"mt": "20"}, "start": 0, "len": 3}
                        },
                        "right": "400",
                    },
                    {
                        "op": "!=",
                        "left": {
                            "var": "pacs.009::Document.FIToFIPmtStsRpt.TxInfAndSts.OrgnlGrpInf.OrgnlCreDtTm"
                        },
                        "right": "",
                    },
                ]
            },
            "then": {
                "value": {
                    "var": "pacs.009::Document.FIToFIPmtStsRpt.TxInfAndSts.OrgnlGrpInf.OrgnlCreDtTm"
                }
            },
        }
    ],
    # ".Document.FIToFIPmtStsRpt.TxInfAndSts.OrgnlGrpInf.OrgnlCreDtTm": [
    #     {
    #         "when": {"all": [{"op": "=", "left": {"literal": "1"}, "right": "1"}]},
    #         "then": {"value": {"literal": "9999-12-31T00:00:00+00:00"}},
    #     }
    # ],
    # # Nota: OrgnlEndToEndId podría venir del PACS009 previo. Solo se emitirá si está disponible.
    # ".Document.FIToFIPmtStsRpt.TxInfAndSts.OrgnlEndToEndId": {
    #     "mode": "append",
    #     "rules": [
    #         {
    #             "when": {"all": [{"op": "!=", "left": {"mt": "E2EID"}, "right": ""}]},
    #             "then": {"value": {"mt": "E2EID"}},
    #         },
    #         {
    #             "when": {"all": [{"op": "=", "left": {"mt": "E2EID"}, "right": ""}]},
    #             "then": {"value": {"literal": "400CAFE250610E2"}},
    #         },
    #     ],
    # },
    # ".Document.FIToFIPmtStsRpt.TxInfAndSts.OrgnlUETR": {
    #     "mode": "append",
    #     "rules": [
    #         {
    #             "when": {"all": [{"op": "!=", "left": {"mt": "121"}, "right": ""}]},
    #             "then": {"value": {"mt": "121"}},
    #         }
    #     ],
    # },
    ".Document.FIToFIPmtStsRpt.TxInfAndSts.TxSts": [
        {
            "when": {
                "all": [
                    {"op": "!=", "left": {"mt": "20"}, "right": ""},
                    {
                        "op": "=",
                        "left": {
                            "substr": {"value": {"mt": "20"}, "start": 0, "len": 3}
                        },
                        "right": "400",
                    },
                ]
            },
            "then": {"value": {"literal": "RCVD"}},
        }
    ],
    ".Document.FIToFIPmtStsRpt.TxInfAndSts.StsRsnInf.Rsn.Cd": {
        "mode": "append",
        "rules": [
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "20"}, "right": ""},
                        {
                            "op": "=",
                            "left": {
                                "substr": {"value": {"mt": "20"}, "start": 0, "len": 3}
                            },
                            "right": "400",
                        },
                    ]
                },
                "then": {"value": {"literal": "NARR"}},
            }
        ],
    },
    ".Document.FIToFIPmtStsRpt.TxInfAndSts.StsRsnInf.AddtlInf": {
        "mode": "append",
        "rules": [
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "20"}, "right": ""},
                        {
                            "op": "=",
                            "left": {
                                "substr": {"value": {"mt": "20"}, "start": 0, "len": 3}
                            },
                            "right": "400",
                        },
                        {"op": "!=", "left": {"mt": "79"}, "right": ""},
                    ]
                },
                "then": {
                    "lines": [
                        {
                            "substr": {
                                "value": {
                                    "fn": "clean_79",
                                    "args": {"value": {"mt": "79"}},
                                },
                                "start": 0,
                                "len": 105,
                            }
                        },
                        {
                            "substr": {
                                "value": {
                                    "fn": "clean_79",
                                    "args": {"value": {"mt": "79"}},
                                },
                                "start": 105,
                            }
                        },
                    ]
                },
            }
        ],
    },
    ".Document.FIToFIPmtStsRpt.TxInfAndSts.InstgAgt.FinInstnId.BICFI": {
        "mode": "append",
        "rules": [
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "20"}, "right": ""},
                        {
                            "op": "=",
                            "left": {
                                "substr": {"value": {"mt": "20"}, "start": 0, "len": 3}
                            },
                            "right": "400",
                        },
                        {"op": "!=", "left": {"mt": "BIC_SENDER_12"}, "right": ""},
                    ]
                },
                "then": {"value": {"mt": "BIC_SENDER_12"}},
            }
        ],
    },
    ".Document.FIToFIPmtStsRpt.TxInfAndSts.InstdAgt.FinInstnId.BICFI": {
        "mode": "append",
        "rules": [
            {
                "when": {
                    "all": [
                        {"op": "!=", "left": {"mt": "20"}, "right": ""},
                        {
                            "op": "=",
                            "left": {
                                "substr": {"value": {"mt": "20"}, "start": 0, "len": 3}
                            },
                            "right": "400",
                        },
                        {"op": "!=", "left": {"mt": "BIC_RECEIVER_12"}, "right": ""},
                    ]
                },
                "then": {"value": {"mt": "BIC_RECEIVER_12"}},
            }
        ],
    },
}


# -----------------------------------------------------------------------
# IO helpers
# -----------------------------------------------------------------------
def find_mt_files(entry_dir: str) -> List[Path]:
    p = Path(entry_dir)
    if not p.exists():
        return []
    files: List[Path] = []
    for ext in ("*.mt299", "*.txt", "*.mt"):
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
    xml_bytes = ET.tostring(tree.getroot(), encoding="utf-8", xml_declaration=True)
    out_path.write_bytes(xml_bytes)


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

    files = find_mt_files(ubicationEntry)
    if not files:
        dbg("No MT299 files found in", ubicationEntry)

    for f in files:
        try:
            txt = read_text(f)
            mt = parse_mt299(txt)

            # 1) Ejecuta reglas y muestra cada field
            fields = build_fields_from_mt(mt, fields_spec)

            # 2) Crea Envelope vacío y aplica SOLO los fields definidos
            root = create_empty_envelope()
            apply_fields_to_xml(root, fields)
            tree = ET.ElementTree(root)

            out_name = f.stem + "_pacs002.xml"
            out_path = Path(ubicationDestiny) / out_name
            write_xml(tree, out_path)
            xml_body = ET.tostring(
                tree.getroot(), encoding="utf-8", xml_declaration=False
            )
            print(
                (b'<?xml version="1.0" encoding="UTF-8"?>\n' + xml_body).decode("utf-8")
            )
        except Exception as e:
            print("ERROR with", f, ":", e)