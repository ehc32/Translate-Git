import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Callable
import pandas as pd
import json
from datetime import datetime, timezone, timedelta
import threading
from zoneinfo import ZoneInfo
import re

# === DEBUG ===
DEBUG = True


def dbg(*a):
    if DEBUG:
        print("[DBG]", *a)


# -----------------------------------------------------------------------
# Archivos de entrada
# -----------------------------------------------------------------------
traslateId = "909"


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

messageDbPath = str(Path(ubicationDb) / "messageDb.json")
valueDbSystemPath = str(Path(ubicationDb) / "valueDbSystem.json")
valueDbUserPath = str(Path(ubicationDb) / "valueDbUser.json")


# (fallbacks si ejecutas en entorno con /mnt/data)
for local_path, mounted in [
    (messageDbPath, "/mnt/data/messageDb.json"),
    (valueDbSystemPath, "/mnt/data/valueDbSystem.json"),
    (valueDbUserPath, "/mnt/data/valueDbUser.json"),
]:
    if not Path(local_path).exists() and Path(mounted).exists():
        if local_path.endswith("messageDb.json"):
            messageDbPath = mounted
        elif local_path.endswith("valueDbSystem.json"):
            valueDbSystemPath = mounted
        elif local_path.endswith("valueDbUser.json"):
            valueDbUserPath = mounted

dbg(
    "Resolved paths:",
    "messageDbPath=",
    messageDbPath,
    "valueDbSystemPath=",
    valueDbSystemPath,
    "valueDbUserPath=",
    valueDbUserPath,
)

# === Params.json unified lists (replaces valueDbSystem.json and valueDbUser.json) ===
# Prefer params.json (same ubication); fallback to old valueDb*.json if params is absent.
# === Params.json unified lists (replaces valueDbSystem.json and valueDbUser.json) ===
# Prefer params.json (same ubication); fallback to old valueDb*.json if params is absent.
paramsPath = str(Path(ubicationDb) / "params.json")
# also check for "Params.json" (capital P) commonly used
if not Path(paramsPath).exists():
    alt_params = Path(ubicationDb) / "Params.json"
    if alt_params.exists():
        paramsPath = str(alt_params)

# (fallbacks si ejecutas en entorno con /mnt/data)
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
    # sometimes params.json uses "value" (single) instead of "values"
    return [str(val)]


def _read_params_to_dfs(path: str):
    """Return two DataFrames: (system_df, user_df) built from params.json structure:
    { "params": [ {type: "system"|"user", key: "...", values:[...]} ] }"""
    import pandas as pd

    sys_map, usr_map = {}, {}

    if not Path(path).exists():
        # fall back to legacy valueDb*.json files if params.json not found
        return None, None

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = data.get("params", [])
    for it in items:
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

    def to_df(d):
        if not d:
            return pd.DataFrame()
        max_len = max((len(v) for v in d.values()), default=0)
        fixed = {k: (v + [""] * (max_len - len(v))) for k, v in d.items()}
        return pd.DataFrame(fixed).fillna("")

    return to_df(sys_map), to_df(usr_map)


# -----------------------------------------------------------------------
# Utilidades XML -> dotmap (rutas con puntos)
# -----------------------------------------------------------------------
def _strip_ns(tag: str) -> str:
    return tag.split("}", 1)[1] if "}" in tag else tag


def _latest_xml(entry_dir: str) -> str:
    p = Path(entry_dir)
    xmls = sorted(p.glob("*.xml"), key=lambda x: x.stat().st_mtime, reverse=True)
    if not xmls:
        raise FileNotFoundError(f"No hay .xml en: {p}")
    return str(xmls[0])


def _walk_dotmap(elem: ET.Element, put: Callable[[str, str], None], prefix: str = ""):
    name = _strip_ns(elem.tag)
    path = f"{prefix}.{name}" if prefix else name

    for k, v in elem.attrib.items():
        put(f"{path}[@{k}]", v)

    t = (elem.text or "").strip()
    if t and len(list(elem)) == 0:
        put(path, t)

    for child in elem:
        _walk_dotmap(child, put, path)


def xml_to_dotmap(xml_path: str) -> Dict[str, Union[str, List[str]]]:
    dotmap: Dict[str, Union[str, List[str]]] = {}

    def put(k: str, v: str):
        if k in dotmap:
            if isinstance(dotmap[k], list):
                dotmap[k].append(v)
            else:
                dotmap[k] = [dotmap[k], v]
        else:
            dotmap[k] = v

    try:
        root = ET.parse(xml_path).getroot()
        _walk_dotmap(root, put, "")
    except ET.ParseError:
        # fallback: múltiples raíces
        data = Path(xml_path).read_text(encoding="utf-8")
        root = ET.fromstring(f"<ROOT>{data}</ROOT>")
        for ch in list(root):
            _walk_dotmap(ch, put, "")
    return dotmap


# --- WATCHER de rutas clave para debug fino ---
_WATCH_SUFFIXES = {
    "53D": [
        ".Document.FIToFICstmrCdtTrf.GrpHdr.SttlmInf.InstgRmbrsmntAgt.FinInstnId.PstlAdr.StrtNm",
        ".Document.FIToFICstmrCdtTrf.GrpHdr.SttlmInf.InstgRmbrsmntAgt.FinInstnId.PstlAdr.TwnNm",
        ".Document.FIToFICstmrCdtTrf.GrpHdr.SttlmInf.InstgRmbrsmntAgt.FinInstnId.PstlAdr.Ctry",
    ],
    "53B": [
        ".Document.FIToFICstmrCdtTrf.GrpHdr.SttlmInf.SttlmAcct.Id.IBAN",
        ".Document.FIToFICstmrCdtTrf.GrpHdr.SttlmInf.SttlmAcct.Id.Othr.Id",
        ".Document.FIToFICstmrCdtTrf.GrpHdr.SttlmInf.InstgRmbrsmntAgt.FinInstnId.ClrSysMmbId.MmbId",
        ".Document.FIToFICstmrCdtTrf.GrpHdr.SttlmInf.InstgRmbrsmntAgtAcct.Id",
    ],
    "53A": [
        ".Document.FIToFICstmrCdtTrf.GrpHdr.SttlmInf.InstgRmbrsmntAgt.FinInstnId.BICFI",
        ".Document.FIToFICstmrCdtTrf.GrpHdr.SttlmInf.InstgRmbrsmntAgtAcct.Id.IBAN",
        ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.PmtId.ClrSysRef",
        ".Document.FIToFICstmrCdtTrf.GrpHdr.SttlmInf.SttlmMtd",
        ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.InstrForNxtAgt.Id",
    ],
}


def load_latest_dotmap(entry_dir: str) -> Dict[str, Union[str, List[str]]]:
    return xml_to_dotmap(_latest_xml(entry_dir))


def _get_by_suffix(
    dotmap: Dict[str, Union[str, List[str]]], suffix: str
) -> Optional[str]:
    watched = any(suffix.endswith(suf) for L in _WATCH_SUFFIXES.values() for suf in L)
    orig_suffix = suffix
    if suffix.startswith("."):
        suffix = suffix[1:]
    for k, v in dotmap.items():
        if k.endswith(suffix):
            val = v[0] if isinstance(v, list) else v
            if watched:
                dbg(
                    f"_get_by_suffix MATCH suffix={orig_suffix!r} <- key={k!r} -> {val!r}"
                )
            return val
    if watched:
        dbg(f"_get_by_suffix MISS  suffix={orig_suffix!r}")
    return None


def _padx(s: Optional[str], n: int, fill: str = "X") -> str:
    v = (s or "").strip()
    return v[:n] if len(v) >= n else v.ljust(n, fill)


TZ_COLOMBIA = ZoneInfo("America/Bogota")


def format_yymmddhhmm(_: str = "", tz: ZoneInfo = TZ_COLOMBIA) -> str:
    # Siempre usa la fecha/hora del momento de procesamiento en la zona indicada
    dt = datetime.now(tz)
    return dt.strftime("%y%m%d%H%M")


def _format_hhmm_offset(dt_str: str) -> str:
    # Convierte un datetime ISO8601 a 'HHMM±HHMM' (e.g., '14:23:00+05:00' -> '1423+0500').
    # Si no trae zona, asume +0000. Soporta 'Z' para UTC.
    # Función genérica reutilizable para cualquier campo.
    s = (dt_str or "").strip()
    if not s:
        return ""
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except Exception:
        # Normaliza casos sin segundos: YYYY-MM-DDTHH:MM±HH:MM
        try:
            plus = s.rfind("+")
            minus = s.rfind("-")
            idx = max(plus, minus)
            if idx > 0:
                base, off = s[:idx], s[idx:]
                if len(base) == 16:  # YYYY-MM-DDTHH:MM
                    s2 = base + ":00" + off
                    dt = datetime.fromisoformat(s2)
                else:
                    return ""
            else:
                return ""
        except Exception:
            return ""

    hhmm = dt.strftime("%H%M")
    off = dt.utcoffset()
    if off is None:
        sign = "+"
        oh = 0
        om = 0
    else:
        total_min = int(off.total_seconds() // 60)
        sign = "+" if total_min >= 0 else "-"
        total_min = abs(total_min)
        oh, om = divmod(total_min, 60)
    return f"{hhmm}{sign}{oh:02d}{om:02d}"


def _parse_iso_dt_norm(dt_str: str) -> datetime | None:
    s = (dt_str or "").strip()
    if not s:
        return None
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(s)
    except Exception:
        try:
            plus = s.rfind("+")
            minus = s.rfind("-")
            idx = max(plus, minus)
            if idx > 0:
                base, off = s[:idx], s[idx:]
                if len(base) == 16:  # YYYY-MM-DDTHH:MM
                    return datetime.fromisoformat(base + ":00" + off)
        except Exception:
            pass
    return None


def format_hhmm(dt_str: str) -> str:
    """Devuelve HHMM (sin offset) a partir de un ISO8601."""
    dt = _parse_iso_dt_norm(dt_str)
    return dt.strftime("%H%M") if dt else ""


def format_yymmdd(dt_str: str) -> str:
    """Devuelve YYMMDD (sin offset) a partir de un ISO8601."""
    dt = _parse_iso_dt_norm(dt_str)
    return dt.strftime("%y%m%d") if dt else ""


# -----------------------------------------------------------------------
# CBPR DateTime (con offset obligatorio) -> YYMMDD
# -----------------------------------------------------------------------
_CBPR_OFFSET_RE = re.compile(r"(\+|-)((0[0-9])|(1[0-4])):[0-5][0-9]$")


def cbpr_datetime_to_yymmdd(dtm_raw: str) -> str:
    """Convierte un dateTime CBPR (con offset obligatorio ±HH:MM, HH 00..14) a YYMMDD
    tomando SOLO la parte de fecha YYYY-MM-DD (sin ajuste por zona horaria).
    Retorna "" si es inválido.
    """
    s = (dtm_raw or "").strip()
    if not s:
        return ""

    # Debe terminar con offset válido ±HH:MM (HH 00..14)
    if _CBPR_OFFSET_RE.search(s) is None:
        return ""

    # La fecha debe venir al inicio: YYYY-MM-DD
    date_part = s[:10]
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", date_part) is None:
        return ""

    # Validez de calendario
    try:
        datetime.strptime(date_part, "%Y-%m-%d")
    except Exception:
        return ""

    # YYMMDD
    return date_part[2:4] + date_part[5:7] + date_part[8:10]


def _format_hhmm_offset(dt_str: str) -> str:
    """(Deja tu función original tal cual) HHMM±HHMM."""
    s = (dt_str or "").strip()
    if not s:
        return ""
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except Exception:
        try:
            plus = s.rfind("+")
            minus = s.rfind("-")
            idx = max(plus, minus)
            if idx > 0:
                base, off = s[:idx], s[idx:]
                if len(base) == 16:
                    dt = datetime.fromisoformat(base + ":00" + off)
                else:
                    return ""
            else:
                return ""
        except Exception:
            return ""
    hhmm = dt.strftime("%H%M")
    off = dt.utcoffset()
    if off is None:
        sign, oh, om = "+", 0, 0
    else:
        total_min = int(off.total_seconds() // 60)
        sign = "+" if total_min >= 0 else "-"
        total_min = abs(total_min)
        oh, om = divmod(total_min, 60)
    return f"{hhmm}{sign}{oh:02d}{om:02d}"


# -----------------------------------------------------------------------
# Carga de Tablas Json
# -----------------------------------------------------------------------
def _read_excel(path: str) -> Dict[str, pd.DataFrame]:
    """
    JSON-only: lee .json y lo devuelve como dict {'Sheet1': DataFrame} para mantener compatibilidad.
    - messageDb.json  -> DataFrame "ancho": columnas por cada 'campo' (además de messageType y codigo_interno)
    - valueDb*.json   -> DataFrame con columnas por LIST (p.ej., 'CCY', 'BICFI'), filas con los ítems (rellena vacíos)
    Si el archivo no existe, retorna {'Sheet1': df vacío}.
    """
    p = Path(path)
    if not Path(path).exists():
        dbg("LOAD JSON: archivo no existe ->", path)
        return {"Sheet1": pd.DataFrame()}

    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    name = p.name.lower()
    dbg("LOAD JSON:", name)

    # ----- messageDb.json -----
    if "messagedb" in name:
        # data: {messageType: {codigo_interno: {campo: valor|[vals]}}}
        rows = []
        for mtype, by_code in (data or {}).items():
            for code, kv in (by_code or {}).items():
                row = {"messageType": str(mtype), "codigo_interno": str(code)}
                for campo, valor in (kv or {}).items():
                    if isinstance(valor, list):
                        row[campo] = "" if not valor else str(valor[0])
                    else:
                        row[campo] = "" if valor is None else str(valor)
                rows.append(row)
        df = pd.DataFrame(rows).fillna("")
        dbg("messageDb columns:", list(df.columns)[:10], "... total:", len(df.columns))
        return {"Sheet1": df}

    # ----- valueDbSystem.json / valueDbUser.json -----
    # data: {LIST_NAME: [values,...]}
    if isinstance(data, dict):
        normalized = {}
        for k, vals in data.items():
            if vals is None:
                normalized[str(k)] = []
            elif isinstance(vals, list):
                normalized[str(k)] = ["" if v is None else str(v) for v in vals]
            else:
                normalized[str(k)] = [str(vals)]

        # Igualar longitudes para evitar ValueError
        max_len = max((len(v) for v in normalized.values()), default=0)
        for k, vals in normalized.items():
            if len(vals) < max_len:
                normalized[k] = vals + [""] * (max_len - len(vals))

        df = pd.DataFrame(normalized).fillna("")
        dbg("valuesDb columns:", list(df.columns))
        for cname in df.columns:
            sample = df[cname].dropna().astype(str).str.strip().tolist()[:10]
            dbg(f"valuesDb[{cname}] sample:", sample)
        return {"Sheet1": df}

    dbg("LOAD JSON: formato no reconocido, retorna vacío")
    return {"Sheet1": pd.DataFrame()}


messageDbSheets = _read_excel(messageDbPath)  # base de datos de mensajes (JSON)

# Try unified params.json first
try:
    _sys_df, _usr_df = _read_params_to_dfs(paramsPath)
except Exception as _e:
    _sys_df, _usr_df = None, None

if _sys_df is not None and _usr_df is not None:
    import pandas as pd

    valueDbSystemSheets = {"Sheet1": _sys_df}
    valueDbUserSheets = {"Sheet1": _usr_df}
else:
    # Legacy fallback to valueDbSystem.json / valueDbUser.json
    valueDbSystemSheets = _read_excel(valueDbSystemPath)  # listas del sistema (JSON)
    valueDbUserSheets = _read_excel(valueDbUserPath)  # listas del usuario (JSON)


def _all_rows(dbsheets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Concatena todas las hojas en un solo DataFrame (agrega columna __sheet__)."""
    frames = []
    for name, df in dbsheets.items():
        if df is not None and not df.empty:
            tmp = df.copy()
            tmp["__sheet__"] = name
            frames.append(tmp)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


messageDbAll = _all_rows(messageDbSheets)
valueDbSystemAll = _all_rows(valueDbSystemSheets)
valueDbUserAll = _all_rows(valueDbUserSheets)


def get_list_from_values_db(db_all: pd.DataFrame, column_name: str) -> List[str]:
    """Retorna los valores no nulos de una columna en cualquier hoja (usa nombre de LIST como nombre de columna)."""
    if db_all.empty:
        dbg(f"get_list_from_values_db: DB vacío, columna={column_name!r}")
        return []
    col = column_name
    if col not in db_all.columns:
        dbg(
            f"get_list_from_values_db: columna no existe -> {col!r}; columnas={list(db_all.columns)}"
        )
        return []
    vals = [str(x).strip() for x in db_all[col].dropna().tolist()]
    dbg(
        f"get_list_from_values_db: columna={col!r} -> {len(vals)} items; sample={vals[:10]}"
    )
    return vals


def db_lookup(
    db_all: pd.DataFrame,
    where: Dict[str, Any],
    return_col: str,
    first_only: bool = True,
) -> Union[None, str, List[str]]:
    """
    Lookup genérico: filtra db_all por igualdad (string) en todas las claves de 'where'
    y devuelve return_col.
    """
    if db_all.empty:
        dbg("db_lookup: DB vacío")
        return None
    df = db_all.copy()
    for k, v in where.items():
        if k not in df.columns:
            dbg(f"db_lookup: columna {k!r} no existe; columnas={list(df.columns)}")
            return None
        df = df[df[k].astype(str) == str(v)]
    if df.empty or return_col not in df.columns:
        dbg(f"db_lookup: sin filas o columna retorno {return_col!r} no existe")
        return None
    vals = df[return_col].dropna().astype(str).tolist()
    if not vals:
        return None
    return vals[0] if first_only else vals


# -----------------------------------------------------------------------
# Constructor general de bloques {N:...}
# -----------------------------------------------------------------------
def build_block(entry_dir: str, block_no: int, spec, vars_ctx=None) -> str:
    dot = load_latest_dotmap(entry_dir)
    vctx = vars_ctx or {}
    parts = []

    for seg in spec:
        if "fixed" in seg:
            parts.append(str(seg["fixed"]))
        elif "var" in seg:
            val = str(vctx.get(seg["var"], "") or "")
            parts.append(
                _padx(val, int(seg["pad"]), seg.get("fill", "X"))
                if "pad" in seg
                else val
            )
        elif "key" in seg or "keys" in seg:
            kpath = seg.get("key") or seg.get("keys")
            val = _get_by_suffix(dot, kpath) or ""

            if "post" in seg:
                post_fn_name = seg["post"]
                if post_fn_name in block_post_processors:
                    val = block_post_processors[post_fn_name](val)
                    dbg(
                        f"POST-PROCESS: {post_fn_name}({_get_by_suffix(dot, kpath)}) -> {val}"
                    )

            if "pad" in seg:
                val = _padx(val, int(seg["pad"]), seg.get("fill", "X"))

            parts.append(val)
        else:
            parts.append("")

    return "{" + f"{block_no}:" + "".join(parts) + "}"


# Especificaciones de Bloques
spec_bloque1_A = [
    {"fixed": "F21"},
    {
        "key": ".AppHdr.To.FIId.FinInstnId.BICFI",
        "post": "bic11",
    },
    {"var": "HDR_SESS_SEQ", "pad": 10, "fill": "0"},
]
spec_bloque1_B = [
    {"fixed": "F01"},
    {
        "key": ".AppHdr.To.FIId.FinInstnId.BICFI",
        "post": "bic11",
    },
    {"var": "HDR_SESS_SEQ", "pad": 10, "fill": "0"},
]
spec_bloque4 = [
    {"fixed": "{177:"},
    {"var": "HDR_YYMMDDHHMM"},
    {"fixed": "}{451:0}"},
]
spec_bloque2 = [
    {"fixed": "I299"},
    {"var": "HDR_HHMM"},
    {"var": "HDR_YYMMDD"},
    {
        "keys": ".AppHdr.To.FIId.FinInstnId.BICFI",
        "pad": 11,
        "fill": "X",
    },
    {"fixed": "X0000000000"},
    {"var": "HDR_YYMMDD"},
    {"fixed": "0000N"},
]
spec_bloque3 = [
    {"fixed": "{121:"},
    {"key": ".Document.FIToFIPmtStsRpt.CdtTrfTxInf.PmtId.UETR"},
    {"fixed": "}"},
]

TZ_COLOMBIA = ZoneInfo("America/Bogota")


def build_header_12(entry_dir: str) -> str:
    dot = load_latest_dotmap(entry_dir)

    # Timestamp para cabecera (orden de preferencia)
    hdr_dt = (
        # 1) Primero intento con la fecha/hora del AppHdr
        _get_by_suffix(dot, ".AppHdr.CreDt")
        # 2) Si no viene en AppHdr, uso la CreDtTm del GrpHdr del camt.054
        or _get_by_suffix(dot, ".Document.BkToCstmrDbtCdtNtfctn.GrpHdr.CreDtTm")
    )

    dt = datetime.now(TZ_COLOMBIA)

    day_of_year = dt.timetuple().tm_yday
    doy_str = f"{day_of_year:03d}"
    hour_str = f"{dt.hour:02d}"
    sec_str = f"{dt.second:02d}"
    millis = dt.microsecond // 1000
    millis_str = f"{millis:03d}"

    sess_seq = f"{doy_str}{hour_str}{sec_str}{millis_str}"

    vars_ctx = {
        "HDR_HHMM": format_hhmm(hdr_dt),
        "HDR_YYMMDD": format_yymmdd(hdr_dt),
        "HDR_YYMMDDHHMM": format_yymmddhhmm(hdr_dt),
        "HDR_SESS_SEQ": sess_seq,
    }

    b1A = build_block(entry_dir, 1, spec_bloque1_A, vars_ctx=vars_ctx)
    b1B = build_block(entry_dir, 1, spec_bloque1_B, vars_ctx=vars_ctx)
    b4 = build_block(entry_dir, 4, spec_bloque4, vars_ctx=vars_ctx)
    b2 = build_block(entry_dir, 2, spec_bloque2, vars_ctx=vars_ctx)

    uetr = _get_by_suffix(
        dot, ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.NtryDtls.TxDtls.Refs.UETR"
    )
    if uetr and str(uetr).strip():
        b3 = build_block(entry_dir, 3, spec_bloque3)
    else:
        dbg("Bloque 3 omitido: no hay UETR")
        b3 = ""

    return b1A + b4 + b1B + b2 + b3 + "{4:"


def normalize_bic11(bic_raw: str) -> str:
    if not bic_raw:
        return "XXXXXXXXXXX"

    bic = bic_raw.strip().upper()

    # 1 - Tomar los primeros 8 caracteres (BIC8)
    bic8 = bic[:8].ljust(8, "X")

    # 2 - Tomar los últimos 3 caracteres del BICFI original (posiciones 8-11)
    branch = bic[8:11] if len(bic) > 8 else ""

    # 3 - Si branch tiene menos de 3 caracteres, rellenar con 'X'
    branch = branch.ljust(3, "X")

    # 4 - Insertar UNA 'X' por defecto entre BIC8 y branch
    return bic8 + "X" + branch


block_post_processors = {
    "bic11": normalize_bic11,
}


def finalize_mt_message(header_12: str, lines_block4: list[str]) -> str:
    return header_12 + "\n" + "\n".join(lines_block4) + "\n-}"


# -----------------------------------------------------------------------
# Motor de reglas
# -----------------------------------------------------------------------
ValueSpec = Dict[
    str, Any
]  # {"xml":".path"} | {"literal":"..."} | {"list":{"db":"system|user","name":"BICFI"}} | {"db_lookup":{...}} | {"var":"nombre"}
CondSpec = Dict[
    str, Any
]  # {"op":"=", "left":ValueSpec, "right":ValueSpec}  (op in ["=","!=","in","not in"])
RuleSpec = Dict[
    str, Any
]  # {"when":{"all":[...], "any":[...]},"set":[...],"then":{"lines":[ValueSpec,...] or "value":ValueSpec}}


class FieldContext:
    def __init__(self):
        self.vars: Dict[str, Any] = {}


def _get_all_by_suffix(
    dotmap: Dict[str, Union[str, List[str]]], suffix: str
) -> List[str]:
    watched = any(suffix.endswith(suf) for L in _WATCH_SUFFIXES.values() for suf in L)
    orig_suffix = suffix
    if suffix.startswith("."):
        suffix = suffix[1:]
    out: List[str] = []
    for k, v in dotmap.items():
        if k.endswith(suffix):
            if isinstance(v, list):
                out.extend([str(x) for x in v])
            else:
                out.append(str(v))
    if watched:
        dbg(f"_get_all_by_suffix suffix={orig_suffix!r} -> {out!r}")
    return out


def _eval_value(
    dot: Dict[str, Union[str, List[str]]], ctx: FieldContext, vs: ValueSpec
) -> Any:
    if "pad" in vs:
        conf = vs["pad"]
        inner = _eval_value(dot, ctx, conf.get("value", {}))
        return _padx(inner, int(conf.get("len", 0)), conf.get("fill", "X"))

    if "substr" in vs:
        conf = vs["substr"]
        s = _eval_value(dot, ctx, conf.get("value", {}))
        start = int(conf.get("start", 0))
        ln = int(conf.get("len", max(0, len(s or ""))))
        return (s or "")[start : start + ln]

    if "map" in vs:
        conf = vs["map"]

        # Valor de entrada
        input_val = _eval_value(dot, ctx, conf.get("input", {}))
        key = "" if input_val is None else str(input_val)

        # Diccionario de mapeo
        mapping = conf.get("map", {})

        # default puede ser un spec o un valor plano
        default_spec = conf.get("default", input_val)
        if isinstance(default_spec, dict):
            default_val = _eval_value(dot, ctx, default_spec)
        else:
            default_val = default_spec

        return mapping.get(key, default_val)

    if "regex" in vs:
        conf = vs["regex"] or {}
        raw = _eval_value(dot, ctx, conf.get("value", {}))
        s = "" if raw is None else str(raw)
        s = s.strip()
        pat = str(conf.get("pattern", ""))
        mode = str(conf.get("mode", "fullmatch")).lower()  # fullmatch | search | match
        flags_str = str(conf.get("flags", "")).upper()
        flags = 0
        if "I" in flags_str:
            flags |= re.IGNORECASE
        if "M" in flags_str:
            flags |= re.MULTILINE
        if "S" in flags_str:
            flags |= re.DOTALL

        try:
            rx = re.compile(pat, flags)
        except re.error:
            dbg(f"REGEX compile error pattern={pat!r}")
            return ""

        ok = False
        if mode == "search":
            ok = rx.search(s) is not None
        elif mode == "match":
            ok = rx.match(s) is not None
        else:
            ok = rx.fullmatch(s) is not None

        if not ok:
            dbg(f"REGEX fail mode={mode} pattern={pat!r} value={s!r}")
            return ""
        return s

    if "trim" in vs:
        raw = _eval_value(dot, ctx, (vs.get("trim") or {}).get("value", {}))
        return ("" if raw is None else str(raw)).strip()

    if "normalize_ws" in vs:
        raw = _eval_value(dot, ctx, (vs.get("normalize_ws") or {}).get("value", {}))
        return normalize_ws("" if raw is None else str(raw))

    if "truncate" in vs:
        conf = vs["truncate"] or {}
        raw = _eval_value(dot, ctx, conf.get("value", {}))
        s = "" if raw is None else str(raw).strip()
        max_len = int(conf.get("max", 0) or 0)
        keep = conf.get("keep")
        keep_len = int(keep) if keep is not None else max_len
        suffix = str(conf.get("suffix", ""))
        if max_len <= 0:
            return s
        if len(s) <= max_len:
            return s
        # Si keep_len es mayor que max_len, lo ajustamos para evitar crecer
        keep_len = min(keep_len, max_len)
        # Si hay sufijo, reserva espacio para que el total no exceda max_len
        if suffix and keep_len + len(suffix) > max_len:
            keep_len = max(0, max_len - len(suffix))
        return s[:keep_len] + suffix
    if "xml" in vs:
        val = _get_by_suffix(dot, vs["xml"]) or ""
        return val

    if "literal" in vs:
        return str(vs["literal"])

    if "var" in vs:
        return ctx.vars.get(str(vs["var"]), "")

    if "list" in vs:
        src = vs["list"].get("db")
        name = vs["list"].get("name")
        if src == "system":
            lst = get_list_from_values_db(valueDbSystemAll, name)
            dbg(
                f"LIST lookup system name={name!r} -> {len(lst)} items; sample={lst[:5]}"
            )
            return lst
        if src == "user":
            lst = get_list_from_values_db(valueDbUserAll, name)
            dbg(
                f"LIST lookup user   name={name!r} -> {len(lst)} items; sample={lst[:5]}"
            )
            return lst
        dbg(f"LIST lookup unknown src={src!r} name={name!r} -> []")
        return []

    if "db_lookup" in vs:
        spec = vs["db_lookup"]
        which = spec.get("db", "messageDb")  # "messageDb" | "system" | "user"
        db = {
            "messageDb": messageDbAll,
            "system": valueDbSystemAll,
            "user": valueDbUserAll,
        }.get(which, messageDbAll)
        where: Dict[str, Any] = {}
        for k, wv in (spec.get("where") or {}).items():
            if isinstance(wv, dict) and "xml" in wv:
                where[k] = _eval_value(dot, ctx, {"xml": wv["xml"]})
            elif isinstance(wv, dict) and "var" in wv:
                where[k] = _eval_value(dot, ctx, {"var": wv["var"]})
            else:
                where[k] = str(wv)
        return_col = spec.get("return")
        if not return_col:
            return ""
        res = db_lookup(db, where, return_col, first_only=True) or ""
        dbg(f"DB_LOOKUP db={which} where={where} return={return_col} -> {res!r}")
        return res

    if "numfmt" in vs:
        conf = vs["numfmt"]
        raw = _eval_value(dot, ctx, conf.get("value", {}))
        s = str(raw or "").strip()

        s = s.replace(",", ".")

        if "." in s:
            intp, frac = s.split(".", 1)
        else:
            intp, frac = s, ""

        intp = intp.lstrip("0") or "0"

        if frac == "":
            return intp + ","
        else:
            return intp + "," + frac

    if "concat" in vs:
        parts = [str(_eval_value(dot, ctx, p)) for p in vs.get("concat", [])]
        return "".join(parts)

    if "cbpr_yymmdd" in vs:
        conf = vs["cbpr_yymmdd"] or {}
        raw = _eval_value(dot, ctx, conf.get("value", {}))
        s = "" if raw is None else str(raw).strip()
        yymmdd = cbpr_datetime_to_yymmdd(s)

        # Deja pista para reglas (opcional)
        ctx.vars["_cbpr_yymmdd"] = yymmdd
        ctx.vars["_cbpr_error"] = "" if yymmdd else "CBPR_DATETIME_INVALID"

        return yymmdd

    if "dtfmt" in vs:
        conf = vs["dtfmt"]
        iso_val = _eval_value(dot, ctx, conf.get("value", {})) or ""
        out = conf.get("out", "HHMM±HHMM")
        if out == "HHMM±HHMM":
            return _format_hhmm_offset(iso_val)

    if "xml_nth" in vs:
        conf = vs["xml_nth"]
        path = conf.get("path", "")
        idx = int(conf.get("index", 0))
        try:
            vals = _get_all_by_suffix(dot, path)
        except NameError:
            v = _get_by_suffix(dot, path)
            return v if idx == 0 else ""
        return vals[idx] if 0 <= idx < len(vals) else ""

    if "xml_all" in vs:
        return _get_all_by_suffix(dot, vs["xml_all"]) or []

    if "zip" in vs:
        lists = [_eval_value(dot, ctx, spec) for spec in vs["zip"]]
        return list(zip(*lists))

    if "template_each" in vs:
        conf = vs["template_each"]
        items = _eval_value(dot, ctx, conf.get("items", {})) or []
        tpl = conf.get("template", {})
        out = []
        for item in items:
            ctx.vars["_item"] = item
            out.append(str(_eval_value(dot, ctx, tpl)))
        ctx.vars.pop("_item", None)
        return out

    if "nth" in vs:
        i = int(vs["nth"])
        cur = ctx.vars.get("_item", [])
        try:
            return cur[i]
        except Exception:
            return ""

    if "format" in vs:
        tpl = vs["format"].get("tpl", "")
        args = [str(_eval_value(dot, ctx, a)) for a in vs["format"].get("args", [])]
        try:
            return tpl.format(*args)
        except Exception:
            return "".join(args)

    return ""


def _eval_condition(dot, ctx, cond: CondSpec) -> bool:
    op = cond.get("op")
    left = _eval_value(dot, ctx, cond.get("left", {}))
    right_val = cond.get("right", {})
    right = (
        _eval_value(dot, ctx, right_val) if isinstance(right_val, dict) else right_val
    )

    ls = str(left).strip()  # normaliza izquierda
    if op == "=":
        rs = str(right).strip()
        res = ls == rs
        dbg(f"COND '='  left={ls!r} right={rs!r} -> {res}")
        return res
    if op == "!=":
        rs = str(right).strip()
        res = ls != rs
        dbg(f"COND '!=' left={ls!r} right={rs!r} -> {res}")
        return res
    if op == "in":
        if isinstance(right, list):
            rlist = [str(x).strip() for x in right]  # normaliza lista
        else:
            rlist = [s.strip() for s in str(right).split(",")]
        res = ls in rlist
        dbg(f"COND 'in' left={ls!r} list_len={len(rlist)} sample={rlist[:10]} -> {res}")
        return res
    if op == "not in":
        if isinstance(right, list):
            rlist = [str(x).strip() for x in right]
        else:
            rlist = [s.strip() for s in str(right).split(",")]
        res = ls not in rlist
        dbg(
            f"COND 'not in' left={ls!r} list_len={len(rlist)} sample={rlist[:10]} -> {res}"
        )
        return res

    # NUEVO: todos iguales (tras normalizar y descartando vacíos)
    if op == "all_equal":
        vals = left
        if isinstance(vals, (str, bytes)) or not isinstance(vals, (list, tuple)):
            vals = [vals]
        norm = [str(v).strip() for v in vals if str(v or "").strip() != ""]
        res = len(set(norm)) <= 1
        dbg(f"COND 'all_equal' vals={norm} -> {res}")
        return res

    dbg(f"COND op={op!r} no soportado -> False")
    return False


def _eval_logic(
    dot: Dict[str, Union[str, List[str]]], ctx: FieldContext, when: Dict[str, Any]
) -> bool:
    # when puede contener "all":[], "any":[]
    all_ok = True
    any_ok = False
    if "all" in when:
        all_ok = all(_eval_condition(dot, ctx, c) for c in when["all"])
    if "any" in when:
        any_ok = any(_eval_condition(dot, ctx, c) for c in when["any"])
    dbg(
        f"_eval_logic summary -> all_present={'all' in when} all_ok={all_ok} | any_present={'any' in when} any_ok={any_ok}"
    )
    # Si no hay 'all', usamos solo 'any'; si hay ambos, combinamos con AND
    return (
        (all_ok and any_ok)
        if "all" in when and "any" in when
        else (any_ok if "any" in when else all_ok)
    )


def normalize_ws(s: str) -> str:
    """Colapsa cualquier whitespace (espacios, tabs, saltos) a un solo espacio y recorta extremos."""
    if s is None:
        return ""
    return " ".join(str(s).split())


def _truncate_lines_for_field(
    field_tag: str, lines: list[str], global_vars: dict | None = None
) -> list[str]:
    """
    Truncado/partición por campo.
    - Para :79: -> máximo 35 líneas, cada una de máximo 50 caracteres.
      Además, normaliza saltos de línea y los elimina (concatena todo).
    """
    tag = str(field_tag or "").strip()

    # Solo aplica reglas especiales a :79:
    if not tag.startswith("79"):
        return lines

    # 1) Normalizar y concatenar todo (como FIN79: sin saltos de línea)
    buf = "".join("" if ln is None else str(ln) for ln in (lines or []))
    buf = buf.replace("\r\n", "\n").replace("\r", "\n")
    buf = buf.replace("\n", "")  # elimina saltos
    buf = normalize_ws(buf)

    # 2) Partir en chunks de 50
    out: list[str] = []
    for i in range(0, len(buf), 35):
        out.append(buf[i : i + 35])

    # 3) Máximo 35 líneas
    if len(out) > 35:
        dbg(f"TRUNCATE 79: {len(out)} líneas -> truncando a 35")
        out = out[:35]

    if global_vars is not None and len(buf) > 35 * 50:
        # Warning si se truncó por capacidad
        global_vars.setdefault("ISSUE_79_LEVEL", "warning")
        global_vars.setdefault("ISSUE_79", "FIELD79_TRUNCATED_TO_CAPACITY")

    # Si quedó vacío, retorna lista vacía
    return [x for x in out if str(x).strip() != ""]


def _collect_then_lines(
    dot, ctx, then_obj, field_tag: str | None = None, global_vars: dict | None = None
):
    lines: list[str] = []
    if "lines" in then_obj:
        for vs in then_obj["lines"]:
            val = _eval_value(dot, ctx, vs)
            # aplanar si es lista
            if isinstance(val, list):
                lines.extend([str(x) for x in val if str(x).strip() != ""])
            else:
                lines.append(str(val))
    elif "value" in then_obj:
        val = _eval_value(dot, ctx, then_obj["value"])
        # aplanar si es lista
        if isinstance(val, list):
            lines.extend([str(x) for x in val if str(x).strip() != ""])
        else:
            lines.append(str(val))

    # Aquí aplicamos el truncado genérico según el campo
    if field_tag is not None:
        lines = _truncate_lines_for_field(field_tag, lines, global_vars=global_vars)

    return lines


def build_fields(entry_dir: str, fields_spec, global_vars=None):
    if global_vars is None:
        global_vars = {}

    if isinstance(entry_dir, dict):
        dot = entry_dir
    elif isinstance(entry_dir, str):
        dot = load_latest_dotmap(entry_dir)
    else:
        raise TypeError(
            f"build_fields: entry_dir debe ser str (ruta) o dict (dot), no {type(entry_dir).__name__}"
        )

    result = {}

    def _eval_logic_with_globals(dot_local, ctx_local, cond_local):
        if not cond_local:
            dbg("_eval_logic_with_globals: cond vacía -> True")
            return True

        custom_keys = {"global_is_set", "global_equals", "not", "any", "all", "exists"}

        def eval_item(item):
            if isinstance(item, dict) and any(k in item for k in custom_keys):
                return _eval_logic_with_globals(dot_local, ctx_local, item)
            return _eval_condition(dot_local, ctx_local, item)

        has_all = "all" in cond_local
        has_any = "any" in cond_local

        all_ok = True
        if has_all:
            all_ok = all(eval_item(c) for c in cond_local["all"])

        any_ok = True
        if has_any:
            any_ok = any(eval_item(c) for c in cond_local["any"])

        if has_all and has_any:
            ok_base = all_ok and any_ok
        elif has_any:
            ok_base = any_ok
        else:
            ok_base = all_ok

        ok_gset = True
        if "global_is_set" in cond_local:
            name = cond_local["global_is_set"]
            ok_gset = name in (global_vars or {})

        ok_geq = True
        if "global_equals" in cond_local:
            geq = cond_local["global_equals"] or {}
            name = geq.get("name")
            val = geq.get("value")
            ok_geq = (global_vars or {}).get(name) == val

        ok_exists = True
        if "exists" in cond_local:
            vspec = cond_local.get("exists") or {}
            v = (
                _eval_value(dot_local, ctx_local, vspec)
                if isinstance(vspec, dict)
                else vspec
            )
            if isinstance(v, list):
                ok_exists = any(str(x or "").strip() != "" for x in v)
            else:
                ok_exists = str(v or "").strip() != ""

        ok_not = True
        if "not" in cond_local:
            ok_not = not _eval_logic_with_globals(
                dot_local, ctx_local, cond_local["not"]
            )

        final_ok = ok_base and ok_gset and ok_geq and ok_exists and ok_not
        dbg(
            f"_eval_logic_with_globals -> base={ok_base} gset={ok_gset} geq={ok_geq} exists={ok_exists} not={ok_not} => FINAL={final_ok} | globals={global_vars} | cond={cond_local}"
        )
        return final_ok

    for field_tag, spec in fields_spec.items():
        # --- modo "lista" (una sola selección) ---
        if isinstance(spec, list):
            ctx = FieldContext()
            assigned = False
            for idx, rule in enumerate(spec, start=1):
                dbg(f"{field_tag}: rule#{idx} EVAL when={rule.get('when')}")
                ok = _eval_logic_with_globals(dot, ctx, rule.get("when", {}))
                dbg(f"{field_tag}: rule#{idx} WHEN -> {ok}")
                if not ok:
                    continue

                # THEN primero
                produced = _collect_then_lines(
                    dot, ctx, rule.get("then", {}), field_tag, global_vars=global_vars
                )
                produced = [ln for ln in produced if str(ln).strip() != ""]
                dbg(f"{field_tag}: rule#{idx} WHEN ok -> produced={produced!r}")

                # SET después del THEN (sin doble fase)
                for action in rule.get("set", []):
                    if "set_var" in action:
                        sv = action["set_var"]
                        name = sv.get("name")

                        def _set_value_to_scope(value):
                            if sv.get("scope") == "global":
                                global_vars[name] = value
                            else:
                                ctx.vars[name] = value

                        if "value" in sv:
                            _set_value_to_scope(_eval_value(dot, ctx, sv["value"]))
                        elif "from_db" in sv:
                            _set_value_to_scope(
                                _eval_value(dot, ctx, {"db_lookup": sv["from_db"]})
                            )
                        else:
                            _set_value_to_scope("")

                if produced:
                    result[field_tag] = produced
                    assigned = True
                    break

            if not assigned:
                dbg(f"{field_tag}: ninguna regla produjo salida")
                result[field_tag] = []
            continue

        # --- modo "append" (múltiples líneas/posiciones) ---
        if isinstance(spec, dict) and spec.get("mode") == "append":
            ctx = FieldContext()
            lines: List[str] = []
            for idx, rule in enumerate(spec.get("rules", []), start=1):
                dbg(f"{field_tag}: rule#{idx} (append) EVAL when={rule.get('when')}")
                ok = _eval_logic_with_globals(dot, ctx, rule.get("when", {}))
                dbg(f"{field_tag}: rule#{idx} (append) WHEN -> {ok}")
                if not ok:
                    continue

                # THEN primero
                produced = _collect_then_lines(
                    dot, ctx, rule.get("then", {}), field_tag, global_vars=global_vars
                )
                produced = [ln for ln in produced if str(ln).strip() != ""]
                dbg(
                    f"{field_tag}: rule#{idx} WHEN ok -> produced(part)={produced!r}, line_no={rule.get('line_no')}"
                )

                # Colocar líneas si hay
                if produced:
                    if "line_no" in rule:
                        start = max(1, int(rule["line_no"])) - 1
                        for i, ln in enumerate(produced):
                            idx_line = start + i
                            while len(lines) <= idx_line:
                                lines.append("")
                            lines[idx_line] = ln
                    else:
                        lines.extend(produced)

                # SET después del THEN (sin doble fase)
                for action in rule.get("set", []):
                    if "set_var" in action:
                        sv = action["set_var"]
                        name = sv.get("name")

                        def _set_value_to_scope(value):
                            if sv.get("scope") == "global":
                                global_vars[name] = value
                            else:
                                ctx.vars[name] = value

                        if "value" in sv:
                            _set_value_to_scope(_eval_value(dot, ctx, sv["value"]))
                        elif "from_db" in sv:
                            _set_value_to_scope(
                                _eval_value(dot, ctx, {"db_lookup": sv["from_db"]})
                            )
                        else:
                            _set_value_to_scope("")

            # Limpieza de trailing blanks
            while lines and lines[-1] == "":
                lines.pop()

            if not any(str(s).strip() for s in lines):
                dbg(f"{field_tag}: append -> sin líneas finales")
                result[field_tag] = []
            else:
                dbg(f"{field_tag}: append -> líneas finales={lines!r}")
                result[field_tag] = lines
            continue

        # Si no coincide ningún formato, deja vacío
        result[field_tag] = []
    return result


# ------------------- ESPECIFICACIONES -------------------

fields_spec_20 = {
    "20": [
        # 1) MsgId ausente/vacío (tras trim) -> error
        {
            "when": {
                "not": {
                    "exists": {
                        "trim": {
                            "value": {"xml": ".Document.FIToFIPmtStsRpt.GrpHdr.MsgId"}
                        }
                    }
                }
            },
            "set": [
                {
                    "set_var": {
                        "scope": "global",
                        "name": "ISSUE_20_LEVEL",
                        "value": {"literal": "error"},
                    }
                },
                {
                    "set_var": {
                        "scope": "global",
                        "name": "ISSUE_20",
                        "value": {"literal": "MSGID_MISSING"},
                    }
                },
            ],
            "then": {"value": {"literal": ""}},
        },
        # 2) MsgId presente pero no cumple restricciones FIN -> error
        {
            "when": {
                "all": [
                    {
                        "exists": {
                            "trim": {
                                "value": {
                                    "xml": ".Document.FIToFIPmtStsRpt.GrpHdr.MsgId"
                                }
                            }
                        }
                    },
                    {
                        "not": {
                            "exists": {
                                "regex": {
                                    "value": {
                                        "trim": {
                                            "value": {
                                                "xml": ".Document.FIToFIPmtStsRpt.GrpHdr.MsgId"
                                            }
                                        }
                                    },
                                    "pattern": "^(?!/)(?!.*//).+(?<!/)$",
                                    "mode": "fullmatch",
                                }
                            }
                        }
                    },
                ]
            },
            "set": [
                {
                    "set_var": {
                        "scope": "global",
                        "name": "ISSUE_20_LEVEL",
                        "value": {"literal": "error"},
                    }
                },
                {
                    "set_var": {
                        "scope": "global",
                        "name": "ISSUE_20",
                        "value": {"literal": "MSGID_FIN_INVALID"},
                    }
                },
            ],
            "then": {"value": {"literal": ""}},
        },
        # 3) Caso OK: FIN válido -> poblar (16x, truncando con '+')
        {
            "when": {
                "exists": {
                    "regex": {
                        "value": {
                            "trim": {
                                "value": {
                                    "xml": ".Document.FIToFIPmtStsRpt.GrpHdr.MsgId"
                                }
                            }
                        },
                        "pattern": "^(?!/)(?!.*//).+(?<!/)$",
                        "mode": "fullmatch",
                    }
                }
            },
            "then": {
                "value": {
                    "truncate": {
                        "value": {
                            "regex": {
                                "value": {
                                    "trim": {
                                        "value": {
                                            "xml": ".Document.FIToFIPmtStsRpt.GrpHdr.MsgId"
                                        }
                                    }
                                },
                                "pattern": "^(?!/)(?!.*//).+(?<!/)$",
                                "mode": "fullmatch",
                            }
                        },
                        "max": 16,
                        "keep": 15,
                        "suffix": "+",
                    }
                }
            },
        },
    ]
}


fields_spec_21 = {
    "21": [
        # 1) OrgnlMsgId ausente/vacío (tras trim) -> error
        {
            "when": {
                "not": {
                    "exists": {
                        "trim": {
                            "value": {
                                "xml": ".Document.FIToFIPmtStsRpt.TxInfAndSts.OrgnlGrpInf.OrgnlMsgId"
                            }
                        }
                    }
                }
            },
            "set": [
                {
                    "set_var": {
                        "scope": "global",
                        "name": "ISSUE_21_LEVEL",
                        "value": {"literal": "error"},
                    }
                },
                {
                    "set_var": {
                        "scope": "global",
                        "name": "ISSUE_21",
                        "value": {"literal": "ORGNLMGSID_MISSING"},
                    }
                },
            ],
            "then": {"value": {"literal": ""}},
        },
        # 2) OrgnlMsgId presente pero no cumple restricciones FIN -> error
        {
            "when": {
                "all": [
                    {
                        "exists": {
                            "trim": {
                                "value": {
                                    "xml": ".Document.FIToFIPmtStsRpt.TxInfAndSts.OrgnlGrpInf.OrgnlMsgId"
                                }
                            }
                        }
                    },
                    {
                        "not": {
                            "exists": {
                                "regex": {
                                    "value": {
                                        "trim": {
                                            "value": {
                                                "xml": ".Document.FIToFIPmtStsRpt.TxInfAndSts.OrgnlGrpInf.OrgnlMsgId"
                                            }
                                        }
                                    },
                                    "pattern": "^(?!/)(?!.*//).+(?<!/)$",
                                    "mode": "fullmatch",
                                }
                            }
                        }
                    },
                ]
            },
            "set": [
                {
                    "set_var": {
                        "scope": "global",
                        "name": "ISSUE_21_LEVEL",
                        "value": {"literal": "error"},
                    }
                },
                {
                    "set_var": {
                        "scope": "global",
                        "name": "ISSUE_21",
                        "value": {"literal": "ORGNLMGSID_FIN_INVALID"},
                    }
                },
            ],
            "then": {"value": {"literal": ""}},
        },
        # 3) Caso OK: FIN válido -> poblar (16x, truncando con '+')
        {
            "when": {
                "exists": {
                    "regex": {
                        "value": {
                            "trim": {
                                "value": {
                                    "xml": ".Document.FIToFIPmtStsRpt.TxInfAndSts.OrgnlGrpInf.OrgnlMsgId"
                                }
                            }
                        },
                        "pattern": "^(?!/)(?!.*//).+(?<!/)$",
                        "mode": "fullmatch",
                    }
                }
            },
            "then": {
                "value": {
                    "truncate": {
                        "value": {
                            "regex": {
                                "value": {
                                    "trim": {
                                        "value": {
                                            "xml": ".Document.FIToFIPmtStsRpt.TxInfAndSts.OrgnlGrpInf.OrgnlMsgId"
                                        }
                                    }
                                },
                                "pattern": "^(?!/)(?!.*//).+(?<!/)$",
                                "mode": "fullmatch",
                            }
                        },
                        "max": 16,
                        "keep": 15,
                        "suffix": "+",
                    }
                }
            },
        },
    ]
}


fields_spec_79 = {
    "79": [
        {
            "when": {
                "exists": {
                    "xml_all": ".Document.FIToFIPmtStsRpt.TxInfAndSts.StsRsnInf.AddtlInf"
                }
            },
            "then": {
                "value": {
                    "xml_all": ".Document.FIToFIPmtStsRpt.TxInfAndSts.StsRsnInf.AddtlInf"
                }
            },
        }
    ]
}

# -----------------------------------------------------------------------
# Ejecución mínima

# -----------------------------------------------------------------------
if __name__ == "__main__":
    from datetime import datetime
    from pathlib import Path

    out_lines = []

    # {1}{2} en una sola línea
    header_12 = build_header_12(ubicationEntry)

    # Contexto global compartido para todas las reglas
    globals_ctx = {}

    # :20:
    fields_20 = build_fields(ubicationEntry, fields_spec_20, global_vars=globals_ctx)
    if fields_20.get("20"):
        out_lines.append(f":20:{fields_20['20'][0]}")

    # :21:
    fields_21 = build_fields(ubicationEntry, fields_spec_21, global_vars=globals_ctx)
    if fields_21.get("21"):
        out_lines.append(f":21:{fields_21['21'][0]}")

    # :79:
    fields_79 = build_fields(ubicationEntry, fields_spec_79, global_vars=globals_ctx)
    lines_79 = [ln for ln in fields_79.get("79", []) if str(ln).strip() != ""]
    dbg("FINAL 79 lines:", lines_79)
    if lines_79:
        out_lines.append(f":79:{lines_79[0]}")
        for ln in lines_79[1:]:
            out_lines.append(ln)
    else:
        dbg(
            "79 NO GENERADO - revisa las trazas anteriores para ver qué condición falló"
        )

    # === escribir archivo ===
    ts = datetime.now().strftime("%Y%m%d%H%M%S")  # AAAAMMDDHHSS
    fname = f"MT299_{ts}.txt"
    out_path = Path(ubicationDestiny) / fname
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mt = finalize_mt_message(build_header_12(ubicationEntry), out_lines)
    out_path.write_bytes(mt.encode("utf-8"))
    print(f"{mt}")
