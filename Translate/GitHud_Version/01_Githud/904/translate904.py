import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Callable
import pandas as pd
import json
import os
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

# === DEBUG ===
DEBUG = False
def dbg(*a):
    if DEBUG:
        print("[DBG]", *a)

# -----------------------------------------------------------------------
# Archivos de entrada 
# -----------------------------------------------------------------------
traslateId = "904"
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

    t = (elem.text or "").strip("\n\r\t")
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

# -----------------------------------------------------------------------
# Normalización de strings (evita falsos positivos con espacios Unicode)
# -----------------------------------------------------------------------
def _norm_ws(v: Any) -> str:
    """Normaliza espacios/controles típicos en textos XML.

    Problema observado: algunos XML traen NBSP (\u00A0) u otros espacios
    Unicode al final de los textos. Python `strip()` no siempre elimina esos
    caracteres, lo que hace que condiciones como `substr != ""` se activen y
    se impriman líneas de continuación vacías (p.ej. `//`).
    """
    if v is None:
        return ""
    s = str(v)
    # espacios Unicode comunes
    s = s.replace("\u00A0", " ").replace("\u2007", " ").replace("\u202F", " ")
    # marcas/carácteres invisibles
    s = s.replace("\u200B", "").replace("\uFEFF", "")
    return s.strip()


def _is_effectively_empty(v: Any) -> bool:
    s = _norm_ws(v)
    # `//` solo nunca es un valor útil en campos SWIFT; es un artefacto.
    return s == "" or s == "//"



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

TZ_COLOMBIA = ZoneInfo("America/Bogota")

def format_hhmm(dt_str: str) -> str:
    """Devuelve HHMM (sin offset) a partir de un ISO8601."""
    dt = _parse_iso_dt_norm(dt_str)
    return dt.strftime("%H%M") if dt else ""


def format_yymmdd(dt_str: str) -> str:
    """Devuelve YYMMDD (sin offset) a partir de un ISO8601."""
    dt = _parse_iso_dt_norm(dt_str)
    return dt.strftime("%y%m%d") if dt else ""

def format_yymmddhhmm(_: str = "", tz: ZoneInfo = TZ_COLOMBIA) -> str:
    # Siempre usa la fecha/hora del momento de procesamiento en la zona indicada
    dt = datetime.now(tz)
    return dt.strftime("%y%m%d%H%M")


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
        elif "key" in seg:
            val = _get_by_suffix(dot, seg["key"]) or ""

            if "post" in seg:
                post_fn_name = seg["post"]
                if post_fn_name in block_post_processors:
                    val = block_post_processors[post_fn_name](val)
                    dbg(
                        f"POST-PROCESS: {post_fn_name}({_get_by_suffix(dot, seg['key'])}) -> {val}"
                    )

            if "pad" in seg:
                val = _padx(val, int(seg["pad"]), seg.get("fill", "X"))

            parts.append(val)
        else:
            parts.append("")

    return "{" + f"{block_no}:" + "".join(parts) + "}"


# Especificaciones de Bloques-----------------------------------------------------------
spec_bloque1_A = [
    {"fixed": "F21"},
    {
        "key": ".Document.FICdtTrf.CdtTrfTxInf.InstdAgt.FinInstnId.BICFI",
        "post": "bic11",
    },
    {"var": "HDR_SESS_SEQ", "pad": 10, "fill": "0"},
]
spec_bloque1_B = [
    {"fixed": "F01"},
    {
        "key": ".Document.FICdtTrf.CdtTrfTxInf.InstdAgt.FinInstnId.BICFI",
        "post": "bic11",
    },
    {"var": "HDR_SESS_SEQ", "pad": 10, "fill": "0"},
]
spec_bloque4 = [ 
    {"fixed": "{177:"}, 
    {"var": "HDR_YYMMDDHHMM"}, 
    {"fixed": "}{451:1}"}, 
]
spec_bloque2 = [
    {"fixed": "O202"},
    {"var": "HDR_HHMM"},
    {"var": "HDR_YYMMDD"},
    {
        "key": ".Document.FICdtTrf.CdtTrfTxInf.InstgAgt.FinInstnId.BICFI",
        "pad": 11,
        "fill": "X",
    },
    {"fixed": "X0000000000"},
    {"var": "HDR_YYMMDD"},
    {"fixed": "0000N"},
]
spec_bloque3 = [
    {
        "key": ".Document.FICdtTrf.CdtTrfTxInf.PmtTpInf.SvcLvl.Cd",
        "post": "svc_level_block3_111", 
    },
    {"fixed": "{121:"},
    {"key": ".Document.FICdtTrf.CdtTrfTxInf.PmtId.UETR"},
    {"fixed": "}"},
]


def build_header_12(entry_dir: str) -> str:
    dot = load_latest_dotmap(entry_dir)

    # Timestamp para cabecera (orden de preferencia)
    hdr_dt = (
        _get_by_suffix(dot, ".AppHdr.CreDtTm")
        or _get_by_suffix(dot, ".Document.FICdtTrf.GrpHdr.CreDtTm")
        or _get_by_suffix(dot, ".Document.FICdtTrf.CdtTrfTxInf.SttlmTmIndctn.DtTm")
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

    uetr = _get_by_suffix(dot, ".Document.FICdtTrf.CdtTrfTxInf.PmtId.UETR")
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



def build_block3_111_from_svc_level(raw: str) -> str:
    code = (raw or "").strip()

    if not code:
        dbg(
            "ERROR 111: SvcLvl/Cd ausente o vacío; "
            "Block 3 no imprimirá {111:...}, solo {121:...}"
        )
        return ""

    # Debe ser 4 alfanuméricos
    if not re.fullmatch(r"[A-Za-z0-9]{4}", code):
        dbg(
            f"ERROR 111: SvcLvl/Cd inválido {code!r}; "
            "debe cumplir 4!c (4 caracteres alfanuméricos). "
            "Block 3 no imprimirá {111:...}, solo {121:...}"
        )
        return ""

    code = code.upper()

    # Tabla según la imagen (solo los que sí se traducen)
    svc_to_111 = {
        "G001": "001",
        "G002": "002",
        "G003": "003",
        "G004": "004",
        "G005": "005",
        "G006": "006",
        "G007": "007",
        "G008": "008",
        "G009": "009",
    }

    translated = svc_to_111.get(code)

    # Si el código no tiene traducción (N/A), no se imprime {111:...}
    if translated is None:
        dbg(
            f"INFO 111: SvcLvl/Cd {code!r} no tiene traducción a 111 "
            "(tabla indica N/A); Block 3 no imprimirá {111:...}, solo {121:...}"
        )
        return ""

    # Construye el campo 111 ya con el valor traducido
    return f"{{111:{translated}}}"



block_post_processors = {
    "bic11": normalize_bic11,
    "svc_level_block3_111": build_block3_111_from_svc_level,  
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

    ls = _norm_ws(left)  # normaliza izquierda (incluye NBSP)
    if op == "=":
        rs = _norm_ws(right)
        res = ls == rs
        dbg(f"COND '='  left={ls!r} right={rs!r} -> {res}")
        return res
    if op == "!=":
        rs = _norm_ws(right)
        res = ls != rs
        dbg(f"COND '!=' left={ls!r} right={rs!r} -> {res}")
        return res
    if op == "in":
        if isinstance(right, list):
            rlist = [_norm_ws(x) for x in right]  # normaliza lista
        else:
            rlist = [_norm_ws(s) for s in str(right).split(",")]
        res = ls in rlist
        dbg(f"COND 'in' left={ls!r} list_len={len(rlist)} sample={rlist[:10]} -> {res}")
        return res
    if op == "not in":
        if isinstance(right, list):
            rlist = [_norm_ws(x) for x in right]
        else:
            rlist = [_norm_ws(s) for s in str(right).split(",")]
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
        norm = [_norm_ws(v) for v in vals if _norm_ws(v) != ""]
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


def _collect_then_lines(dot, ctx, then_obj):
    lines = []
    if "lines" in then_obj:
        for vs in then_obj["lines"]:
            val = _eval_value(dot, ctx, vs)
            # aplanar si es lista
            if isinstance(val, list):
                lines.extend([str(x) for x in val if not _is_effectively_empty(x)])
            else:
                lines.append(str(val))
    elif "value" in then_obj:
        val = _eval_value(dot, ctx, then_obj["value"])
        # aplanar si es lista
        if isinstance(val, list):
            lines.extend([str(x) for x in val if not _is_effectively_empty(x)])
        else:
            lines.append(str(val))
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

        custom_keys = {"global_is_set", "global_equals", "not", "any", "all"}

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

        ok_not = True
        if "not" in cond_local:
            ok_not = not _eval_logic_with_globals(
                dot_local, ctx_local, cond_local["not"]
            )

        final_ok = ok_base and ok_gset and ok_geq and ok_not
        dbg(
            f"_eval_logic_with_globals -> base={ok_base} gset={ok_gset} geq={ok_geq} not={ok_not} => FINAL={final_ok} | globals={global_vars} | cond={cond_local}"
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
                produced = _collect_then_lines(dot, ctx, rule.get("then", {}))
                produced = [ln for ln in produced if not _is_effectively_empty(ln)]
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
                produced = _collect_then_lines(dot, ctx, rule.get("then", {}))
                produced = [ln for ln in produced if not _is_effectively_empty(ln)]
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

fields_spec_23B = {"23B": {"mode": "append", "rules": []}}
fields_spec_23E = {"23E": {"mode": "append", "rules": []}}
fields_spec_33B = {"33B": {"mode": "append", "rules": []}}
fields_spec_36 = {"36": {"mode": "append", "rules": []}}
fields_spec_50A = {"50A": {"mode": "append", "rules": []}}
fields_spec_50F = {"50F": {"mode": "append", "rules": []}}
fields_spec_52C = {"52C": {"mode": "append", "rules": []}}
fields_spec_55A = {"55A": {"mode": "append", "rules": []}}
fields_spec_55D = {"55D": {"mode": "append", "rules": []}}
fields_spec_56C = {"56C": {"mode": "append", "rules": []}}
fields_spec_57C = {"57C": {"mode": "append", "rules": []}}
fields_spec_59 = {"59": {"mode": "append", "rules": []}}
fields_spec_59F = {"59F": {"mode": "append", "rules": []}}
fields_spec_70 = {"70": {"mode": "append", "rules": []}}
fields_spec_71A = {"71A": {"mode": "append", "rules": []}}
fields_spec_71F = {"71F": {"mode": "append", "rules": []}}
fields_spec_71G = {"71G": {"mode": "append", "rules": []}}

fields_spec_20 = {
    "20": {
        "mode": "append",
        "rules": [
            {
                "when": {
                    "all": [
                        {
                            "op": "!=",
                            "left": {
                                "xml": ".Document.FICdtTrf.CdtTrfTxInf.PmtId.InstrId"
                            },
                            "right": "",
                        }
                    ]
                },
                "then": {
                    "value": {"xml": ".Document.FICdtTrf.CdtTrfTxInf.PmtId.InstrId"}
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_20",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
                "line_no": 1,
            },
            # Reglas anteriores
            {  # 1) BizMsgIdr
                "when": {
                    "all": [
                        {"op": "!=", "left": {"xml": ".AppHdr.BizMsgIdr"}, "right": ""}
                    ]
                },
                "then": {
                    "value": {
                        "substr": {
                            "value": {"xml": ".AppHdr.BizMsgIdr"},
                            "start": 0,
                            "len": 16,
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_20",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
                "line_no": 2,
            },
            {  # 2) GrpHdr.MsgId (soporta FICdtTrf y FIToFICstmrCdtTrf)
                "when": {
                    "not": {"global_is_set": "has_20"},
                    "any": [
                        {
                            "op": "!=",
                            "left": {"xml": ".Document.FICdtTrf.GrpHdr.MsgId"},
                            "right": "",
                        },
                        {
                            "op": "!=",
                            "left": {"xml": ".Document.FIToFICstmrCdtTrf.GrpHdr.MsgId"},
                            "right": "",
                        },
                    ],
                },
                "then": {
                    "value": {
                        "substr": {
                            "value": {
                                "first_non_empty": [
                                    {"xml": ".Document.FICdtTrf.GrpHdr.MsgId"},
                                    {"xml": ".Document.FIToFICstmrCdtTrf.GrpHdr.MsgId"},
                                ]
                            },
                            "start": 0,
                            "len": 16,
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_20",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
                "line_no": 3,
            },
            {  # 3) Fallback con fecha de creación -> YYMMDDHHMMSS (≤16)
                "when": {
                    "not": {"global_is_set": "has_20"},
                    "any": [
                        {
                            "op": "!=",
                            "left": {"xml": ".Document.FICdtTrf.GrpHdr.CreDtTm"},
                            "right": "",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "xml": ".Document.FIToFICstmrCdtTrf.GrpHdr.CreDtTm"
                            },
                            "right": "",
                        },
                    ],
                },
                "then": {
                    "value": {
                        "substr": {
                            "value": {
                                "concat": [
                                    {
                                        "dtfmt": {
                                            "value": {
                                                "first_non_empty": [
                                                    {
                                                        "xml": ".Document.FICdtTrf.GrpHdr.CreDtTm"
                                                    },
                                                    {
                                                        "xml": ".Document.FIToFICstmrCdtTrf.GrpHdr.CreDtTm"
                                                    },
                                                ]
                                            },
                                            "out": "YYMMDD",
                                        }
                                    },
                                    {
                                        "dtfmt": {
                                            "value": {
                                                "first_non_empty": [
                                                    {
                                                        "xml": ".Document.FICdtTrf.GrpHdr.CreDtTm"
                                                    },
                                                    {
                                                        "xml": ".Document.FIToFICstmrCdtTrf.GrpHdr.CreDtTm"
                                                    },
                                                ]
                                            },
                                            "out": "HHMMSS",
                                        }
                                    },
                                ]
                            },
                            "start": 0,
                            "len": 16,
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_20",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
                "line_no": 4,
            },
        ],
    }
}


fields_spec_21 = {
    "21": {
        "mode": "append",
        "rules": [
            {  # 1) EndToEndId inválido → NOTPROVIDED
                "when": {
                    "all": [
                        {"not": {"global_is_set": "has_21"}},
                        {
                            "op": "!=",
                            "left": {
                                "xml": ".Document.FICdtTrf.CdtTrfTxInf.PmtId.EndToEndId"
                            },
                            "right": "",
                        },
                        {
                            "any": [
                                {
                                    "op": "starts_with",
                                    "left": {
                                        "xml": ".Document.FICdtTrf.CdtTrfTxInf.PmtId.EndToEndId"
                                    },
                                    "right": "/",
                                },
                                {
                                    "op": "ends_with",
                                    "left": {
                                        "xml": ".Document.FICdtTrf.CdtTrfTxInf.PmtId.EndToEndId"
                                    },
                                    "right": "/",
                                },
                                {
                                    "op": "contains",
                                    "left": {
                                        "xml": ".Document.FICdtTrf.CdtTrfTxInf.PmtId.EndToEndId"
                                    },
                                    "right": "//",
                                },
                            ]
                        },
                    ]
                },
                "then": {"value": {"literal": "NOTPROVIDED"}},
                "set": [
                    {
                        "set_var": {
                            "name": "has_21",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
                "line_no": 1,
            },
            {  # 2) EndToEndId normal → usarlo
                "when": {
                    "all": [
                        {"not": {"global_is_set": "has_21"}},
                        {
                            "op": "!=",
                            "left": {
                                "xml": ".Document.FICdtTrf.CdtTrfTxInf.PmtId.EndToEndId"
                            },
                            "right": "",
                        },
                        {
                            "not": {
                                "any": [
                                    {
                                        "op": "starts_with",
                                        "left": {
                                            "xml": ".Document.FICdtTrf.CdtTrfTxInf.PmtId.EndToEndId"
                                        },
                                        "right": "/",
                                    },
                                    {
                                        "op": "ends_with",
                                        "left": {
                                            "xml": ".Document.FICdtTrf.CdtTrfTxInf.PmtId.EndToEndId"
                                        },
                                        "right": "/",
                                    },
                                    {
                                        "op": "contains",
                                        "left": {
                                            "xml": ".Document.FICdtTrf.CdtTrfTxInf.PmtId.EndToEndId"
                                        },
                                        "right": "//",
                                    },
                                ]
                            }
                        },
                    ]
                },
                "then": {
                    "value": {
                        "substr": {
                            "value": {
                                "xml": ".Document.FICdtTrf.CdtTrfTxInf.PmtId.EndToEndId"
                            },
                            "start": 0,
                            "len": 16,
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_21",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
                "line_no": 2,
            },
            {  # 3) Fallback a InstrId (solo si EndToEndId vacío)
                "when": {
                    "all": [
                        {"not": {"global_is_set": "has_21"}},
                        {
                            "op": "=",
                            "left": {
                                "xml": ".Document.FICdtTrf.CdtTrfTxInf.PmtId.EndToEndId"
                            },
                            "right": "",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "xml": ".Document.FICdtTrf.CdtTrfTxInf.PmtId.InstrId"
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "substr": {
                            "value": {
                                "xml": ".Document.FICdtTrf.CdtTrfTxInf.PmtId.InstrId"
                            },
                            "start": 0,
                            "len": 16,
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_21",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
                "line_no": 4,
            },
            {  # 4) Fallback adicional a ClrSysRef o TxId
                "when": {
                    "all": [
                        {"not": {"global_is_set": "has_21"}},
                        {
                            "any": [
                                {
                                    "op": "!=",
                                    "left": {
                                        "xml": ".Document.FICdtTrf.CdtTrfTxInf.PmtId.ClrSysRef"
                                    },
                                    "right": "",
                                },
                                {
                                    "op": "!=",
                                    "left": {
                                        "xml": ".Document.FICdtTrf.CdtTrfTxInf.PmtId.TxId"
                                    },
                                    "right": "",
                                },
                            ]
                        },
                    ]
                },
                "then": {
                    "value": {
                        "substr": {
                            "value": {
                                "first_non_empty": [
                                    {
                                        "xml": ".Document.FICdtTrf.CdtTrfTxInf.PmtId.ClrSysRef"
                                    },
                                    {
                                        "xml": ".Document.FICdtTrf.CdtTrfTxInf.PmtId.TxId"
                                    },
                                ]
                            },
                            "start": 0,
                            "len": 16,
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_21",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
                "line_no": 5,
            },
        ],
    }
}

fields_spec_13C = {
    "13C": {
        "mode": "append",
        "rules": [
            {  # 1) SNDTIME (Send time indication)
                "when": {
                    "all": [
                        {
                            "op": "!=",
                            "left": {
                                "xml": ".Document.FICdtTrf.CdtTrfTxInf.SttlmTmIndctn.DtTm"
                            },
                            "right": "",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "xml": ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.SttlmTmIndctn.DtTm"
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "concat": [
                            {"literal": "/SNDTIME/"},
                            {
                                "dtfmt": {
                                    "value": {
                                        "first_non_empty": [
                                            {
                                                "xml": ".Document.FICdtTrf.CdtTrfTxInf.SttlmTmIndctn.DtTm"
                                            },
                                            {
                                                "xml": ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.SttlmTmIndctn.DtTm"
                                            },
                                        ]
                                    },
                                    "out": "HHMM±HHMM",
                                }
                            },
                        ]
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_13C",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
                "line_no": 1,
            },
            {  # 2) RNCTIME (Receive time indication)
                "when": {
                    "all": [
                        {
                            "op": "!=",
                            "left": {
                                "xml": ".Document.FICdtTrf.CdtTrfTxInf.SttlmTmIndctn.CdtDtTm"
                            },
                            "right": "",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "xml": ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.SttlmTmIndctn.CdtDtTm"
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "concat": [
                            {"literal": "/RNCTIME/"},
                            {
                                "dtfmt": {
                                    "value": {
                                        "first_non_empty": [
                                            {
                                                "xml": ".Document.FICdtTrf.CdtTrfTxInf.SttlmTmIndctn.CdtDtTm"
                                            },
                                            {
                                                "xml": ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.SttlmTmIndctn.CdtDtTm"
                                            },
                                        ]
                                    },
                                    "out": "HHMM±HHMM",
                                }
                            },
                        ]
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_13C",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
                "line_no": 2,
            },
            {  # 3) CLSTIME (Close time indication, opcional)
                "when": {
                    "all": [
                        {
                            "op": "!=",
                            "left": {
                                "xml": ".Document.FICdtTrf.CdtTrfTxInf.SttlmTmReq.CLSTm"
                            },
                            "right": "",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "xml": ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.SttlmTmReq.CLSTm"
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "concat": [
                            {"literal": "/CLSTIME/"},
                            {
                                "dtfmt": {
                                    "value": {
                                        "first_non_empty": [
                                            {
                                                "xml": ".Document.FICdtTrf.CdtTrfTxInf.SttlmTmReq.CLSTm"
                                            },
                                            {
                                                "xml": ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.SttlmTmReq.CLSTm"
                                            },
                                        ]
                                    },
                                    "out": "HHMM±HHMM",
                                }
                            },
                        ]
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_13C",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
                "line_no": 3,
            },
            {  # 4) TILTIME (Till time indication, opcional)
                "when": {
                    "all": [
                        {
                            "op": "!=",
                            "left": {
                                "xml": ".Document.FICdtTrf.CdtTrfTxInf.SttlmTmReq.TillTm"
                            },
                            "right": "",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "xml": ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.SttlmTmReq.TillTm"
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "concat": [
                            {"literal": "/TILTIME/"},
                            {
                                "dtfmt": {
                                    "value": {
                                        "first_non_empty": [
                                            {
                                                "xml": ".Document.FICdtTrf.CdtTrfTxInf.SttlmTmReq.TillTm"
                                            },
                                            {
                                                "xml": ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.SttlmTmReq.TillTm"
                                            },
                                        ]
                                    },
                                    "out": "HHMM±HHMM",
                                }
                            },
                        ]
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_13C",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
                "line_no": 4,
            },
            {  # 5) FROTIME (From time indication, opcional)
                "when": {
                    "all": [
                        {
                            "op": "!=",
                            "left": {
                                "xml": ".Document.FICdtTrf.CdtTrfTxInf.SttlmTmReq.FrTm"
                            },
                            "right": "",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "xml": ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.SttlmTmReq.FrTm"
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "concat": [
                            {"literal": "/FROTIME/"},
                            {
                                "dtfmt": {
                                    "value": {
                                        "first_non_empty": [
                                            {
                                                "xml": ".Document.FICdtTrf.CdtTrfTxInf.SttlmTmReq.FrTm"
                                            },
                                            {
                                                "xml": ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.SttlmTmReq.FrTm"
                                            },
                                        ]
                                    },
                                    "out": "HHMM±HHMM",
                                }
                            },
                        ]
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_13C",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
                "line_no": 5,
            },
            {  # 6) REJTIME (Reject time indication, opcional)
                "when": {
                    "all": [
                        {
                            "op": "!=",
                            "left": {
                                "xml": ".Document.FICdtTrf.CdtTrfTxInf.SttlmTmReq.RjctTm"
                            },
                            "right": "",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "xml": ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.SttlmTmReq.RjctTm"
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "concat": [
                            {"literal": "/REJTIME/"},
                            {
                                "dtfmt": {
                                    "value": {
                                        "first_non_empty": [
                                            {
                                                "xml": ".Document.FICdtTrf.CdtTrfTxInf.SttlmTmReq.RjctTm"
                                            },
                                            {
                                                "xml": ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.SttlmTmReq.RjctTm"
                                            },
                                        ]
                                    },
                                    "out": "HHMM±HHMM",
                                }
                            },
                        ]
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_13C",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
                "line_no": 6,
            },
        ],
    }
}

fields_spec_32A = {
    "32A": {
        "mode": "append",
        "rules": [
            {
                "when": {
                    "all": [
                        {
                            "op": "!=",
                            "left": {
                                "xml": ".Document.FICdtTrf.CdtTrfTxInf.IntrBkSttlmDt"
                            },
                            "right": "",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "xml": ".Document.FICdtTrf.CdtTrfTxInf.IntrBkSttlmAmt[@Ccy]"
                            },
                            "right": "",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "xml": ".Document.FICdtTrf.CdtTrfTxInf.IntrBkSttlmAmt"
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "concat": [
                            # Fecha YYYY-MM-DD → YYMMDD
                            {
                                "substr": {
                                    "value": {
                                        "xml": ".Document.FICdtTrf.CdtTrfTxInf.IntrBkSttlmDt"
                                    },
                                    "start": 2,
                                    "len": 2,
                                }
                            },  # YY
                            {
                                "substr": {
                                    "value": {
                                        "xml": ".Document.FICdtTrf.CdtTrfTxInf.IntrBkSttlmDt"
                                    },
                                    "start": 5,
                                    "len": 2,
                                }
                            },  # MM
                            {
                                "substr": {
                                    "value": {
                                        "xml": ".Document.FICdtTrf.CdtTrfTxInf.IntrBkSttlmDt"
                                    },
                                    "start": 8,
                                    "len": 2,
                                }
                            },  # DD
                            {
                                "substr": {
                                    "value": {
                                        "xml": ".Document.FICdtTrf.CdtTrfTxInf.IntrBkSttlmAmt[@Ccy]"
                                    },
                                    "start": 0,
                                    "len": 3,
                                }
                            },
                            {
                                "substr": {
                                    "value": {
                                        "numfmt": {
                                            "value": {
                                                "xml": ".Document.FICdtTrf.CdtTrfTxInf.IntrBkSttlmAmt"
                                            }
                                        }
                                    },
                                    "start": 0,
                                    "len": 15,
                                }
                            },
                        ]
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_32A",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
                "line_no": 1,
            }
        ],
    }
}

fields_spec_52A = {
    # :52A: Ordering Institution (O)
    "52A": {
        "mode": "append",
        "rules": [
            {
                # 2) FICdtTrf - Othr.Id del beneficiario + BIC del beneficiario
                #    (caso de tu XML: /74859632100 + CAFECOBB)
                "when": {
                    "all": [
                        {"not": {"global_is_set": "has_52A"}},
                        # BIC del beneficiario presente
                        {
                            "op": "!=",
                            "left": {
                                "xml": ".Document.FICdtTrf.CdtTrfTxInf.Dbtr.FinInstnId.BICFI"
                            },
                            "right": "",
                        },
                        # Cuenta del beneficiario (Othr.Id) presente
                        {
                            "op": "!=",
                            "left": {
                                "xml": ".Document.FICdtTrf.CdtTrfTxInf.DbtrAcct.Id.Othr.Id"
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "lines": [
                        # Línea 1: Party Identifier = /[Othr.Id]
                        {
                            "concat": [
                                {"literal": "/"},
                                {
                                    "xml": ".Document.FICdtTrf.CdtTrfTxInf.DbtrAcct.Id.Othr.Id"
                                },
                            ]
                        },
                        # Línea 2: Identifier Code = BIC
                        {"xml": ".Document.FICdtTrf.CdtTrfTxInf.Dbtr.FinInstnId.BICFI"},
                    ]
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_52A",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
                "line_no": 2,
            },
            {  # 1) Opción con IBAN y BIC (se deja igual)
                "when": {
                    "all": [
                        {
                            "op": "!=",
                            "left": {
                                "xml": ".Document.FICdtTrf.CdtTrfTxInf.DbtrAcct.Id.IBAN"
                            },
                            "right": "",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "xml": ".Document.FICdtTrf.CdtTrfTxInf.Dbtr.FinInstnId.BICFI"
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "concat": [
                            {"literal": "/"},
                            {
                                "substr": {
                                    "value": {
                                        "xml": ".Document.FICdtTrf.CdtTrfTxInf.DbtrAcct.Id.IBAN"
                                    },
                                    "start": 0,
                                    "len": 34,
                                }
                            },
                            {"literal": "\n"},
                            {
                                "xml": ".Document.FICdtTrf.CdtTrfTxInf.Dbtr.FinInstnId.BICFI"
                            },
                        ]
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_52A",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
                "line_no": 1,
            },
            {  # 2) Solo BIC (sin IBAN)
                "when": {
                    "not": {"global_is_set": "has_52A"},
                    "all": [
                        {
                            "op": "!=",
                            "left": {
                                "xml": ".Document.FICdtTrf.CdtTrfTxInf.Dbtr.FinInstnId.BICFI"
                            },
                            "right": "",
                        }
                    ],
                },
                "then": {
                    "value": {
                        "substr": {
                            "value": {
                                "xml": ".Document.FICdtTrf.CdtTrfTxInf.Dbtr.FinInstnId.BICFI"
                            },
                            "start": 0,
                            "len": 11,
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_52A",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
                "line_no": 2,
            },
            {  # 3) Solo IBAN (como Identifier Code, siguiendo la regla de IBAN)
                "when": {
                    "not": {"global_is_set": "has_52A"},
                    "all": [
                        {
                            "op": "!=",
                            "left": {
                                "xml": ".Document.FICdtTrf.CdtTrfTxInf.DbtrAcct.Id.IBAN"
                            },
                            "right": "",
                        }
                    ],
                },
                "then": {
                    "value": {
                        "substr": {
                            "value": {
                                "xml": ".Document.FICdtTrf.CdtTrfTxInf.DbtrAcct.Id.IBAN"
                            },
                            "start": 0,
                            "len": 34,
                        }
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_52A",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
                "line_no": 3,
            },
            {  # 4) Fallback: NOTPROVIDED si no hay nada mapeable
                "when": {"not": {"global_is_set": "has_52A"}},
                "then": {"value": {"literal": "NOTPROVIDED"}},
                "set": [
                    {
                        "set_var": {
                            "name": "has_52A",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
                "line_no": 4,
            },
        ],
    }
}

fields_spec_52D = {
    "52D": {
        "mode": "append",
        "rules": [
            {
                "when": {
                    "not": {"global_is_set": "has_52A"},
                    "all": [
                        {
                            "op": "=",
                            "left": {
                                "xml": ".Document.FICdtTrf.CdtTrfTxInf.Dbtr.FinInstnId.BICFI"
                            },
                            "right": "",
                        },
                        {
                            "any": [
                                {
                                    "op": "!=",
                                    "left": {
                                        "xml": ".Document.FICdtTrf.CdtTrfTxInf.Dbtr.FinInstnId.Nm"
                                    },
                                    "right": "",
                                },
                                {
                                    "op": "!=",
                                    "left": {
                                        "xml": ".Document.FICdtTrf.CdtTrfTxInf.Dbtr.FinInstnId.PstlAdr.AdrLine"
                                    },
                                    "right": "",
                                },
                            ]
                        },
                    ],
                },
                "then": {
                    "lines": [
                        # 1IBAN (si existe)
                        {
                            "optional": {
                                "when": {
                                    "op": "!=",
                                    "left": {
                                        "xml": ".Document.FICdtTrf.CdtTrfTxInf.DbtrAcct.Id.IBAN"
                                    },
                                    "right": "",
                                },
                                "value": {
                                    "concat": [
                                        {"literal": "/"},
                                        {
                                            "substr": {
                                                "value": {
                                                    "xml": ".Document.FICdtTrf.CdtTrfTxInf.DbtrAcct.Id.IBAN"
                                                },
                                                "start": 0,
                                                "len": 34,
                                            }
                                        },
                                    ]
                                },
                            }
                        },
                        # 2Nombre de la institución
                        {
                            "optional": {
                                "when": {
                                    "op": "!=",
                                    "left": {
                                        "xml": ".Document.FICdtTrf.CdtTrfTxInf.Dbtr.FinInstnId.Nm"
                                    },
                                    "right": "",
                                },
                                "value": {
                                    "substr": {
                                        "value": {
                                            "xml": ".Document.FICdtTrf.CdtTrfTxInf.Dbtr.FinInstnId.Nm"
                                        },
                                        "start": 0,
                                        "len": 35,
                                    }
                                },
                            }
                        },
                        # 3Línea de dirección (AdrLine)
                        {
                            "optional": {
                                "when": {
                                    "op": "!=",
                                    "left": {
                                        "xml": ".Document.FICdtTrf.CdtTrfTxInf.Dbtr.FinInstnId.PstlAdr.AdrLine"
                                    },
                                    "right": "",
                                },
                                "value": {
                                    "substr": {
                                        "value": {
                                            "xml": ".Document.FICdtTrf.CdtTrfTxInf.Dbtr.FinInstnId.PstlAdr.AdrLine"
                                        },
                                        "start": 0,
                                        "len": 35,
                                    }
                                },
                            }
                        },
                        # 4Ciudad (TwnNm)
                        {
                            "optional": {
                                "when": {
                                    "op": "!=",
                                    "left": {
                                        "xml": ".Document.FICdtTrf.CdtTrfTxInf.Dbtr.FinInstnId.PstlAdr.TwnNm"
                                    },
                                    "right": "",
                                },
                                "value": {
                                    "substr": {
                                        "value": {
                                            "xml": ".Document.FICdtTrf.CdtTrfTxInf.Dbtr.FinInstnId.PstlAdr.TwnNm"
                                        },
                                        "start": 0,
                                        "len": 35,
                                    }
                                },
                            }
                        },
                        # 5País (Ctry)
                        {
                            "optional": {
                                "when": {
                                    "op": "!=",
                                    "left": {
                                        "xml": ".Document.FICdtTrf.CdtTrfTxInf.Dbtr.FinInstnId.PstlAdr.Ctry"
                                    },
                                    "right": "",
                                },
                                "value": {
                                    "substr": {
                                        "value": {
                                            "xml": ".Document.FICdtTrf.CdtTrfTxInf.Dbtr.FinInstnId.PstlAdr.Ctry"
                                        },
                                        "start": 0,
                                        "len": 35,
                                    }
                                },
                            }
                        },
                    ]
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_52D",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
                "line_no": 1,
            }
        ],
    }
}

fields_spec_53A = {
    "53A": {
        "mode": "append",
        "rules": [
            {
                # :53A se genera solo si:
                #   - no existe ya 53A ni 53B
                #   - SttlmMtd = INGA o INDA
                #   - SttlmAcct está ausente (sin IBAN ni Othr.Id)
                #   - hay un BIC válido en algún InstrForNxtAgt/InstrInf con /FIN53/
                "when": {
                    "all": [
                        {"not": {"global_is_set": "has_53A"}},
                        {"not": {"global_is_set": "has_53B"}},
                        # SettlementMethod = INGA o INDA
                        {
                            "any": [
                                {
                                    "op": "=",
                                    "left": {
                                        "xml": ".Document.FICdtTrf.GrpHdr.SttlmInf.SttlmMtd"
                                    },
                                    "right": "INGA",
                                },
                                {
                                    "op": "=",
                                    "left": {
                                        "xml": ".Document.FICdtTrf.GrpHdr.SttlmInf.SttlmMtd"
                                    },
                                    "right": "INDA",
                                },
                            ]
                        },
                        # SettlementAccount ausente (sin IBAN ni Othr.Id)
                        {
                            "all": [
                                {
                                    "op": "=",
                                    "left": {
                                        "xml": ".Document.FICdtTrf.GrpHdr.SttlmInf.SttlmAcct.Id.IBAN"
                                    },
                                    "right": "",
                                },
                                {
                                    "op": "=",
                                    "left": {
                                        "xml": ".Document.FICdtTrf.GrpHdr.SttlmInf.SttlmAcct.Id.Othr.Id"
                                    },
                                    "right": "",
                                },
                            ]
                        },
                        # Existe un BIC en /FIN53/ (regex_group no vacío)
                        {
                            "op": "!=",
                            "left": {
                                "regex_group": {
                                    "text": {
                                        "concat": [
                                            {
                                                "xml": ".Document.FICdtTrf.CdtTrfTxInf.InstrForNxtAgt[0].InstrInf"
                                            },
                                            {"literal": " "},
                                            {
                                                "xml": ".Document.FICdtTrf.CdtTrfTxInf.InstrForNxtAgt[1].InstrInf"
                                            },
                                            {"literal": " "},
                                            {
                                                "xml": ".Document.FICdtTrf.CdtTrfTxInf.InstrForNxtAgt[2].InstrInf"
                                            },
                                            {"literal": " "},
                                            {
                                                "xml": ".Document.FICdtTrf.CdtTrfTxInf.InstrForNxtAgt[3].InstrInf"
                                            },
                                        ]
                                    },
                                    # Busca /FIN53/ seguido de 8 a 11 caracteres alfanuméricos
                                    "pattern": "/FIN53/([A-Z0-9]{8,11})",
                                    "group": 1,
                                }
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "lines": [
                        # Única línea de 53A: Identifier Code = BIC extraído
                        {
                            "value": {
                                "regex_group": {
                                    "text": {
                                        "concat": [
                                            {
                                                "xml": ".Document.FICdtTrf.CdtTrfTxInf.InstrForNxtAgt[0].InstrInf"
                                            },
                                            {"literal": " "},
                                            {
                                                "xml": ".Document.FICdtTrf.CdtTrfTxInf.InstrForNxtAgt[1].InstrInf"
                                            },
                                            {"literal": " "},
                                            {
                                                "xml": ".Document.FICdtTrf.CdtTrfTxInf.InstrForNxtAgt[2].InstrInf"
                                            },
                                            {"literal": " "},
                                            {
                                                "xml": ".Document.FICdtTrf.CdtTrfTxInf.InstrForNxtAgt[3].InstrInf"
                                            },
                                        ]
                                    },
                                    "pattern": "/FIN53/([A-Z0-9]{8,11})",
                                    "group": 1,
                                }
                            }
                        }
                    ]
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_53A",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
                "line_no": 1,
            }
        ],
    }
}

fields_spec_53B = {
    "53B": {
        "mode": "append",
        "rules": [
            {  # 1) INGA + IBAN (prioridad IBAN, prefijo /C)
                "when": {
                    "all": [
                        {"not": {"global_is_set": "has_53A"}},
                        {"not": {"global_is_set": "has_53B"}},
                        {
                            "op": "=",
                            "left": {
                                "xml": ".Document.FICdtTrf.GrpHdr.SttlmInf.SttlmMtd"
                            },
                            "right": "INGA",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "xml": ".Document.FICdtTrf.GrpHdr.SttlmInf.SttlmAcct.Id.IBAN"
                            },
                            "right": "",
                        },
                        {
                            # NO hay BIC en /FIN53/
                            "op": "=",
                            "left": {
                                "regex_group": {
                                    "text": {
                                        "concat": [
                                            {
                                                "xml": ".Document.FICdtTrf.GrpHdr.InstrForNxtAgt[0].InstrInf"
                                            },
                                            {"literal": " "},
                                            {
                                                "xml": ".Document.FICdtTrf.GrpHdr.InstrForNxtAgt[1].InstrInf"
                                            },
                                            {"literal": " "},
                                            {
                                                "xml": ".Document.FICdtTrf.GrpHdr.InstrForNxtAgt[2].InstrInf"
                                            },
                                            {"literal": " "},
                                            {
                                                "xml": ".Document.FICdtTrf.GrpHdr.InstrForNxtAgt[3].InstrInf"
                                            },
                                        ]
                                    },
                                    "pattern": "/FIN53/([A-Z0-9]{8,11})",
                                    "group": 1,
                                }
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "concat": [
                            {"literal": "/C"},
                            {
                                "substr": {
                                    "value": {
                                        "xml": ".Document.FICdtTrf.GrpHdr.SttlmInf.SttlmAcct.Id.IBAN"
                                    },
                                    "start": 0,
                                    "len": 34,
                                }
                            },
                        ]
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_53B",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
                "line_no": 1,
            },
            {  # 2) INDA + IBAN (prioridad IBAN, sin prefijo /C)
                "when": {
                    "all": [
                        {"not": {"global_is_set": "has_53A"}},
                        {"not": {"global_is_set": "has_53B"}},
                        {
                            "op": "=",
                            "left": {
                                "xml": ".Document.FICdtTrf.GrpHdr.SttlmInf.SttlmMtd"
                            },
                            "right": "INDA",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "xml": ".Document.FICdtTrf.GrpHdr.SttlmInf.SttlmAcct.Id.IBAN"
                            },
                            "right": "",
                        },
                        {
                            "op": "=",
                            "left": {
                                "regex_group": {
                                    "text": {
                                        "concat": [
                                            {
                                                "xml": ".Document.FICdtTrf.GrpHdr.InstrForNxtAgt[0].InstrInf"
                                            },
                                            {"literal": " "},
                                            {
                                                "xml": ".Document.FICdtTrf.GrpHdr.InstrForNxtAgt[1].InstrInf"
                                            },
                                            {"literal": " "},
                                            {
                                                "xml": ".Document.FICdtTrf.GrpHdr.InstrForNxtAgt[2].InstrInf"
                                            },
                                            {"literal": " "},
                                            {
                                                "xml": ".Document.FICdtTrf.GrpHdr.InstrForNxtAgt[3].InstrInf"
                                            },
                                        ]
                                    },
                                    "pattern": "/FIN53/([A-Z0-9]{8,11})",
                                    "group": 1,
                                }
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "concat": [
                            {"literal": "/"},
                            {
                                "substr": {
                                    "value": {
                                        "xml": ".Document.FICdtTrf.GrpHdr.SttlmInf.SttlmAcct.Id.IBAN"
                                    },
                                    "start": 0,
                                    "len": 34,
                                }
                            },
                        ]
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_53B",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
                "line_no": 2,
            },
            {  # 3) INGA + Othr.Id (solo si no se usó IBAN)
                "when": {
                    "all": [
                        {"not": {"global_is_set": "has_53A"}},
                        {"not": {"global_is_set": "has_53B"}},
                        {
                            "op": "=",
                            "left": {
                                "xml": ".Document.FICdtTrf.GrpHdr.SttlmInf.SttlmMtd"
                            },
                            "right": "INGA",
                        },
                        {
                            "op": "=",
                            "left": {
                                "xml": ".Document.FICdtTrf.GrpHdr.SttlmInf.SttlmAcct.Id.IBAN"
                            },
                            "right": "",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "xml": ".Document.FICdtTrf.GrpHdr.SttlmInf.SttlmAcct.Id.Othr.Id"
                            },
                            "right": "",
                        },
                        {
                            "op": "=",
                            "left": {
                                "regex_group": {
                                    "text": {
                                        "concat": [
                                            {
                                                "xml": ".Document.FICdtTrf.GrpHdr.InstrForNxtAgt[0].InstrInf"
                                            },
                                            {"literal": " "},
                                            {
                                                "xml": ".Document.FICdtTrf.GrpHdr.InstrForNxtAgt[1].InstrInf"
                                            },
                                            {"literal": " "},
                                            {
                                                "xml": ".Document.FICdtTrf.GrpHdr.InstrForNxtAgt[2].InstrInf"
                                            },
                                            {"literal": " "},
                                            {
                                                "xml": ".Document.FICdtTrf.GrpHdr.InstrForNxtAgt[3].InstrInf"
                                            },
                                        ]
                                    },
                                    "pattern": "/FIN53/([A-Z0-9]{8,11})",
                                    "group": 1,
                                }
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "concat": [
                            {"literal": "/C"},
                            {
                                "substr": {
                                    "value": {
                                        "xml": ".Document.FICdtTrf.GrpHdr.SttlmInf.SttlmAcct.Id.Othr.Id"
                                    },
                                    "start": 0,
                                    "len": 34,
                                }
                            },
                        ]
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_53B",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
                "line_no": 3,
            },
            {  # 4) INDA + Othr.Id (solo si no se usó IBAN)
                "when": {
                    "all": [
                        {"not": {"global_is_set": "has_53A"}},
                        {"not": {"global_is_set": "has_53B"}},
                        {
                            "op": "=",
                            "left": {
                                "xml": ".Document.FICdtTrf.GrpHdr.SttlmInf.SttlmMtd"
                            },
                            "right": "INDA",
                        },
                        {
                            "op": "=",
                            "left": {
                                "xml": ".Document.FICdtTrf.GrpHdr.SttlmInf.SttlmAcct.Id.IBAN"
                            },
                            "right": "",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "xml": ".Document.FICdtTrf.GrpHdr.SttlmInf.SttlmAcct.Id.Othr.Id"
                            },
                            "right": "",
                        },
                        {
                            "op": "=",
                            "left": {
                                "regex_group": {
                                    "text": {
                                        "concat": [
                                            {
                                                "xml": ".Document.FICdtTrf.GrpHdr.InstrForNxtAgt[0].InstrInf"
                                            },
                                            {"literal": " "},
                                            {
                                                "xml": ".Document.FICdtTrf.GrpHdr.InstrForNxtAgt[1].InstrInf"
                                            },
                                            {"literal": " "},
                                            {
                                                "xml": ".Document.FICdtTrf.GrpHdr.InstrForNxtAgt[2].InstrInf"
                                            },
                                            {"literal": " "},
                                            {
                                                "xml": ".Document.FICdtTrf.GrpHdr.InstrForNxtAgt[3].InstrInf"
                                            },
                                        ]
                                    },
                                    "pattern": "/FIN53/([A-Z0-9]{8,11})",
                                    "group": 1,
                                }
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "concat": [
                            {"literal": "/"},
                            {
                                "substr": {
                                    "value": {
                                        "xml": ".Document.FICdtTrf.GrpHdr.SttlmInf.SttlmAcct.Id.Othr.Id"
                                    },
                                    "start": 0,
                                    "len": 34,
                                }
                            },
                        ]
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_53B",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
                "line_no": 4,
            },
        ],
    }
}

fields_spec_53D = {
    "53D": {
        "mode": "append",
        "rules": [
            {
                "when": {
                    "not": {
                        "any": [
                            {"global_is_set": "has_53A"},
                            {"global_is_set": "has_53B"},
                        ]
                    },
                    "all": [
                        {
                            "op": "=",
                            "left": {
                                "xml": ".Document.FICdtTrf.GrpHdr.SttlmInf.InstgRmbrsmntAgt.FinInstnId.BICFI"
                            },
                            "right": "",
                        },
                        {
                            "any": [
                                {
                                    "op": "!=",
                                    "left": {
                                        "xml": ".Document.FICdtTrf.GrpHdr.SttlmInf.InstgRmbrsmntAgt.FinInstnId.Nm"
                                    },
                                    "right": "",
                                },
                                {
                                    "op": "!=",
                                    "left": {
                                        "xml": ".Document.FICdtTrf.GrpHdr.SttlmInf.InstgRmbrsmntAgt.FinInstnId.PstlAdr.AdrLine"
                                    },
                                    "right": "",
                                },
                                {
                                    "op": "!=",
                                    "left": {
                                        "xml": ".Document.FICdtTrf.GrpHdr.SttlmInf.InstgRmbrsmntAgtAcct.Id.Othr.Id"
                                    },
                                    "right": "",
                                },
                                {
                                    "op": "!=",
                                    "left": {
                                        "xml": ".Document.FICdtTrf.GrpHdr.SttlmInf.InstgRmbrsmntAgtAcct.Id.IBAN"
                                    },
                                    "right": "",
                                },
                            ]
                        },
                    ],
                },
                "then": {
                    "lines": [
                        # 1Party Identifier → prefijo "/"
                        {
                            "optional": {
                                "when": {
                                    "any": [
                                        {
                                            "op": "!=",
                                            "left": {
                                                "xml": ".Document.FICdtTrf.GrpHdr.SttlmInf.InstgRmbrsmntAgtAcct.Id.Othr.Id"
                                            },
                                            "right": "",
                                        },
                                        {
                                            "op": "!=",
                                            "left": {
                                                "xml": ".Document.FICdtTrf.GrpHdr.SttlmInf.InstgRmbrsmntAgtAcct.Id.IBAN"
                                            },
                                            "right": "",
                                        },
                                    ]
                                },
                                "value": {
                                    "concat": [
                                        {"literal": "/"},
                                        {
                                            "first_non_empty": [
                                                {
                                                    "xml": ".Document.FICdtTrf.GrpHdr.SttlmInf.InstgRmbrsmntAgtAcct.Id.Othr.Id"
                                                },
                                                {
                                                    "xml": ".Document.FICdtTrf.GrpHdr.SttlmInf.InstgRmbrsmntAgtAcct.Id.IBAN"
                                                },
                                            ]
                                        },
                                    ]
                                },
                            }
                        },
                        # 2Nombre del corresponsal
                        {
                            "optional": {
                                "when": {
                                    "op": "!=",
                                    "left": {
                                        "xml": ".Document.FICdtTrf.GrpHdr.SttlmInf.InstgRmbrsmntAgt.FinInstnId.Nm"
                                    },
                                    "right": "",
                                },
                                "value": {
                                    "substr": {
                                        "value": {
                                            "xml": ".Document.FICdtTrf.GrpHdr.SttlmInf.InstgRmbrsmntAgt.FinInstnId.Nm"
                                        },
                                        "start": 0,
                                        "len": 35,
                                    }
                                },
                            }
                        },
                        # 3Dirección
                        {
                            "optional": {
                                "when": {
                                    "op": "!=",
                                    "left": {
                                        "xml": ".Document.FICdtTrf.GrpHdr.SttlmInf.InstgRmbrsmntAgt.FinInstnId.PstlAdr.AdrLine"
                                    },
                                    "right": "",
                                },
                                "value": {
                                    "substr": {
                                        "value": {
                                            "xml": ".Document.FICdtTrf.GrpHdr.SttlmInf.InstgRmbrsmntAgt.FinInstnId.PstlAdr.AdrLine"
                                        },
                                        "start": 0,
                                        "len": 35,
                                    }
                                },
                            }
                        },
                        # 4Ciudad
                        {
                            "optional": {
                                "when": {
                                    "op": "!=",
                                    "left": {
                                        "xml": ".Document.FICdtTrf.GrpHdr.SttlmInf.InstgRmbrsmntAgt.FinInstnId.PstlAdr.TwnNm"
                                    },
                                    "right": "",
                                },
                                "value": {
                                    "substr": {
                                        "value": {
                                            "xml": ".Document.FICdtTrf.GrpHdr.SttlmInf.InstgRmbrsmntAgt.FinInstnId.PstlAdr.TwnNm"
                                        },
                                        "start": 0,
                                        "len": 35,
                                    }
                                },
                            }
                        },
                        # 5País
                        {
                            "optional": {
                                "when": {
                                    "op": "!=",
                                    "left": {
                                        "xml": ".Document.FICdtTrf.GrpHdr.SttlmInf.InstgRmbrsmntAgt.FinInstnId.PstlAdr.Ctry"
                                    },
                                    "right": "",
                                },
                                "value": {
                                    "substr": {
                                        "value": {
                                            "xml": ".Document.FICdtTrf.GrpHdr.SttlmInf.InstgRmbrsmntAgt.FinInstnId.PstlAdr.Ctry"
                                        },
                                        "start": 0,
                                        "len": 35,
                                    }
                                },
                            }
                        },
                    ]
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_53D",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
                "line_no": 1,
            }
        ],
    }
}


fields_spec_54A = {
    # :54A: Receiver's Correspondent (O)
    "54A": {
        "mode": "append",
        "rules": [
            {  # 1Opción principal: BIC del corresponsal del receptor
                "when": {
                    "all": [
                        {
                            "op": "!=",
                            "left": {
                                "xml": ".Document.FICdtTrf.GrpHdr.SttlmInf.InstdRmbrsmntAgt.FinInstnId.BICFI"
                            },
                            "right": "",
                        }
                    ]
                },
                "then": {
                    "value": {
                        "xml": ".Document.FICdtTrf.GrpHdr.SttlmInf.InstdRmbrsmntAgt.FinInstnId.BICFI"
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_54A",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
                "line_no": 1,
            },
            {
                "when": {
                    "not": {"global_is_set": "has_54A"},
                    "any": [
                        {
                            "op": "!=",
                            "left": {
                                "xml": ".Document.FICdtTrf.GrpHdr.SttlmInf.InstdRmbrsmntAgtAcct.Id.Othr.Id"
                            },
                            "right": "",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "xml": ".Document.FICdtTrf.GrpHdr.SttlmInf.InstdRmbrsmntAgtAcct.Id.IBAN"
                            },
                            "right": "",
                        },
                    ],
                },
                "then": {
                    "value": {
                        "concat": [
                            {"literal": "/"},
                            {
                                "first_non_empty": [
                                    {
                                        "xml": ".Document.FICdtTrf.GrpHdr.SttlmInf.InstdRmbrsmntAgtAcct.Id.Othr.Id"
                                    },
                                    {
                                        "xml": ".Document.FICdtTrf.GrpHdr.SttlmInf.InstdRmbrsmntAgtAcct.Id.IBAN"
                                    },
                                ]
                            },
                        ]
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_54B",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
                "line_no": 2,
            },
            {
                "when": {
                    "not": {
                        "any": [
                            {"global_is_set": "has_54A"},
                            {"global_is_set": "has_54B"},
                        ]
                    },
                    "any": [
                        {
                            "op": "!=",
                            "left": {
                                "xml": ".Document.FICdtTrf.GrpHdr.SttlmInf.InstdRmbrsmntAgt.FinInstnId.Nm"
                            },
                            "right": "",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "xml": ".Document.FICdtTrf.GrpHdr.SttlmInf.InstdRmbrsmntAgt.FinInstnId.PstlAdr.AdrLine"
                            },
                            "right": "",
                        },
                    ],
                },
                "then": {
                    "lines": [
                        # 1Nombre
                        {
                            "optional": {
                                "when": {
                                    "op": "!=",
                                    "left": {
                                        "xml": ".Document.FICdtTrf.GrpHdr.SttlmInf.InstdRmbrsmntAgt.FinInstnId.Nm"
                                    },
                                    "right": "",
                                },
                                "value": {
                                    "substr": {
                                        "value": {
                                            "xml": ".Document.FICdtTrf.GrpHdr.SttlmInf.InstdRmbrsmntAgt.FinInstnId.Nm"
                                        },
                                        "start": 0,
                                        "len": 35,
                                    }
                                },
                            }
                        },
                        # 2Dirección
                        {
                            "optional": {
                                "when": {
                                    "op": "!=",
                                    "left": {
                                        "xml": ".Document.FICdtTrf.GrpHdr.SttlmInf.InstdRmbrsmntAgt.FinInstnId.PstlAdr.AdrLine"
                                    },
                                    "right": "",
                                },
                                "value": {
                                    "substr": {
                                        "value": {
                                            "xml": ".Document.FICdtTrf.GrpHdr.SttlmInf.InstdRmbrsmntAgt.FinInstnId.PstlAdr.AdrLine"
                                        },
                                        "start": 0,
                                        "len": 35,
                                    }
                                },
                            }
                        },
                        # 3Ciudad
                        {
                            "optional": {
                                "when": {
                                    "op": "!=",
                                    "left": {
                                        "xml": ".Document.FICdtTrf.GrpHdr.SttlmInf.InstdRmbrsmntAgt.FinInstnId.PstlAdr.TwnNm"
                                    },
                                    "right": "",
                                },
                                "value": {
                                    "substr": {
                                        "value": {
                                            "xml": ".Document.FICdtTrf.GrpHdr.SttlmInf.InstdRmbrsmntAgt.FinInstnId.PstlAdr.TwnNm"
                                        },
                                        "start": 0,
                                        "len": 35,
                                    }
                                },
                            }
                        },
                        # 4País
                        {
                            "optional": {
                                "when": {
                                    "op": "!=",
                                    "left": {
                                        "xml": ".Document.FICdtTrf.GrpHdr.SttlmInf.InstdRmbrsmntAgt.FinInstnId.PstlAdr.Ctry"
                                    },
                                    "right": "",
                                },
                                "value": {
                                    "substr": {
                                        "value": {
                                            "xml": ".Document.FICdtTrf.GrpHdr.SttlmInf.InstdRmbrsmntAgt.FinInstnId.PstlAdr.Ctry"
                                        },
                                        "start": 0,
                                        "len": 35,
                                    }
                                },
                            }
                        },
                    ]
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_54D",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
                "line_no": 3,
            },
        ],
    }
}


fields_spec_54B = {
    "54B": {
        "mode": "append",
        "rules": [
            {
                # 1FICdtTrf
                "when": {
                    "not": {"global_is_set": "has_54A"},
                    "all": [
                        {
                            "op": "=",
                            "left": {
                                "xml": ".Document.FICdtTrf.GrpHdr.SttlmInf.InstdRmbrsmntAgt.FinInstnId.BICFI"
                            },
                            "right": "",
                        },
                        {
                            "op": "=",
                            "left": {
                                "xml": ".Document.FICdtTrf.GrpHdr.SttlmInf.InstdRmbrsmntAgt.FinInstnId.Nm"
                            },
                            "right": "",
                        },
                        {
                            "op": "=",
                            "left": {
                                "xml": ".Document.FICdtTrf.GrpHdr.SttlmInf.InstdRmbrsmntAgt.FinInstnId.PstlAdr.AdrLine"
                            },
                            "right": "",
                        },
                        {
                            "any": [
                                {
                                    "op": "!=",
                                    "left": {
                                        "xml": ".Document.FICdtTrf.GrpHdr.SttlmInf.InstdRmbrsmntAgtAcct.Id.Othr.Id"
                                    },
                                    "right": "",
                                },
                                {
                                    "op": "!=",
                                    "left": {
                                        "xml": ".Document.FICdtTrf.GrpHdr.SttlmInf.InstdRmbrsmntAgtAcct.Id.IBAN"
                                    },
                                    "right": "",
                                },
                            ]
                        },
                    ],
                },
                "then": {
                    "value": {
                        "concat": [
                            {"literal": "/"},
                            {
                                "first_non_empty": [
                                    {
                                        "xml": ".Document.FICdtTrf.GrpHdr.SttlmInf.InstdRmbrsmntAgtAcct.Id.Othr.Id"
                                    },
                                    {
                                        "xml": ".Document.FICdtTrf.GrpHdr.SttlmInf.InstdRmbrsmntAgtAcct.Id.IBAN"
                                    },
                                ]
                            },
                        ]
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_54B",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
                "line_no": 1,
            },
            {
                # 2FIToFICstmrCdtTrf (compatibilidad)
                "when": {
                    "not": {
                        "any": [
                            {"global_is_set": "has_54A"},
                            {"global_is_set": "has_54B"},
                        ]
                    },
                    "all": [
                        {
                            "op": "=",
                            "left": {
                                "xml": ".Document.FIToFICstmrCdtTrf.GrpHdr.SttlmInf.InstdRmbrsmntAgt.FinInstnId.BICFI"
                            },
                            "right": "",
                        },
                        {
                            "op": "=",
                            "left": {
                                "xml": ".Document.FIToFICstmrCdtTrf.GrpHdr.SttlmInf.InstdRmbrsmntAgt.FinInstnId.Nm"
                            },
                            "right": "",
                        },
                        {
                            "op": "=",
                            "left": {
                                "xml": ".Document.FIToFICstmrCdtTrf.GrpHdr.SttlmInf.InstdRmbrsmntAgt.FinInstnId.PstlAdr.AdrLine"
                            },
                            "right": "",
                        },
                        {
                            "any": [
                                {
                                    "op": "!=",
                                    "left": {
                                        "xml": ".Document.FIToFICstmrCdtTrf.GrpHdr.SttlmInf.InstdRmbrsmntAgtAcct.Id.Othr.Id"
                                    },
                                    "right": "",
                                },
                                {
                                    "op": "!=",
                                    "left": {
                                        "xml": ".Document.FIToFICstmrCdtTrf.GrpHdr.SttlmInf.InstdRmbrsmntAgtAcct.Id.IBAN"
                                    },
                                    "right": "",
                                },
                            ]
                        },
                    ],
                },
                "then": {
                    "value": {
                        "concat": [
                            {"literal": "/"},
                            {
                                "first_non_empty": [
                                    {
                                        "xml": ".Document.FIToFICstmrCdtTrf.GrpHdr.SttlmInf.InstdRmbrsmntAgtAcct.Id.Othr.Id"
                                    },
                                    {
                                        "xml": ".Document.FIToFICstmrCdtTrf.GrpHdr.SttlmInf.InstdRmbrsmntAgtAcct.Id.IBAN"
                                    },
                                ]
                            },
                        ]
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_54B",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
                "line_no": 2,
            },
        ],
    }
}

fields_spec_54D = {
    "54D": {
        "mode": "append",
        "rules": [
            {
                "when": {
                    "not": {
                        "any": [
                            {"global_is_set": "has_54A"},
                            {"global_is_set": "has_54B"},
                        ]
                    },
                    "all": [
                        {
                            "op": "=",
                            "left": {
                                "xml": ".Document.FICdtTrf.GrpHdr.SttlmInf.InstdRmbrsmntAgt.FinInstnId.BICFI"
                            },
                            "right": "",
                        },
                        {
                            "any": [
                                {
                                    "op": "!=",
                                    "left": {
                                        "xml": ".Document.FICdtTrf.GrpHdr.SttlmInf.InstdRmbrsmntAgt.FinInstnId.Nm"
                                    },
                                    "right": "",
                                },
                                {
                                    "op": "!=",
                                    "left": {
                                        "xml": ".Document.FICdtTrf.GrpHdr.SttlmInf.InstdRmbrsmntAgt.FinInstnId.PstlAdr.AdrLine"
                                    },
                                    "right": "",
                                },
                            ]
                        },
                    ],
                },
                "then": {
                    "lines": [
                        # 1Identificador opcional (Othr.Id o IBAN)
                        {
                            "optional": {
                                "when": {
                                    "any": [
                                        {
                                            "op": "!=",
                                            "left": {
                                                "xml": ".Document.FICdtTrf.GrpHdr.SttlmInf.InstdRmbrsmntAgtAcct.Id.Othr.Id"
                                            },
                                            "right": "",
                                        },
                                        {
                                            "op": "!=",
                                            "left": {
                                                "xml": ".Document.FICdtTrf.GrpHdr.SttlmInf.InstdRmbrsmntAgtAcct.Id.IBAN"
                                            },
                                            "right": "",
                                        },
                                    ]
                                },
                                "value": {
                                    "concat": [
                                        {"literal": "/"},
                                        {
                                            "first_non_empty": [
                                                {
                                                    "xml": ".Document.FICdtTrf.GrpHdr.SttlmInf.InstdRmbrsmntAgtAcct.Id.Othr.Id"
                                                },
                                                {
                                                    "xml": ".Document.FICdtTrf.GrpHdr.SttlmInf.InstdRmbrsmntAgtAcct.Id.IBAN"
                                                },
                                            ]
                                        },
                                    ]
                                },
                            }
                        },
                        # 2Nombre
                        {
                            "optional": {
                                "when": {
                                    "op": "!=",
                                    "left": {
                                        "xml": ".Document.FICdtTrf.GrpHdr.SttlmInf.InstdRmbrsmntAgt.FinInstnId.Nm"
                                    },
                                    "right": "",
                                },
                                "value": {
                                    "substr": {
                                        "value": {
                                            "xml": ".Document.FICdtTrf.GrpHdr.SttlmInf.InstdRmbrsmntAgt.FinInstnId.Nm"
                                        },
                                        "start": 0,
                                        "len": 35,
                                    }
                                },
                            }
                        },
                        # 3Dirección (AdrLine)
                        {
                            "optional": {
                                "when": {
                                    "op": "!=",
                                    "left": {
                                        "xml": ".Document.FICdtTrf.GrpHdr.SttlmInf.InstdRmbrsmntAgt.FinInstnId.PstlAdr.AdrLine"
                                    },
                                    "right": "",
                                },
                                "value": {
                                    "substr": {
                                        "value": {
                                            "xml": ".Document.FICdtTrf.GrpHdr.SttlmInf.InstdRmbrsmntAgt.FinInstnId.PstlAdr.AdrLine"
                                        },
                                        "start": 0,
                                        "len": 35,
                                    }
                                },
                            }
                        },
                        # 4Ciudad (TwnNm)
                        {
                            "optional": {
                                "when": {
                                    "op": "!=",
                                    "left": {
                                        "xml": ".Document.FICdtTrf.GrpHdr.SttlmInf.InstdRmbrsmntAgt.FinInstnId.PstlAdr.TwnNm"
                                    },
                                    "right": "",
                                },
                                "value": {
                                    "substr": {
                                        "value": {
                                            "xml": ".Document.FICdtTrf.GrpHdr.SttlmInf.InstdRmbrsmntAgt.FinInstnId.PstlAdr.TwnNm"
                                        },
                                        "start": 0,
                                        "len": 35,
                                    }
                                },
                            }
                        },
                        # 5País (Ctry)
                        {
                            "optional": {
                                "when": {
                                    "op": "!=",
                                    "left": {
                                        "xml": ".Document.FICdtTrf.GrpHdr.SttlmInf.InstdRmbrsmntAgt.FinInstnId.PstlAdr.Ctry"
                                    },
                                    "right": "",
                                },
                                "value": {
                                    "substr": {
                                        "value": {
                                            "xml": ".Document.FICdtTrf.GrpHdr.SttlmInf.InstdRmbrsmntAgt.FinInstnId.PstlAdr.Ctry"
                                        },
                                        "start": 0,
                                        "len": 35,
                                    }
                                },
                            }
                        },
                    ]
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_54D",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
                "line_no": 1,
            }
        ],
    }
}

fields_spec_56A = {
    "56A": {
        "mode": "append",
        "rules": [
            {  # 1 FICdtTrf principal — con BICFI
                "when": {
                    "not": {
                        "any": [
                            {"global_is_set": "has_56B"},
                            {"global_is_set": "has_56D"},
                        ]
                    },
                    "all": [
                        {
                            "op": "!=",
                            "left": {
                                "xml": ".Document.FICdtTrf.CdtTrfTxInf.IntrmyAgt1.FinInstnId.BICFI"
                            },
                            "right": "",
                        }
                    ],
                },
                "then": {
                    "value": {
                        "xml": ".Document.FICdtTrf.CdtTrfTxInf.IntrmyAgt1.FinInstnId.BICFI"
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_56A",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
                "line_no": 1,
            },
            {  # 2 FIToFICstmrCdtTrf (compatibilidad) — con BICFI
                "when": {
                    "not": {
                        "any": [
                            {"global_is_set": "has_56A"},
                            {"global_is_set": "has_56B"},
                            {"global_is_set": "has_56D"},
                        ]
                    },
                    "all": [
                        {
                            "op": "!=",
                            "left": {
                                "xml": ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.IntrmyAgt1.FinInstnId.BICFI"
                            },
                            "right": "",
                        }
                    ],
                },
                "then": {
                    "value": {
                        "xml": ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.IntrmyAgt1.FinInstnId.BICFI"
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_56A",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
                "line_no": 2,
            },
            # === Nuevas reglas según BrnchId (Nm / Id) ===
            {  # 3 FICdtTrf — IntrmyAgt1.BranchId.Id (BIC o identificador limpio)
                # Condición: no se haya llenado 56A/56B/56D y exista Id
                "when": {
                    "not": {
                        "any": [
                            {"global_is_set": "has_56A"},
                            {"global_is_set": "has_56B"},
                            {"global_is_set": "has_56D"},
                        ]
                    },
                    "all": [
                        {
                            "op": "!=",
                            "left": {
                                "xml": ".Document.FICdtTrf.CdtTrfTxInf.IntrmyAgt1.BrnchId.Id"
                            },
                            "right": "",
                        }
                    ],
                },
                "then": {
                    # Se asigna directamente el Id como Party Identifier / Identifier Code en :56A:
                    # (validación de formato BIC / ClearingSystemMemberID se asume en una capa previa)
                    "value": {
                        "xml": ".Document.FICdtTrf.CdtTrfTxInf.IntrmyAgt1.BrnchId.Id"
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_56A",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
                "line_no": 1,
            },
            {  # 4 FICdtTrf — IntrmyAgt1.BranchId.Nm (Party Identifier texto)
                # Solo se usará si no se ha llenado 56A/56B/56D previamente
                "when": {
                    "not": {
                        "any": [
                            {"global_is_set": "has_56A"},
                            {"global_is_set": "has_56B"},
                            {"global_is_set": "has_56D"},
                        ]
                    },
                    "all": [
                        {
                            "op": "!=",
                            "left": {
                                "xml": ".Document.FICdtTrf.CdtTrfTxInf.IntrmyAgt1.BrnchId.Nm"
                            },
                            "right": "",
                        }
                    ],
                },
                "then": {
                    # Se usa el nombre de la sucursal como Party Identifier en :56A:
                    "value": {
                        "xml": ".Document.FICdtTrf.CdtTrfTxInf.IntrmyAgt1.BrnchId.Nm"
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_56A",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
                "line_no": 1,
            },
            {  # 5 FIToFICstmrCdtTrf — IntrmyAgt1.BranchId.Id (BIC o identificador)
                "when": {
                    "not": {
                        "any": [
                            {"global_is_set": "has_56A"},
                            {"global_is_set": "has_56B"},
                            {"global_is_set": "has_56D"},
                        ]
                    },
                    "all": [
                        {
                            "op": "!=",
                            "left": {
                                "xml": ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.IntrmyAgt1.BrnchId.Id"
                            },
                            "right": "",
                        }
                    ],
                },
                "then": {
                    "value": {
                        "xml": ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.IntrmyAgt1.BrnchId.Id"
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_56A",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
                "line_no": 2,
            },
            {  # 6 FIToFICstmrCdtTrf — IntrmyAgt1.BranchId.Nm (Party Identifier texto)
                "when": {
                    "not": {
                        "any": [
                            {"global_is_set": "has_56A"},
                            {"global_is_set": "has_56B"},
                            {"global_is_set": "has_56D"},
                        ]
                    },
                    "all": [
                        {
                            "op": "!=",
                            "left": {
                                "xml": ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.IntrmyAgt1.BrnchId.Nm"
                            },
                            "right": "",
                        }
                    ],
                },
                "then": {
                    "value": {
                        "xml": ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.IntrmyAgt1.BrnchId.Nm"
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_56A",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
                "line_no": 2,
            },
        ],
    }
}


fields_spec_56D = {
    "56D": {
        "mode": "append",
        "rules": [
            {
                # 1 FICdtTrf — IntrmyAgt1.BranchId (Nm/Id/PstlAdr)
                # Condición: no hay 56A/56B/56D, no se usa opción A (BICFI vacío)
                # y existe Nm o Id en BrnchId.
                "when": {
                    "not": {
                        "any": [
                            {"global_is_set": "has_56A"},
                            {"global_is_set": "has_56B"},
                            {"global_is_set": "has_56D"},
                        ]
                    },
                    "all": [
                        {
                            "op": "=",
                            "left": {
                                "xml": ".Document.FICdtTrf.CdtTrfTxInf.IntrmyAgt1.FinInstnId.BICFI"
                            },
                            "right": "",
                        },
                        {
                            "any": [
                                {
                                    "op": "!=",
                                    "left": {
                                        "xml": ".Document.FICdtTrf.CdtTrfTxInf.IntrmyAgt1.BrnchId.Nm"
                                    },
                                    "right": "",
                                },
                                {
                                    "op": "!=",
                                    "left": {
                                        "xml": ".Document.FICdtTrf.CdtTrfTxInf.IntrmyAgt1.BrnchId.Id"
                                    },
                                    "right": "",
                                },
                            ]
                        },
                    ],
                },
                "then": {
                    "lines": [
                        # 1 Party Identifier (Id si existe; se asume no BIC porque no usamos opción A)
                        {
                            "optional": {
                                "when": {
                                    "op": "!=",
                                    "left": {
                                        "xml": ".Document.FICdtTrf.CdtTrfTxInf.IntrmyAgt1.BrnchId.Id"
                                    },
                                    "right": "",
                                },
                                "value": {
                                    "substr": {
                                        "value": {
                                            "xml": ".Document.FICdtTrf.CdtTrfTxInf.IntrmyAgt1.BrnchId.Id"
                                        },
                                        "start": 0,
                                        "len": 35,
                                    }
                                },
                            }
                        },
                        # 2 Name (Nm)
                        {
                            "optional": {
                                "when": {
                                    "op": "!=",
                                    "left": {
                                        "xml": ".Document.FICdtTrf.CdtTrfTxInf.IntrmyAgt1.BrnchId.Nm"
                                    },
                                    "right": "",
                                },
                                "value": {
                                    "substr": {
                                        "value": {
                                            "xml": ".Document.FICdtTrf.CdtTrfTxInf.IntrmyAgt1.BrnchId.Nm"
                                        },
                                        "start": 0,
                                        "len": 35,
                                    }
                                },
                            }
                        },
                        # 3 Dirección: AdrLine o NOTPROVIDED
                        {
                            "value": {
                                "substr": {
                                    "value": {
                                        "first_non_empty": [
                                            {
                                                "xml": ".Document.FICdtTrf.CdtTrfTxInf.IntrmyAgt1.BrnchId.PstlAdr.AdrLine"
                                            },
                                            {"literal": "NOTPROVIDED"},
                                        ]
                                    },
                                    "start": 0,
                                    "len": 35,
                                }
                            }
                        },
                    ]
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_56D",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
                "line_no": 1,
            },
            {
                # 2 FIToFICstmrCdtTrf — IntrmyAgt1.BranchId (Nm/Id/PstlAdr) — compatibilidad
                "when": {
                    "not": {
                        "any": [
                            {"global_is_set": "has_56A"},
                            {"global_is_set": "has_56B"},
                            {"global_is_set": "has_56D"},
                        ]
                    },
                    "all": [
                        {
                            "op": "=",
                            "left": {
                                "xml": ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.IntrmyAgt1.FinInstnId.BICFI"
                            },
                            "right": "",
                        },
                        {
                            "any": [
                                {
                                    "op": "!=",
                                    "left": {
                                        "xml": ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.IntrmyAgt1.BrnchId.Nm"
                                    },
                                    "right": "",
                                },
                                {
                                    "op": "!=",
                                    "left": {
                                        "xml": ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.IntrmyAgt1.BrnchId.Id"
                                    },
                                    "right": "",
                                },
                            ]
                        },
                    ],
                },
                "then": {
                    "lines": [
                        {
                            "optional": {
                                "when": {
                                    "op": "!=",
                                    "left": {
                                        "xml": ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.IntrmyAgt1.BrnchId.Id"
                                    },
                                    "right": "",
                                },
                                "value": {
                                    "substr": {
                                        "value": {
                                            "xml": ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.IntrmyAgt1.BrnchId.Id"
                                        },
                                        "start": 0,
                                        "len": 35,
                                    }
                                },
                            }
                        },
                        {
                            "optional": {
                                "when": {
                                    "op": "!=",
                                    "left": {
                                        "xml": ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.IntrmyAgt1.BrnchId.Nm"
                                    },
                                    "right": "",
                                },
                                "value": {
                                    "substr": {
                                        "value": {
                                            "xml": ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.IntrmyAgt1.BrnchId.Nm"
                                        },
                                        "start": 0,
                                        "len": 35,
                                    }
                                },
                            }
                        },
                        {
                            "value": {
                                "substr": {
                                    "value": {
                                        "first_non_empty": [
                                            {
                                                "xml": ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.IntrmyAgt1.BrnchId.PstlAdr.AdrLine"
                                            },
                                            {"literal": "NOTPROVIDED"},
                                        ]
                                    },
                                    "start": 0,
                                    "len": 35,
                                }
                            }
                        },
                    ]
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_56D",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
                "line_no": 2,
            },
            # === Tus reglas originales (FinInstnId) se mantienen como fallback ===
            {
                # 3 FICdtTrf principal — FinInstnId (Nm / PstlAdr)
                "when": {
                    "not": {
                        "any": [
                            {"global_is_set": "has_56A"},
                            {"global_is_set": "has_56B"},
                            {"global_is_set": "has_56D"},
                        ]
                    },
                    "all": [
                        {
                            "op": "=",
                            "left": {
                                "xml": ".Document.FICdtTrf.CdtTrfTxInf.IntrmyAgt1.FinInstnId.BICFI"
                            },
                            "right": "",
                        },
                        {
                            "any": [
                                {
                                    "op": "!=",
                                    "left": {
                                        "xml": ".Document.FICdtTrf.CdtTrfTxInf.IntrmyAgt1.FinInstnId.Nm"
                                    },
                                    "right": "",
                                },
                                {
                                    "op": "!=",
                                    "left": {
                                        "xml": ".Document.FICdtTrf.CdtTrfTxInf.IntrmyAgt1.FinInstnId.PstlAdr.AdrLine"
                                    },
                                    "right": "",
                                },
                            ]
                        },
                    ],
                },
                "then": {
                    "lines": [
                        # 1 Identificador (opcional) — /IBAN o /Othr.Id
                        {
                            "optional": {
                                "when": {
                                    "any": [
                                        {
                                            "op": "!=",
                                            "left": {
                                                "xml": ".Document.FICdtTrf.CdtTrfTxInf.IntrmyAgt1Acct.Id.Othr.Id"
                                            },
                                            "right": "",
                                        },
                                        {
                                            "op": "!=",
                                            "left": {
                                                "xml": ".Document.FICdtTrf.CdtTrfTxInf.IntrmyAgt1Acct.Id.IBAN"
                                            },
                                            "right": "",
                                        },
                                    ]
                                },
                                "value": {
                                    "concat": [
                                        {"literal": "/"},
                                        {
                                            "first_non_empty": [
                                                {
                                                    "xml": ".Document.FICdtTrf.CdtTrfTxInf.IntrmyAgt1Acct.Id.Othr.Id"
                                                },
                                                {
                                                    "xml": ".Document.FICdtTrf.CdtTrfTxInf.IntrmyAgt1Acct.Id.IBAN"
                                                },
                                            ]
                                        },
                                    ]
                                },
                            }
                        },
                        # 2 Nombre
                        {
                            "optional": {
                                "when": {
                                    "op": "!=",
                                    "left": {
                                        "xml": ".Document.FICdtTrf.CdtTrfTxInf.IntrmyAgt1.FinInstnId.Nm"
                                    },
                                    "right": "",
                                },
                                "value": {
                                    "substr": {
                                        "value": {
                                            "xml": ".Document.FICdtTrf.CdtTrfTxInf.IntrmyAgt1.FinInstnId.Nm"
                                        },
                                        "start": 0,
                                        "len": 35,
                                    }
                                },
                            }
                        },
                        # 3 Dirección (AdrLine)
                        {
                            "optional": {
                                "when": {
                                    "op": "!=",
                                    "left": {
                                        "xml": ".Document.FICdtTrf.CdtTrfTxInf.IntrmyAgt1.FinInstnId.PstlAdr.AdrLine"
                                    },
                                    "right": "",
                                },
                                "value": {
                                    "substr": {
                                        "value": {
                                            "xml": ".Document.FICdtTrf.CdtTrfTxInf.IntrmyAgt1.FinInstnId.PstlAdr.AdrLine"
                                        },
                                        "start": 0,
                                        "len": 35,
                                    }
                                },
                            }
                        },
                        # 4 Ciudad
                        {
                            "optional": {
                                "when": {
                                    "op": "!=",
                                    "left": {
                                        "xml": ".Document.FICdtTrf.CdtTrfTxInf.IntrmyAgt1.FinInstnId.PstlAdr.TwnNm"
                                    },
                                    "right": "",
                                },
                                "value": {
                                    "substr": {
                                        "value": {
                                            "xml": ".Document.FICdtTrf.CdtTrfTxInf.IntrmyAgt1.FinInstnId.PstlAdr.TwnNm"
                                        },
                                        "start": 0,
                                        "len": 35,
                                    }
                                },
                            }
                        },
                        # 5 País
                        {
                            "optional": {
                                "when": {
                                    "op": "!=",
                                    "left": {
                                        "xml": ".Document.FICdtTrf.CdtTrfTxInf.IntrmyAgt1.FinInstnId.PstlAdr.Ctry"
                                    },
                                    "right": "",
                                },
                                "value": {
                                    "substr": {
                                        "value": {
                                            "xml": ".Document.FICdtTrf.CdtTrfTxInf.IntrmyAgt1.FinInstnId.PstlAdr.Ctry"
                                        },
                                        "start": 0,
                                        "len": 35,
                                    }
                                },
                            }
                        },
                    ]
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_56D",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
                "line_no": 3,
            },
            {
                # 4 FIToFICstmrCdtTrf (compatibilidad) — FinInstnId
                "when": {
                    "not": {
                        "any": [
                            {"global_is_set": "has_56A"},
                            {"global_is_set": "has_56B"},
                            {"global_is_set": "has_56D"},
                        ]
                    },
                    "all": [
                        {
                            "op": "=",
                            "left": {
                                "xml": ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.IntrmyAgt1.FinInstnId.BICFI"
                            },
                            "right": "",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "xml": ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.IntrmyAgt1.FinInstnId.Nm"
                            },
                            "right": "",
                        },
                    ],
                },
                "then": {
                    "lines": [
                        {
                            "substr": {
                                "value": {
                                    "xml": ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.IntrmyAgt1.FinInstnId.Nm"
                                },
                                "start": 0,
                                "len": 35,
                            }
                        },
                        {
                            "substr": {
                                "value": {
                                    "xml": ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.IntrmyAgt1.FinInstnId.PstlAdr.TwnNm"
                                },
                                "start": 0,
                                "len": 35,
                            }
                        },
                        {
                            "substr": {
                                "value": {
                                    "xml": ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.IntrmyAgt1.FinInstnId.PstlAdr.Ctry"
                                },
                                "start": 0,
                                "len": 35,
                            }
                        },
                    ]
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_56D",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
                "line_no": 4,
            },
            # === Nuevas reglas IntrmyAgt2.FinInstnId.Othr.Id ===
            {
                # 5 FICdtTrf — IntrmyAgt2.FinInstnId.Othr.Id
                "when": {
                    "not": {
                        "any": [
                            {"global_is_set": "has_56A"},
                            {"global_is_set": "has_56B"},
                            {"global_is_set": "has_56D"},
                        ]
                    },
                    "all": [
                        {
                            "op": "!=",
                            "left": {
                                "xml": ".Document.FICdtTrf.CdtTrfTxInf.IntrmyAgt2.FinInstnId.Othr.Id"
                            },
                            "right": "",
                        }
                    ],
                },
                "then": {
                    "lines": [
                        {
                            "substr": {
                                "value": {
                                    "xml": ".Document.FICdtTrf.CdtTrfTxInf.IntrmyAgt2.FinInstnId.Othr.Id"
                                },
                                "start": 0,
                                "len": 35,
                            }
                        }
                    ]
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_56D",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
                "line_no": 5,
            },
            {
                # 6 FIToFICstmrCdtTrf — IntrmyAgt2.FinInstnId.Othr.Id (compatibilidad)
                "when": {
                    "not": {
                        "any": [
                            {"global_is_set": "has_56A"},
                            {"global_is_set": "has_56B"},
                            {"global_is_set": "has_56D"},
                        ]
                    },
                    "all": [
                        {
                            "op": "!=",
                            "left": {
                                "xml": ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.IntrmyAgt2.FinInstnId.Othr.Id"
                            },
                            "right": "",
                        }
                    ],
                },
                "then": {
                    "lines": [
                        {
                            "substr": {
                                "value": {
                                    "xml": ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.IntrmyAgt2.FinInstnId.Othr.Id"
                                },
                                "start": 0,
                                "len": 35,
                            }
                        }
                    ]
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_56D",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
                "line_no": 6,
            },
        ],
    }
}

fields_spec_57A = {
    "57A": {
        "mode": "append",
        "rules": [
            # -----------------------------------------
            # 1) Party Identifier desde IBAN (FICdtTrf)
            # :57A:/IBAN/[IBAN]
            # -----------------------------------------
            {
                "when": {
                    "not": {
                        "any": [
                            {"global_is_set": "has_57B"},
                            {"global_is_set": "has_57D"},
                        ]
                    },
                    "all": [
                        {
                            "op": "!=",
                            "left": {
                                "xml": ".Document.FICdtTrf.CdtTrfTxInf.CdtrAgtAcct.Id.IBAN"
                            },
                            "right": "",
                        }
                    ],
                },
                "then": {
                    "value": {
                        "concat": [
                            {"literal": "/IBAN/"},
                            {
                                "xml": ".Document.FICdtTrf.CdtTrfTxInf.CdtrAgtAcct.Id.IBAN"
                            },
                        ]
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_57A",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
                "line_no": 1,
            },
            # ----------------------------------------------------
            # 2) Party Identifier desde IBAN (FIToFICstmrCdtTrf)
            #   Solo si aún no se llenó 57A (compatibilidad)
            # ----------------------------------------------------
            {
                "when": {
                    "not": {
                        "any": [
                            {"global_is_set": "has_57A"},
                            {"global_is_set": "has_57B"},
                            {"global_is_set": "has_57D"},
                        ]
                    },
                    "all": [
                        {
                            "op": "!=",
                            "left": {
                                "xml": ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.CdtrAgtAcct.Id.IBAN"
                            },
                            "right": "",
                        }
                    ],
                },
                "then": {
                    "value": {
                        "concat": [
                            {"literal": "/IBAN/"},
                            {
                                "xml": ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.CdtrAgtAcct.Id.IBAN"
                            },
                        ]
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_57A",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
                "line_no": 1,
            },
            # ---------------------------------------------------
            # 3) BIC FICdtTrf como Identifier Code
            #    Caso: ya hay Party Identifier (IBAN) -> línea 2
            # ---------------------------------------------------
            {
                "when": {
                    "not": {
                        "any": [
                            {"global_is_set": "has_57B"},
                            {"global_is_set": "has_57D"},
                        ]
                    },
                    "all": [
                        {"global_is_set": "has_57A"},
                        {
                            "op": "!=",
                            "left": {
                                "xml": ".Document.FICdtTrf.CdtTrfTxInf.CdtrAgt.FinInstnId.BICFI"
                            },
                            "right": "",
                        },
                    ],
                },
                "then": {
                    "value": {
                        "xml": ".Document.FICdtTrf.CdtTrfTxInf.CdtrAgt.FinInstnId.BICFI"
                    }
                },
                "line_no": 2,
            },
            # ---------------------------------------------------
            # 4) BIC FICdtTrf como Identifier Code
            #    Caso: NO hay Party Identifier -> línea 1
            # ---------------------------------------------------
            {
                "when": {
                    "not": {
                        "any": [
                            {"global_is_set": "has_57A"},
                            {"global_is_set": "has_57B"},
                            {"global_is_set": "has_57D"},
                        ]
                    },
                    "all": [
                        {
                            "op": "!=",
                            "left": {
                                "xml": ".Document.FICdtTrf.CdtTrfTxInf.CdtrAgt.FinInstnId.BICFI"
                            },
                            "right": "",
                        }
                    ],
                },
                "then": {
                    "value": {
                        "xml": ".Document.FICdtTrf.CdtTrfTxInf.CdtrAgt.FinInstnId.BICFI"
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_57A",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
                "line_no": 1,
            },
            # ---------------------------------------------------
            # 5) BIC FIToFICstmrCdtTrf como Identifier Code
            #    Caso: ya hay Party Identifier -> línea 2
            #    (solo si no hubo BIC en FICdtTrf)
            # ---------------------------------------------------
            {
                "when": {
                    "not": {
                        "any": [
                            {"global_is_set": "has_57B"},
                            {"global_is_set": "has_57D"},
                        ]
                    },
                    "all": [
                        {"global_is_set": "has_57A"},
                        {
                            "op": "!=",
                            "left": {
                                "xml": ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.CdtrAgt.FinInstnId.BICFI"
                            },
                            "right": "",
                        },
                    ],
                },
                "then": {
                    "value": {
                        "xml": ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.CdtrAgt.FinInstnId.BICFI"
                    }
                },
                "line_no": 2,
            },
            # ---------------------------------------------------
            # 6) BIC FIToFICstmrCdtTrf como Identifier Code
            #    Caso: NO hay Party Identifier -> línea 1
            # ---------------------------------------------------
            {
                "when": {
                    "not": {
                        "any": [
                            {"global_is_set": "has_57A"},
                            {"global_is_set": "has_57B"},
                            {"global_is_set": "has_57D"},
                        ]
                    },
                    "all": [
                        {
                            "op": "!=",
                            "left": {
                                "xml": ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.CdtrAgt.FinInstnId.BICFI"
                            },
                            "right": "",
                        }
                    ],
                },
                "then": {
                    "value": {
                        "xml": ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.CdtrAgt.FinInstnId.BICFI"
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_57A",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
                "line_no": 1,
            },
        ],
    }
}


fields_spec_57B = {
    "57B": {
        "mode": "append",
        "rules": [
            {
                # 1FICdtTrf
                "when": {
                    "not": {
                        "any": [
                            {"global_is_set": "has_57A"},
                            {"global_is_set": "has_57D"},
                        ]
                    },
                    "all": [
                        {
                            "op": "=",
                            "left": {
                                "xml": ".Document.FICdtTrf.CdtTrfTxInf.CdtrAgt.FinInstnId.BICFI"
                            },
                            "right": "",
                        },
                        {
                            "op": "=",
                            "left": {
                                "xml": ".Document.FICdtTrf.CdtTrfTxInf.CdtrAgt.FinInstnId.Nm"
                            },
                            "right": "",
                        },
                        {
                            "op": "=",
                            "left": {
                                "xml": ".Document.FICdtTrf.CdtTrfTxInf.CdtrAgt.FinInstnId.PstlAdr.AdrLine"
                            },
                            "right": "",
                        },
                        {
                            "any": [
                                {
                                    "op": "!=",
                                    "left": {
                                        "xml": ".Document.FICdtTrf.CdtTrfTxInf.CdtrAgtAcct.Id.Othr.Id"
                                    },
                                    "right": "",
                                },
                                {
                                    "op": "!=",
                                    "left": {
                                        "xml": ".Document.FICdtTrf.CdtTrfTxInf.CdtrAgtAcct.Id.IBAN"
                                    },
                                    "right": "",
                                },
                            ]
                        },
                    ],
                },
                "then": {
                    "value": {
                        "concat": [
                            {"literal": "/"},
                            {
                                "first_non_empty": [
                                    {
                                        "xml": ".Document.FICdtTrf.CdtTrfTxInf.CdtrAgtAcct.Id.Othr.Id"
                                    },
                                    {
                                        "xml": ".Document.FICdtTrf.CdtTrfTxInf.CdtrAgtAcct.Id.IBAN"
                                    },
                                ]
                            },
                        ]
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_57B",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
                "line_no": 1,
            },
            {
                # 2FIToFICstmrCdtTrf (compatibilidad)
                "when": {
                    "not": {
                        "any": [
                            {"global_is_set": "has_57A"},
                            {"global_is_set": "has_57B"},
                            {"global_is_set": "has_57D"},
                        ]
                    },
                    "all": [
                        {
                            "op": "=",
                            "left": {
                                "xml": ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.CdtrAgt.FinInstnId.BICFI"
                            },
                            "right": "",
                        },
                        {
                            "op": "=",
                            "left": {
                                "xml": ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.CdtrAgt.FinInstnId.Nm"
                            },
                            "right": "",
                        },
                        {
                            "op": "=",
                            "left": {
                                "xml": ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.CdtrAgt.FinInstnId.PstlAdr.AdrLine"
                            },
                            "right": "",
                        },
                        {
                            "any": [
                                {
                                    "op": "!=",
                                    "left": {
                                        "xml": ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.CdtrAgtAcct.Id.Othr.Id"
                                    },
                                    "right": "",
                                },
                                {
                                    "op": "!=",
                                    "left": {
                                        "xml": ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.CdtrAgtAcct.Id.IBAN"
                                    },
                                    "right": "",
                                },
                            ]
                        },
                    ],
                },
                "then": {
                    "value": {
                        "concat": [
                            {"literal": "/"},
                            {
                                "first_non_empty": [
                                    {
                                        "xml": ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.CdtrAgtAcct.Id.Othr.Id"
                                    },
                                    {
                                        "xml": ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.CdtrAgtAcct.Id.IBAN"
                                    },
                                ]
                            },
                        ]
                    }
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_57B",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
                "line_no": 2,
            },
        ],
    }
}

fields_spec_57D = {
    "57D": {
        "mode": "append",
        "rules": [
            {
                # 1) FICdtTrf principal - Name & Address cuando no hay BIC
                "when": {
                    "not": {
                        "any": [
                            {"global_is_set": "has_57A"},
                            {"global_is_set": "has_57B"},
                            {"global_is_set": "has_57D"},
                        ]
                    },
                    "all": [
                        {
                            "op": "=",
                            "left": {
                                "xml": ".Document.FICdtTrf.CdtTrfTxInf.CdtrAgt.FinInstnId.BICFI"
                            },
                            "right": "",
                        },
                        {
                            "any": [
                                {
                                    "op": "!=",
                                    "left": {
                                        "xml": ".Document.FICdtTrf.CdtTrfTxInf.CdtrAgt.FinInstnId.Nm"
                                    },
                                    "right": "",
                                },
                                {
                                    "op": "!=",
                                    "left": {
                                        "xml": ".Document.FICdtTrf.CdtTrfTxInf.CdtrAgt.FinInstnId.PstlAdr.AdrLine"
                                    },
                                    "right": "",
                                },
                                {
                                    "op": "!=",
                                    "left": {
                                        "xml": ".Document.FICdtTrf.CdtTrfTxInf.CdtrAgt.FinInstnId.PstlAdr.TwnNm"
                                    },
                                    "right": "",
                                },
                                {
                                    "op": "!=",
                                    "left": {
                                        "xml": ".Document.FICdtTrf.CdtTrfTxInf.CdtrAgt.FinInstnId.PstlAdr.Ctry"
                                    },
                                    "right": "",
                                },
                            ]
                        },
                    ],
                },
                "then": {
                    "lines": [
                        # 1ª línea: /NameAndAddress + primer dato disponible (Nm / AdrLine / TwnNm / Ctry)
                        {
                            "optional": {
                                "when": {
                                    "any": [
                                        {
                                            "op": "!=",
                                            "left": {
                                                "xml": ".Document.FICdtTrf.CdtTrfTxInf.CdtrAgt.FinInstnId.Nm"
                                            },
                                            "right": "",
                                        },
                                        {
                                            "op": "!=",
                                            "left": {
                                                "xml": ".Document.FICdtTrf.CdtTrfTxInf.CdtrAgt.FinInstnId.PstlAdr.AdrLine"
                                            },
                                            "right": "",
                                        },
                                        {
                                            "op": "!=",
                                            "left": {
                                                "xml": ".Document.FICdtTrf.CdtTrfTxInf.CdtrAgt.FinInstnId.PstlAdr.TwnNm"
                                            },
                                            "right": "",
                                        },
                                        {
                                            "op": "!=",
                                            "left": {
                                                "xml": ".Document.FICdtTrf.CdtTrfTxInf.CdtrAgt.FinInstnId.PstlAdr.Ctry"
                                            },
                                            "right": "",
                                        },
                                    ]
                                },
                                "value": {
                                    "substr": {
                                        "value": {
                                            "concat": [
                                                {"literal": "/NameAndAddress "},
                                                {
                                                    "first_non_empty": [
                                                        {
                                                            "xml": ".Document.FICdtTrf.CdtTrfTxInf.CdtrAgt.FinInstnId.Nm"
                                                        },
                                                        {
                                                            "xml": ".Document.FICdtTrf.CdtTrfTxInf.CdtrAgt.FinInstnId.PstlAdr.AdrLine"
                                                        },
                                                        {
                                                            "xml": ".Document.FICdtTrf.CdtTrfTxInf.CdtrAgt.FinInstnId.PstlAdr.TwnNm"
                                                        },
                                                        {
                                                            "xml": ".Document.FICdtTrf.CdtTrfTxInf.CdtrAgt.FinInstnId.PstlAdr.Ctry"
                                                        },
                                                    ]
                                                },
                                            ]
                                        },
                                        "start": 0,
                                        "len": 35,
                                    }
                                },
                            }
                        },
                        # 2ª línea: AdrLine
                        {
                            "optional": {
                                "when": {
                                    "op": "!=",
                                    "left": {
                                        "xml": ".Document.FICdtTrf.CdtTrfTxInf.CdtrAgt.FinInstnId.PstlAdr.AdrLine"
                                    },
                                    "right": "",
                                },
                                "value": {
                                    "substr": {
                                        "value": {
                                            "xml": ".Document.FICdtTrf.CdtTrfTxInf.CdtrAgt.FinInstnId.PstlAdr.AdrLine"
                                        },
                                        "start": 0,
                                        "len": 35,
                                    }
                                },
                            }
                        },
                        # 3ª línea: TwnNm
                        {
                            "optional": {
                                "when": {
                                    "op": "!=",
                                    "left": {
                                        "xml": ".Document.FICdtTrf.CdtTrfTxInf.CdtrAgt.FinInstnId.PstlAdr.TwnNm"
                                    },
                                    "right": "",
                                },
                                "value": {
                                    "substr": {
                                        "value": {
                                            "xml": ".Document.FICdtTrf.CdtTrfTxInf.CdtrAgt.FinInstnId.PstlAdr.TwnNm"
                                        },
                                        "start": 0,
                                        "len": 35,
                                    }
                                },
                            }
                        },
                        # 4ª línea: Ctry
                        {
                            "optional": {
                                "when": {
                                    "op": "!=",
                                    "left": {
                                        "xml": ".Document.FICdtTrf.CdtTrfTxInf.CdtrAgt.FinInstnId.PstlAdr.Ctry"
                                    },
                                    "right": "",
                                },
                                "value": {
                                    "substr": {
                                        "value": {
                                            "xml": ".Document.FICdtTrf.CdtTrfTxInf.CdtrAgt.FinInstnId.PstlAdr.Ctry"
                                        },
                                        "start": 0,
                                        "len": 35,
                                    }
                                },
                            }
                        },
                    ]
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_57D",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
                "line_no": 1,
            },
            {
                # 2) FIToFICstmrCdtTrf (compatibilidad) - Name & Address cuando no hay BIC
                "when": {
                    "not": {
                        "any": [
                            {"global_is_set": "has_57A"},
                            {"global_is_set": "has_57B"},
                            {"global_is_set": "has_57D"},
                        ]
                    },
                    "all": [
                        {
                            "op": "=",
                            "left": {
                                "xml": ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.CdtrAgt.FinInstnId.BICFI"
                            },
                            "right": "",
                        },
                        {
                            "any": [
                                {
                                    "op": "!=",
                                    "left": {
                                        "xml": ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.CdtrAgt.FinInstnId.Nm"
                                    },
                                    "right": "",
                                },
                                {
                                    "op": "!=",
                                    "left": {
                                        "xml": ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.CdtrAgt.FinInstnId.PstlAdr.AdrLine"
                                    },
                                    "right": "",
                                },
                                {
                                    "op": "!=",
                                    "left": {
                                        "xml": ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.CdtrAgt.FinInstnId.PstlAdr.TwnNm"
                                    },
                                    "right": "",
                                },
                                {
                                    "op": "!=",
                                    "left": {
                                        "xml": ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.CdtrAgt.FinInstnId.PstlAdr.Ctry"
                                    },
                                    "right": "",
                                },
                            ]
                        },
                    ],
                },
                "then": {
                    "lines": [
                        # 1ª línea: /NameAndAddress + primer dato disponible
                        {
                            "substr": {
                                "value": {
                                    "concat": [
                                        {"literal": "/NameAndAddress "},
                                        {
                                            "first_non_empty": [
                                                {
                                                    "xml": ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.CdtrAgt.FinInstnId.Nm"
                                                },
                                                {
                                                    "xml": ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.CdtrAgt.FinInstnId.PstlAdr.AdrLine"
                                                },
                                                {
                                                    "xml": ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.CdtrAgt.FinInstnId.PstlAdr.TwnNm"
                                                },
                                                {
                                                    "xml": ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.CdtrAgt.FinInstnId.PstlAdr.Ctry"
                                                },
                                            ]
                                        },
                                    ]
                                },
                                "start": 0,
                                "len": 35,
                            }
                        },
                        # 2ª línea: AdrLine
                        {
                            "optional": {
                                "when": {
                                    "op": "!=",
                                    "left": {
                                        "xml": ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.CdtrAgt.FinInstnId.PstlAdr.AdrLine"
                                    },
                                    "right": "",
                                },
                                "value": {
                                    "substr": {
                                        "value": {
                                            "xml": ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.CdtrAgt.FinInstnId.PstlAdr.AdrLine"
                                        },
                                        "start": 0,
                                        "len": 35,
                                    }
                                },
                            }
                        },
                        # 3ª línea: TwnNm
                        {
                            "optional": {
                                "when": {
                                    "op": "!=",
                                    "left": {
                                        "xml": ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.CdtrAgt.FinInstnId.PstlAdr.TwnNm"
                                    },
                                    "right": "",
                                },
                                "value": {
                                    "substr": {
                                        "value": {
                                            "xml": ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.CdtrAgt.FinInstnId.PstlAdr.TwnNm"
                                        },
                                        "start": 0,
                                        "len": 35,
                                    }
                                },
                            }
                        },
                        # 4ª línea: Ctry
                        {
                            "optional": {
                                "when": {
                                    "op": "!=",
                                    "left": {
                                        "xml": ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.CdtrAgt.FinInstnId.PstlAdr.Ctry"
                                    },
                                    "right": "",
                                },
                                "value": {
                                    "substr": {
                                        "value": {
                                            "xml": ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.CdtrAgt.FinInstnId.PstlAdr.Ctry"
                                        },
                                        "start": 0,
                                        "len": 35,
                                    }
                                },
                            }
                        },
                    ]
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_57D",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
                "line_no": 1,
            },
            {
                # 3) Fallback FICdtTrf - usar ClrSysMmbId/MmbId como NameAndAddress cuando no hay BIC ni Name/Address
                "when": {
                    "not": {
                        "any": [
                            {"global_is_set": "has_57A"},
                            {"global_is_set": "has_57B"},
                            {"global_is_set": "has_57D"},
                        ]
                    },
                    "all": [
                        {
                            "op": "=",
                            "left": {
                                "xml": ".Document.FICdtTrf.CdtTrfTxInf.CdtrAgt.FinInstnId.BICFI"
                            },
                            "right": "",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "xml": ".Document.FICdtTrf.CdtTrfTxInf.CdtrAgt.FinInstnId.ClrSysMmbId.MmbId"
                            },
                            "right": "",
                        },
                    ],
                },
                "then": {
                    "lines": [
                        {
                            "substr": {
                                "value": {
                                    "concat": [
                                        {"literal": "/NameAndAddress "},
                                        {
                                            "xml": ".Document.FICdtTrf.CdtTrfTxInf.CdtrAgt.FinInstnId.ClrSysMmbId.MmbId"
                                        },
                                    ]
                                },
                                "start": 0,
                                "len": 35,
                            }
                        }
                    ]
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_57D",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
                "line_no": 1,
            },
        ],
    }
}

fields_spec_58A = {
    "58A": {
        "mode": "append",
        "rules": [
            {
                # 1) FICdtTrf - IBAN del beneficiario + BIC del beneficiario
                "when": {
                    "all": [
                        {"not": {"global_is_set": "has_58A"}},
                        # BIC del beneficiario presente
                        {
                            "op": "!=",
                            "left": {
                                "xml": ".Document.FICdtTrf.CdtTrfTxInf.Cdtr.FinInstnId.BICFI"
                            },
                            "right": "",
                        },
                        # Cuenta del beneficiario (IBAN) presente
                        {
                            "op": "!=",
                            "left": {
                                "xml": ".Document.FICdtTrf.CdtTrfTxInf.CdtrAcct.Id.IBAN"
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "lines": [
                        # Línea 1: Party Identifier = /IBAN/[IBAN]
                        {
                            "concat": [
                                {"literal": "/IBAN/"},
                                {
                                    "xml": ".Document.FICdtTrf.CdtTrfTxInf.CdtrAcct.Id.IBAN"
                                },
                            ]
                        },
                        # Línea 2: Identifier Code = BIC
                        {"xml": ".Document.FICdtTrf.CdtTrfTxInf.Cdtr.FinInstnId.BICFI"},
                    ]
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_58A",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
                "line_no": 1,
            },
            {
                # 2) FICdtTrf - Othr.Id del beneficiario + BIC del beneficiario
                #    (caso de tu XML: /74859632100 + CAFECOBB)
                "when": {
                    "all": [
                        {"not": {"global_is_set": "has_58A"}},
                        # BIC del beneficiario presente
                        {
                            "op": "!=",
                            "left": {
                                "xml": ".Document.FICdtTrf.CdtTrfTxInf.Cdtr.FinInstnId.BICFI"
                            },
                            "right": "",
                        },
                        # Cuenta del beneficiario (Othr.Id) presente
                        {
                            "op": "!=",
                            "left": {
                                "xml": ".Document.FICdtTrf.CdtTrfTxInf.CdtrAcct.Id.Othr.Id"
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "lines": [
                        # Línea 1: Party Identifier = /[Othr.Id]
                        {
                            "concat": [
                                {"literal": "/"},
                                {
                                    "xml": ".Document.FICdtTrf.CdtTrfTxInf.CdtrAcct.Id.Othr.Id"
                                },
                            ]
                        },
                        # Línea 2: Identifier Code = BIC
                        {"xml": ".Document.FICdtTrf.CdtTrfTxInf.Cdtr.FinInstnId.BICFI"},
                    ]
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_58A",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
                "line_no": 2,
            },
            {
                # 3) FICdtTrf - Solo BIC del beneficiario (fallback)
                "when": {
                    "all": [
                        {"not": {"global_is_set": "has_58A"}},
                        # BIC del beneficiario presente
                        {
                            "op": "!=",
                            "left": {
                                "xml": ".Document.FICdtTrf.CdtTrfTxInf.Cdtr.FinInstnId.BICFI"
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "lines": [
                        # Línea única: Identifier Code = BIC
                        {"xml": ".Document.FICdtTrf.CdtTrfTxInf.Cdtr.FinInstnId.BICFI"}
                    ]
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_58A",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
                "line_no": 3,
            },
        ],
    }
}

fields_spec_58D = {
    "58D": {
        "mode": "append",
        "rules": [
            # Regla 1: con dirección postal completa
            {
                "when": {
                    "all": [
                        {"not": {"global_is_set": "has_58A"}},
                        {
                            "op": "!=",
                            "left": {
                                "xml": ".Document.FICdtTrf.CdtTrfTxInf.CdtrAcct.Id.Othr.Id"
                            },
                            "right": "",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "xml": ".Document.FICdtTrf.CdtTrfTxInf.Cdtr.FinInstnId.Nm"
                            },
                            "right": "",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "xml": ".Document.FICdtTrf.CdtTrfTxInf.Cdtr.FinInstnId.PstlAdr.StrtNm"
                            },
                            "right": "",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "xml": ".Document.FICdtTrf.CdtTrfTxInf.Cdtr.FinInstnId.PstlAdr.TwnNm"
                            },
                            "right": "",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "xml": ".Document.FICdtTrf.CdtTrfTxInf.Cdtr.FinInstnId.PstlAdr.Ctry"
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "lines": [
                        {
                            "concat": [
                                {"literal": "/"},
                                {
                                    "substr": {
                                        "value": {
                                            "xml": ".Document.FICdtTrf.CdtTrfTxInf.CdtrAcct.Id.Othr.Id"
                                        },
                                        "start": 0,
                                        "len": 35,
                                    }
                                },
                            ]
                        },
                        {
                            "substr": {
                                "value": {
                                    "xml": ".Document.FICdtTrf.CdtTrfTxInf.Cdtr.FinInstnId.Nm"
                                },
                                "start": 0,
                                "len": 35,
                            }
                        },
                        {
                            "substr": {
                                "value": {
                                    "xml": ".Document.FICdtTrf.CdtTrfTxInf.Cdtr.FinInstnId.PstlAdr.StrtNm"
                                },
                                "start": 0,
                                "len": 35,
                            }
                        },
                        {
                            "concat": [
                                {
                                    "substr": {
                                        "value": {
                                            "xml": ".Document.FICdtTrf.CdtTrfTxInf.Cdtr.FinInstnId.PstlAdr.Ctry"
                                        },
                                        "start": 0,
                                        "len": 35,
                                    }
                                },
                                {"literal": "/ "},
                                {
                                    "substr": {
                                        "value": {
                                            "xml": ".Document.FICdtTrf.CdtTrfTxInf.Cdtr.FinInstnId.PstlAdr.TwnNm"
                                        },
                                        "start": 0,
                                        "len": 35,
                                    }
                                },
                            ]
                        },
                    ]
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_58D",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
                "line_no": 1,
            },
            # Regla 2 (fallback): sin dirección postal completa, se comporta como antes
            {
                "when": {
                    "all": [
                        {"not": {"global_is_set": "has_58A"}},
                        {
                            "op": "!=",
                            "left": {
                                "xml": ".Document.FICdtTrf.CdtTrfTxInf.CdtrAcct.Id.Othr.Id"
                            },
                            "right": "",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "xml": ".Document.FICdtTrf.CdtTrfTxInf.Cdtr.FinInstnId.Nm"
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "lines": [
                        {
                            "concat": [
                                {"literal": "/"},
                                {
                                    "substr": {
                                        "value": {
                                            "xml": ".Document.FICdtTrf.CdtTrfTxInf.CdtrAcct.Id.Othr.Id"
                                        },
                                        "start": 0,
                                        "len": 35,
                                    }
                                },
                            ]
                        },
                        {
                            "substr": {
                                "value": {
                                    "xml": ".Document.FICdtTrf.CdtTrfTxInf.Cdtr.FinInstnId.Nm"
                                },
                                "start": 0,
                                "len": 35,
                            }
                        },
                    ]
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_58D",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
                "line_no": 1,
            },
        ],
    }
}

fields_spec_72 = {
    "72": {
        "mode": "append",
        "rules": [
            # ============================
            # /INS/ desde PrvsInstgAgt1 (nombre completo, multilinea)
            # ============================
            {
                "when": {
                    "all": [
                        {
                            "op": "!=",
                            "left": {
                                "xml": ".Document.FICdtTrf.CdtTrfTxInf.PrvsInstgAgt1.FinInstnId.Nm"
                            },
                            "right": "",
                        }
                    ]
                },
                "then": {
                    "lines": [
                        {
                            "concat": [
                                {"literal": "/INS/"},
                                {
                                    "substr": {
                                        "value": {
                                            "xml": ".Document.FICdtTrf.CdtTrfTxInf.PrvsInstgAgt1.FinInstnId.Nm"
                                        },
                                        "start": 0,
                                        "len": 31,
                                    }
                                },
                            ]
                        }
                    ]
                },
                "set": [
                    {
                        "set_var": {
                            "name": "has_72",
                            "value": {"literal": "1"},
                            "scope": "global",
                        }
                    }
                ],
                "line_no": 1,
            },
            {
                "when": {
                    "all": [
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "xml": ".Document.FICdtTrf.CdtTrfTxInf.PrvsInstgAgt1.FinInstnId.Nm"
                                    },
                                    "start": 31,
                                    "len": 1,
                                }
                            },
                            "right": "",
                        }
                    ]
                },
                "then": {
                    "value": {
                        "concat": [
                            {"literal": "//"},
                            {
                                "substr": {
                                    "value": {
                                        "xml": ".Document.FICdtTrf.CdtTrfTxInf.PrvsInstgAgt1.FinInstnId.Nm"
                                    },
                                    "start": 31,
                                    "len": 33,
                                }
                            },
                        ]
                    }
                },
                "line_no": 2,
            },
            {
                "when": {
                    "all": [
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "xml": ".Document.FICdtTrf.CdtTrfTxInf.PrvsInstgAgt1.FinInstnId.Nm"
                                    },
                                    "start": 64,
                                    "len": 1,
                                }
                            },
                            "right": "",
                        }
                    ]
                },
                "then": {
                    "value": {
                        "concat": [
                            {"literal": "//"},
                            {
                                "substr": {
                                    "value": {
                                        "xml": ".Document.FICdtTrf.CdtTrfTxInf.PrvsInstgAgt1.FinInstnId.Nm"
                                    },
                                    "start": 64,
                                    "len": 33,
                                }
                            },
                        ]
                    }
                },
                "line_no": 3,
            },
            # ============================
            # SVCLVL - FIToFICstmrCdtTrf
            # ============================
            {
                "when": {
                    "all": [
                        {
                            "op": "!=",
                            "left": {
                                "xml": ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.PmtTpInf.SvcLvl.Cd"
                            },
                            "right": "",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "xml": ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.PmtTpInf.SvcLvl.Cd"
                                    },
                                    "start": 0,
                                    "len": 3,
                                }
                            },
                            "right": "G00",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "concat": [
                            {"literal": "/SVCLVL/"},
                            {
                                "substr": {
                                    "value": {
                                        "xml": ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.PmtTpInf.SvcLvl.Cd"
                                    },
                                    "start": 0,
                                    "len": 27,
                                }
                            },
                        ]
                    }
                },
                "line_no": 4,
            },
            {
                "when": {
                    "all": [
                        {
                            "op": "=",
                            "left": {
                                "xml": ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.PmtTpInf.SvcLvl.Cd"
                            },
                            "right": "",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "xml": ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.PmtTpInf.SvcLvl.Prtry"
                            },
                            "right": "",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "xml": ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.PmtTpInf.SvcLvl.Prtry"
                                    },
                                    "start": 0,
                                    "len": 3,
                                }
                            },
                            "right": "G00",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "concat": [
                            {"literal": "/SVCLVL/"},
                            {
                                "substr": {
                                    "value": {
                                        "xml": ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.PmtTpInf.SvcLvl.Prtry"
                                    },
                                    "start": 0,
                                    "len": 27,
                                }
                            },
                        ]
                    }
                },
                "line_no": 5,
            },
            # ============================
            # SVCLVL - FICdtTrf (pacs.009)
            # ============================
            {
                # SvcLvl.Cd
                "when": {
                    "all": [
                        {
                            "op": "!=",
                            "left": {
                                "xml": ".Document.FICdtTrf.CdtTrfTxInf.PmtTpInf.SvcLvl.Cd"
                            },
                            "right": "",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "xml": ".Document.FICdtTrf.CdtTrfTxInf.PmtTpInf.SvcLvl.Cd"
                                    },
                                    "start": 0,
                                    "len": 3,
                                }
                            },
                            "right": "G00",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "concat": [
                            {"literal": "/SVCLVL/"},
                            {
                                "substr": {
                                    "value": {
                                        "xml": ".Document.FICdtTrf.CdtTrfTxInf.PmtTpInf.SvcLvl.Cd"
                                    },
                                    "start": 0,
                                    "len": 27,
                                }
                            },
                        ]
                    }
                },
                "line_no": 6,
            },
            {
                # SvcLvl.Prtry
                "when": {
                    "all": [
                        {
                            "op": "=",
                            "left": {
                                "xml": ".Document.FICdtTrf.CdtTrfTxInf.PmtTpInf.SvcLvl.Cd"
                            },
                            "right": "",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "xml": ".Document.FICdtTrf.CdtTrfTxInf.PmtTpInf.SvcLvl.Prtry"
                            },
                            "right": "",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "xml": ".Document.FICdtTrf.CdtTrfTxInf.PmtTpInf.SvcLvl.Prtry"
                                    },
                                    "start": 0,
                                    "len": 3,
                                }
                            },
                            "right": "G00",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "concat": [
                            {"literal": "/SVCLVL/"},
                            {
                                "substr": {
                                    "value": {
                                        "xml": ".Document.FICdtTrf.CdtTrfTxInf.PmtTpInf.SvcLvl.Prtry"
                                    },
                                    "start": 0,
                                    "len": 27,
                                }
                            },
                        ]
                    }
                },
                "line_no": 7,
            },
            # ============================
            # LOCINS / CATPURP (FIToFICstmrCdtTrf)
            # ============================
            {
                "when": {
                    "all": [
                        {
                            "op": "!=",
                            "left": {
                                "xml": ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.PmtTpInf.LclInstrm.Cd"
                            },
                            "right": "",
                        }
                    ]
                },
                "then": {
                    "value": {
                        "concat": [
                            {"literal": "/LOCINS/"},
                            {
                                "substr": {
                                    "value": {
                                        "xml": ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.PmtTpInf.LclInstrm.Cd"
                                    },
                                    "start": 0,
                                    "len": 27,
                                }
                            },
                        ]
                    }
                },
                "line_no": 8,
            },
            {
                "when": {
                    "all": [
                        {
                            "op": "!=",
                            "left": {
                                "xml": ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.PmtTpInf.CtgyPurp.Cd"
                            },
                            "right": "",
                        }
                    ]
                },
                "then": {
                    "value": {
                        "concat": [
                            {"literal": "/CATPURP/"},
                            {
                                "substr": {
                                    "value": {
                                        "xml": ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.PmtTpInf.CtgyPurp.Cd"
                                    },
                                    "start": 0,
                                    "len": 26,
                                }
                            },
                        ]
                    }
                },
                "line_no": 9,
            },
            # ============================
            # CATPURP - FICdtTrf (pacs.009)
            # ============================
            {
                "when": {
                    "all": [
                        {
                            "op": "!=",
                            "left": {
                                "xml": ".Document.FICdtTrf.CdtTrfTxInf.PmtTpInf.CtgyPurp.Cd"
                            },
                            "right": "",
                        }
                    ]
                },
                "then": {
                    "value": {
                        "concat": [
                            {"literal": "/CATPURP/"},
                            {
                                "substr": {
                                    "value": {
                                        "xml": ".Document.FICdtTrf.CdtTrfTxInf.PmtTpInf.CtgyPurp.Cd"
                                    },
                                    "start": 0,
                                    "len": 26,
                                }
                            },
                        ]
                    }
                },
                "line_no": 10,
            },
            {
                "when": {
                    "all": [
                        {
                            "op": "=",
                            "left": {
                                "xml": ".Document.FICdtTrf.CdtTrfTxInf.PmtTpInf.CtgyPurp.Cd"
                            },
                            "right": "",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "xml": ".Document.FICdtTrf.CdtTrfTxInf.PmtTpInf.CtgyPurp.Prtry"
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "concat": [
                            {"literal": "/CATPURP/"},
                            {
                                "substr": {
                                    "value": {
                                        "xml": ".Document.FICdtTrf.CdtTrfTxInf.PmtTpInf.CtgyPurp.Prtry"
                                    },
                                    "start": 0,
                                    "len": 26,
                                }
                            },
                        ]
                    }
                },
                "line_no": 11,
            },
            # ============================
            # PrvsInstgAgt - evitar "/" solo
            # (solo Agt2 y Agt3, Agt1 va por /INS/)
            # ============================
            {
                "when": {
                    "all": [
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "xml": ".Document.FICdtTrf.CdtTrfTxInf.PrvsInstgAgt2.FinInstnId.Nm"
                                    },
                                    "start": 0,
                                    "len": 1,
                                }
                            },
                            "right": "",
                        }
                    ]
                },
                "then": {
                    "value": {
                        "concat": [
                            {"literal": "/"},
                            {
                                "substr": {
                                    "value": {
                                        "xml": ".Document.FICdtTrf.CdtTrfTxInf.PrvsInstgAgt2.FinInstnId.Nm"
                                    },
                                    "start": 0,
                                    "len": 34,
                                }
                            },
                        ]
                    }
                },
                "line_no": 13,
            },
            {
                "when": {
                    "all": [
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "xml": ".Document.FICdtTrf.CdtTrfTxInf.PrvsInstgAgt3.FinInstnId.Nm"
                                    },
                                    "start": 0,
                                    "len": 1,
                                }
                            },
                            "right": "",
                        }
                    ]
                },
                "then": {
                    "value": {
                        "concat": [
                            {"literal": "/"},
                            {
                                "substr": {
                                    "value": {
                                        "xml": ".Document.FICdtTrf.CdtTrfTxInf.PrvsInstgAgt3.FinInstnId.Nm"
                                    },
                                    "start": 0,
                                    "len": 34,
                                }
                            },
                        ]
                    }
                },
                "line_no": 14,
            },
            # ============================
            # IntrmyAgt2 - evitar "/" solo en base
            # ============================
            {
                "when": {
                    "all": [
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "xml": ".Document.FICdtTrf.CdtTrfTxInf.IntrmyAgt2.FinInstnId.Othr.Id"
                                    },
                                    "start": 0,
                                    "len": 1,
                                }
                            },
                            "right": "",
                        }
                    ]
                },
                "then": {
                    "value": {
                        "concat": [
                            {"literal": "/"},
                            {
                                "substr": {
                                    "value": {
                                        "xml": ".Document.FICdtTrf.CdtTrfTxInf.IntrmyAgt2.FinInstnId.Othr.Id"
                                    },
                                    "start": 0,
                                    "len": 34,
                                }
                            },
                        ]
                    }
                },
                "line_no": 15,
            },
            {
                "when": {
                    "all": [
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "xml": ".Document.FICdtTrf.CdtTrfTxInf.IntrmyAgt2.FinInstnId.Othr.Id"
                                    },
                                    "start": 34,
                                    "len": 1,
                                }
                            },
                            "right": "",
                        }
                    ]
                },
                "then": {
                    "value": {
                        "concat": [
                            {"literal": "//"},
                            {
                                "substr": {
                                    "value": {
                                        "xml": ".Document.FICdtTrf.CdtTrfTxInf.IntrmyAgt2.FinInstnId.Othr.Id"
                                    },
                                    "start": 34,
                                    "len": 33,
                                }
                            },
                        ]
                    }
                },
                "line_no": 16,
            },
            {
                "when": {
                    "all": [
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "xml": ".Document.FICdtTrf.CdtTrfTxInf.IntrmyAgt2.FinInstnId.Othr.Id"
                                    },
                                    "start": 67,
                                    "len": 1,
                                }
                            },
                            "right": "",
                        }
                    ]
                },
                "then": {
                    "value": {
                        "concat": [
                            {"literal": "//"},
                            {
                                "substr": {
                                    "value": {
                                        "xml": ".Document.FICdtTrf.CdtTrfTxInf.IntrmyAgt2.FinInstnId.Othr.Id"
                                    },
                                    "start": 67,
                                    "len": 33,
                                }
                            },
                        ]
                    }
                },
                "line_no": 17,
            },
            {
                "when": {
                    "all": [
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "xml": ".Document.FICdtTrf.CdtTrfTxInf.IntrmyAgt2.FinInstnId.Othr.Id"
                                    },
                                    "start": 100,
                                    "len": 1,
                                }
                            },
                            "right": "",
                        }
                    ]
                },
                "then": {
                    "value": {
                        "concat": [
                            {"literal": "//"},
                            {
                                "substr": {
                                    "value": {
                                        "xml": ".Document.FICdtTrf.CdtTrfTxInf.IntrmyAgt2.FinInstnId.Othr.Id"
                                    },
                                    "start": 100,
                                    "len": 33,
                                }
                            },
                        ]
                    }
                },
                "line_no": 18,
            },
            {
                "when": {
                    "all": [
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "xml": ".Document.FICdtTrf.CdtTrfTxInf.IntrmyAgt2.FinInstnId.Othr.Id"
                                    },
                                    "start": 133,
                                    "len": 1,
                                }
                            },
                            "right": "",
                        }
                    ]
                },
                "then": {
                    "value": {
                        "concat": [
                            {"literal": "//"},
                            {
                                "substr": {
                                    "value": {
                                        "xml": ".Document.FICdtTrf.CdtTrfTxInf.IntrmyAgt2.FinInstnId.Othr.Id"
                                    },
                                    "start": 133,
                                    "len": 33,
                                }
                            },
                        ]
                    }
                },
                "line_no": 19,
            },
            {
                "when": {
                    "all": [
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "xml": ".Document.FICdtTrf.CdtTrfTxInf.IntrmyAgt2.FinInstnId.Othr.Id"
                                    },
                                    "start": 166,
                                    "len": 1,
                                }
                            },
                            "right": "",
                        }
                    ]
                },
                "then": {
                    "value": {
                        "concat": [
                            {"literal": "//"},
                            {
                                "substr": {
                                    "value": {
                                        "xml": ".Document.FICdtTrf.CdtTrfTxInf.IntrmyAgt2.FinInstnId.Othr.Id"
                                    },
                                    "start": 166,
                                    "len": 33,
                                }
                            },
                        ]
                    }
                },
                "line_no": 20,
            },
            # ============================
            # IntrmyAgt3 - evitar "/" solo en base
            # ============================
            {
                "when": {
                    "all": [
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "xml": ".Document.FICdtTrf.CdtTrfTxInf.IntrmyAgt3.FinInstnId.Othr.Id"
                                    },
                                    "start": 0,
                                    "len": 1,
                                }
                            },
                            "right": "",
                        }
                    ]
                },
                "then": {
                    "value": {
                        "concat": [
                            {"literal": "/"},
                            {
                                "substr": {
                                    "value": {
                                        "xml": ".Document.FICdtTrf.CdtTrfTxInf.IntrmyAgt3.FinInstnId.Othr.Id"
                                    },
                                    "start": 0,
                                    "len": 34,
                                }
                            },
                        ]
                    }
                },
                "line_no": 21,
            },
            {
                "when": {
                    "all": [
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "xml": ".Document.FICdtTrf.CdtTrfTxInf.IntrmyAgt3.FinInstnId.Othr.Id"
                                    },
                                    "start": 34,
                                    "len": 1,
                                }
                            },
                            "right": "",
                        }
                    ]
                },
                "then": {
                    "value": {
                        "concat": [
                            {"literal": "//"},
                            {
                                "substr": {
                                    "value": {
                                        "xml": ".Document.FICdtTrf.CdtTrfTxInf.IntrmyAgt3.FinInstnId.Othr.Id"
                                    },
                                    "start": 34,
                                    "len": 33,
                                }
                            },
                        ]
                    }
                },
                "line_no": 22,
            },
            {
                "when": {
                    "all": [
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "xml": ".Document.FICdtTrf.CdtTrfTxInf.IntrmyAgt3.FinInstnId.Othr.Id"
                                    },
                                    "start": 67,
                                    "len": 1,
                                }
                            },
                            "right": "",
                        }
                    ]
                },
                "then": {
                    "value": {
                        "concat": [
                            {"literal": "//"},
                            {
                                "substr": {
                                    "value": {
                                        "xml": ".Document.FICdtTrf.CdtTrfTxInf.IntrmyAgt3.FinInstnId.Othr.Id"
                                    },
                                    "start": 67,
                                    "len": 33,
                                }
                            },
                        ]
                    }
                },
                "line_no": 23,
            },
            {
                "when": {
                    "all": [
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "xml": ".Document.FICdtTrf.CdtTrfTxInf.IntrmyAgt3.FinInstnId.Othr.Id"
                                    },
                                    "start": 100,
                                    "len": 1,
                                }
                            },
                            "right": "",
                        }
                    ]
                },
                "then": {
                    "value": {
                        "concat": [
                            {"literal": "//"},
                            {
                                "substr": {
                                    "value": {
                                        "xml": ".Document.FICdtTrf.CdtTrfTxInf.IntrmyAgt3.FinInstnId.Othr.Id"
                                    },
                                    "start": 100,
                                    "len": 33,
                                }
                            },
                        ]
                    }
                },
                "line_no": 24,
            },
            {
                "when": {
                    "all": [
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "xml": ".Document.FICdtTrf.CdtTrfTxInf.IntrmyAgt3.FinInstnId.Othr.Id"
                                    },
                                    "start": 133,
                                    "len": 1,
                                }
                            },
                            "right": "",
                        }
                    ]
                },
                "then": {
                    "value": {
                        "concat": [
                            {"literal": "//"},
                            {
                                "substr": {
                                    "value": {
                                        "xml": ".Document.FICdtTrf.CdtTrfTxInf.IntrmyAgt3.FinInstnId.Othr.Id"
                                    },
                                    "start": 133,
                                    "len": 33,
                                }
                            },
                        ]
                    }
                },
                "line_no": 25,
            },
            {
                "when": {
                    "all": [
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "xml": ".Document.FICdtTrf.CdtTrfTxInf.IntrmyAgt3.FinInstnId.Othr.Id"
                                    },
                                    "start": 166,
                                    "len": 1,
                                }
                            },
                            "right": "",
                        }
                    ]
                },
                "then": {
                    "value": {
                        "concat": [
                            {"literal": "//"},
                            {
                                "substr": {
                                    "value": {
                                        "xml": ".Document.FICdtTrf.CdtTrfTxInf.IntrmyAgt3.FinInstnId.Othr.Id"
                                    },
                                    "start": 166,
                                    "len": 33,
                                }
                            },
                        ]
                    }
                },
                "line_no": 26,
            },
            # ============================
            # InstrForCdtrAgt / PURP / ACC
            # (evitar etiquetas solas)
            # ============================
            {
                # /PURP/ desde InstrForCdtrAgt.Cd
                "when": {
                    "all": [
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "xml": ".Document.FICdtTrf.CdtTrfTxInf.InstrForCdtrAgt.Cd"
                                    },
                                    "start": 0,
                                    "len": 1,
                                }
                            },
                            "right": "",
                        }
                    ]
                },
                "then": {
                    "value": {
                        "concat": [
                            {"literal": "/PURP/"},
                            {
                                "substr": {
                                    "value": {
                                        "xml": ".Document.FICdtTrf.CdtTrfTxInf.InstrForCdtrAgt.Cd"
                                    },
                                    "start": 0,
                                    "len": 29,
                                }
                            },
                        ]
                    }
                },
                "line_no": 27,
            },
            {
                "when": {
                    "all": [
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "xml": ".Document.FICdtTrf.CdtTrfTxInf.InstrForCdtrAgt.Cd"
                                    },
                                    "start": 29,
                                    "len": 1,
                                }
                            },
                            "right": "",
                        }
                    ]
                },
                "then": {
                    "value": {
                        "concat": [
                            {"literal": "/PURP/"},
                            {
                                "substr": {
                                    "value": {
                                        "xml": ".Document.FICdtTrf.CdtTrfTxInf.InstrForCdtrAgt.Cd"
                                    },
                                    "start": 0,
                                    "len": 29,
                                }
                            },
                            {"literal": "+"},
                        ]
                    }
                },
                "line_no": 28,
            },
            {
                # /ACC/ desde InstrForCdtrAgt.InstrInf
                "when": {
                    "all": [
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "xml": ".Document.FICdtTrf.CdtTrfTxInf.InstrForCdtrAgt.InstrInf"
                                    },
                                    "start": 0,
                                    "len": 1,
                                }
                            },
                            "right": "",
                        }
                    ]
                },
                "then": {
                    "value": {
                        "concat": [
                            {"literal": "/ACC/"},
                            {
                                "substr": {
                                    "value": {
                                        "xml": ".Document.FICdtTrf.CdtTrfTxInf.InstrForCdtrAgt.InstrInf"
                                    },
                                    "start": 0,
                                    "len": 30,
                                }
                            },
                        ]
                    }
                },
                "line_no": 29,
            },
            {
                "when": {
                    "all": [
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "xml": ".Document.FICdtTrf.CdtTrfTxInf.InstrForCdtrAgt.InstrInf"
                                    },
                                    "start": 30,
                                    "len": 1,
                                }
                            },
                            "right": "",
                        }
                    ]
                },
                "then": {
                    "value": {
                        "concat": [
                            {"literal": "/ACC/"},
                            {
                                "substr": {
                                    "value": {
                                        "xml": ".Document.FICdtTrf.CdtTrfTxInf.InstrForCdtrAgt.InstrInf"
                                    },
                                    "start": 0,
                                    "len": 30,
                                }
                            },
                            {"literal": "+"},
                        ]
                    }
                },
                "line_no": 30,
            },
            # ============================
            # Purp (original)
            # ============================
            {
                "when": {
                    "all": [
                        {
                            "op": "!=",
                            "left": {"xml": ".Document.FICdtTrf.CdtTrfTxInf.Purp.Cd"},
                            "right": "",
                        }
                    ]
                },
                "then": {
                    "value": {
                        "concat": [
                            {"literal": "/PURP/"},
                            {
                                "substr": {
                                    "value": {
                                        "xml": ".Document.FICdtTrf.CdtTrfTxInf.Purp.Cd"
                                    },
                                    "start": 0,
                                    "len": 29,
                                }
                            },
                        ]
                    }
                },
                "line_no": 31,
            },
            {
                "when": {
                    "all": [
                        {
                            "op": "=",
                            "left": {"xml": ".Document.FICdtTrf.CdtTrfTxInf.Purp.Cd"},
                            "right": "",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "xml": ".Document.FICdtTrf.CdtTrfTxInf.Purp.Prtry"
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "concat": [
                            {"literal": "/PURP/"},
                            {
                                "substr": {
                                    "value": {
                                        "xml": ".Document.FICdtTrf.CdtTrfTxInf.Purp.Prtry"
                                    },
                                    "start": 0,
                                    "len": 29,
                                }
                            },
                        ]
                    }
                },
                "line_no": 32,
            },
            {
                "when": {
                    "all": [
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "xml": ".Document.FICdtTrf.CdtTrfTxInf.Purp.Cd"
                                    },
                                    "start": 29,
                                    "len": 1,
                                }
                            },
                            "right": "",
                        }
                    ]
                },
                "then": {
                    "value": {
                        "concat": [
                            {"literal": "/PURP/"},
                            {
                                "substr": {
                                    "value": {
                                        "xml": ".Document.FICdtTrf.CdtTrfTxInf.Purp.Cd"
                                    },
                                    "start": 0,
                                    "len": 29,
                                }
                            },
                            {"literal": "+"},
                        ]
                    }
                },
                "line_no": 33,
            },
            {
                "when": {
                    "all": [
                        {
                            "op": "=",
                            "left": {"xml": ".Document.FICdtTrf.CdtTrfTxInf.Purp.Cd"},
                            "right": "",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "xml": ".Document.FICdtTrf.CdtTrfTxInf.Purp.Prtry"
                                    },
                                    "start": 29,
                                    "len": 1,
                                }
                            },
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "concat": [
                            {"literal": "/PURP/"},
                            {
                                "substr": {
                                    "value": {
                                        "xml": ".Document.FICdtTrf.CdtTrfTxInf.Purp.Prtry"
                                    },
                                    "start": 0,
                                    "len": 29,
                                }
                            },
                            {"literal": "+"},
                        ]
                    }
                },
                "line_no": 34,
            },
            # ============================
            # /REC/ desde InstrForNxtAgt.InstrInf
            # y /INS/ como fallback con DbtrAgt.BICFI
            # ============================
            {
            # /REC/ - Unir todo InstrInf[0..3] y luego partir en líneas:
            #   Línea 1: 35 caracteres
            #   Líneas 2..4: '//' + 33 caracteres (total 35)
            "when": {
                "op": "!=",
                "left": {
                    "substr": {
                        "value": {
                            "concat": [
                                {"xml_nth": {"path": ".Document.FICdtTrf.CdtTrfTxInf.InstrForNxtAgt.InstrInf", "index": 0}},
                                {"xml_nth": {"path": ".Document.FICdtTrf.CdtTrfTxInf.InstrForNxtAgt.InstrInf", "index": 1}},
                                {"xml_nth": {"path": ".Document.FICdtTrf.CdtTrfTxInf.InstrForNxtAgt.InstrInf", "index": 2}},
                                {"xml_nth": {"path": ".Document.FICdtTrf.CdtTrfTxInf.InstrForNxtAgt.InstrInf", "index": 3}}
                            ]
                        },
                        "start": 0,
                        "len": 35
                    }
                },
                "right": ""
            },
            "then": {
                "value": {
                    "substr": {
                        "value": {
                            "concat": [
                                {"xml_nth": {"path": ".Document.FICdtTrf.CdtTrfTxInf.InstrForNxtAgt.InstrInf", "index": 0}},
                                {"xml_nth": {"path": ".Document.FICdtTrf.CdtTrfTxInf.InstrForNxtAgt.InstrInf", "index": 1}},
                                {"xml_nth": {"path": ".Document.FICdtTrf.CdtTrfTxInf.InstrForNxtAgt.InstrInf", "index": 2}},
                                {"xml_nth": {"path": ".Document.FICdtTrf.CdtTrfTxInf.InstrForNxtAgt.InstrInf", "index": 3}}
                            ]
                        },
                        "start": 0,
                        "len": 35
                    }
                }
            },
            "line_no": 35
        },
        {
            # /REC/ - Línea 2: // + 33 chars (35 total)
            "when": {
                "op": "!=",
                "left": {
                    "substr": {
                        "value": {
                            "concat": [
                                {"xml_nth": {"path": ".Document.FICdtTrf.CdtTrfTxInf.InstrForNxtAgt.InstrInf", "index": 0}},
                                {"xml_nth": {"path": ".Document.FICdtTrf.CdtTrfTxInf.InstrForNxtAgt.InstrInf", "index": 1}},
                                {"xml_nth": {"path": ".Document.FICdtTrf.CdtTrfTxInf.InstrForNxtAgt.InstrInf", "index": 2}},
                                {"xml_nth": {"path": ".Document.FICdtTrf.CdtTrfTxInf.InstrForNxtAgt.InstrInf", "index": 3}}
                            ]
                        },
                        "start": 35,
                        "len": 1
                    }
                },
                "right": ""
            },
            "then": {
                "value": {
                    "concat": [
                        {"literal": "//"},
                        {"substr": {
                        "value": {
                            "concat": [
                                {"xml_nth": {"path": ".Document.FICdtTrf.CdtTrfTxInf.InstrForNxtAgt.InstrInf", "index": 0}},
                                {"xml_nth": {"path": ".Document.FICdtTrf.CdtTrfTxInf.InstrForNxtAgt.InstrInf", "index": 1}},
                                {"xml_nth": {"path": ".Document.FICdtTrf.CdtTrfTxInf.InstrForNxtAgt.InstrInf", "index": 2}},
                                {"xml_nth": {"path": ".Document.FICdtTrf.CdtTrfTxInf.InstrForNxtAgt.InstrInf", "index": 3}}
                            ]
                        },
                        "start": 35,
                        "len": 33
                    }}
                    ]
                }
            },
            "line_no": 36
        },
        {
            # /REC/ - Línea 3
            "when": {
                "op": "!=",
                "left": {
                    "substr": {
                        "value": {
                            "concat": [
                                {"xml_nth": {"path": ".Document.FICdtTrf.CdtTrfTxInf.InstrForNxtAgt.InstrInf", "index": 0}},
                                {"xml_nth": {"path": ".Document.FICdtTrf.CdtTrfTxInf.InstrForNxtAgt.InstrInf", "index": 1}},
                                {"xml_nth": {"path": ".Document.FICdtTrf.CdtTrfTxInf.InstrForNxtAgt.InstrInf", "index": 2}},
                                {"xml_nth": {"path": ".Document.FICdtTrf.CdtTrfTxInf.InstrForNxtAgt.InstrInf", "index": 3}}
                            ]
                        },
                        "start": 68,
                        "len": 1
                    }
                },
                "right": ""
            },
            "then": {
                "value": {
                    "concat": [
                        {"literal": "//"},
                        {"substr": {
                        "value": {
                            "concat": [
                                {"xml_nth": {"path": ".Document.FICdtTrf.CdtTrfTxInf.InstrForNxtAgt.InstrInf", "index": 0}},
                                {"xml_nth": {"path": ".Document.FICdtTrf.CdtTrfTxInf.InstrForNxtAgt.InstrInf", "index": 1}},
                                {"xml_nth": {"path": ".Document.FICdtTrf.CdtTrfTxInf.InstrForNxtAgt.InstrInf", "index": 2}},
                                {"xml_nth": {"path": ".Document.FICdtTrf.CdtTrfTxInf.InstrForNxtAgt.InstrInf", "index": 3}}
                            ]
                        },
                        "start": 68,
                        "len": 33
                    }}
                    ]
                }
            },
            "line_no": 37
        },
        {
            # /REC/ - Línea 4
            "when": {
                "op": "!=",
                "left": {
                    "substr": {
                        "value": {
                            "concat": [
                                {"xml_nth": {"path": ".Document.FICdtTrf.CdtTrfTxInf.InstrForNxtAgt.InstrInf", "index": 0}},
                                {"xml_nth": {"path": ".Document.FICdtTrf.CdtTrfTxInf.InstrForNxtAgt.InstrInf", "index": 1}},
                                {"xml_nth": {"path": ".Document.FICdtTrf.CdtTrfTxInf.InstrForNxtAgt.InstrInf", "index": 2}},
                                {"xml_nth": {"path": ".Document.FICdtTrf.CdtTrfTxInf.InstrForNxtAgt.InstrInf", "index": 3}}
                            ]
                        },
                        "start": 101,
                        "len": 1
                    }
                },
                "right": ""
            },
            "then": {
                "value": {
                    "concat": [
                        {"literal": "//"},
                        {"substr": {
                        "value": {
                            "concat": [
                                {"xml_nth": {"path": ".Document.FICdtTrf.CdtTrfTxInf.InstrForNxtAgt.InstrInf", "index": 0}},
                                {"xml_nth": {"path": ".Document.FICdtTrf.CdtTrfTxInf.InstrForNxtAgt.InstrInf", "index": 1}},
                                {"xml_nth": {"path": ".Document.FICdtTrf.CdtTrfTxInf.InstrForNxtAgt.InstrInf", "index": 2}},
                                {"xml_nth": {"path": ".Document.FICdtTrf.CdtTrfTxInf.InstrForNxtAgt.InstrInf", "index": 3}}
                            ]
                        },
                        "start": 101,
                        "len": 33
                    }}
                    ]
                }
            },
            "line_no": 38
        },
            {
                # Fallback: solo si NO viene InstrForNxtAgt.InstrInf, usar /INS/BICFI
                "when": {
                    "all": [
                        {
                            "op": "=",
                            "left": {
                                "xml": ".Document.FICdtTrf.CdtTrfTxInf.InstrForNxtAgt.InstrInf"
                            },
                            "right": "",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "xml": ".Document.FICdtTrf.CdtTrfTxInf.DbtrAgt.FinInstnId.BICFI"
                            },
                            "right": "",
                        },
                        {
                            "op": "=",
                            "left": {"var": {"name": "has_72", "scope": "global"}},
                            "right": "",
                        },
                    ]
                },
                "then": {
                    "value": {
                        "concat": [
                            {"literal": "/INS/"},
                            {
                                "substr": {
                                    "value": {
                                        "xml": ".Document.FICdtTrf.CdtTrfTxInf.DbtrAgt.FinInstnId.BICFI"
                                    },
                                    "start": 0,
                                    "len": 11,
                                }
                            },
                        ]
                    }
                },
                "line_no": 38,
            },
            # ============================
            # Remittance Info (original)
            # ============================
            {
                "when": {
                    "op": "!=",
                    "left": { "xml": ".Document.FICdtTrf.CdtTrfTxInf.RmtInf.Ustrd" },
                    "right": ""
                },
                "then": {
                    "value": {
                        "substr": {
                            "value": {
                                "xml": ".Document.FICdtTrf.CdtTrfTxInf.RmtInf.Ustrd"
                            },
                            "start": 0,
                            "len": 35
                        }
                    }
                },
                "line_no": 39
            },
            {
                "when": {
                    "all": [
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "xml": ".Document.FICdtTrf.CdtTrfTxInf.RmtInf.Ustrd"
                                    },
                                    "start": 35,
                                    "len": 1
                                }
                            },
                            "right": ""
                        }
                    ]
                },
                "then": {
                    "value": {
                        "concat": [
                            { "literal": "//" },
                            {
                                "substr": {
                                    "value": {
                                        "xml": ".Document.FICdtTrf.CdtTrfTxInf.RmtInf.Ustrd"
                                    },
                                    "start": 35,
                                    "len": 33
                                }
                            }
                        ]
                    }
                },
                "line_no": 40
            },
            {
                "when": {
                    "all": [
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "xml": ".Document.FICdtTrf.CdtTrfTxInf.RmtInf.Ustrd"
                                    },
                                    "start": 68,
                                    "len": 1
                                }
                            },
                            "right": ""
                        }
                    ]
                },
                "then": {
                    "value": {
                        "concat": [
                            { "literal": "//" },
                            {
                                "substr": {
                                    "value": {
                                        "xml": ".Document.FICdtTrf.CdtTrfTxInf.RmtInf.Ustrd"
                                    },
                                    "start": 68,
                                    "len": 33
                                }
                            }
                        ]
                    }
                },
                "line_no": 41
            }
        ],
    }
}

# -----------------------------------------------------------------------
# Ejecución mínima
# -----------------------------------------------------------------------
if __name__ == "__main__":
    from datetime import datetime
    from pathlib import Path

    # --- DEBUG de insumos clave para 53A ---
    dot_dbg = load_latest_dotmap(ubicationEntry)
    bic_instg = _get_by_suffix(
        dot_dbg,
        ".Document.FIToFICstmrCdtTrf.GrpHdr.SttlmInf.InstgRmbrsmntAgt.FinInstnId.BICFI",
    )
    iban_instg = _get_by_suffix(
        dot_dbg,
        ".Document.FIToFICstmrCdtTrf.GrpHdr.SttlmInf.InstgRmbrsmntAgtAcct.Id.IBAN",
    )
    clrsysref = _get_by_suffix(
        dot_dbg, ".Document.FIToFICstmrCdtTrf.CdtTrfTxInf.PmtId.ClrSysRef"
    )
    sttlm_mtd = _get_by_suffix(
        dot_dbg, ".Document.FIToFICstmrCdtTrf.GrpHdr.SttlmInf.SttlmMtd"
    )
    bic_list = get_list_from_values_db(valueDbSystemAll, "BICFI")
    dbg(
        f"XML.BICFI(InstgRmbrsmntAgt.FinInstnId)={bic_instg!r} | IBAN={iban_instg!r} | ClrSysRef={clrsysref!r} | SttlmMtd={sttlm_mtd!r}"
    )
    dbg(f"valueDbSystem.BICFI -> len={len(bic_list)} sample={bic_list[:10]}")

    out_lines = []

    # {1}{2} en una sola línea
    header_12 = build_header_12(ubicationEntry)

    # Contexto global compartido para todas las reglas
    globals_ctx = {}

    # :13C:
    fields_13C = build_fields(ubicationEntry, fields_spec_13C, global_vars=globals_ctx)
    lines_13C = [ln for ln in fields_13C.get("13C", []) if str(ln).strip() != ""]
    dbg(f"FINAL 13C lines (count={len(lines_13C)}):", lines_13C)
    if lines_13C:
        for ln in lines_13C:
            out_lines.append(f":13C:{ln}")
    else:
        dbg(
            "13C NO GENERADO - revisa las trazas anteriores para ver qué condición falló"
        )

    # :20:
    fields_20 = build_fields(ubicationEntry, fields_spec_20, global_vars=globals_ctx)
    if fields_20.get("20"):
        out_lines.append(f":20:{fields_20['20'][0]}")

    # :21: CAMPO NUEVO AGREGADO --> 8/10/2025 -----------------------------------------
    fields_21 = build_fields(ubicationEntry, fields_spec_21, global_vars=globals_ctx)
    lines_21 = [ln for ln in fields_21.get("21", []) if str(ln).strip() != ""]
    dbg(f"FINAL 21 lines (count={len(lines_21)}):", lines_21)
    if lines_21:
        out_lines.append(f":21:{lines_21[0]}")
        for ln in lines_21[1:]:
            out_lines.append(ln)
    else:
        dbg(
            "21 NO GENERADO - revisa las trazas anteriores para ver qué condición falló"
        )
    # -------------------------------------------------------------------------------------

    # :23B:
    fields_23B = build_fields(ubicationEntry, fields_spec_23B, global_vars=globals_ctx)
    if fields_23B.get("23B"):
        out_lines.append(f":23B:{fields_23B['23B'][0]}")

    # :23E:
    fields_23E = build_fields(ubicationEntry, fields_spec_23E, global_vars=globals_ctx)
    lines_23E = [ln for ln in fields_23E.get("23E", []) if str(ln).strip() != ""]
    dbg(f"FINAL 23E lines (count={len(lines_23E)}):", lines_23E)
    if lines_23E:
        for ln in lines_23E:
            out_lines.append(f":23E:{ln}")
    else:
        dbg(
            "23E NO GENERADO - revisa las trazas anteriores para ver qué condición falló"
        )

    # :32A:
    fields_32A = build_fields(ubicationEntry, fields_spec_32A, global_vars=globals_ctx)
    lines_32A = [ln for ln in fields_32A.get("32A", []) if str(ln).strip() != ""]
    dbg(f"FINAL 32A lines (count={len(lines_32A)}):", lines_32A)
    if lines_32A:
        out_lines.append(f":32A:{lines_32A[0]}")
    else:
        dbg(
            "32A NO GENERADO - revisa las trazas anteriores para ver qué condición falló"
        )
    dbg(f"GLOBALS DESPUÉS DE 32A: {globals_ctx}")

    # :33B:  (usa el MISMO globals_ctx)
    fields_33B = build_fields(ubicationEntry, fields_spec_33B, global_vars=globals_ctx)
    lines_33B = [ln for ln in fields_33B.get("33B", []) if str(ln).strip() != ""]
    dbg(f"FINAL 33B lines (count={len(lines_33B)}):", lines_33B)
    if lines_33B:
        out_lines.append(f":33B:{lines_33B[0]}")
    else:
        dbg(
            "33B NO GENERADO - revisa las trazas anteriores para ver qué condición falló"
        )

    # :36:
    fields_36 = build_fields(ubicationEntry, fields_spec_36, global_vars=globals_ctx)
    lines_36 = [ln for ln in fields_36.get("36", []) if str(ln).strip() != ""]
    dbg(f"FINAL 36 lines (count={len(lines_36)}):", lines_36)
    if lines_36:
        out_lines.append(f":36:{lines_36[0]}")
    else:
        dbg(
            "36 NO GENERADO - revisa las trazas anteriores para ver qué condición falló"
        )

    # :50A:
    fields_50A = build_fields(ubicationEntry, fields_spec_50A, global_vars=globals_ctx)
    lines_50A = [ln for ln in fields_50A.get("50A", []) if str(ln).strip() != ""]
    dbg(f"FINAL 50A lines (count={len(lines_50A)}):", lines_50A)
    if lines_50A:
        out_lines.append(f":50A:{lines_50A[0]}")
    else:
        dbg(
            "50A NO GENERADO - revisa las trazas anteriores para ver qué condición falló"
        )

    # :50F:
    fields_50F = build_fields(ubicationEntry, fields_spec_50F, global_vars=globals_ctx)
    lines_50F = [ln for ln in fields_50F.get("50F", []) if str(ln).strip() != ""]
    dbg(f"FINAL 50F lines (count={len(lines_50F)}):", lines_50F)
    if lines_50F:
        out_lines.append(f":50F:{lines_50F[0]}")
    else:
        dbg(
            "50F NO GENERADO - revisa las trazas anteriores para ver qué condición falló"
        )

    # :52A:
    fields_52A = build_fields(ubicationEntry, fields_spec_52A, global_vars=globals_ctx)
    lines_52A = [ln for ln in fields_52A.get("52A", []) if str(ln).strip() != ""]
    dbg("FINAL 52A lines:", lines_52A)
    if lines_52A:
        out_lines.append(f":52A:{lines_52A[0]}")
        for ln in lines_52A[1:]:
            out_lines.append(ln)
    else:
        dbg(
            "52A NO GENERADO - revisa las trazas anteriores para ver qué condición falló"
        )

    # :52C:
    fields_52C = build_fields(ubicationEntry, fields_spec_52C, global_vars=globals_ctx)
    lines_52C = [ln for ln in fields_52C.get("52C", []) if str(ln).strip() != ""]
    dbg("FINAL 52C lines:", lines_52C)
    if lines_52C:
        out_lines.append(f":52C:{lines_52C[0]}")
        for ln in lines_52C[1:]:
            out_lines.append(ln)
    else:
        dbg(
            "52C NO GENERADO - revisa las trazas anteriores para ver qué condición falló"
        )

    # :52D:
    fields_52D = build_fields(ubicationEntry, fields_spec_52D, global_vars=globals_ctx)
    lines_52D = [ln for ln in fields_52D.get("52D", []) if str(ln).strip() != ""]
    dbg("FINAL 52D lines:", lines_52D)
    if lines_52D:
        out_lines.append(f":52D:{lines_52D[0]}")
        for ln in lines_52D[1:]:
            out_lines.append(ln)
    else:
        dbg(
            "52D NO GENERADO - revisa las trazas anteriores para ver qué condición falló"
        )

    # :53A:
    fields_53A = build_fields(ubicationEntry, fields_spec_53A, global_vars=globals_ctx)
    lines_53A = [ln for ln in fields_53A.get("53A", []) if str(ln).strip() != ""]
    dbg("FINAL 53A lines:", lines_53A)
    if lines_53A:
        out_lines.append(f":53A:{lines_53A[0]}")
        for ln in lines_53A[1:]:
            out_lines.append(ln)
    else:
        dbg(
            "53A NO GENERADO - revisa las trazas anteriores para ver qué condición falló"
        )

    # :53B:
    fields_53B = build_fields(ubicationEntry, fields_spec_53B, global_vars=globals_ctx)
    lines_53B = [ln for ln in fields_53B.get("53B", []) if str(ln).strip() != ""]
    dbg("FINAL 53B lines:", lines_53B)
    if lines_53B:
        out_lines.append(f":53B:{lines_53B[0]}")
        for ln in lines_53B[1:]:
            out_lines.append(ln)
    else:
        dbg(
            "53B NO GENERADO - revisa las trazas anteriores para ver qué condición falló"
        )

    # --- DEBUG específico de 53D: inputs y globales ---
    strt = _get_by_suffix(
        dot_dbg,
        ".Document.FIToFICstmrCdtTrf.GrpHdr.SttlmInf.InstgRmbrsmntAgt.FinInstnId.PstlAdr.StrtNm",
    )
    twn = _get_by_suffix(
        dot_dbg,
        ".Document.FIToFICstmrCdtTrf.GrpHdr.SttlmInf.InstgRmbrsmntAgt.FinInstnId.PstlAdr.TwnNm",
    )
    ctry = _get_by_suffix(
        dot_dbg,
        ".Document.FIToFICstmrCdtTrf.GrpHdr.SttlmInf.InstgRmbrsmntAgt.FinInstnId.PstlAdr.Ctry",
    )
    dbg(
        f"53D INPUTS -> has_53B={bool(lines_53B)} | StrtNm={strt!r} | TwnNm={twn!r} | Ctry={ctry!r}"
    )

    # :53D:
    fields_53D = build_fields(ubicationEntry, fields_spec_53D, global_vars=globals_ctx)
    lines_53D = [ln for ln in fields_53D.get("53D", []) if str(ln).strip() != ""]
    dbg("FINAL 53D lines:", lines_53D)
    if lines_53D:
        out_lines.append(f":53D:{lines_53D[0]}")
        for ln in lines_53D[1:]:
            out_lines.append(ln)
    else:
        dbg(
            "53D NO GENERADO - revisa las trazas anteriores para ver qué condición falló"
        )

    # :54A:
    fields_54A = build_fields(ubicationEntry, fields_spec_54A, global_vars=globals_ctx)
    lines_54A = [ln for ln in fields_54A.get("54A", []) if str(ln).strip() != ""]
    dbg("FINAL 54A lines:", lines_54A)
    if lines_54A:
        out_lines.append(f":54A:{lines_54A[0]}")
        for ln in lines_54A[1:]:
            out_lines.append(ln)
    else:
        dbg(
            "54A NO GENERADO - revisa las trazas anteriores para ver qué condición falló"
        )

    # :54B: Campo Agregado 9/10/25-----------------------------------------------------
    fields_54B = build_fields(ubicationEntry, fields_spec_54B, global_vars=globals_ctx)
    lines_54B = [ln for ln in fields_54B.get("54B", []) if str(ln).strip() != ""]
    dbg("FINAL 54B lines:", lines_54B)
    if lines_54B:
        out_lines.append(f":54B:{lines_54B[0]}")
        for ln in lines_54B[1:]:
            out_lines.append(ln)
    else:
        dbg(
            "54B NO GENERADO - revisa las trazas anteriores para ver qué condición falló"
        )
    # -------------------------------------------------------------------------------------

    # :54D:
    fields_54D = build_fields(ubicationEntry, fields_spec_54D, global_vars=globals_ctx)
    lines_54D = [ln for ln in fields_54D.get("54D", []) if str(ln).strip() != ""]
    dbg("FINAL 54D lines:", lines_54D)
    if lines_54D:
        out_lines.append(f":54D:{lines_54D[0]}")
        for ln in lines_54D[1:]:
            out_lines.append(ln)
    else:
        dbg(
            "54D NO GENERADO - revisa las trazas anteriores para ver qué condición falló"
        )

    # :55A:
    fields_55A = build_fields(ubicationEntry, fields_spec_55A, global_vars=globals_ctx)
    lines_55A = [ln for ln in fields_55A.get("55A", []) if str(ln).strip() != ""]
    dbg("FINAL 55A lines:", lines_55A)
    if lines_55A:
        out_lines.append(f":55A:{lines_55A[0]}")
        for ln in lines_55A[1:]:
            out_lines.append(ln)
    else:
        dbg(
            "55A NO GENERADO - revisa las trazas anteriores para ver qué condición falló"
        )

    # :55D:
    fields_55D = build_fields(ubicationEntry, fields_spec_55D, global_vars=globals_ctx)
    lines_55D = [ln for ln in fields_55D.get("55D", []) if str(ln).strip() != ""]
    dbg("FINAL 55D lines:", lines_55D)
    if lines_55D:
        out_lines.append(f":55D:{lines_55D[0]}")
        for ln in lines_55D[1:]:
            out_lines.append(ln)
    else:
        dbg(
            "55D NO GENERADO - revisa las trazas anteriores para ver qué condición falló"
        )

    # :56A:
    fields_56A = build_fields(ubicationEntry, fields_spec_56A, global_vars=globals_ctx)
    lines_56A = [ln for ln in fields_56A.get("56A", []) if str(ln).strip() != ""]
    dbg("FINAL 56A lines:", lines_56A)
    if lines_56A:
        out_lines.append(f":56A:{lines_56A[0]}")
        for ln in lines_56A[1:]:
            out_lines.append(ln)
    else:
        dbg(
            "56A NO GENERADO - revisa las trazas anteriores para ver qué condición falló"
        )

    # :56C:
    fields_56C = build_fields(ubicationEntry, fields_spec_56C, global_vars=globals_ctx)
    lines_56C = [ln for ln in fields_56C.get("56C", []) if str(ln).strip() != ""]
    dbg("FINAL 56C lines:", lines_56C)
    if lines_56C:
        out_lines.append(f":56C:{lines_56C[0]}")
        for ln in lines_56C[1:]:
            out_lines.append(ln)
    else:
        dbg(
            "56C NO GENERADO - revisa las trazas anteriores para ver qué condición falló"
        )

    # :57A:
    fields_57A = build_fields(ubicationEntry, fields_spec_57A, global_vars=globals_ctx)
    lines_57A = [ln for ln in fields_57A.get("57A", []) if str(ln).strip() != ""]
    dbg("FINAL 57A lines:", lines_57A)
    if lines_57A:
        out_lines.append(f":57A:{lines_57A[0]}")
        for ln in lines_57A[1:]:
            out_lines.append(ln)
    else:
        dbg(
            "57A NO GENERADO - revisa las trazas anteriores para ver qué condición falló"
        )

    # :57B: Campo Agregado 9/10/25-----------------------------------------------------
    fields_57B = build_fields(ubicationEntry, fields_spec_57B, global_vars=globals_ctx)
    lines_57B = [ln for ln in fields_57B.get("57B", []) if str(ln).strip() != ""]
    dbg("FINAL 57B lines:", lines_57B)
    if lines_57B:
        out_lines.append(f":57B:{lines_57B[0]}")
        for ln in lines_57B[1:]:
            out_lines.append(ln)
    else:
        dbg(
            "57B NO GENERADO - revisa las trazas anteriores para ver qué condición falló"
        )
    # -------------------------------------------------------------------------------------

    # :57C:
    fields_57C = build_fields(ubicationEntry, fields_spec_57C, global_vars=globals_ctx)
    lines_57C = [ln for ln in fields_57C.get("57C", []) if str(ln).strip() != ""]
    dbg("FINAL 57C lines:", lines_57C)
    if lines_57C:
        out_lines.append(f":57C:{lines_57C[0]}")
        for ln in lines_57C[1:]:
            out_lines.append(ln)
    else:
        dbg(
            "57C NO GENERADO - revisa las trazas anteriores para ver qué condición falló"
        )

    # :58A:  (imprimir si existe; 58D queda como fallback)-----------------------------
    fields_58A = build_fields(ubicationEntry, fields_spec_58A, global_vars=globals_ctx)
    lines_58A = [ln for ln in fields_58A.get("58A", []) if str(ln).strip() != ""]
    dbg(f"FINAL 58A lines (count={len(lines_58A)}):", lines_58A)
    if lines_58A:
        out_lines.append(f":58A:{lines_58A[0]}")
        for ln in lines_58A[1:]:
            out_lines.append(ln)
    else:
        dbg(
            "58A NO GENERADO - revisa las trazas anteriores para ver qué condición falló"
        )
    # ----------------------------------------------------------------------------------

    # :58D: CAMPO AGREGADO 8/10/25------------------------------------------------------
    fields_58D = build_fields(ubicationEntry, fields_spec_58D, global_vars=globals_ctx)
    lines_58D = [ln for ln in fields_58D.get("58D", []) if str(ln).strip() != ""]
    dbg(f"FINAL 58D lines (count={len(lines_58D)}):", lines_58D)
    if lines_58D:
        out_lines.append(f":58D:{lines_58D[0]}")
        for ln in lines_58D[1:]:
            out_lines.append(ln)
    else:
        dbg(
            "58D NO GENERADO - revisa las trazas anteriores para ver qué condición falló"
        )
    # -------------------------------------------------------------------------------------

    # :59:
    fields_59 = build_fields(ubicationEntry, fields_spec_59, global_vars=globals_ctx)
    lines_59 = [ln for ln in fields_59.get("59", []) if str(ln).strip() != ""]
    dbg("FINAL 59 lines:", lines_59)
    if lines_59:
        out_lines.append(f":59:{lines_59[0]}")
        for ln in lines_59[1:]:
            out_lines.append(ln)
    else:
        dbg(
            "59 NO GENERADO - revisa las trazas anteriores para ver qué condición falló"
        )

    # :59F:
    fields_59F = build_fields(ubicationEntry, fields_spec_59F, global_vars=globals_ctx)
    lines_59F = [ln for ln in fields_59F.get("59F", []) if str(ln).strip() != ""]
    dbg("FINAL 59F lines:", lines_59F)
    if lines_59F:
        out_lines.append(f":59F:{lines_59F[0]}")
        for ln in lines_59F[1:]:
            out_lines.append(ln)
    else:
        dbg(
            "59F NO GENERADO - revisa las trazas anteriores para ver qué condición falló"
        )

    # :70:
    fields_70 = build_fields(ubicationEntry, fields_spec_70, global_vars=globals_ctx)
    lines_70 = [ln for ln in fields_70.get("70", []) if str(ln).strip() != ""]
    dbg("FINAL 70 lines:", lines_70)
    if lines_70:
        out_lines.append(f":70:{lines_70[0]}")
        for ln in lines_70[1:]:
            out_lines.append(ln)
    else:
        dbg(
            "70 NO GENERADO - revisa las trazas anteriores para ver qué condición falló"
        )

    # :71A:
    fields_71A = build_fields(ubicationEntry, fields_spec_71A, global_vars=globals_ctx)
    lines_71A = [ln for ln in fields_71A.get("71A", []) if str(ln).strip() != ""]
    dbg("FINAL 71A lines:", lines_71A)
    if lines_71A:
        out_lines.append(f":71A:{lines_71A[0]}")
        for ln in lines_71A[1:]:
            out_lines.append(ln)
    else:
        dbg(
            "71A NO GENERADO - revisa las trazas anteriores para ver qué condición falló"
        )

    # :71F:
    fields_71F = build_fields(ubicationEntry, fields_spec_71F, global_vars=globals_ctx)
    lines_71F = [ln for ln in fields_71F.get("71F", []) if str(ln).strip() != ""]
    dbg("FINAL 71F lines:", lines_71F)
    if lines_71F:
        for ln in lines_71F:
            out_lines.append(f":71F:{ln}")
    else:
        dbg(
            "71F NO GENERADO - revisa las trazas anteriores para ver qué condición falló"
        )

    # :71G:
    fields_71G = build_fields(ubicationEntry, fields_spec_71G, global_vars=globals_ctx)
    lines_71G = [ln for ln in fields_71G.get("71G", []) if str(ln).strip() != ""]
    dbg("FINAL 71G lines:", lines_71G)
    if lines_71G:
        out_lines.append(f":71G:{lines_71G[0]}")
        for ln in lines_71G[1:]:
            out_lines.append(ln)
    else:
        dbg(
            "71G NO GENERADO - revisa las trazas anteriores para ver qué condición falló"
        )

    # :72:
    fields_72 = build_fields(ubicationEntry, fields_spec_72, global_vars=globals_ctx)
    lines_72 = [ln for ln in fields_72.get("72", []) if str(ln).strip() != ""]
    dbg("FINAL 72 lines:", lines_72)
    if lines_72:
        out_lines.append(f":72:{lines_72[0]}")
        for ln in lines_72[1:]:
            out_lines.append(ln)
    else:
        dbg(
            "72 NO GENERADO - revisa las trazas anteriores para ver qué condición falló"
        )

    # === escribir archivo ===
    ts = datetime.now().strftime("%Y%m%d%H%M%S")  # AAAAMMDDHHSS
    fname = f"MT202_{ts}.txt"
    out_path = Path(ubicationDestiny) / fname
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mt = finalize_mt_message(build_header_12(ubicationEntry), out_lines)
    out_path.write_text(mt, encoding="utf-8")

    print(f"{mt}")
