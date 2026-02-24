import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Callable
import pandas as pd
import json
from datetime import datetime, timezone, timedelta
import threading
from zoneinfo import ZoneInfo

# === DEBUG ===
DEBUG = False
TRUNCATION_WARNINGS = []


def dbg(*a):
    if DEBUG:
        print("[DBG]", *a)


# -----------------------------------------------------------------------
# Archivos de entrada
# -----------------------------------------------------------------------
traslateId = "011"


ubicationEntry = r"C:\Users\heoctor\Desktop\WANT\BANREP\Translete-Update\Translate Sin Comentarios\011\entry"
ubicationDestiny = r"C:\Users\heoctor\Desktop\WANT\BANREP\Translete-Update\Translate Sin Comentarios\011\destiny"
ubicationDb = r"C:\Users\heoctor\Desktop\WANT\BANREP\Archivos"

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


def _get_all_by_suffix(
    dotmap: Dict[str, Union[str, List[str]]], suffix: str
) -> List[str]:
    """Devuelve TODOS los valores que coinciden con el suffix (para múltiples AdrLine, etc.)"""
    if suffix.startswith("."):
        suffix = suffix[1:]
    for k, v in dotmap.items():
        if k.endswith(suffix):
            if isinstance(v, list):
                return v
            else:
                return [v]
    return []


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


def format_hhmm(dt_str: str) -> str:
    """Devuelve HHMM (sin offset) a partir de un ISO8601."""
    dt = _parse_iso_dt_norm(dt_str)
    return dt.strftime("%H%M") if dt else ""


def format_yymmdd(dt_str: str) -> str:
    """Devuelve YYMMDD (sin offset) a partir de un ISO8601."""
    dt = _parse_iso_dt_norm(dt_str)
    return dt.strftime("%y%m%d") if dt else ""


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


# Especificaciones de Bloques
spec_bloque1 = [
    {"fixed": "F01"},
    {
        "key": ".AppHdr.To.FIId.FinInstnId.BICFI",
        "post": "bic11",
    },
    {"var": "HDR_SESS_SEQ", "pad": 10, "fill": "0"},
]
spec_bloque2 = [
    {"fixed": "O900"},
    {"var": "HDR_HHMM"},
    {"var": "HDR_YYMMDD"},
    {
        "key": ".AppHdr.Fr.FIId.FinInstnId.BICFI",
        "pad": 11,
        "fill": "X",
    },
    {"fixed": "X0000000000"},
    {"var": "HDR_YYMMDD"},
    {"fixed": "0000N"},
]
spec_bloque3 = [
    {"fixed": "{121:"},
    {"key": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.NtryDtls.TxDtls.Refs.UETR"},
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
        "HDR_SESS_SEQ": sess_seq,
    }

    b1 = build_block(entry_dir, 1, spec_bloque1, vars_ctx=vars_ctx)
    b2 = build_block(entry_dir, 2, spec_bloque2, vars_ctx=vars_ctx)

    uetr = _get_by_suffix(
        dot, ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.NtryDtls.TxDtls.Refs.UETR"
    )
    if uetr and str(uetr).strip():
        b3 = build_block(entry_dir, 3, spec_bloque3)
    else:
        dbg("Bloque 3 omitido: no hay UETR")
        b3 = ""

    return b1 + b2 + b3 + "{4:"


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


block_post_processors = {"bic11": normalize_bic11}


def parse_apphdr_boolean(value):
    # 1) value es None
    if value is None:
        return False, None, "PSSBLDPLCT_NULL"

    # 2) trim
    s = str(value).strip()

    # 3) vacío tras trim
    if s == "":
        return False, None, "PSSBLDPLCT_EMPTY"

    # 4) comparar case-insensitive
    lower = s.lower()
    if lower == "true":
        return True, True, None
    if lower == "false":
        return True, False, None

    # 5) cualquier otro caso -> inválido
    return False, None, "PSSBLDPLCT_INVALID_VALUE"


def generate_chk(content: str) -> str:
    """
    Genera un checksum de 12 caracteres hexadecimales para el trailer {CHK:}.
    Usa un hash MD5 truncado del contenido del mensaje (Block 4 normalmente).
    """
    import hashlib

    hash_md5 = hashlib.md5(content.encode("utf-8")).hexdigest().upper()
    return hash_md5[:12]  # Tomar los primeros 12 caracteres


def build_block5(entry_dir: str | dict, block4_content: str = "") -> str:
    """
    Construye el Block 5 del mensaje MT.

    - Lee /AppHdr/PssblDplct:
        * Si es 'true' (case-insensitive, con trim) -> añade {PDE:}
        * Si es 'false' -> no añade {PDE:}
        * Si es inválido -> no añade {PDE:} y loguea error

    - Siempre incluye {CHK:checksum} al final.
    """
    # dot: DotMap o dict con el XML ya cargado
    if isinstance(entry_dir, dict):
        dot = entry_dir
    else:
        dot = load_latest_dotmap(entry_dir)

    trailers = []

    # 1) Leer /AppHdr/PssblDplct
    #    Usamos tu helper _get_by_suffix, que busca por sufijo en el DotMap.
    pssbl_dplct = _get_by_suffix(dot, ".AppHdr.PssblDplct")

    # 2) Normalizar y validar con parse_apphdr_boolean
    is_valid, result_boolean, error_code = parse_apphdr_boolean(pssbl_dplct)

    # 3) Decidir presencia de {PDE:}
    if is_valid and result_boolean is True:
        # Solo si el booleano es True añadimos PDE
        trailers.append("{PDE:}")
        dbg(f"Block 5: Añadido {{PDE:}} porque PssblDplct={pssbl_dplct!r}")
    elif not is_valid and error_code not in (None, "PSSBLDPLCT_NULL"):
        # Si hay error de valor (EMPTY o INVALID_VALUE), solo logueamos
        dbg(f"Block 5: PssblDplct inválido={pssbl_dplct!r}, error={error_code}")
    else:
        # Ausente, false o NULL -> no se añade PDE
        dbg(f"Block 5: PssblDplct={pssbl_dplct!r} -> No se añade {{PDE:}}")

    # 4) Siempre añadir {CHK:} al final
    chk_value = generate_chk(block4_content or "")
    trailers.append(f"{{CHK:{chk_value}}}")

    # 5) Construir Block 5 completo
    return "{5:" + "".join(trailers) + "}"


def finalize_mt_message(
    header_12: str, lines_block4: list[str], block5: str = ""
) -> str:
    """
    Construye el mensaje MT completo con todos los bloques.
    """
    msg = header_12 + "\n" + "\n".join(lines_block4) + "\n-}"
    if block5:
        msg += block5
    return msg


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
        elif out == "YYMMDD":
            return format_yymmdd(iso_val)

    if "fatf_id" in vs:
        # Lógica específica para :50F: Party Identifier
        # /Code/Ctry/Issuer/Identifier
        conf = vs["fatf_id"]
        base_path = conf.get("base_path", "")

        # Resolver paths relativos
        def get_val(subpath):
            full_path = base_path + subpath
            return str(_get_by_suffix(dot, full_path) or "").strip()

        # 1. Code (Othr/SchmeNm/Prtry) -> default CUST
        code = get_val(".Id.OrgId.Othr.SchmeNm.Prtry")
        if not code:
            code = get_val(".Id.PrvtId.Othr.SchmeNm.Prtry")

        if not code or len(code) != 4 or not code.isalpha():
            code = "CUST"
        else:
            code = code.upper()

        # 2. Ctry (PstlAdr/Ctry)
        ctry = get_val(".PstlAdr.Ctry").upper()

        # 3. Issuer (Othr/Issr)
        issuer = get_val(".Id.OrgId.Othr.Issr")
        if not issuer:
            issuer = get_val(".Id.PrvtId.Othr.Issr")
        issuer = issuer.upper()

        # 4. Identifier (Othr/Id)
        ident = get_val(".Id.OrgId.Othr.Id")
        if not ident:
            ident = get_val(".Id.PrvtId.Othr.Id")

        # Construir segmentos
        parts = ["", code]  # Empieza con /
        if ctry:
            parts.append(ctry)
        if issuer:
            parts.append(issuer)
        parts.append(ident)

        res = "/".join(parts)

        def _clean_finx(s: str) -> str:
            import re

            return re.sub(r"[^A-Za-z0-9/ ]", "", s).strip()

        res = _clean_finx(res)[:35]
        return res

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


def _truncate_lines_for_field(
    field_tag: str, lines: list[str], src_path: str | None = None
) -> list[str]:
    """
    Trunca o divide cada línea en bloques de máx. 35 caracteres para campos:
    50, 51, 52, 53, 54, 55, 56, 57, 59, 70, 72, 77
    (se aplica también a variantes: 50A, 50F, 50K, 52D, 53A, 53B, etc.)

    Si una línea supera los 35 caracteres, se parte en varias líneas
    de 35 caracteres cada una.
    """
    # Prefijos de campos a los que se les aplica el límite de 35 caracteres
    fields_35_prefixes = (
        "50",
        "51",
        "52",
        "53",
        "54",
        "55",
        "56",
        "57",
        "59",
        "70",
        "72",
        "77",
    )

    # Si el campo no está en la lista, se devuelve tal cual
    if not any(str(field_tag).startswith(p) for p in fields_35_prefixes):
        return lines

    out: list[str] = []
    for ln in lines:
        s = "" if ln is None else str(ln)

        # START MODIFICATION: Trace truncated/split lines (Simulate SWIFT Exception Report)
        if len(s) > 35:
            entry = {
                "mt_tag": field_tag,
                "original": s,
                "max_len": 35,
                "target_path": src_path or "",
            }
            if not any(
                isinstance(x, dict)
                and x.get("mt_tag") == entry["mt_tag"]
                and x.get("original") == entry["original"]
                for x in TRUNCATION_WARNINGS
            ):
                TRUNCATION_WARNINGS.append(entry)
        # END MODIFICATION

        # Partir la cadena en bloques de 35 caracteres
        for i in range(0, len(s), 35):
            out.append(s[i : i + 35])
    return out


def _dotpath_to_envelope(dot_path: str) -> str:
    """Convierte '.Document.X.Y' en 'Envelope/Document/X/Y' para el reporte."""
    p = str(dot_path or "")
    if p.startswith("."):
        p = p[1:]
    # compat: no mostrar Body en paths (tu estándar usa Envelope/AppHdr y Envelope/Document)
    if p.startswith("Body."):
        p = p[5:]

    return "Envelope/" + p.replace(".", "/")


def _collect_then_lines(dot, ctx, then_obj, field_tag: str | None = None):
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
        # intentar extraer la ruta XML fuente desde then_obj (si existe)
        src_path = None

        def _extract_path(obj):
            if isinstance(obj, dict):
                if "xml" in obj and isinstance(obj["xml"], str):
                    return obj["xml"]
                if "xml_nth" in obj and isinstance(obj["xml_nth"], dict):
                    return obj["xml_nth"].get("path")
                for v in obj.values():
                    p = _extract_path(v)
                    if p:
                        return p
            if isinstance(obj, list):
                for it in obj:
                    p = _extract_path(it)
                    if p:
                        return p
            return None

        if "value" in then_obj:
            src_path = _extract_path(then_obj["value"])
        elif "lines" in then_obj:
            for vs in then_obj["lines"]:
                src_path = _extract_path(vs)
                if src_path:
                    break

        lines = _truncate_lines_for_field(field_tag, lines, src_path)

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
                produced = _collect_then_lines(
                    dot, ctx, rule.get("then", {}), field_tag
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
                    dot, ctx, rule.get("then", {}), field_tag
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
        {
            # 1) MsgId > 16 -> truncar a 15 y añadir "+" SI es válido (sin barras inválidas)
            "when": {
                "all": [
                    {
                        "op": "!=",
                        "left": {"xml": ".Document.BkToCstmrDbtCdtNtfctn.GrpHdr.MsgId"},
                        "right": "",
                    },
                    # longitud > 16  => el caracter 17 (posición 16) existe
                    {
                        "op": "!=",
                        "left": {
                            "substr": {
                                "value": {
                                    "xml": ".Document.BkToCstmrDbtCdtNtfctn.GrpHdr.MsgId"
                                },
                                "start": 16,
                                "len": 1,
                            }
                        },
                        "right": "",
                    },
                ],
                "not": {
                    "any": [
                        # NO debe empezar con "/"
                        {
                            "op": "=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "xml": ".Document.BkToCstmrDbtCdtNtfctn.GrpHdr.MsgId"
                                    },
                                    "start": 0,
                                    "len": 1,
                                }
                            },
                            "right": "/",
                        },
                        # NO debe contener "//" en los primeros 15 caracteres
                        # (repite el bloque start=0..13)
                        {
                            "op": "=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "xml": ".Document.BkToCstmrDbtCdtNtfctn.GrpHdr.MsgId"
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
                                "substr": {
                                    "value": {
                                        "xml": ".Document.BkToCstmrDbtCdtNtfctn.GrpHdr.MsgId"
                                    },
                                    "start": 1,
                                    "len": 2,
                                }
                            },
                            "right": "//",
                        },
                        {
                            "op": "=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "xml": ".Document.BkToCstmrDbtCdtNtfctn.GrpHdr.MsgId"
                                    },
                                    "start": 2,
                                    "len": 2,
                                }
                            },
                            "right": "//",
                        },
                        # ... seguir con start=3,4,...,13 siempre len=2
                    ]
                },
            },
            "then": {
                "value": {
                    "concat": [
                        {
                            "substr": {
                                "value": {
                                    "xml": ".Document.BkToCstmrDbtCdtNtfctn.GrpHdr.MsgId"
                                },
                                "start": 0,
                                "len": 15,
                            }
                        },
                        {"literal": "+"},
                    ]
                }
            },
            "set": [
                {
                    "set_var": {
                        "name": "has_20",
                        "value": {"literal": "1"},
                        "scope": "global",
                    }
                },
                {
                    # flag para que el manejador de errores registre la TRUNCACIÓN
                    "set_var": {
                        "name": "trunc_20_910",
                        "value": {"literal": "1"},
                        "scope": "global",
                    }
                },
            ],
        },
        {
            # 2) MsgId <= 16 -> copiar MsgId SI es válido (sin barras inválidas)
            "when": {
                "all": [
                    {
                        "op": "!=",
                        "left": {"xml": ".Document.BkToCstmrDbtCdtNtfctn.GrpHdr.MsgId"},
                        "right": "",
                    },
                    # longitud <= 16 => el caracter 17 (posición 16) NO existe
                    {
                        "op": "=",
                        "left": {
                            "substr": {
                                "value": {
                                    "xml": ".Document.BkToCstmrDbtCdtNtfctn.GrpHdr.MsgId"
                                },
                                "start": 16,
                                "len": 1,
                            }
                        },
                        "right": "",
                    },
                ],
                "not": {
                    "any": [
                        # NO debe empezar con "/"
                        {
                            "op": "=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "xml": ".Document.BkToCstmrDbtCdtNtfctn.GrpHdr.MsgId"
                                    },
                                    "start": 0,
                                    "len": 1,
                                }
                            },
                            "right": "/",
                        },
                        # NO debe terminar con "/"
                        {
                            "op": "=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "xml": ".Document.BkToCstmrDbtCdtNtfctn.GrpHdr.MsgId"
                                    },
                                    "start": -1,  # último caracter
                                    "len": 1,
                                }
                            },
                            "right": "/",
                        },
                        # NO debe contener "//" en ninguna posición (0..14)
                        {
                            "op": "=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "xml": ".Document.BkToCstmrDbtCdtNtfctn.GrpHdr.MsgId"
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
                                "substr": {
                                    "value": {
                                        "xml": ".Document.BkToCstmrDbtCdtNtfctn.GrpHdr.MsgId"
                                    },
                                    "start": 1,
                                    "len": 2,
                                }
                            },
                            "right": "//",
                        },
                        {
                            "op": "=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "xml": ".Document.BkToCstmrDbtCdtNtfctn.GrpHdr.MsgId"
                                    },
                                    "start": 2,
                                    "len": 2,
                                }
                            },
                            "right": "//",
                        },
                        # ... seguir con start=3,4,...,14 siempre len=2
                    ]
                },
            },
            "then": {"value": {"xml": ".Document.BkToCstmrDbtCdtNtfctn.GrpHdr.MsgId"}},
            "set": [
                {
                    "set_var": {
                        "name": "has_20",
                        "value": {"literal": "1"},
                        "scope": "global",
                    }
                }
            ],
        },
        {
            # 3) MsgId informado pero NO pasa validación -> :20: = "NOTPROVIDED" + T14001
            # (cubre tanto el caso >16 como <=16, porque las válidas ya cayeron en las reglas 1 y 2)
            "when": {
                "all": [
                    {
                        "op": "!=",
                        "left": {"xml": ".Document.BkToCstmrDbtCdtNtfctn.GrpHdr.MsgId"},
                        "right": "",
                    }
                ]
            },
            "then": {"value": {"literal": "NOTPROVIDED"}},
            "set": [
                {
                    "set_var": {
                        "name": "has_20",
                        "value": {"literal": "1"},
                        "scope": "global",
                    }
                },
                {
                    # flag de error T14001 (barras inválidas en :20:)
                    "set_var": {
                        "name": "err_T14001_20",
                        "value": {"literal": "1"},
                        "scope": "global",
                    }
                },
            ],
        },
        {
            # 4) MsgId vacío -> también "NOTPROVIDED" (sin T14001, opcional según negocio)
            "when": {
                "all": [
                    {
                        "op": "=",
                        "left": {"xml": ".Document.BkToCstmrDbtCdtNtfctn.GrpHdr.MsgId"},
                        "right": "",
                    }
                ]
            },
            "then": {"value": {"literal": "NOTPROVIDED"}},
            "set": [
                {
                    "set_var": {
                        "name": "has_20",
                        "value": {"literal": "1"},
                        "scope": "global",
                    }
                }
            ],
        },
    ]
}
fields_spec_21 = {
    "21": [
        {
            # Regla 1: Entry de CRÉDITO (CRDT) con InstrId informado
            "when": {
                "all": [
                    {
                        "op": "=",
                        "left": {
                            "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.CdtDbtInd"
                        },
                        "right": "CRDT",
                    },
                    {
                        "op": "!=",
                        "left": {
                            "xml": (
                                ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.NtryDtls.TxDtls.Refs.InstrIdd"
                            )
                        },
                        "right": "",
                    },
                ]
            },
            "then": {
                "value": {
                    # TR002A simplificado: usar InstrId truncado a 16 caracteres
                    "substr": {
                        "value": {
                            "xml": (
                                ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.NtryDtls.TxDtls.Refs.InstrId"
                            )
                        },
                        "start": 0,
                        "len": 16,
                    }
                }
            },
            "set": [
                {
                    "set_var": {
                        "name": "rule_21_mode",
                        "value": {"literal": "TR002A"},
                        "scope": "global",
                    }
                }
            ],
        },
        {
            # Regla 2: Entry NO CRÉDITO (DBIT u otros) con InstrId informado
            #   -> :21: = InstrId truncado a 16 chars (equivalente a TR002B simplificado)
            "when": {
                "all": [
                    {
                        "op": "!=",
                        "left": {
                            "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.CdtDbtInd"
                        },
                        "right": "CRDT",
                    },
                    {
                        "op": "!=",
                        "left": {
                            "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.NtryDtls.TxDtls.Refs.InstrId"
                        },
                        "right": "",
                    },
                ]
            },
            "then": {
                "value": {
                    # TR002B simplificado: usar InstrId truncado a 16 caracteres
                    "substr": {
                        "value": {
                            "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.NtryDtls.TxDtls.Refs.InstrId"
                        },
                        "start": 0,
                        "len": 16,
                    }
                }
            },
            "set": [
                {
                    # Flag global para saber que se aplicó rama "no crédito"
                    "set_var": {
                        "name": "rule_21_mode",
                        "value": {"literal": "TR002B"},
                        "scope": "global",
                    }
                }
            ],
        },
        {
            # Regla 3 (catch-all): sin InstrId -> :21:NOTPROVIDED
            # Mantiene la presencia obligatoria del campo :21:
            "when": {
                "all": [
                    {
                        "op": "=",
                        "left": {
                            "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.NtryDtls.TxDtls.Refs.InstrId"
                        },
                        "right": "",
                    }
                ]
            },
            "then": {"value": {"literal": "NOTPROVIDED"}},
            "set": [
                {
                    "set_var": {
                        "name": "rule_21_mode",
                        "value": {
                            "map": {
                                "input": {
                                    "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.CdtDbtInd"
                                },
                                "map": {
                                    "CRDT": "TR002A",
                                },
                                "default": "TR002B",
                            }
                        },
                        "scope": "global",
                    }
                }
            ],
        },
    ]
}
fields_spec_25 = {
    "25": [
        {
            "when": {
                "all": [
                    {
                        "op": "!=",
                        "left": {
                            "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Acct.Id.IBAN"
                        },
                        "right": "",
                    }
                ]
            },
            "then": {
                "value": {"xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Acct.Id.IBAN"}
            },
        },
        {
            "when": {
                "all": [
                    {
                        "op": "=",
                        "left": {
                            "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Acct.Id.IBAN"
                        },
                        "right": "",
                    },
                    {
                        "op": "!=",
                        "left": {
                            "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Acct.Id.Othr.Id"
                        },
                        "right": "",
                    },
                ]
            },
            "then": {
                "value": {
                    "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Acct.Id.Othr.Id"
                }
            },
        },
        {
            "when": {
                "all": [
                    {
                        "op": "=",
                        "left": {
                            "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Acct.Id.IBAN"
                        },
                        "right": "",
                    },
                    {
                        "op": "=",
                        "left": {
                            "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Acct.Id.Othr.Id"
                        },
                        "right": "",
                    },
                ]
            }
        },
    ]
}
fields_spec_13D = {
    "13D": [
        {
            "when": {
                "all": [
                    # 1) Debe existir BookgDt/DtTm
                    {
                        "op": "!=",
                        "left": {
                            "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.BookgDt.DtTm"
                        },
                        "right": "",
                    },
                    # 2) Deben existir partes de fecha (YYYY-MM-DD)
                    {
                        "op": "!=",
                        "left": {
                            "substr": {
                                "value": {
                                    "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.BookgDt.DtTm"
                                },
                                "start": 0,
                                "len": 4,  # YYYY
                            }
                        },
                        "right": "",
                    },
                    {
                        "op": "!=",
                        "left": {
                            "substr": {
                                "value": {
                                    "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.BookgDt.DtTm"
                                },
                                "start": 5,
                                "len": 2,  # MM
                            }
                        },
                        "right": "",
                    },
                    {
                        "op": "!=",
                        "left": {
                            "substr": {
                                "value": {
                                    "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.BookgDt.DtTm"
                                },
                                "start": 8,
                                "len": 2,  # DD
                            }
                        },
                        "right": "",
                    },
                    # 3) La conversión de hora/offset debe ser válida (si falla -> "")
                    {
                        "op": "!=",
                        "left": {
                            "dtfmt": {
                                "value": {
                                    "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.BookgDt.DtTm"
                                },
                                "out": "HHMM±HHMM",
                            }
                        },
                        "right": "",
                    },
                    # 4) MTDate (YYYYMMDD) != 99991231 (centinela)
                    {
                        "op": "!=",
                        "left": {
                            "concat": [
                                {
                                    "substr": {
                                        "value": {
                                            "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.BookgDt.DtTm"
                                        },
                                        "start": 0,
                                        "len": 4,  # YYYY
                                    }
                                },
                                {
                                    "substr": {
                                        "value": {
                                            "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.BookgDt.DtTm"
                                        },
                                        "start": 5,
                                        "len": 2,  # MM
                                    }
                                },
                                {
                                    "substr": {
                                        "value": {
                                            "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.BookgDt.DtTm"
                                        },
                                        "start": 8,
                                        "len": 2,  # DD
                                    }
                                },
                            ]
                        },
                        "right": "99991231",
                    },
                ]
            },
            "then": {
                "value": {
                    "concat": [
                        # dateYYMMDD = MTDate (YYYYMMDD) sin siglo => YYMMDD
                        {
                            "substr": {
                                "value": {
                                    "concat": [
                                        {
                                            "substr": {
                                                "value": {
                                                    "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.BookgDt.DtTm"
                                                },
                                                "start": 0,
                                                "len": 4,  # YYYY
                                            }
                                        },
                                        {
                                            "substr": {
                                                "value": {
                                                    "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.BookgDt.DtTm"
                                                },
                                                "start": 5,
                                                "len": 2,  # MM
                                            }
                                        },
                                        {
                                            "substr": {
                                                "value": {
                                                    "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.BookgDt.DtTm"
                                                },
                                                "start": 8,
                                                "len": 2,  # DD
                                            }
                                        },
                                    ]
                                },
                                "start": 2,
                                "len": 6,
                            }
                        },
                        # MTTimeOffset = HHMM±HHMM
                        {
                            "dtfmt": {
                                "value": {
                                    "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.BookgDt.DtTm"
                                },
                                "out": "HHMM±HHMM",
                            }
                        },
                    ]
                }
            },
        }
    ]
}
fields_spec_32A = {
    "32A": [
        # ------------------------------------------------------------------
        # 1) Caso preferente: ValDt/Dt (YYYY-MM-DD)
        # ------------------------------------------------------------------
        {
            "when": {
                "all": [
                    # Debe existir ValDt/Dt
                    {
                        "op": "!=",
                        "left": {
                            "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.ValDt.Dt"
                        },
                        "right": "",
                    },
                    # Debe existir Amt/@Ccy
                    {
                        "op": "!=",
                        "left": {
                            "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.Amt[@Ccy]"
                        },
                        "right": "",
                    },
                    # Debe existir Amt (importe)
                    {
                        "op": "!=",
                        "left": {
                            "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.Amt"
                        },
                        "right": "",
                    },
                ]
            },
            "then": {
                "value": {
                    "concat": [
                        # MTDate en YYMMDD (a partir de YYYY-MM-DD → YYYYMMDD → YYMMDD)
                        {
                            "substr": {
                                "value": {
                                    "concat": [
                                        {
                                            "substr": {
                                                "value": {
                                                    "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.ValDt.Dt"
                                                },
                                                "start": 0,
                                                "len": 4,  # YYYY
                                            }
                                        },
                                        {
                                            "substr": {
                                                "value": {
                                                    "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.ValDt.Dt"
                                                },
                                                "start": 5,
                                                "len": 2,  # MM
                                            }
                                        },
                                        {
                                            "substr": {
                                                "value": {
                                                    "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.ValDt.Dt"
                                                },
                                                "start": 8,
                                                "len": 2,  # DD
                                            }
                                        },
                                    ]
                                },
                                "start": 2,  # quitar siglo -> YYMMDD
                                "len": 6,
                            }
                        },
                        # Divisa (Ccy)
                        {
                            "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.Amt[@Ccy]"
                        },
                        # Importe formateado en formato MT (numfmt)
                        {
                            "numfmt": {
                                "value": {
                                    "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.Amt"
                                }
                            }
                        },
                    ]
                }
            },
        },
        # ------------------------------------------------------------------
        # 2) Si no hay ValDt/Dt, usar ValDt/DtTm (YYYY-MM-DDThh:mm:ss...)
        # ------------------------------------------------------------------
        {
            "when": {
                "all": [
                    # NO existe ValDt/Dt
                    {
                        "op": "=",
                        "left": {
                            "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.ValDt.Dt"
                        },
                        "right": "",
                    },
                    # Sí existe ValDt/DtTm
                    {
                        "op": "!=",
                        "left": {
                            "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.ValDt.DtTm"
                        },
                        "right": "",
                    },
                    # Debe existir Amt/@Ccy
                    {
                        "op": "!=",
                        "left": {
                            "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.Amt[@Ccy]"
                        },
                        "right": "",
                    },
                    # Debe existir Amt (importe)
                    {
                        "op": "!=",
                        "left": {
                            "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.Amt"
                        },
                        "right": "",
                    },
                ]
            },
            "then": {
                "value": {
                    "concat": [
                        # MTDate en YYMMDD a partir de ValDt/DtTm (YYYY-MM-DDThh:mm...)
                        {
                            "substr": {
                                "value": {
                                    "concat": [
                                        {
                                            "substr": {
                                                "value": {
                                                    "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.ValDt.DtTm"
                                                },
                                                "start": 0,
                                                "len": 4,  # YYYY
                                            }
                                        },
                                        {
                                            "substr": {
                                                "value": {
                                                    "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.ValDt.DtTm"
                                                },
                                                "start": 5,
                                                "len": 2,  # MM
                                            }
                                        },
                                        {
                                            "substr": {
                                                "value": {
                                                    "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.ValDt.DtTm"
                                                },
                                                "start": 8,
                                                "len": 2,  # DD
                                            }
                                        },
                                    ]
                                },
                                "start": 2,
                                "len": 6,
                            }
                        },
                        # Divisa (Ccy)
                        {
                            "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.Amt[@Ccy]"
                        },
                        # Importe formateado
                        {
                            "numfmt": {
                                "value": {
                                    "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.Amt"
                                }
                            }
                        },
                    ]
                }
            },
        },
        # ------------------------------------------------------------------
        # 3) Si no hay ValDt/Dt ni ValDt/DtTm, usar IntrBkSttlmDt
        # ------------------------------------------------------------------
        {
            "when": {
                "all": [
                    # NO existe ValDt/Dt
                    {
                        "op": "=",
                        "left": {
                            "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.ValDt.Dt"
                        },
                        "right": "",
                    },
                    # NO existe ValDt/DtTm
                    {
                        "op": "=",
                        "left": {
                            "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.ValDt.DtTm"
                        },
                        "right": "",
                    },
                    # Sí existe IntrBkSttlmDt
                    {
                        "op": "!=",
                        "left": {
                            "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.NtryDtls.TxDtls.RltdDts.IntrBkSttlmDt"
                        },
                        "right": "",
                    },
                    # Debe existir Amt/@Ccy
                    {
                        "op": "!=",
                        "left": {
                            "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.Amt[@Ccy]"
                        },
                        "right": "",
                    },
                    # Debe existir Amt (importe)
                    {
                        "op": "!=",
                        "left": {
                            "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.Amt"
                        },
                        "right": "",
                    },
                ]
            },
            "then": {
                "value": {
                    "concat": [
                        # MTDate en YYMMDD a partir de IntrBkSttlmDt (YYYY-MM-DD)
                        {
                            "substr": {
                                "value": {
                                    "concat": [
                                        {
                                            "substr": {
                                                "value": {
                                                    "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.NtryDtls.TxDtls.RltdDts.IntrBkSttlmDt"
                                                },
                                                "start": 0,
                                                "len": 4,  # YYYY
                                            }
                                        },
                                        {
                                            "substr": {
                                                "value": {
                                                    "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.NtryDtls.TxDtls.RltdDts.IntrBkSttlmDt"
                                                },
                                                "start": 5,
                                                "len": 2,  # MM
                                            }
                                        },
                                        {
                                            "substr": {
                                                "value": {
                                                    "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.NtryDtls.TxDtls.RltdDts.IntrBkSttlmDt"
                                                },
                                                "start": 8,
                                                "len": 2,  # DD
                                            }
                                        },
                                    ]
                                },
                                "start": 2,
                                "len": 6,
                            }
                        },
                        # Divisa (Ccy)
                        {
                            "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.Amt[@Ccy]"
                        },
                        # Importe formateado
                        {
                            "numfmt": {
                                "value": {
                                    "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.Amt"
                                }
                            }
                        },
                    ]
                }
            },
        },
    ]
}
fields_spec_52A = {
    "52A": [
        # ------------------------------------------------------------
        # 1) BICFI + DbtrAcct/IBAN  => PartyId="/IBAN" + línea BIC
        # ------------------------------------------------------------
        {
            "when": {
                "all": [
                    {
                        "op": "!=",
                        "left": {
                            "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.NtryDtls.TxDtls.RltdPties.Dbtr.Agt.FinInstnId.BICFI"
                        },
                        "right": "",
                    },
                    {
                        "op": "!=",
                        "left": {
                            "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.NtryDtls.TxDtls.RltdPties.DbtrAcct.Id.IBAN"
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
                                "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.NtryDtls.TxDtls.RltdPties.DbtrAcct.Id.IBAN"
                            },
                        ]
                    },
                    {
                        "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.NtryDtls.TxDtls.RltdPties.Dbtr.Agt.FinInstnId.BICFI"
                    },
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
        },
        # ------------------------------------------------------------
        # 2) BICFI + DbtrAcct/Othr/Id con SchmeNm/Cd="CUID" y len=6
        #     => PartyId="//CH" + Account  + línea BIC
        # ------------------------------------------------------------
        {
            "when": {
                "all": [
                    {
                        "op": "!=",
                        "left": {
                            "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.NtryDtls.TxDtls.RltdPties.Dbtr.Agt.FinInstnId.BICFI"
                        },
                        "right": "",
                    },
                    # IBAN vacío (para respetar prioridad IBAN)
                    {
                        "op": "=",
                        "left": {
                            "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.NtryDtls.TxDtls.RltdPties.DbtrAcct.Id.IBAN"
                        },
                        "right": "",
                    },
                    # Othr/Id presente
                    {
                        "op": "!=",
                        "left": {
                            "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.NtryDtls.TxDtls.RltdPties.DbtrAcct.Id.Othr.Id"
                        },
                        "right": "",
                    },
                    # SchmeNm/Cd = CUID
                    {
                        "op": "=",
                        "left": {
                            "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.NtryDtls.TxDtls.RltdPties.DbtrAcct.Id.Othr.SchmeNm.Cd"
                        },
                        "right": "CUID",
                    },
                    # Length(Account) == 6  => char[5] existe y char[6] no existe
                    {
                        "op": "!=",
                        "left": {
                            "substr": {
                                "value": {
                                    "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.NtryDtls.TxDtls.RltdPties.DbtrAcct.Id.Othr.Id"
                                },
                                "start": 5,
                                "len": 1,
                            }
                        },
                        "right": "",
                    },
                    {
                        "op": "=",
                        "left": {
                            "substr": {
                                "value": {
                                    "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.NtryDtls.TxDtls.RltdPties.DbtrAcct.Id.Othr.Id"
                                },
                                "start": 6,
                                "len": 1,
                            }
                        },
                        "right": "",
                    },
                ]
            },
            "then": {
                "lines": [
                    {
                        "concat": [
                            {"literal": "//CH"},
                            {
                                "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.NtryDtls.TxDtls.RltdPties.DbtrAcct.Id.Othr.Id"
                            },
                        ]
                    },
                    {
                        "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.NtryDtls.TxDtls.RltdPties.Dbtr.Agt.FinInstnId.BICFI"
                    },
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
        },
        # ------------------------------------------------------------
        # 3) BICFI + DbtrAcct/Othr/Id (resto de casos) => "/"+Account + línea BIC
        # ------------------------------------------------------------
        {
            "when": {
                "all": [
                    {
                        "op": "!=",
                        "left": {
                            "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.NtryDtls.TxDtls.RltdPties.Dbtr.Agt.FinInstnId.BICFI"
                        },
                        "right": "",
                    },
                    # IBAN vacío (prioridad)
                    {
                        "op": "=",
                        "left": {
                            "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.NtryDtls.TxDtls.RltdPties.DbtrAcct.Id.IBAN"
                        },
                        "right": "",
                    },
                    # Othr/Id presente
                    {
                        "op": "!=",
                        "left": {
                            "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.NtryDtls.TxDtls.RltdPties.DbtrAcct.Id.Othr.Id"
                        },
                        "right": "",
                    },
                ],
                # Evitar que caiga aquí el caso CUID+len6 (ya cubierto arriba)
                "not": {
                    "all": [
                        {
                            "op": "=",
                            "left": {
                                "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.NtryDtls.TxDtls.RltdPties.DbtrAcct.Id.Othr.SchmeNm.Cd"
                            },
                            "right": "CUID",
                        },
                        {
                            "op": "!=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.NtryDtls.TxDtls.RltdPties.DbtrAcct.Id.Othr.Id"
                                    },
                                    "start": 5,
                                    "len": 1,
                                }
                            },
                            "right": "",
                        },
                        {
                            "op": "=",
                            "left": {
                                "substr": {
                                    "value": {
                                        "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.NtryDtls.TxDtls.RltdPties.DbtrAcct.Id.Othr.Id"
                                    },
                                    "start": 6,
                                    "len": 1,
                                }
                            },
                            "right": "",
                        },
                    ]
                },
            },
            "then": {
                "lines": [
                    {
                        "concat": [
                            {"literal": "/"},
                            {
                                "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.NtryDtls.TxDtls.RltdPties.DbtrAcct.Id.Othr.Id"
                            },
                        ]
                    },
                    {
                        "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.NtryDtls.TxDtls.RltdPties.Dbtr.Agt.FinInstnId.BICFI"
                    },
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
        },
        # ------------------------------------------------------------
        # 4) BICFI y SIN cuenta válida => usar ClearingSystemMemberIdentification si existe
        #    => "//" + Cd + MmbId + línea BIC
        # ------------------------------------------------------------
        {
            "when": {
                "all": [
                    {
                        "op": "!=",
                        "left": {
                            "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.NtryDtls.TxDtls.RltdPties.Dbtr.Agt.FinInstnId.BICFI"
                        },
                        "right": "",
                    },
                    # sin IBAN
                    {
                        "op": "=",
                        "left": {
                            "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.NtryDtls.TxDtls.RltdPties.DbtrAcct.Id.IBAN"
                        },
                        "right": "",
                    },
                    # sin Other/Id
                    {
                        "op": "=",
                        "left": {
                            "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.NtryDtls.TxDtls.RltdPties.DbtrAcct.Id.Othr.Id"
                        },
                        "right": "",
                    },
                    # clearing code + member id presentes (ajusta paths si tu dotmap difiere)
                    {
                        "op": "!=",
                        "left": {
                            "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.NtryDtls.TxDtls.RltdPties.Dbtr.Agt.FinInstnId.ClrSysMmbId.ClrSysId.Cd"
                        },
                        "right": "",
                    },
                    {
                        "op": "!=",
                        "left": {
                            "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.NtryDtls.TxDtls.RltdPties.Dbtr.Agt.FinInstnId.ClrSysMmbId.MmbId"
                        },
                        "right": "",
                    },
                ]
            },
            "then": {
                "lines": [
                    {
                        "concat": [
                            {"literal": "//"},
                            {
                                "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.NtryDtls.TxDtls.RltdPties.Dbtr.Agt.FinInstnId.ClrSysMmbId.ClrSysId.Cd"
                            },
                            {
                                "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.NtryDtls.TxDtls.RltdPties.Dbtr.Agt.FinInstnId.ClrSysMmbId.MmbId"
                            },
                        ]
                    },
                    {
                        "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.NtryDtls.TxDtls.RltdPties.Dbtr.Agt.FinInstnId.BICFI"
                    },
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
        },
        # ------------------------------------------------------------
        # 5) Solo BICFI (sin PartyId por cuenta ni clearing) => :52A: solo BIC
        # ------------------------------------------------------------
        {
            "when": {
                "all": [
                    {
                        "op": "!=",
                        "left": {
                            "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.NtryDtls.TxDtls.RltdPties.Dbtr.Agt.FinInstnId.BICFI"
                        },
                        "right": "",
                    }
                ]
            },
            "then": {
                "value": {
                    "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.NtryDtls.TxDtls.RltdPties.Dbtr.Agt.FinInstnId.BICFI"
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
        },
    ]
}

fields_spec_52D = {
    "52D": [
        # =========================================================
        # BLOQUE A) ORIGEN: RltdPties/Dbtr/Agt/FinInstnId  (TU XML)
        # =========================================================
        # A1) Si hay BICFI -> NO aplica 52D (se asume 52A en otra regla)
        # (No se define regla aquí; la condición BICFI="" se pone en todas)
        # A2) PartyId por CUENTA (IBAN)
        {
            "when": {
                "all": [
                    {
                        "op": "=",
                        "left": {
                            "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.NtryDtls.TxDtls.RltdPties.Dbtr.Agt.FinInstnId.BICFI"
                        },
                        "right": "",
                    },
                    {
                        "op": "!=",
                        "left": {
                            "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.NtryDtls.TxDtls.RltdPties.DbtrAcct.Id.IBAN"
                        },
                        "right": "",
                    },
                ]
            },
            "then": {
                "lines": [
                    {  # Party Identifier: /IBAN
                        "expr": {
                            "concat": [
                                {"const": "/"},
                                {
                                    "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.NtryDtls.TxDtls.RltdPties.DbtrAcct.Id.IBAN"
                                },
                            ]
                        }
                    },
                    {  # Name (solo si existe)
                        "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.NtryDtls.TxDtls.RltdPties.Dbtr.Agt.FinInstnId.Nm"
                    },
                    {  # AdrLine(s)
                        "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.NtryDtls.TxDtls.RltdPties.Dbtr.Agt.FinInstnId.PstlAdr.AdrLine"
                    },
                    {  # Country (si existe)
                        "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.NtryDtls.TxDtls.RltdPties.Dbtr.Agt.FinInstnId.PstlAdr.Ctry"
                    },
                ]
            },
        },
        # A3) PartyId por CUENTA (Othr/Id)
        {
            "when": {
                "all": [
                    {
                        "op": "=",
                        "left": {
                            "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.NtryDtls.TxDtls.RltdPties.Dbtr.Agt.FinInstnId.BICFI"
                        },
                        "right": "",
                    },
                    {
                        "op": "=",
                        "left": {
                            "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.NtryDtls.TxDtls.RltdPties.DbtrAcct.Id.IBAN"
                        },
                        "right": "",
                    },
                    {
                        "op": "!=",
                        "left": {
                            "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.NtryDtls.TxDtls.RltdPties.DbtrAcct.Id.Othr.Id"
                        },
                        "right": "",
                    },
                ]
            },
            "then": {
                "lines": [
                    {  # Party Identifier: /OtherId
                        "expr": {
                            "concat": [
                                {"const": "/"},
                                {
                                    "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.NtryDtls.TxDtls.RltdPties.DbtrAcct.Id.Othr.Id"
                                },
                            ]
                        }
                    },
                    {
                        "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.NtryDtls.TxDtls.RltdPties.Dbtr.Agt.FinInstnId.Nm"
                    },
                    {
                        "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.NtryDtls.TxDtls.RltdPties.Dbtr.Agt.FinInstnId.PstlAdr.AdrLine"
                    },
                    {
                        "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.NtryDtls.TxDtls.RltdPties.Dbtr.Agt.FinInstnId.PstlAdr.Ctry"
                    },
                ]
            },
        },
        # A4) PartyId por CLEARING (//Cd+MmbId)
        {
            "when": {
                "all": [
                    {
                        "op": "=",
                        "left": {
                            "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.NtryDtls.TxDtls.RltdPties.Dbtr.Agt.FinInstnId.BICFI"
                        },
                        "right": "",
                    },
                    {
                        "op": "=",
                        "left": {
                            "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.NtryDtls.TxDtls.RltdPties.DbtrAcct.Id.IBAN"
                        },
                        "right": "",
                    },
                    {
                        "op": "=",
                        "left": {
                            "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.NtryDtls.TxDtls.RltdPties.DbtrAcct.Id.Othr.Id"
                        },
                        "right": "",
                    },
                    {
                        "op": "!=",
                        "left": {
                            "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.NtryDtls.TxDtls.RltdPties.Dbtr.Agt.FinInstnId.ClrSysMmbId.ClrSysId.Cd"
                        },
                        "right": "",
                    },
                    {
                        "op": "!=",
                        "left": {
                            "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.NtryDtls.TxDtls.RltdPties.Dbtr.Agt.FinInstnId.ClrSysMmbId.MmbId"
                        },
                        "right": "",
                    },
                ]
            },
            "then": {
                "lines": [
                    {  # Party Identifier: //CdMmbId
                        "expr": {
                            "concat": [
                                {"const": "//"},
                                {
                                    "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.NtryDtls.TxDtls.RltdPties.Dbtr.Agt.FinInstnId.ClrSysMmbId.ClrSysId.Cd"
                                },
                                {
                                    "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.NtryDtls.TxDtls.RltdPties.Dbtr.Agt.FinInstnId.ClrSysMmbId.MmbId"
                                },
                            ]
                        }
                    },
                    {
                        "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.NtryDtls.TxDtls.RltdPties.Dbtr.Agt.FinInstnId.Nm"
                    },
                    {
                        "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.NtryDtls.TxDtls.RltdPties.Dbtr.Agt.FinInstnId.PstlAdr.AdrLine"
                    },
                    {
                        "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.NtryDtls.TxDtls.RltdPties.Dbtr.Agt.FinInstnId.PstlAdr.Ctry"
                    },
                ]
            },
        },
        # A5) MACHETAZO: SIN CUENTA NI CLEARING -> PartyId = Nm (sin "/"), y NO repetir Nm abajo
        # (Esto te produce exactamente: 123456 + BANCO DAVIVIENDA en tu caso)
        {
            "when": {
                "all": [
                    {
                        "op": "=",
                        "left": {
                            "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.NtryDtls.TxDtls.RltdPties.Dbtr.Agt.FinInstnId.BICFI"
                        },
                        "right": "",
                    },
                    {
                        "op": "=",
                        "left": {
                            "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.NtryDtls.TxDtls.RltdPties.DbtrAcct.Id.IBAN"
                        },
                        "right": "",
                    },
                    {
                        "op": "=",
                        "left": {
                            "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.NtryDtls.TxDtls.RltdPties.DbtrAcct.Id.Othr.Id"
                        },
                        "right": "",
                    },
                    {
                        "op": "=",
                        "left": {
                            "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.NtryDtls.TxDtls.RltdPties.Dbtr.Agt.FinInstnId.ClrSysMmbId.ClrSysId.Cd"
                        },
                        "right": "",
                    },
                    {
                        "op": "=",
                        "left": {
                            "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.NtryDtls.TxDtls.RltdPties.Dbtr.Agt.FinInstnId.ClrSysMmbId.MmbId"
                        },
                        "right": "",
                    },
                    {
                        "op": "!=",
                        "left": {
                            "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.NtryDtls.TxDtls.RltdPties.Dbtr.Agt.FinInstnId.Nm"
                        },
                        "right": "",
                    },
                ]
            },
            "then": {
                "lines": [
                    {  # Party Identifier: Nm (sin "/")
                        "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.NtryDtls.TxDtls.RltdPties.Dbtr.Agt.FinInstnId.Nm"
                    },
                    {  # SOLO dirección (para no repetir Nm)
                        "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.NtryDtls.TxDtls.RltdPties.Dbtr.Agt.FinInstnId.PstlAdr.AdrLine"
                    },
                    {
                        "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.NtryDtls.TxDtls.RltdPties.Dbtr.Agt.FinInstnId.PstlAdr.Ctry"
                    },
                ]
            },
        },
        # A6) Último fallback: no hay nada usable -> /NOTPROVIDED + NOTPROVIDED
        {
            "when": {
                "all": [
                    {
                        "op": "=",
                        "left": {
                            "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.NtryDtls.TxDtls.RltdPties.Dbtr.Agt.FinInstnId.BICFI"
                        },
                        "right": "",
                    },
                    {
                        "op": "=",
                        "left": {
                            "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.NtryDtls.TxDtls.RltdPties.DbtrAcct.Id.IBAN"
                        },
                        "right": "",
                    },
                    {
                        "op": "=",
                        "left": {
                            "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.NtryDtls.TxDtls.RltdPties.DbtrAcct.Id.Othr.Id"
                        },
                        "right": "",
                    },
                    {
                        "op": "=",
                        "left": {
                            "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.NtryDtls.TxDtls.RltdPties.Dbtr.Agt.FinInstnId.ClrSysMmbId.ClrSysId.Cd"
                        },
                        "right": "",
                    },
                    {
                        "op": "=",
                        "left": {
                            "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.NtryDtls.TxDtls.RltdPties.Dbtr.Agt.FinInstnId.ClrSysMmbId.MmbId"
                        },
                        "right": "",
                    },
                    {
                        "op": "=",
                        "left": {
                            "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.NtryDtls.TxDtls.RltdPties.Dbtr.Agt.FinInstnId.Nm"
                        },
                        "right": "",
                    },
                ]
            },
            "then": {
                "lines": [
                    {"const": "/NOTPROVIDED"},
                    {"const": "NOTPROVIDED"},
                ]
            },
        },
        # =========================================================
        # BLOQUE B) ORIGEN: RltdAgts/DbtrAgt/FinInstnId (MT900/910)
        # =========================================================
        # B1) PartyId por CLEARING (//Cd+MmbId)
        {
            "when": {
                "all": [
                    {
                        "op": "=",
                        "left": {
                            "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.NtryDtls.TxDtls.RltdAgts.DbtrAgt.FinInstnId.BICFI"
                        },
                        "right": "",
                    },
                    {
                        "op": "!=",
                        "left": {
                            "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.NtryDtls.TxDtls.RltdAgts.DbtrAgt.FinInstnId.ClrSysMmbId.ClrSysId.Cd"
                        },
                        "right": "",
                    },
                    {
                        "op": "!=",
                        "left": {
                            "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.NtryDtls.TxDtls.RltdAgts.DbtrAgt.FinInstnId.ClrSysMmbId.MmbId"
                        },
                        "right": "",
                    },
                ]
            },
            "then": {
                "lines": [
                    {
                        "expr": {
                            "concat": [
                                {"const": "//"},
                                {
                                    "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.NtryDtls.TxDtls.RltdAgts.DbtrAgt.FinInstnId.ClrSysMmbId.ClrSysId.Cd"
                                },
                                {
                                    "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.NtryDtls.TxDtls.RltdAgts.DbtrAgt.FinInstnId.ClrSysMmbId.MmbId"
                                },
                            ]
                        }
                    },
                    {
                        "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.NtryDtls.TxDtls.RltdAgts.DbtrAgt.FinInstnId.Nm"
                    },
                    {
                        "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.NtryDtls.TxDtls.RltdAgts.DbtrAgt.FinInstnId.PstlAdr.AdrLine"
                    },
                    {
                        "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.NtryDtls.TxDtls.RltdAgts.DbtrAgt.FinInstnId.PstlAdr.Ctry"
                    },
                ]
            },
        },
        # B2) MACHETAZO: SIN CLEARING -> PartyId = Nm (sin "/"), y NO repetir Nm abajo
        {
            "when": {
                "all": [
                    {
                        "op": "=",
                        "left": {
                            "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.NtryDtls.TxDtls.RltdAgts.DbtrAgt.FinInstnId.BICFI"
                        },
                        "right": "",
                    },
                    {
                        "op": "=",
                        "left": {
                            "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.NtryDtls.TxDtls.RltdAgts.DbtrAgt.FinInstnId.ClrSysMmbId.ClrSysId.Cd"
                        },
                        "right": "",
                    },
                    {
                        "op": "=",
                        "left": {
                            "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.NtryDtls.TxDtls.RltdAgts.DbtrAgt.FinInstnId.ClrSysMmbId.MmbId"
                        },
                        "right": "",
                    },
                    {
                        "op": "!=",
                        "left": {
                            "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.NtryDtls.TxDtls.RltdAgts.DbtrAgt.FinInstnId.Nm"
                        },
                        "right": "",
                    },
                ]
            },
            "then": {
                "lines": [
                    {  # Party Identifier: Nm (sin "/")
                        "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.NtryDtls.TxDtls.RltdAgts.DbtrAgt.FinInstnId.Nm"
                    },
                    {
                        "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.NtryDtls.TxDtls.RltdAgts.DbtrAgt.FinInstnId.PstlAdr.AdrLine"
                    },
                    {
                        "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.NtryDtls.TxDtls.RltdAgts.DbtrAgt.FinInstnId.PstlAdr.Ctry"
                    },
                ]
            },
        },
        # B3) Último fallback
        {
            "when": {
                "all": [
                    {
                        "op": "=",
                        "left": {
                            "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.NtryDtls.TxDtls.RltdAgts.DbtrAgt.FinInstnId.BICFI"
                        },
                        "right": "",
                    },
                    {
                        "op": "=",
                        "left": {
                            "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.NtryDtls.TxDtls.RltdAgts.DbtrAgt.FinInstnId.Nm"
                        },
                        "right": "",
                    },
                    {
                        "op": "=",
                        "left": {
                            "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.NtryDtls.TxDtls.RltdAgts.DbtrAgt.FinInstnId.PstlAdr.AdrLine"
                        },
                        "right": "",
                    },
                    {
                        "op": "=",
                        "left": {
                            "xml": ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.NtryDtls.TxDtls.RltdAgts.DbtrAgt.FinInstnId.PstlAdr.Ctry"
                        },
                        "right": "",
                    },
                ]
            },
            "then": {
                "lines": [
                    {"const": "/NOTPROVIDED"},
                    {"const": "NOTPROVIDED"},
                ]
            },
        },
    ]
}


fields_spec_72 = {
    "72": [
        {
            # --------------------------------------------------------------
            # REGLA 1: AddtlTxInf presente y longitud > 210
            #   -> truncar a 209 caracteres y añadir '+'
            # --------------------------------------------------------------
            "when": {
                "all": [
                    # AddtlTxInf no vacío
                    {
                        "op": "!=",
                        "left": {
                            "xml": (
                                ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.NtryDtls.TxDtls.AddtlTxInf"
                            )
                        },
                        "right": "",
                    },
                    # length(Information) > 210  <==>
                    # existe el carácter en posición 211 (índice 210, base 0)
                    {
                        "op": "!=",
                        "left": {
                            "substr": {
                                "value": {
                                    "xml": (
                                        ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.NtryDtls.TxDtls.AddtlTxInf"
                                    )
                                },
                                "start": 210,  # carácter 211 en base 1
                                "len": 1,
                            }
                        },
                        "right": "",
                    },
                ]
            },
            "then": {
                # Aquí hacemos: Information = substring(1, 209) + "+"
                #   - substring base 0 -> start=0, len=209  => chars 1..209
                #   - luego concatenamos el literal "+"
                "value": {
                    "concat": [
                        {
                            "substr": {
                                "value": {
                                    "xml": (
                                        ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.NtryDtls.TxDtls.AddtlTxInf"
                                    )
                                },
                                "start": 0,
                                "len": 209,
                            }
                        },
                        {"literal": "+"},
                    ]
                }
            },
        },
        {
            # --------------------------------------------------------------
            # REGLA 2: AddtlTxInf presente y longitud <= 210
            #   -> copiar el texto tal cual, sin truncar
            # --------------------------------------------------------------
            "when": {
                "all": [
                    # AddtlTxInf no vacío
                    {
                        "op": "!=",
                        "left": {
                            "xml": (
                                ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.NtryDtls.TxDtls.AddtlTxInf"
                            )
                        },
                        "right": "",
                    },
                    # length(Information) <= 210  <==>
                    # NO existe carácter en posición 211 (índice 210)
                    {
                        "op": "=",
                        "left": {
                            "substr": {
                                "value": {
                                    "xml": (
                                        ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.NtryDtls.TxDtls.AddtlTxInf"
                                    )
                                },
                                "start": 210,
                                "len": 1,
                            }
                        },
                        "right": "",
                    },
                ]
            },
            "then": {
                "value": {
                    "xml": (
                        ".Document.BkToCstmrDbtCdtNtfctn.Ntfctn.Ntry.NtryDtls.TxDtls.AddtlTxInf"
                    )
                }
            },
        },
    ]
}


if __name__ == "__main__":
    out_lines = []
    globals_ctx = {}

    # :20:
    fields_20 = build_fields(ubicationEntry, fields_spec_20, global_vars=globals_ctx)
    if fields_20.get("20"):
        out_lines.append(f":20:{fields_20['20'][0]}")
    else:
        dbg(
            "20 NO GENERADO - revisa las trazas anteriores para ver qué condición falló"
        )

    # :21:
    fields_21 = build_fields(ubicationEntry, fields_spec_21, global_vars=globals_ctx)
    lines_21 = [ln for ln in fields_21.get("21", []) if str(ln).strip() != ""]
    if lines_21:
        # solo debe haber una línea para :21:, formato 16x
        out_lines.append(f":21:{lines_21[0]}")
    else:
        dbg("21 NO GENERADO - revisa las trazas para ver qué condición falló")

    # :25:
    fields_25 = build_fields(ubicationEntry, fields_spec_25, global_vars=globals_ctx)
    lines_25 = [ln for ln in fields_25.get("25", []) if str(ln).strip() != ""]
    dbg(f"FINAL 25 lines (count={len(lines_25)}):", lines_25)
    if lines_25:
        for ln in lines_25:
            out_lines.append(f":25:{ln}")
    else:
        dbg(
            "25 NO GENERADO - revisa las trazas anteriores para ver qué condición falló"
        )

    # :13D:
    fields_13D = build_fields(ubicationEntry, fields_spec_13D, global_vars=globals_ctx)
    if fields_13D.get("13D"):
        out_lines.append(f":13D:{fields_13D['13D'][0]}")
    else:
        dbg(
            "13D NO GENERADO - revisa las trazas anteriores para ver qué condición falló"
        )

    # :32A:
    fields_32A = build_fields(ubicationEntry, fields_spec_32A, global_vars=globals_ctx)
    if fields_32A.get("32A"):
        out_lines.append(f":32A:{fields_32A['32A'][0]}")
    else:
        dbg(
            "32A NO GENERADO - revisa las trazas anteriores para ver qué condición falló"
        )

    # :52A:
    fields_52A = build_fields(ubicationEntry, fields_spec_52A, global_vars=globals_ctx)
    lines_52A = [ln for ln in fields_52A.get("52A", []) if str(ln).strip() != ""]
    dbg(f"FINAL 52A lines (count={len(lines_52A)}):", lines_52A)

    if lines_52A:
        # Si hay 52A, se imprime 52A y NO se imprime 52D
        for i, ln in enumerate(lines_52A):
            if i == 0:
                out_lines.append(f":52A:{ln}")
            else:
                out_lines.append(ln)
    else:
        dbg("52A NO GENERADO -> se intenta 52D (Ordering Institution)")

        # :52D: (solo si NO se generó 52A)
        fields_52D = build_fields(
            ubicationEntry, fields_spec_52D, global_vars=globals_ctx
        )
        lines_52D = [ln for ln in fields_52D.get("52D", []) if str(ln).strip() != ""]
        dbg(f"FINAL 52D lines (count={len(lines_52D)}):", lines_52D)

        if lines_52D:
            for i, ln in enumerate(lines_52D):
                if i == 0:
                    out_lines.append(f":52D:{ln}")
                else:
                    out_lines.append(ln)
        else:
            dbg("52D NO GENERADO - no hubo datos suficientes (o no existía el agente)")

    fields_72 = build_fields(ubicationEntry, fields_spec_72, global_vars=globals_ctx)
    lines_72 = [ln for ln in fields_72.get("72", []) if str(ln).strip() != ""]
    if lines_72:
        for i, ln in enumerate(lines_72):
            if i == 0:
                out_lines.append(f":72:{ln}")
            else:
                out_lines.append(ln)
    else:
        dbg(
            "72 NO GENERADO - revisa las trazas anteriores para ver qué condición falló"
        )

    # === escribir archivo ===
    # === Generar Reporte de Excepciones y Prepend al Mensaje ===
    exception_report = ""
    if TRUNCATION_WARNINGS:
        report_lines = []
        report_lines.append("Exception Report:")
        report_lines.append("")
        report_lines.append("Envelope")
        report_lines.append("CAMT.054 to MT900 Translation Warnings:")

        report_lines.append(
            " - NetworkValidation: WARNING.TINPUT: The input message contains potential truncation errors."
        )
        report_lines.append(
            " - Translation:      WARNING.TOUTUG: Validation of usage guidelines performed locally. \n"
        )

        for i, t in enumerate(TRUNCATION_WARNINGS):
            if isinstance(t, dict):
                mt_tag = t.get("mt_tag", "UNKNOWN")
                original = t.get("original", "") or ""
                max_len = int(t.get("max_len", 0) or 0)
                exceeded_by = max(len(original) - max_len, 0)
                path = _dotpath_to_envelope(t.get("target_path", ""))
                report_lines.append(f"[{i}] Field {mt_tag}")
                report_lines.append(
                    f"    Reason : content truncated (max {max_len}, exceeded by {exceeded_by} chars)"
                )
                report_lines.append(
                    "    Code   : Translation: TRUNC_N.T0000T: Field content has been truncated."
                )
                report_lines.append(f"    Path   : {path}")
                report_lines.append(f"    Original Value: '{original}'")
            else:
                s = str(t)
                lines_t = s.splitlines()
                first = lines_t[0] if lines_t else s
                report_lines.append(f"[{i}] {first}")
                m_orig = re.search(r"Original Value: '(.+)'", s)
                orig = m_orig.group(1) if m_orig else ""
                m_max = re.search(r"exceeded\s+(\d+)\s+chars", s)
                if m_max:
                    max_len = int(m_max.group(1))
                    exceeded_by = max(len(orig) - max_len, 0) if orig else 0
                    report_lines.append("---")
                    report_lines.append(
                        f" tipo asi (max {max_len}, exceeded by {exceeded_by} chars)"
                    )
                if orig:
                    report_lines.append(f"   Original Value: '{orig}'")
            report_lines.append("---")
            report_lines.append("")
        exception_report = "\n".join(report_lines)

    # === Escribir archivo ===
    ts = datetime.now().strftime("%Y%m%d%H%M%S")  # AAAAMMDDHHSS
    fname = f"MT900_{ts}.txt"
    out_path = Path(ubicationDestiny) / fname
    out_path.parent.mkdir(parents=True, exist_ok=True)

    block4_content = "\n".join(out_lines)

    block5 = build_block5(ubicationEntry, block4_content)

    mt = finalize_mt_message(build_header_12(ubicationEntry), out_lines, block5)
    full_output = exception_report + mt

    out_path.write_bytes(full_output.encode("utf-8"))
    print(f"{full_output}")