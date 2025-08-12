# src/fraud_scorer/pipelines/normalizers.py
import re
from datetime import datetime

_DATE_IN = ["%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y", "%d.%m.%Y", "%d/%m/%y", "%d-%m-%y"]

def norm_date(s: str) -> str:
    t = (s or "").strip()
    if not t: return ""
    for f in _DATE_IN:
        try:
            return datetime.strptime(t, f).strftime("%d/%m/%Y")
        except Exception:
            pass
    # patrones "del 01/01/2025 al 31/12/2025" (separar afuera si quieres inicio/fin)
    m = re.search(r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}).{0,12}(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})", t)
    if m:
        try:
            return datetime.strptime(m.group(1), "%d/%m/%Y").strftime("%d/%m/%Y")
        except Exception:
            pass
    return t

def norm_money(s: str) -> str:
    t = (s or "").strip()
    if not t: return ""
    t = re.sub(r"\s+", " ", t)
    if re.fullmatch(r"[\$]?\s?\d[\d,\.]*", t):
        t = t.replace(",", "")
        try:
            val = float(re.sub(r"[^\d\.]", "", t))
            return f"${val:,.2f} MXN"
        except Exception:
            pass
    return t
