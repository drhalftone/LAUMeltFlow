"""Case configuration. Replaces CASE.INC + READ_CASE in 1Dheat.f.

Reads a Fortran-style namelist file (heat.case) into a Case dataclass.
Field names match Martin's Fortran variables with the CASE_ prefix
stripped and lowercased:
    CASE_NBRN          -> nbrn
    CASE_L0            -> l0
    CASE_GAS_ENERGY    -> gas_energy

Defaults match READ_CASE in 1Dheat.f (lines 2033-2077). Fields declared
in CASE.INC but not initialized there get neutral Python defaults
(0 / False / "").
"""

from __future__ import annotations
from dataclasses import dataclass, fields
from pathlib import Path
import re


# Fortran double-precision literal (5.0D-2, -1.D0, etc.) -> Python float
_FORTRAN_DOUBLE = re.compile(
    r"([+-]?\d+\.?\d*|\.\d+)[dDeE]([+-]?\d+)"
)


@dataclass
class Case:
    """Runtime configuration. One instance per simulation run.

    Defaults reproduce READ_CASE in 1Dheat.f. They get overwritten by
    whatever is in the heat.case namelist file.
    """

    # --- File paths ---
    name: str = "default"
    cp_file: str = "heat.dat"
    rho_file: str = "rho.dat"
    time_file: str = "time_shuttle.dat"
    sol_k_file: str = "sol_kappa_01.dat"
    eff_k_file: str = "eff_kappa_01.dat"
    solid_file: str = "solid.dat"
    sol_k_char_file: str = ""        # Fortran defaults to sol_k_file
    eff_k_char_file: str = ""        # Fortran defaults to eff_k_file
    cp_char_file: str = ""           # Fortran defaults to cp_file
    tc_file: str = " "
    gas_file: str = " "
    aero_file: str = " "
    bprime_dir: str = " "
    bprime_file: str = ""
    grid_file: str = ""
    init_profile_file: str = ""

    # --- Integers ---
    nbrn: int = 300
    m: int = 0                       # 0 = Cartesian, 1 = cylindrical, 2 = spherical
    bnd0: int = 2                    # left BC type
    bndn: int = 2                    # right BC type
    nbrtc: int = 4                   # number of thermocouples
    gbnd0: int = 1                   # left gas BC type
    gbndn: int = 1                   # right gas BC type

    # --- Logicals ---
    use_time_bcn: bool = True
    use_aero_bcn: bool = False
    use_recession: bool = False
    use_spec_recession: bool = False
    gas_energy: bool = False
    solid_kats: bool = False
    tau_linear: bool = False
    use_hwcorr: bool = False

    # --- Reals (geometry / time) ---
    l0: float = 0.0254               # domain length [m]
    time_final: float = 30.0         # [s]
    time_init: float = 0.0           # [s]
    cfl: float = 15.0
    init_temp: float = 300.0         # [K]

    # --- Reals (boundary conditions) ---
    qa0: float = 0.0                 # left  heat flux        [W/m^2]
    qan: float = 1.15e6              # right heat flux        [W/m^2]
    tw0: float = 300.0               # left  wall temperature [K]
    twn: float = 300.0               # right wall temperature [K]

    # --- Reals (gas properties) ---
    gcp: float = 1200.0              # gas heat capacity [J/kg/K]
    pamb: float = 101325.0           # ambient pressure  [Pa]
    gmu: float = 3.0e-5              # gas viscosity     [Pa*s]
    rgas: float = 287.0              # gas constant      [J/kg/K]
    k0_v: float = 1.0e-10            # virgin permeability
    k0_c: float = 0.0                # char permeability; Fortran defaults to k0_v
    eps_v: float = 0.85              # virgin emissivity
    eps_c: float = 0.85              # char emissivity
    hgas_tref: float = -1.0          # reference T for gas enthalpy
    phi_c: float = 0.85              # char porosity

    # --- Reals (extras declared in CASE.INC but not set by READ_CASE) ---
    h0: float = 0.0
    hn: float = 0.0
    tr0: float = 0.0
    trn: float = 0.0
    qstar: float = 0.0
    spec_sdot: float = 0.0
    tabl_min: float = 0.0
    hwc_tcw: float = 0.0
    hwc_scale: float = 0.0
    rec_scale: float = 0.0

    def __post_init__(self):
        # Mirror the post-default Fortran logic in READ_CASE:
        #   CASE_SOL_K_CHAR_FILE = CASE_SOL_K_FILE   (if not later overridden)
        #   CASE_EFF_K_CHAR_FILE = CASE_EFF_K_FILE
        #   CASE_CP_CHAR_FILE    = CASE_CP_FILE
        #   CASE_K0_C            = CASE_K0_V
        if not self.sol_k_char_file:
            self.sol_k_char_file = self.sol_k_file
        if not self.eff_k_char_file:
            self.eff_k_char_file = self.eff_k_file
        if not self.cp_char_file:
            self.cp_char_file = self.cp_file
        if self.k0_c == 0.0:
            self.k0_c = self.k0_v


# ----------------------- Namelist parser ----------------------- #


def _parse_value(raw: str) -> object:
    """Convert a Fortran-namelist value token to a Python value."""
    s = raw.strip().rstrip(",")

    # Quoted string
    if s.startswith(("'", '"')) and len(s) >= 2:
        return s[1:-1]

    # Logicals
    up = s.upper()
    if up in (".TRUE.", "T", ".T."):
        return True
    if up in (".FALSE.", "F", ".F."):
        return False

    # Fortran double 5.0D-2 / -1.D0 / 1.D+8 -> Python e-notation
    if "d" in s.lower():
        s = _FORTRAN_DOUBLE.sub(lambda m: f"{m.group(1)}e{m.group(2)}", s)

    # Try int, then float
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    # Last resort: treat as bare string
    return s


def _strip_case_prefix(key: str) -> str:
    """CASE_NBRN -> nbrn"""
    k = key.strip().lower()
    if k.startswith("case_"):
        k = k[5:]
    return k


def parse_namelist(text: str) -> dict:
    """Parse a Fortran namelist body into a dict of {field: value}.

    Recognizes the `&NAME ... /` block. Returns lowercased field names
    with the CASE_ prefix dropped.
    """
    # Find the &...NML body
    m = re.search(r"&\w+\s*(.*?)/", text, flags=re.DOTALL)
    body = m.group(1) if m else text

    # Strip comments (! to end-of-line) and join continuation lines
    lines = []
    for line in body.splitlines():
        line = line.split("!", 1)[0].strip()
        if line:
            lines.append(line)
    joined = " ".join(lines)

    # Split on commas that are at the top level (not inside quotes).
    # The format is "KEY = VALUE [, KEY = VALUE]*". The cleanest split is on
    # the "KEY =" pattern.
    out = {}
    # Find all KEY = VALUE pairs. VALUE extends to the next KEY= or end.
    pat = re.compile(r"(\w+)\s*=\s*(.*?)(?=\s*\w+\s*=|$)")
    for match in pat.finditer(joined):
        key = _strip_case_prefix(match.group(1))
        raw = match.group(2).strip().rstrip(",").strip()
        out[key] = _parse_value(raw)
    return out


def load_case(path: str | Path) -> Case:
    """Read a heat.case file and return a populated Case.

    If the file doesn't exist, returns the default Case (matches Fortran
    behavior: READ_CASE returns silently if heat.case is missing).
    """
    p = Path(path)
    case = Case()
    if not p.exists():
        return case

    text = p.read_text()
    parsed = parse_namelist(text)

    valid_fields = {f.name for f in fields(case)}
    for key, value in parsed.items():
        if key in valid_fields:
            setattr(case, key, value)
        # Silently ignore unknown keys (forward compatibility)

    # Re-apply the post-default mirroring after overrides
    case.__post_init__()
    return case


if __name__ == "__main__":
    # Quick sanity check: load each example case
    import sys
    repo_root = Path(__file__).resolve().parents[1]
    for name in ("aw1", "aw2_tc21", "aw2_tc22"):
        p = repo_root / "heat_2026-04-11_1837" / "examples" / name / "heat.case"
        c = load_case(p)
        print(f"\n=== {name} ===")
        print(f"  name      = {c.name!r}")
        print(f"  nbrn      = {c.nbrn}")
        print(f"  l0        = {c.l0}")
        print(f"  cfl       = {c.cfl}")
        print(f"  time_final= {c.time_final}")
        print(f"  bnd0/bndn = {c.bnd0}/{c.bndn}")
        print(f"  gas_energy= {c.gas_energy}")
        print(f"  use_time_bcn = {c.use_time_bcn}")
        print(f"  cp_file   = {c.cp_file!r}")
