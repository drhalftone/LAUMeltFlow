# Martin's 1D Heat-Shield Code — Summary

Working notes on the Fortran code in `heat_2026-04-11_1837/`. Goal: enough understanding to (eventually) port to Python and build a surrogate.

## 1. What it actually simulates

A **1D slab of charring thermal protection system (TPS) material** under aerothermal heating. The code tracks, as a function of depth `x` and time `t`:

- **Temperature** `T(x,t)` — main field
- **Solid bulk density** `ρ(x,t)` — drops as virgin material decomposes into char + gases (pyrolysis)
- **Per-species solid densities** `ρᵢ(x,t)` for `NBRS = 3` reactive components (TACOT 3.0 model)
- **Gas pressure** `P_g(x,t)` and **mass flux** `ṁ(x,t)` — gases produced by pyrolysis flow through the porous char toward the surface
- **Optional: radiation source** `Q̇_rad(x,t)` when `-r` flag is set

It's the kind of thing you'd use to predict a heat-shield's interior temperature history and char/pyrolysis-front depths during reentry, given a wall heat flux history.

## 2. The 5 physics layers

The code is structured so each layer is optional, toggled by CLI flag:

| Flag | Module | What it adds |
|---|---|---|
| (default) | Heat conduction + pyrolysis | Always on. Solves the energy + species equations. |
| `-g` / `gas` | Gas flow (`LGAS`) | Solves Darcy-like equation for gas pressure, computes `ṁ`, advects gas enthalpy. |
| `-r` / `rad` | Radiation (`LRAD`) | Multi-band P1 / Marshak radiative transfer adds volumetric source `Q̇_rad`. |
| `-rb` / `radblu` | Radiation + laser | Same as `-r` but with a 1.07 µm laser source instead of 10.6 µm. |
| `-e` / `eff` | Effective conductivity (`LEFF`) | Uses Rosseland-style effective `k` that bakes in radiation — skips the Planck term. |
| `-d` / `debug` | Debug | Verbose per-step printout. |
| `-i` / `info` | Info | Mid-verbosity narration. |

## 3. Governing equations (loosely)

**Energy** (per cell, finite-volume form):

```
(d/dt)(ρ h_s · ΔV) = [k·∂T/∂x · A]_left^right  −  Q̇_rad · ΔV  −  ∇·(ṁ · h_g)  +  Δh_chem
```

The code stores enthalpy `h_s = h_s_form(τ) + ∫ cp(T',τ) dT'` and uses a fixed-point iteration (3 sweeps in [1Dheat.f:618-629](../heat_2026-04-11_1837/1Dheat.f#L618-L629)) to back out `T_new` from updated `ρ_new · h_new`.

**Pyrolysis** (Arrhenius decomposition, per species `k`, inlined at [1Dheat.f:482-513](../heat_2026-04-11_1837/1Dheat.f#L482-L513)):

```
W = (ρᵢ − ρᶜᵢ) / ρᵛᵢ                  (normalized progress, 0 = char, 1 = virgin)
ψ = 1 case:   W_new = W · exp(−A·Δt·exp(−E/T))
ψ ≠ 1 case:   W_new = (W^(1−ψ) − Δt(1−ψ)·A·exp(−E/T))^(1/(1−ψ))
```

Parameters per species: `A`, `ψ`, `E`, `γ` come from [solid.dat](../heat_2026-04-11_1837/examples/aw1/solid.dat). Char fraction `τ` (which drives material-property mixing between virgin and char) is computed in `CALC_TAU` ([1Dheat.f:1182](../heat_2026-04-11_1837/1Dheat.f#L1182)) — basically `τ = 1 − (ρ_reactive_remaining / ρ_virgin_reactive)`, clamped to `[0,1]`. `τ=0` means virgin, `τ=1` means fully charred.

**Material properties** are `(virgin, char)` table pairs linearly mixed by `τ`:

```
k(T,τ) = (1−τ)·k_virgin(T) + τ·k_char(T)        ← CALC_K_MIX
cp(T,τ) = (1−τ)·cp_virgin(T) + τ·cp_char(T)     ← CALC_CP_MIX
```

Two K tables can be loaded: solid (`sol_kappa_*.dat`) and effective (`eff_kappa_*.dat`). The `-e` flag selects effective.

**Gas flow** (in [1Dheat.f:2484, SOLVE_GAS](../heat_2026-04-11_1837/1Dheat.f#L2484)). Darcy flow through porous medium, solved as a tridiagonal system for pressure:

```
ṁ = −(K₀·ρ_g)/μ · ∂P/∂x · A         ← face flux
∂/∂t(φ·ρ_g) + ∂ṁ/∂x  = G_src        ← cell continuity, G_src from pyrolysis
```

Permeability `K₀` mixes between virgin (`CASE_K0_V`) and char (`CASE_K0_C`) by the char fraction at each face.

**Radiation** ([1Drad.f:101, RAD](../heat_2026-04-11_1837/1Drad.f#L101)). Spectral P1-style equations solved tridiagonally for incident radiation `G(x)` per wavelength band (`NBRB = 101` bands log-spaced 0.1–100 µm), then integrated to a volumetric source `Q̇_rad`. Uses OpenMP `parallel do` over wavelengths. Output `UTOT(I)` = `Q̇_rad` in cell `I`, and `QWIN`/`QWOU` = net radiative flux into the two boundaries.

## 4. Numerics

- **Discretization:** 1D finite volume. `MAXN = 1001` max cells, configured per case via `CASE_NBRN` (e.g. 100 for `aw1`).
- **Geometry:** controlled by `CASE_M`: `0 = Cartesian slab`, `1 = cylindrical`, `2 = spherical`. The `DA(I)` (face area) and `DV(I)` (cell volume) computations at [1Dheat.f:184-193](../heat_2026-04-11_1837/1Dheat.f#L184-L193) generalize the volume integral with `r^m`.
- **Grid:** uniform by default (`F = 1`). Stretched grid available by setting `F ≠ 1` (geometric ratio).
- **Time stepping:** explicit, **CFL-limited** based on the diffusion scale `Δx²·ρ·cp/k` divided by `CFL` (typically 10). Computed once at startup ([1Dheat.f:258-263](../heat_2026-04-11_1837/1Dheat.f#L258-L263)). `Δt` is **clipped** down so that integration always lands exactly on BC event times (entries in `TTIME`/`TAEROT`).
- **Boundary conditions** (cell-centered with ghost cells at `T(0)` and `T(NBRN+1)`):
  - `BND_TYPE = 1` — fixed wall temperature (Dirichlet). Ghost: `T(0) = 2·T_W − T(1)`.
  - `BND_TYPE = 2` — fixed heat flux (Neumann), with optional aerothermal load (B' chemistry + blowing correction) and radiative emission.
- **Inner convergence loop:** when radiation is on, the code iterates the energy + radiation solve up to `NBRJ = 10` times per timestep, with tolerance `TOL = 1.D-1`% on `Q̇_rad` ([1Dheat.f:479, 670-674](../heat_2026-04-11_1837/1Dheat.f#L479)).

## 5. The time-step algorithm

Pseudocode of the main loop ([1Dheat.f:348-715](../heat_2026-04-11_1837/1Dheat.f#L348-L715)):

```
for IT in 1..NBRT:
    1.  Δt = min(Δt_nom, time-to-next-BC-event)
    2.  Update aerothermal BCs from time tables (CALC_AERO_*, CALC_TEMP)
    3.  Set ghost cells T(0), T(NBRN+1) from BND_TYPE
    4.  Sample probes (thermocouples)  →  write history
    5.  If LRAD: call RAD(T) → QDOT[I]

    for J in 1..NBRJ:   # radiation convergence loop
        a. Pyrolysis step: for each cell, each species, advance W → W_new
           (Arrhenius, in closed form for ψ=1 or general ψ)
           Accumulate gas source GSRC[I] = Σ_k ρ_w_k · max(0, ρ_old − ρ_new) / Δt
        b. If LGAS: SOLVE_GAS → P_g, ṁ; compute HFACE = ṁ · h_g; apply blowing BC at wall
        c. Energy update per cell:
           - face conductivities k1, k2 from CALC_K_MIX(T_face, τ_face)
           - fluxes FLUXM, FLUXP
           - tentative ΔT = ΔV_inv · Δt · (−FLUXM·A_L + FLUXP·A_R − Q̇·ΔV − ∇·(ṁh_g))
           - assemble new total energy E_new
           - Newton-style 3 iterations to back out T_new from ρ_new·h_s(T_new,τ_new) = E_new
        d. If LRAD and J < NBRJ: re-run RAD with T_new, compute relative change in Q̇,
           exit loop if < TOL else iterate.

    6. UPDATE: T ← T_new, ρᵢ ← ρᵢ_new, ρ ← Σ ρ_w·ρᵢ, ρ_g_storage ← φ·ρ_g
    7. Energy bookkeeping (ADDENE, REMENE, SRCENE, RADQWI, RADQWO)
```

After the loop: write `final_*.out` snapshots and the global energy balance.

## 6. File and code map

### Sources
- [1Dheat.f](../heat_2026-04-11_1837/1Dheat.f) — main program + ~40 subroutines/functions
- [1Drad.f](../heat_2026-04-11_1837/1Drad.f) — radiation module (~7 subroutines)
- Includes:
  - [PARA.INC](../heat_2026-04-11_1837/PARA.INC) — table-size parameters + COMMON `/TABLE/` for cp, k, gas props, aero BCs, B' tables, radiation spectra, flags `LRAD`/`LBLU`/`LINFO`/`LEFF`
  - [DOMA.INC](../heat_2026-04-11_1837/DOMA.INC) — domain (`MAXN`, `DX`, `DV`, `DA`, `X`, gradient ratios `GRADM`/`GRADP`)
  - [DECOMP.INC](../heat_2026-04-11_1837/DECOMP.INC) — pyrolysis params (`MAXS=3`, `RHOVi`, `RHOCi`, `Ai`, `PSIi`, `Ei`, `GAMi`, porosity `PHI`)
  - [CP.INC](../heat_2026-04-11_1837/CP.INC), [KAPPA.INC](../heat_2026-04-11_1837/KAPPA.INC) — alternate forms used by some routines (some duplication with PARA.INC)
  - [RADA.INC](../heat_2026-04-11_1837/RADA.INC) — radiation common: `LAM`, `KAPP`, `SIGP`, `WLW`, `RFLUXM/P`, `PCOEF1/2`
  - **`CASE.INC`** — ⚠️ **missing from Martin's tarball.** Declares ~40 `CASE_*` variables (case config) and their COMMON block. Without this file the code does not compile. NAMELIST at [1Dheat.f:2016](../heat_2026-04-11_1837/1Dheat.f#L2016) lists all the variables that must be declared.

### Key subroutines (rough taxonomy)

| Group | Routines |
|---|---|
| Case I/O | `READ_CASE`, `READ_GAS`, `READ_AERO`, `READ_BPRIME_HW`, `NORMALIZE_BGFILE`, `SKIP_DATA_HEADERS` |
| Material tables | `READ_KAPPA`, `READ_CP`, `READ_RHO`, `READ_TEMP`, `READ_TC_LOCATIONS` |
| Material properties | `CALC_K`, `CALC_K_MIX`, `CALC_CP`, `CALC_CP_MIX`, `CALC_HS_FORM`, `CALC_HS_MIX`, `CALC_EPS_RHO` |
| Pyrolysis state | `CALC_TAU`, `CALC_BETA`, `CALC_POROSITY`, `CALC_RHOVB`, `CALC_RHOCB`, `DECOMP` |
| Gas properties | `CALC_GCP`, `CALC_GMU`, `CALC_RGAS`, `CALC_HGAS`, `CALC_HGAS_ABS`, `CALC_UGAS`, `CALC_RHOG`, `CALC_RHOGP`, `CALC_CVG` |
| Gas solver | `SOLVE_GAS` (tridiagonal Darcy), `TRIDIG` (generic Thomas algorithm) |
| Aerothermal BC | `CALC_TEMP`, `CALC_AERO_CH`, `CALC_AERO_HR`, `CALC_AERO_P`, `CALC_AERO_QR`, `CALC_AERO_LAM`, `CALC_AERO_TEXT`, `CALC_BLOWCORR`, `CALC_WALL_HW` |
| Diagnostics | `WRITE_THERM_HISTORY`, `WRITE_GAS_HISTORY`, `SAMPLE_PROBES`, `GET_FRONT_DEPTHS`, `EVALERR` |
| Radiation | `INIT_RAD`, `RAD`, `TRAPEZOID`, `PLANCK`, `READ_KABS`, `READ_BABS`, `CALC_KABS`, `CALC_BABS`, `CALC_QL` |
| Interpolation | `INTERP_TABLE`, `INTERP_ENERGY_TABLE`, `INTERP_FACE_FROM_CELL`, `INTERP_PRESSURE_PROFILE`, `INTERP_FACE_PROFILE` |

## 7. Configuration

Each run reads **`heat.case`** in the working directory — a Fortran namelist (`&CASECFG_NML`). Example from [examples/aw1/heat.case](../heat_2026-04-11_1837/examples/aw1/heat.case):

```fortran
&CASECFG_NML
  CASE_NAME = 'aw1'
  CASE_L0 = 5.0D-2          ! domain length [m]
  CASE_NBRN = 100           ! cells
  CASE_M = 0                ! Cartesian
  CASE_TIME_INIT = 0.D0
  CASE_TIME_FINAL = 60.D0
  CASE_CFL = 10.D0
  CASE_INIT_TEMP = 298.15D0
  CASE_BND0 = 2             ! left BC: fixed heat flux
  CASE_BNDN = 1             ! right BC: fixed wall T
  CASE_USE_TIME_BCN = .TRUE. ! wall T comes from time_aw1.dat
  CASE_TWN = 1644.D0        ! wall T setpoint (used only if not USE_TIME_BCN)
  CASE_GBND0 = 0; CASE_GBNDN = 1   ! gas BC: closed inside, ambient outside
  CASE_GAS_ENERGY = .TRUE.  ! include gas u·ρ in energy budget
  CASE_NBRTC = 5            ! number of thermocouples
  CASE_CP_FILE = 'cp_virgin.dat'
  ... (all file paths and material constants)
/
```

Data files referenced by the namelist (per-case, in the same directory):

| File | Format | Purpose |
|---|---|---|
| `time_*.dat` | `t, T_wall` | Time-history of wall temperature for `BND_TYPEN=1` |
| `BLinputFile.txt` | `t, ρ_e·u_e·CH, h_r, P_w, q_rad, λ, T_ext` | Boundary-layer aerothermal inputs for `CASE_USE_AERO_BCN` (column header in plain text) |
| `solid.dat` | structured | Porosity φ, NBRS, then per-species `(ρv, ρc, A, ψ, E, γ, Tmin)` (TACOT 3.0 params) |
| `cp_virgin.dat`, `cp_char.dat` | `T, cp` | Heat capacity tables |
| `sol_kappa_*.dat`, `eff_kappa_*.dat` | `T, k` | Solid and effective thermal conductivity tables |
| `rho.dat` | scalar | Initial total density override (mostly redundant) |
| `gas_props.dat` | `T, MW, cp, μ, h, ρ_g` | Pyrolysis-gas thermodynamic table |
| `tc_locations.dat` | depths or fractions | Thermocouple x positions |
| `surfaceChemistry/p=*/Bg=*.txt` | B'g/B'c tables | Surface ablation chemistry (when aero BC active) |

## 8. Example cases

Three problems in [examples/](../heat_2026-04-11_1837/examples/):

| Case | Setup | Reference |
|---|---|---|
| **aw1** | 5cm slab, ramp wall T from 298 K → 1644 K over 1 s, hold to 60 s. No aero, no gas, no radiation. | Steady soak |
| **aw2_tc21** | Time-varying boundary-layer flux from `BLinputFile.txt` + aerothermal + gas flow + B' chemistry. Thermocouple 21 location. | `reference/2.1/thermocouple.txt`, `PyrolysisFront.txt`, `BLinputFile.txt` |
| **aw2_tc22** | Same as aw2_tc21 but at TC 22 depth. | `reference/2.2/...` |

`aw2_*` cases use the surface chemistry B' tables under [examples/aw2_tc21/surfaceChemistry/p=101325/](../heat_2026-04-11_1837/examples/aw2_tc21/surfaceChemistry/p=101325/). Multiple B'g values (mass-injection scaling) — the code interpolates.

## 9. Outputs

| File | When | Content |
|---|---|---|
| `con.out` / `red.out` / `blu.out` | always (name picked by `-r`/`-rb`) | Time-series of `T`, `ρ`, and thermocouple probes |
| `gas.out` | with `-g` | Same as con.out but gas-aware |
| `gas_diag.out` | with `-g` | `t, ṁ_hot, char98_depth, virgin98_depth` |
| `decomp.out`, `press.out`, `vel.out` | with `-g` | Per-cell pyrolysis sources, pressure, velocity histories |
| `final_con.out` / `final_gas.out` | end of run | Final state at all faces (X, T, k, cp, A, Q, ρ) |
| `final_gas_state.out`, `final_gas_species.out` | with `-g` at end | Final pressure / ṁ / per-species ρ |

`GET_FRONT_DEPTHS` ([1Dheat.f:1500](../heat_2026-04-11_1837/1Dheat.f#L1500)) computes the char-front (where ρ has dropped to within 2 % of fully char) and virgin-front (within 2 % of virgin), useful for comparing against `PyrolysisFront.txt` references.

## 10. Build & run

```bash
cd heat_2026-04-11_1837
make heat                  # gfortran -O2 -fopenmp -o heat 1Drad.o 1Dheat.o
cd examples/aw1
../../heat                 # reads ./heat.case
../../heat -r              # with radiation
../../heat -g -r           # with gas flow + radiation
```

Outputs land in the current directory. `make clean` removes them.

## 11. Notable details and gotchas

1. **CASE.INC is missing.** Has to be reconstructed or obtained from Martin before the Fortran will compile. We can re-derive the variable list from the `NAMELIST /CASECFG_NML/` statement at [1Dheat.f:2016](../heat_2026-04-11_1837/1Dheat.f#L2016), but the COMMON block name and SAVE attribute are guessed.
2. **Spectral table override at [1Drad.f:69](../heat_2026-04-11_1837/1Drad.f#L69):** `KAPP(IBB) = 19131.751463370114D3` — Martin is hardcoding a gray absorption coefficient over the data-driven value, probably for testing. Look out for this if results don't match expectations.
3. **Two `INTEG_TABLE` interpolators exist** with similar signatures; `INTERP_ENERGY_TABLE` precomputes ∫cp dT as a separate table for enthalpy lookups.
4. **Temperature bracketing.** `TN(I)` is clamped to `[max(CASE_INIT_TEMP, min(TTC, TTCC)), max(TTC, TTCC)]` — bounded by the cp tables. This will silently mask blow-up.
5. **Energy-balance check** is printed at the end: `(ADDENE − REMENE + SRCENE) / (FINENE − INIENE)`. A clean run should give ≈ 1.
6. **OpenMP in radiation only.** `RAD` parallelizes over wavelength bands (6 threads). Heat conduction is serial.
7. **AppleDouble files (`._*`)** in Martin's tarball can be ignored — Mac metadata, not source.
8. **`CASE_USE_AERO_BCN`** flag is what switches on the full aerothermal surface model (CH, blowing correction, wall enthalpy, pyrolysis-gas blowing, radiation emission). With it off, only `CASE_QAN` / `CASE_TWN` are used as a simple flux/temp BC.

## 12. For Python port (later)

When time comes to convert this:

- **Module split**: `case.py` (read namelist → dataclass), `domain.py` (grid + DA/DV/GRADM/GRADP setup), `materials.py` (table lookups + mixing rules), `pyrolysis.py` (Arrhenius update + τ), `gas.py` (Darcy solver), `radiation.py` (spectral P1, np.einsum-friendly), `bcs.py` (aerothermal BC pieces), `solver.py` (main loop).
- **Vectorize** the per-cell `DO I = 1, NBRN` loops — they're all stencil ops on flat 1D arrays. NumPy slicing handles this naturally.
- **Verification target**: reproduce `examples/aw1/reference/...` thermocouple traces to within whatever tolerance the Fortran energy-balance check gives.
- **Surrogate-friendly choices**: keep the time-stepper purely functional (input state → output state) so it can be wrapped as a training data generator with minimal changes. Make the per-cell update a pure function of `(neighbor states, dx, materials, bcs)` so a GNN can replicate it directly.
- **The 3-iteration enthalpy back-out** ([1Dheat.f:618-629](../heat_2026-04-11_1837/1Dheat.f#L618-L629)) is one of the few places that isn't a pure stencil — it's a cell-local Newton step. Keep it as such; don't try to make it implicit globally.
