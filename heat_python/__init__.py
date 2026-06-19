"""Python port of Martin's 1D heat-shield Fortran solver.

Maps to heat_2026-04-11_1837/:
    case.py        <- CASE.INC + READ_CASE
    domain.py      <- DOMA.INC + grid setup in 1Dheat.f
    materials.py   <- READ_KAPPA, READ_CP, CALC_K_MIX, etc.
    pyrolysis.py   <- CALC_TAU, DECOMP, CALC_POROSITY, Arrhenius update
    bcs.py         <- boundary condition logic
    gas.py         <- SOLVE_GAS (Darcy flow)         [phase 4]
    radiation.py   <- 1Drad.f                         [phase 4]
    solver.py      <- main time loop (PROGRAM HEAT)
    main.py        <- CLI entry point
"""

__version__ = "0.1.0"
