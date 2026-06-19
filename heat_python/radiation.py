"""Radiation transport. Replaces 1Drad.f (INIT_RAD, RAD, READ_KABS, READ_BABS,
PLANCK, TRAPEZOID, CALC_KABS, CALC_BABS).   [phase 4]

Multi-band P1 / Marshak radiative transfer, solved tridiagonally per
wavelength band. Output is the volumetric source Q_rad per cell.
Only active when -r or -rb flag is set.
"""
