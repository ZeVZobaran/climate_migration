# POLOCENTRO environment and crop-suitability data

This directory contains reproducible builders for two extensions to the
POLOCENTRO study:

1. a static FAO/IIASA GAEZ v4 crop-suitability control on the study's 3,800
   1970--2010 AMCs; and
2. annual MapBiomas Brazil Collection 11 land-cover outcomes on those AMCs.

Run from the repository root:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-polocentro-environment.txt
.venv\Scripts\python.exe -u code\polocentro_environment\build_gaez_amc_suitability.py
.venv\Scripts\python.exe -u code\polocentro_environment\build_mapbiomas_amc_panel.py
.venv\Scripts\python.exe -u code\polocentro_environment\validate_outputs.py
```

Outputs are written to `data/processed/polocentro_environment`. Join the GAEZ
wide table to any AMC table by string `amc_code`. Join the MapBiomas AMC panel
by string `amc_code` and integer `year`.

The builders intentionally do not merge these variables into
`amc_year_panel.parquet`: they preserve separate source grains and timing so a
later econometric specification must make the identifying choices explicitly.

Important interpretation: GAEZ crop suitability is a modeled 1961--1990
climate-normal endowment, not annual observations. MapBiomas begins in 1985,
ten years after the January 1975 POLOCENTRO assignment, so it supplies
post-treatment environmental outcomes but no pre-program land-cover baseline.
