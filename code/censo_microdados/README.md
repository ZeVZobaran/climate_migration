# Harmonized census microdata

The output is a repeated cross-section with one row per sampled person and
census. It is not a panel: people cannot be linked between censuses.

Primary migration outcome: residence five years before the census. Municipal
five-year origins are available from 1991 onward. The 1970 and 1980 files are
retained for demographic, socioeconomic, birthplace, duration, and less exact
migration measures, but `migrant_5yr` is deliberately null for those years.

Run a small validation build:

```powershell
.venv\Scripts\python.exe code\censo_microdados\build_census_parquet.py --sample-rows 10000
```

Run the national build by omitting `--sample-rows`. Outputs are Zstandard-
compressed Parquet under `data/processed/censo_microdados/persons`, partitioned
by census year and current UF. Source categorical codes are preserved in
`*_code` columns. Nominal incomes must not be compared across years without a
separate currency/price harmonization step.

Resume an interrupted national build without rebuilding valid partitions:

```powershell
.venv\Scripts\python.exe -u code\censo_microdados\build_census_parquet.py --chunksize 500000 --resume
```

If files produced by an older version of the pipeline need their physical
Parquet schemas harmonized, run:

```powershell
.venv\Scripts\python.exe -u code\censo_microdados\build_census_parquet.py --normalize-existing --chunksize 500000
```

The completed build contains 111,783,901 records in 135 Parquet files. All
files share one physical 78-column schema; Hive partition discovery adds
`current_uf`, producing a 79-column logical Arrow dataset.

## Migration-group socioeconomic profiles

Generate survey-weighted distribution plots comparing immigrants, emigrants,
and stayers for every state-year and for a person-weighted pool of censuses:

```powershell
.venv\Scripts\python.exe -u code\censo_microdados\migration_profiles.py
```

Plots are written to `figs/censo_microdados/migration_profiles`. Reusable group
sizes and weighted distributions are written under
`data/processed/censo_microdados/migration_profiles`. Use `--plots-only` to
regenerate SVGs without rescanning the person microdata.

## Migration-flow types

Classify migrants into local, neighboring-state, Northeast-to-Southeast,
South-to-agricultural-frontier, the two reverse flows, long-distance DF/GO
inflows, and other movements:

```powershell
.venv\Scripts\python.exe -u code\censo_microdados\migration_types.py
```

Weighted census shares and the underlying state-to-state corridor table are
written under `data/processed/censo_microdados/migration_types`.
