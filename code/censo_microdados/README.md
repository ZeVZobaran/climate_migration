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

The currently materialized base build contains 111,783,901 records in 135
Parquet files. Its files share the older physical 78-column schema; Hive
partition discovery adds `current_uf`, producing a 79-column logical Arrow
dataset. A new base rebuild also preserves `origin_5yr_urban_code`, adding one
physical column. The restricted characteristics builder reads that field
directly from the raw source when it is absent from the older base build.

## AMC analysis tables

The harmonized AMC migration-flow, resident-characteristics, municipal GDP,
and combined AMC-year tables are documented in
[`AMC_PANELS.md`](AMC_PANELS.md). These outputs cover 1970--2010 AMCs,
retain origin UF, and use the agreed 1970 and 1980 interstate-migration proxy
definitions alongside exact fixed-date origins from 1991 onward.

## Analysis-ready characteristics datasets, 1991--2010

The restricted analysis datasets requested for the migration-system exercise
are under `data/processed/censo_microdados/characteristics_panels`. Despite the
directory's historical name, these are three repeated cross-sections, not a
longitudinal panel: each file has one sampled person per row and people cannot
be linked across censuses.

Build the official prior-census population crosswalk, construct the datasets,
and run the independent full-row audit with:

```powershell
.venv\Scripts\python.exe -u code\censo_microdados\build_municipality_population_crosswalk.py
.venv\Scripts\python.exe -u code\censo_microdados\build_characteristics_panels.py
.venv\Scripts\python.exe -u code\censo_microdados\validate_characteristics_panels.py
```

The outputs are `persons_1991.parquet`, `persons_2000.parquet`, and
`persons_2010.parquet`, plus `metadata.json` and `validation_report.json`.
They retain only predetermined demographic and migration characteristics:
current and five-year-prior geography and urban/rural status, sex, race/color,
age, education, survey weight, origin/destination mesoregions, lagged municipal
population, Morten--Oliveira travel measures, and origin-indexed
migration-system accessibility measures. Current occupation, income,
employment, and other potentially post-migration outcomes are excluded.

The valid migration-system universe requires an identified Brazilian
municipality both currently and five years earlier. All sampled people remain
in the files, but migration-system variables are null outside that universe
(including people younger than five and foreign or unidentified origins).
For valid origins, category precedence is `stayer`, `large_city`,
`agri_frontier`, then `other`. Large-city status has both 500,000 and 1,000,000
threshold variants. Frontier variants are North only; North plus Mato Grosso
(main); and North plus Mato Grosso and Mato Grosso do Sul (broad robustness).

Large-city status uses municipality population in the preceding census:
1980 for 1991, 1991 for 2000, and 2000 for 2010. Municipalities not yet present
in the preceding census inherit the population of their maximum-area-overlap
predecessor. The predecessor code, assignment method, and overlap share are
retained so every such assignment is auditable.

Travel measures are joined at the mesoregion origin--destination level using
Morten--Oliveira years 1990, 2000, and 2010 for the respective census files.
`mo_fm_road` is the primary measure; `mo_fm_road_rad` and `mo_dist_km` are also
retained. Five-year-prior urban/rural status is unavailable in the 2010 source
and is deliberately null rather than imputed.

Migration-system accessibility measures summarize the full mesoregion choice
set faced by a person's five-year origin, not just the destination actually
chosen. The four alternatives are `stayer`, `city`, `agri_frontier`, and
`other`, with the same precedence as the main 500,000-inhabitant, North plus
Mato Grosso system: stayer first, then city, then agricultural frontier, then
other. Because Morten--Oliveira flows and travel times are mesoregional, the
stayer alternative is the origin mesoregion itself and a city mesoregion is any
mesoregion containing at least one lagged-population municipality at or above
500,000 residents. For each origin and alternative, the files include prior
Morten--Oliveira corridor flows (`prev_corridor_*`) and unweighted minimum and
median `fm_road` travel times (`min_tt_to_*`, `med_tt_to_*`). Prior flow years
are 1980, 1991, and 2000 for the 1991, 2000, and 2010 census files,
respectively. The 137-row yearly lookups used to populate those person-level
columns are saved as `migration_system_accessibility_YEAR.parquet`.

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

## Macro-region migration matrices

Generate weighted 5x5 origin-destination matrices for North, Northeast,
Center-West, Southeast, and South:

```powershell
.venv\Scripts\python.exe -u code\censo_microdados\regional_migration_matrices.py
```

For every census and for a person-weighted pool of all censuses, the script
writes absolute weighted populations, origin-row emigrant shares, and
destination-column immigrant shares under
`data/processed/censo_microdados/regional_migration_matrices`. A readable copy
of all matrices is written to `reports/regional_migration_matrices.md`.
