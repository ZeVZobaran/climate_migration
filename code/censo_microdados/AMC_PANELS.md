# AMC analysis tables

The builder creates four analysis tables under
`data/processed/amcs`. All census geography is harmonized to the
3,800 minimum comparable areas (AMCs) for 1970--2010.

Run or resume the complete build with:

```powershell
.venv\Scripts\python.exe -u code\censo_microdados\build_amc_panels.py --rebuild-crosswalk
```

## Outputs

- `amc_origin_uf_year_flows.parquet`: one AMC--origin-UF--census-year cell.
  It contains survey-weighted population and interstate-migrant counts, cell
  shares, and the socioeconomic composition of each origin cell.
- `amc_year_characteristics.parquet`: one AMC--census-year cell containing
  characteristics of all usual residents, not only migrants.
- `municipality_year_gdp.parquet`: one source municipality--year cell with
  total GDP and agriculture, industry, private-services, and public-
  administration value added. Monetary values are R$ thousands at constant
  2010 prices. The available years are 1959, 1970, 1975, 1980, 1985, 1996, 2000,
  2005, and 2010; 2015 is deliberately excluded.
  The 1959 workbook has no separate public-administration value-added values,
  so that component is null in 1959 rather than imputed or set to zero.
- `amc_year_panel.parquet`: an outer AMC--year panel combining census
  characteristics and AMC-aggregated GDP. It retains data-availability and
  municipality-coverage fields, so missing components are not silently
  treated as zeros.

`municipality_to_amc_crosswalk.parquet` makes every geographic assignment
auditable. `metadata.json` records definitions, source paths, diagnostics, and
validation results.

## Migration definitions

The flow universe is usual residents aged five or older with positive survey
weight and a mapped destination AMC. The destination is current residence.
Origin is a UF, assigned as follows:

- 1970: previous UF when residence in the current UF is under five years and
  previous UF differs from current UF.
- 1980: birth UF when residence in the current UF is under five years and
  birth UF differs from current UF. This is a proxy, not exact five-year
  residence.
- 1991, 2000, and 2010: reported UF of residence five years before the census.

For people not identified as interstate migrants, origin UF is set to current
UF. Consequently, a current-UF cell is explicitly labelled as a residual: it
mixes stayers, intrastate migrants, and cases not identifiable as interstate
migrants. It must not be interpreted as a pure non-migrant cell.

Each socioeconomic estimate has its own `*_valid_weighted_population`
denominator because variables differ in availability and valid coding across
censuses. Household appliance and electricity fields in the existing 2000 and
2010 person files have incomplete geographic coverage; use their denominators
to restrict or weight comparisons. Raw income levels are in census-specific
currencies and are not comparable over time without a separate price and
currency harmonization.

## Program-evaluation use

For first-pass treatment-assignment balance tests, use covariates dated before
program assignment, especially the 1970 census/GDP baseline where appropriate.
Later census characteristics may themselves respond to treatment and should
not be used as ordinary controls in causal specifications without a clear
timing argument. The separate flow, resident-characteristics, municipality GDP,
and AMC panel tables support migration outcomes, individual-composition
outcomes, production outcomes, and pre-treatment selection checks without
forcing them into a single grain.

## POLOCENTRO treatment assignment

Build the January 1975 geographic treatment vectors with:

```powershell
code\censo_microdados\download_polocentro_sources.ps1
.venv\Scripts\python.exe -u code\censo_microdados\build_polocentro_treatment.py
```

The legal assignment source is Article 2 of Decreto 75.320/1975. Banco Central
Circular 259/1975, Annex 1 is used to operationalize the named corridors and
bounded areas because it supplies widths and approximate surface areas. The
outputs are:

- `polocentro_1975_municipality_treatment.parquet` and `.csv`: all 5,565 2010
  municipalities, their AMC, reconstructed land-area exposure, binary
  assignments, and the relevant POLOCENTRO area names.
- `polocentro_1975_amc_treatment.parquet` and `.csv`: the same assignment at
  the 3,800-AMC grain, directly joinable to `amc_year_panel.parquet` by
  `amc_code`.
- `polocentro_1975_areas.gpkg`: the twelve reconstructed program areas. Vão do
  Paracatu has separate decree-literal (BR-356) and operational Circular 259
  (BR-365) geometries.
- `polocentro_1975_metadata.json`: sources, construction diagnostics, counts,
  and interpretation warnings.

The recommended binary is `polocentro_operational_core`: a municipality or AMC
is assigned when a municipal seat lies inside a reconstructed program area or
at least 10% of its land area overlaps one. Prefer
`polocentro_operational_area_share` as a continuous exposure measure, and use
the seat-only, majority-area, and any-overlap variables as robustness checks.
The reconstruction represents the original January 1975 selection; it does not
fold in the October 1975 expansion or exceptional out-of-area credit allowed by
the implementing regulation.

The road/rail corridors are the highest-confidence assignments. Areas described
only by rivers have lower boundary precision, especially Xavantina and Rio
Verde; the GeoPackage and `boundary_confidence` field make this explicit. These
variables are suitable for municipality/AMC research, not parcel-level
eligibility determinations.
