# POLOCENTRO extension: crop suitability and environmental outcomes

## Executive finding

Both extensions are feasible, but they have different timing and econometric
roles.

| Information set | Product used | Coverage | Study role | Central limitation |
|---|---|---:|---|---|
| Crop suitability | FAO/IIASA GAEZ v4, continuous all-land suitability (`sx`), rainfed, CRU TS 3.2 historical climate | Static 1961--1990 climate normal; 5 arc-minutes; 3,800 AMCs | Predetermined geographic control and treatment-selection covariate | It is not an annual series, and the 30-year climate normal overlaps the post-1975 period |
| Land cover | MapBiomas Brazil Collection 11 municipality statistics | Annual 1985--2025; 41 years; 3,799 of 3,800 AMCs | Environmental stock and post-1985 net-change outcomes | Starts ten years after assignment, so it supplies no pre-program land-cover baseline or pre-trend |

The common claim that the FAO Crop Suitability Index is "available from 1980
onward" mixes two different GAEZ outputs. GAEZ's agro-climatic-resources module
contains annual climate series, but crop-suitability and attainable-yield maps
are scenario/reference-period outputs. GAEZ v4 includes the historical
30-year periods 1961--1990, 1971--2000, and 1981--2010. The earliest period is
the best match for POLOCENTRO and is the one built here. The
[FAO GAEZ v4 overview](https://www.fao.org/gaez/gaezv4/en) and
[GAEZ v4 repository guide](https://gaez-v4-data.fao.org/data/GAEZ%20v4%20data%20repository%20user%20guide.pdf)
document the product structure, resolution, crops, variables, and reference
periods.

MapBiomas Collection 11 has annual municipality-level coverage statistics for
1985--2025 in the official
[MapBiomas statistics downloads](https://brasil.mapbiomas.org/downloads/estatisticas/).
The environmental-outcome extension is therefore readily implementable, but
only as a post-treatment series.

## 1. FAO crop suitability

### Product choice

The generated control uses the GAEZ v4 Module V variable `sx`: the continuous
crop suitability index for **all land in the grid cell**, rather than the
corresponding index restricted to current cropland. This distinction matters:
current-cropland restriction would condition the control on a land-use stock
that POLOCENTRO may itself have changed.

The files use:

- historical CRU TS 3.2 climate, 1961--1990;
- rainfed production;
- high and low management/input levels;
- no CO2-fertilization variant;
- eight POLOCENTRO-relevant crops: soybean, maize, dryland rice, Phaseolus
  bean, wheat, cotton, coffee, and sugarcane;
- exact polygon/raster intersections with spherical-area weighting;
- the native GAEZ 0--10,000 scale rescaled to 0--100.

GAEZ v5 is newer, but it uses a 2020 baseline and the climate normals
1981--2000 and 2001--2020. Those periods are less suitable for a program
assigned in 1975, so v4 is the deliberate research-design choice. FAO describes
the updated v5 baseline and climate periods on its
[current GAEZ page](https://www.fao.org/gaez/en/).

### Recommended use

The preferred main covariate is crop-specific high-input, rainfed suitability,
especially soybean suitability. Use the low-input counterpart and the
eight-crop composite as robustness controls. High-minus-low suitability can
also be used as a predetermined measure of how strongly modern inputs change
productive potential, but it should be motivated as treatment-effect
heterogeneity rather than silently included in every specification.

Do not use GAEZ actual yields, current-cropland suitability, or yield gaps as
ordinary baseline controls: those products embed observed production or
post-treatment land use.

### Descriptive check

The reconstructed operational-core treatment covers 81 AMCs. Their mean
high-input soybean suitability is 62.7 on the 0--100 scale, compared with 50.2
for other AMCs. The corresponding eight-crop means are 51.1 and 39.1. These are
descriptive, not causal, but they confirm that agronomic selection is material
and should be addressed in treatment-assignment/balance models.

## 2. MapBiomas environmental outcomes

### Coverage and variables

Collection 11 covers 1985--2025. The generated panel sums MapBiomas leaf
classes to each AMC and provides annual hectares and shares for:

- native vegetation, split into forest and non-forest native formations;
- anthropic use;
- farming, pasture, agriculture, soybean, sugarcane, rice, cotton, coffee, and
  forest plantation;
- urban area and mining;
- natural water and not-observed area;
- year-to-year native-vegetation net change and signed net loss;
- cumulative net native-vegetation loss and anthropic expansion since 1985.

`native_vegetation_net_loss_ha` is the previous year's native-vegetation stock
minus the current stock. A positive number is net loss and a negative number is
net gain. It is **not gross deforestation**: clearing and regeneration can
offset within an AMC-year. MapBiomas's separate deforestation/secondary-
vegetation method identifies confirmed pixel transitions from 1987 onward
using 1985 as the base and 1986 for confirmation; see the official
[method description](https://brasil.mapbiomas.org/metodo-desmatamento/).

### Boundary harmonization

The official Collection 11 workbook includes current IBGE geocodes, which
allows a direct match for 5,564 study municipalities. Five post-2010
municipalities are assigned to their 2010 parent AMC. Boa Esperança do Norte's
two parent municipalities are both in AMC 1098. Paraíso das Águas was created
from parts of three municipalities belonging to different AMCs, so its
statistics are split by exact 2024 polygon-area overlap:

- AMC 1047: 64.09%;
- AMC 1065: 24.01%;
- AMC 1068: 11.90%.

This allocation preserves total area but assumes uniform land-cover
composition within Paraíso das Águas. The official
[IBGE 2024 Mato Grosso do Sul boundaries](https://geoftp.ibge.gov.br/organizacao_do_territorio/malhas_territoriais/malhas_municipais/municipio_2024/UFs/MS/)
are retained for reproducibility.

Collection 11 does not include Fernando de Noronha as a municipality. It is a
standalone AMC (7061), leaving 3,799 of 3,800 AMCs covered. This omission is
irrelevant for the POLOCENTRO treatment region and is explicitly recorded in
metadata rather than imputed.

### Descriptive check

Within the 81 operational-core AMCs, native vegetation falls from 17.709
million hectares in 1985 to 10.599 million in 2025, while anthropic use rises
from 10.248 million to 17.551 million hectares. The unweighted mean AMC native-
vegetation share falls from 51.5% to 31.8%; for other AMCs it falls from 46.3%
to 37.8%. These differences are not treatment effects: the GAEZ balance results
show that POLOCENTRO areas were selected on agronomic potential, and MapBiomas
does not observe the first ten treatment years.

## 3. Identification implications

MapBiomas cannot support a conventional pre/post difference-in-differences or
event study around the 1975 assignment. The defensible first specifications
are:

1. **Post-1985 long difference:** regress 1985--2025 native-vegetation loss or
   anthropic expansion on POLOCENTRO exposure, conditioning on the 1985 land-
   cover stock, GAEZ crop suitability, 1970 socioeconomic covariates, and the
   treatment-selection model. Interpret this as differential loss among land
   still present in 1985, not the total 1975 program effect.
2. **Post-1985 trajectory:** model annual outcomes from 1985 onward with AMC and
   year fixed effects and treatment exposure interacted with a post-1985 time
   trend or flexible year indicators. This describes divergence after 1985;
   the treatment main effect is absorbed and pre-trends remain untestable.
3. **Dose response:** prefer continuous
   `polocentro_operational_area_share`, with the operational-core binary as a
   readable summary and the existing treatment definitions as robustness
   checks.

If a genuinely pre-1975 environmental baseline is indispensable, a separate
remote-sensing reconstruction from early Landsat MSS imagery (available from
the 1970s but not as a consistent turnkey national land-cover product) would
be required. That is a classification/validation project, not a simple data
download, and its comparability with 30 m MapBiomas Landsat-era maps would need
to be demonstrated.

## 4. Generated datasets

All analysis files are under `data/processed/polocentro_environment`.

| File | Grain | Merge key | Purpose |
|---|---|---|---|
| `gaez_v4_6190_amc_crop_suitability.parquet` | AMC x crop x input level (60,800 rows) | `amc_code` plus crop/input fields | Full suitability control and heterogeneity data |
| `gaez_v4_6190_amc_crop_suitability_wide.parquet` | AMC (3,800 rows) | `amc_code` | Direct regression merge with crop-specific and composite controls |
| `mapbiomas_collection11_amc_year.parquet` | AMC x year (155,759 rows) | `amc_code`, `year` | Main environmental outcomes |
| `mapbiomas_collection11_municipality_year.parquet` | Current municipality x year (228,370 rows) | `geocode`, `year` | Audit and alternative municipality analysis |
| `mapbiomas_collection11_municipality_to_amc.parquet` | Municipality-to-AMC allocation | `geocode`, `amc_code` | Boundary-harmonization audit trail |

Compressed CSV equivalents are supplied for the main GAEZ long file and
MapBiomas AMC panel. JSON metadata record definitions, checksums, source URLs,
class lists, coverage, and cautions. The builders are in
`code/polocentro_environment`.

GAEZ v4 data are distributed under CC BY-NC-SA 4.0; MapBiomas data are open
under CC BY 4.0 with source attribution. These licensing differences should be
preserved if a replication package is redistributed.
