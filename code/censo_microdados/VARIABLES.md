# Harmonized variable groups

Every row is one sampled person in one census. `person_weight` is the person
expansion factor. Columns ending in `_code` preserve the census-specific code;
labels and harmonized derivatives should never overwrite those source codes.

## Geography and migration

- Current UF, municipality, mesoregion, microregion, and weighting area where available.
- Birth UF/country and whether born in the current municipality.
- Duration in the current municipality and UF.
- Last-residence UF/municipality/country and urban/rural status where available.
- Five-year origin UF/municipality/country, `migrant_5yr`, and
  `internal_migrant_5yr`.
- `migrant_5yr` is intentionally unavailable in 1970 and 1980. Municipal
  five-year origin begins in 1991.

## Person characteristics

- Sex, age, race/color, household relationship, and marital status.
- Literacy, school attendance, detailed education source codes, exact years of
  schooling where supplied, and harmonized education level where supplied.
- Labor-force and employment status, occupation, industry, employment
  position, hours worked, workplace municipality, and nominal earnings.

## Household and housing characteristics

- Household size and household/sample weight where available.
- Dwelling type, tenure, rooms, bedrooms, bathrooms, wall material, water,
  sanitation, and electricity.
- Radio, television, refrigerator, washing machine, telephone, computer,
  internet, and automobile availability where asked.
- Nominal household income, minimum-wage income, and per-capita income where
  supplied.

Nominal earnings and income are not comparable across census years without a
separate currency and price-level harmonization. Housing assets also change
meaning as technologies diffuse, so source-year codes are retained.

## Restricted characteristics dataset

The `characteristics_panels/persons_YEAR.parquet` files form a deliberately
narrow, analysis-ready view for 1991, 2000, and 2010. Key variable families are:

- `current_*` and `origin_5yr_*`: current and five-year-prior municipality,
  UF, mesoregion, and urban/rural status. The 2010 prior urban/rural fields are
  null because the census source does not report them.
- `sex*`, `race*`, `age_years`, and `education*`: source codes, readable labels where
  available, and common education-attainment categories.
- `five_year_internal_od_observed`, `stayer_5yr`, and `internal_migrant_5yr`:
  mutually consistent indicators based on identified Brazilian municipal
  origins and destinations. Source migration indicators are kept separately.
- `origin_northeast`, `origin_south`, and `origin_region_group`: origin-region
  indicators using the full official Northeast and South state groupings.
- `previous_census_population` and `previous_population_*`: prior-census
  population and auditable direct/predecessor assignment fields.
- `destination_large_city_500k` and `destination_large_city_1m`: inclusive
  thresholds (`>=`) based on lagged municipality population.
- `destination_frontier_north_only`, `destination_frontier_main_north_mt`, and
  `destination_frontier_broad_north_mt_ms`: the three destination robustness
  definitions. The main definition excludes Mato Grosso do Sul.
- `migration_system_{500k,1m}_{north_only,main_north_mt,broad_north_mt_ms}`:
  six categorical specifications applying the exact precedence `stayer` >
  `large_city` > `agri_frontier` > `other`. They are null outside the valid
  internal origin--destination universe.
- `mo_fm_road`, `mo_fm_road_rad`, and `mo_dist_km`: Morten--Oliveira
  mesoregion-to-mesoregion travel measures. Census years 1991, 2000, and 2010
  use network years 1990, 2000, and 2010, respectively.
- `prev_corridor_{stayer,city,agri_frontier,other}`: aggregate
  Morten--Oliveira `N_od_flow_all` flows from the person's five-year origin
  mesoregion to all mesoregions in each migration-system alternative in the
  preceding census flow year: 1980 for 1991, 1991 for 2000, and 2000 for 2010.
- `min_tt_to_{stayer,city,agri_frontier,other}` and
  `med_tt_to_{stayer,city,agri_frontier,other}`: unweighted minimum and median
  Morten--Oliveira `fm_road` travel time from the person's origin mesoregion to
  all mesoregions in each alternative. These are choice-set measures indexed by
  origin, not characteristics of the chosen destination.

All sampled people are retained and `person_weight` remains the expansion
factor. Post-decision characteristics such as current occupation, employment,
and income are intentionally absent.

For these choice-set columns, alternatives are constructed at the mesoregion
level under the main 500,000-inhabitant, North plus Mato Grosso definition.
Precedence is `stayer` > `city` > `agri_frontier` > `other`. The stayer option
is the origin mesoregion itself, and a city mesoregion is any mesoregion that
contains at least one lagged-population municipality at or above 500,000
residents.
