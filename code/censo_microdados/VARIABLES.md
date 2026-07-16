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
