# Migration-group socioeconomic profiles

This directory contains **159 survey-weighted SVG profiles** comparing
immigrants, emigrants, and stayers.

- `state_year/<year>/<UF>.svg`: one profile for each observed state and census.
- `state_pooled/<UF>.svg`: person-weighted pool across all available censuses.

Every panel is normalized separately within migration group. The plot legend
reports the full classified group population in thousands using `person_weight`.

## Migration definitions

- **1970:** previous state of residence. A missing previous state is treated as
  no interstate move. Guanabara is merged into RJ and Fernando de Noronha into
  PE. The historical MT and GO boundaries remain.
- **1980:** state of birth.
- **1991, 2000, 2010:** state of residence five years before the census.

The common analysis universe is age five or older. International migrants and
records without an identifiable internal origin are excluded from the
three-group comparison. Each internal mover appears as an immigrant in the
destination profile and as an emigrant in the origin profile.

## Dimensions

Age, national income rank, sex, literacy, years of schooling, household size,
rooms, and refrigerator access are shown. Years of schooling are only available
in harmonized form for 1991 and 2000; household size is unavailable in 1991.

Income is plotted as no income plus national, survey-weighted positive-income
quintiles for adults in each census. This makes pooled plots comparable despite
currency changes. The 1980 source contains main-job rather than total personal
income, which is stated in every plot.

Reusable numeric results are written to
`data/processed/censo_microdados/migration_profiles` by
`code/censo_microdados/migration_profiles.py`.
