# POLOCENTRO BCB operational-area maps

Generate both 16:9 presentation figures from the repository root:

```powershell
code\polocentro_maps\run_polocentro_map.ps1
```

The default run creates a Brazil-wide locator map and a tighter map where AMC
boundaries and full/partial exposure are easier to read. Only the Banco Central
operational reconstruction is drawn; the decree-literal Paracatu variant is not
included. Titles, subtitles, coverage counts, and source notes are hidden by
default so the outputs contain only map elements and legends. They can be
restored with the corresponding `presentation` switches in the configuration.
The Brazil view also draws the exact bounding extent used by the detailed view.

The easiest way to revise either figure is to edit
`polocentro_map_config.json`. Separate `brazil` and `zoom` sections expose:

- title and subtitle;
- map padding and AMC/boundary line widths;
- per-view tight cropping and inside/outside legend placement;
- label, north-arrow, and scale-bar visibility;
- scale-bar length and position;
- figure colors, transparency, size, DPI, and output formats;
- area-label x/y offsets in kilometres.

Useful command-line overrides:

```powershell
# Generate only the Brazil-wide figure.
code\polocentro_maps\run_polocentro_map.ps1 --view brazil

# Generate only the detailed AMC figure, without area labels, as PNG and SVG.
code\polocentro_maps\run_polocentro_map.ps1 `
  --view zoom --no-labels --formats png svg
```

Use `--help` for all overrides. Outputs are written to
`figs/polocentro_maps` by default.
