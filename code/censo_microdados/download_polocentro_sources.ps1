$ErrorActionPreference = "Stop"

$destination = "data\censo_microdados\amc"
New-Item -ItemType Directory -Force -Path $destination | Out-Null

curl.exe -L `
  "https://geoftp.ibge.gov.br/cartas_e_mapas/bases_cartograficas_continuas/bc250/versao2017/shapefile/Transporte_v2017.zip" `
  -o "$destination\Transporte_v2017.zip"

curl.exe -L `
  "https://geoftp.ibge.gov.br/cartas_e_mapas/bases_cartograficas_continuas/bc250/versao2017/shapefile/Localidades_v2017.zip" `
  -o "$destination\Localidades_v2017.zip"

curl.exe -L `
  "https://normativos.bcb.gov.br/Lists/Normativos/Attachments/40950/Circ_0259_v1_O.pdf" `
  -o "$destination\bcb_circular_259_1975.pdf"

curl.exe -L `
  --data-binary "@code\censo_microdados\polocentro_waterways.overpassql" `
  "https://overpass-api.de/api/interpreter" `
  -o "$destination\polocentro_waterways.json"
