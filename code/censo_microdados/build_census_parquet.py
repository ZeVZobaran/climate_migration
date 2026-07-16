"""Build harmonized Brazilian census person records as partitioned Parquet.

The output is a repeated cross-section: one row per sampled person and census.
Source codes are retained in ``*_code`` columns; harmonized variables are only
created where the source supports a defensible common interpretation.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


@dataclass(frozen=True)
class F:
    start: int
    width: int
    decimals: int = 0


UF_FROM_1980 = {
    **{i: code for i, code in enumerate(
        [11, 12, 13, 14, 15, 16, 21, 22, 23, 24, 25, 26, 27, 28, 29,
         31, 32, 33, 35, 41, 42, 43, 50, 51, 52, 53], 1)},
}
UF_ABBR = {"AC":12,"AL":27,"AM":13,"AP":16,"BA":29,"CE":23,"DF":53,
           "ES":32,"FN":26,"GB":33,"GO":52,"MA":21,"MG":31,"MT":51,
           "PA":15,"PB":25,"PE":26,"PI":22,"PR":41,"RJ":33,"RN":24,
           "RO":11,"RR":14,"RS":43,"SC":42,"SE":28,"SP":35}

P70 = {
    **{f"V{i:03d}": F(s, w) for i, s, w in [
        (1,1,3),(2,4,3),(3,7,2),(4,9,1),(5,10,2),(6,12,1),(7,13,1),
        (8,14,1),(9,15,1),(10,16,1),(11,17,1),(12,18,1),(13,19,1),
        (14,20,1),(15,21,1),(16,22,1),(17,23,1),(18,24,1),(19,25,1),
        (20,26,2),(21,28,2),(22,30,1),(23,31,1),(24,32,1),(25,33,1),
        (26,34,1),(27,35,2),(28,37,1),(29,38,1),(30,39,2),(31,41,1),
        (32,42,1),(33,43,2),(34,45,1),(35,46,1),(36,47,1),(37,48,1),
        (38,49,1),(39,50,2),(40,52,1),(41,53,4),(42,57,1),(43,58,1),
        (44,59,3),(45,62,3),(46,65,1),(47,66,1),(48,67,1),(49,68,1),
        (50,69,2),(51,71,1),(52,72,1),(53,73,2),(54,75,2)]}
}

P00 = {k:F(*v) for k,v in {
 "uf":(1,2),"meso":(3,4),"micro":(7,5),"muni":(12,7),"control":(39,8),
 "person_no":(47,2),"area":(51,13),"urban":(66,1),"self_response":(68,1),
 "sex":(69,1),"relationship":(71,2),"family_no":(77,1),"age":(79,3),
 "age_months":(83,2),"race":(87,1),"always_muni":(103,1),
 "years_muni":(105,2),"born_muni":(108,1),"born_uf":(110,1),
 "nationality":(112,1),"birth_origin":(119,2),"years_uf":(122,2),
 "last_origin_uf":(125,2),"status_5yr":(128,1),"origin_5yr_muni":(130,7),
 "origin_5yr_uf":(138,2),"work_muni":(141,7),"literacy":(149,1),
 "school":(151,1),"education_course":(153,2),"education_grade":(156,1),
 "education_years":(168,2),"marital_status":(174,1),"worked":(176,1),
 "occupation":(188,4),"industry":(193,5),"employment_position":(199,1),
 "income_main":(209,6),"income_all_jobs":(249,6),"income_all_jobs_sm":(255,6,2),
 "hours":(267,3),"income_total":(309,6),"income_total_sm":(315,6,2),
 "person_weight":(335,11,8)}.items()}

D00 = {k:F(*v) for k,v in {
 "uf":(1,2),"muni":(12,7),"control":(39,8),"dwelling_type":(74,1),
 "rooms":(76,2),"bedrooms":(79,1),"tenure":(81,1),"water":(85,1),
 "bathrooms":(89,1),"sanitation":(93,1),"electricity":(97,1),"radio":(99,1),
 "refrigerator":(101,1),"washing_machine":(105,1),"telephone":(109,1),
 "computer":(111,1),"televisions":(113,1),"automobiles":(115,1),
 "household_size":(119,2),"household_income":(145,6),
 "household_income_sm":(151,6,2),"household_weight":(157,11,8)}.items()}

P10 = {k:F(*v) for k,v in {
 "uf":(1,2),"muni":(3,5),"area":(8,13),"control":(21,8),
 "person_weight":(29,16,13),"meso":(46,2),"micro":(48,3),"urban":(53,1),
 "relationship":(54,2),"person_no":(56,2),"sex":(58,1),"age":(62,3),
 "race":(68,1),"born_muni":(74,1),"born_uf":(75,1),"nationality":(76,1),
 "birth_uf":(82,7),"birth_country":(89,7),"years_uf":(96,3),"years_muni":(99,3),
 "last_origin_type":(102,1),"last_origin_uf":(103,7),"last_origin_muni":(110,7),
 "last_origin_country":(117,7),"origin_5yr_type":(124,1),"origin_5yr_uf":(125,7),
 "origin_5yr_muni":(132,7),"origin_5yr_country":(139,7),"literacy":(146,1),
 "school":(147,1),"education_course":(154,2),"education_completed":(156,1),
 "education_level":(158,1),"marital_status":(194,1),"worked":(195,1),
 "occupation":(200,4),"industry":(204,5),"employment_position":(209,1),
 "income_main":(219,6),"income_main_sm":(225,6,2),"income_all_jobs":(247,7),
 "income_all_jobs_sm":(254,9,5),"income_total":(263,7),"income_total_sm":(270,9,5),
 "household_income":(279,7),"household_income_pc":(296,8,2),"hours":(313,3),
 "work_location":(328,1),"work_uf":(329,7),"work_muni":(336,7),
 "labor_force":(391,1),"employment_status":(392,1)}.items()}

D10 = {k:F(*v) for k,v in {
 "uf":(1,2),"muni":(3,5),"area":(8,13),"control":(21,8),
 "household_weight":(29,16,13),"dwelling_type":(56,2),"tenure":(58,1),
 "wall_material":(74,1),"rooms":(75,2),"bedrooms":(80,2),"bathrooms":(85,1),
 "sanitation":(87,1),"water":(88,2),"electricity":(92,1),"radio":(94,1),
 "television":(95,1),"washing_machine":(96,1),"refrigerator":(97,1),
 "telephone_mobile":(98,1),"telephone_fixed":(99,1),"computer":(100,1),
 "internet":(101,1),"automobile":(103,1),"household_size":(105,2),
 "household_income":(109,7),"household_income_sm":(116,10,5),
 "household_income_pc":(126,8,2),"household_income_pc_sm":(134,9,5)}.items()}

DBF80_P = ["UF","MUNIC","NDOM","SEXO","PARENDOM","PARENFAM","FAMILIA","COR",
 "MINACION","MIUFNASC","MINASCMU","MIMUMOZN","MIANTEZN","MITEMPUF","MITEMPMU",
 "EDSABELE","EDSERIE","EDGRAU","EDCURSNS","EDULSERI","EDULGRAU","EDCURSTP",
 "ESTCONJ","TMUNTRAB","TRUL12M","TSITDESO","TOCUPACA","TATIVIDA","TPOSICAO",
 "THORTRAB","RPRINDIN","RPRINPRM","ROUTROCU","IDADEMES","IDADEANO","PESOP"]
DBF80_D = ["UF","MUNIC","CONTADOM","SITUACAO","ESPECIE","TIPO","PAREDES","PISO",
 "AGUA","SANESCOA","SANUSO","CONDOCUP","TPRESID","COMODOS","COMODOR","FOGAO",
 "TELEFONE","ILUMINA","RADIO","GELADEIR","TV","AUTOMOVE","PESOD"]
DBF91 = ["UFNUM","MESONUM","MICRONUM","MUNICNUM","SITSET","AGUA","BANHEIRO",
 "COBERTUR","COMODOR","COMODOS","CONDOCUP","ESPECIE","GELADEIR","ILUMINA","LIXO",
 "PAREDES","PESO","RADIO","RDOMICIV","SANESCOA","SANUSO","TELEFONE","TVCORES",
 "TVPRETO","ATIVIDAD","EDANOEST","EDCURSO","EDGRAU","EDSABELE","EDULGRAU",
 "EDULSERI","HORTRAB","IDADEANO","IDADEMES","MIANMOMU","MIANMOUF","MIANTEMU",
 "MIANTEUF","MIANTEZN","MIMO86MU","MIMO86UF","MIMO86ZN","MINACION","MINASCMU",
 "MIUFPAIS","OCUPACAO","PARENDOM","PESSOAN","POSOCUP","RACACOR","RPRINCIV",
 "RTOTALPV","SCATUAL","SEXO","SITDESO","TRUL12M"]


def num(s: pd.Series, decimals: int = 0) -> pd.Series:
    x = pd.to_numeric(s.str.strip().replace("", pd.NA), errors="coerce")
    return x / (10 ** decimals) if decimals else x


def fixed_chunks(path: Path, fields: dict[str,F], chunksize: int, limit: int|None):
    items=list(fields.items()); nrows=limit
    return pd.read_fwf(path, colspecs=[(f.start-1,f.start-1+f.width) for _,f in items],
                       names=[k for k,_ in items], dtype="string", chunksize=chunksize,
                       nrows=nrows, compression="infer", encoding="latin1")


def dbf_chunks(path: Path, wanted: list[str], chunksize: int, limit: int|None):
    with path.open("rb") as fh:
        head=fh.read(32); nrec=struct.unpack("<I",head[4:8])[0]
        hlen=struct.unpack("<H",head[8:10])[0]; rlen=struct.unpack("<H",head[10:12])[0]
        desc=[]; offset=1
        while True:
            d=fh.read(32)
            if d[0]==13: break
            name=d[:11].split(b"\0")[0].decode("latin1").upper(); width=d[16]
            desc.append((name,offset,width)); offset+=width
        selected=[x for x in desc if x[0] in wanted]
        fh.seek(hlen); remaining=min(nrec,limit) if limit else nrec
        while remaining:
            n=min(chunksize,remaining); raw=fh.read(n*rlen); n=len(raw)//rlen
            if not n: break
            a=np.frombuffer(raw[:n*rlen],dtype=np.uint8).reshape(n,rlen); data={}
            for name,start,width in selected:
                b=np.ascontiguousarray(a[:,start:start+width]).view(f"S{width}").reshape(n)
                data[name]=pd.Series(b.astype(f"U{width}"),dtype="string")
            out=pd.DataFrame(data); keep=a[:,0]!=42
            yield out.loc[keep].reset_index(drop=True); remaining-=n


def source_code(df, col):
    return df[col].astype("string").str.strip().replace("",pd.NA) if col in df else pd.Series(pd.NA,index=df.index,dtype="string")


def common(df: pd.DataFrame, year: int, uf: str, source: Path) -> pd.DataFrame:
    out=pd.DataFrame(index=df.index)
    out["census_year"]=np.int32(year); out["current_uf_code"]=source_code(df,"current_uf")
    out["current_uf_code_vintage"]=uf; out["current_municipality_code"]=source_code(df,"current_muni")
    out["current_meso_code"]=source_code(df,"meso"); out["current_micro_code"]=source_code(df,"micro")
    out["area_weighting_code"]=source_code(df,"area"); out["household_id"]=source_code(df,"household_id")
    out["person_number"]=source_code(df,"person_no"); out["person_weight"]=df.get("person_weight")
    for c in ["urban","sex","age","age_months","race","relationship","marital_status",
              "literacy","school","education_years","education_level","education_course",
              "labor_force","employment_status","worked","occupation","industry",
              "employment_position","hours","income_main","income_all_jobs","income_total",
              "income_total_sm","work_muni","nationality","born_muni","birth_uf","birth_country",
              "years_muni","years_uf","last_origin_uf","last_origin_muni","last_origin_country",
              "last_origin_urban","origin_5yr_uf","origin_5yr_muni","origin_5yr_country"]:
        out[c+"_code" if c not in {"age","age_months","education_years","hours","income_main","income_all_jobs","income_total","income_total_sm","years_muni","years_uf"} else c]=df.get(c)
    for c in ["person_weight","age","age_months","education_years","hours","income_main",
              "income_all_jobs","income_total","income_total_sm","years_muni","years_uf"]:
        out[c]=pd.to_numeric(out[c],errors="coerce")
    out["source_file"]=str(source); out["source_row_in_file"]=df.index+1
    return out


def add_household(out, hh):
    for c in ["household_weight","household_size","dwelling_type","tenure","wall_material",
              "rooms","bedrooms","bathrooms","water","sanitation","electricity","radio",
              "television","refrigerator","washing_machine","telephone","computer","internet",
              "automobile","household_income","household_income_sm","household_income_pc",
              "household_income_pc_sm"]:
        out[c+"_code" if c not in {"household_weight","household_size","rooms","bedrooms",
             "bathrooms","household_income","household_income_sm","household_income_pc",
             "household_income_pc_sm"} else c]=hh.get(c)
    for c in ["household_weight","household_size","rooms","bedrooms","bathrooms",
              "household_income","household_income_sm","household_income_pc","household_income_pc_sm"]:
        out[c]=pd.to_numeric(out[c],errors="coerce")
    return out


def harmonize(out, year):
    out["usual_resident"]=True
    sex=out["sex_code"]
    out["sex"] = sex.map(({"0":"male","1":"female"} if year==1970 else
                           {"1":"male","3":"female"} if year==1980 else {"1":"male","2":"female"}))
    out["age_years"]=pd.to_numeric(out["age"],errors="coerce")
    if year==1970:
        typ=out.pop("age_type_code") if "age_type_code" in out else pd.Series(pd.NA,index=out.index)
        raw=out["age_years"]
        out["age_months"]=raw.where(typ.isin(["1","2"])); out["age_years"]=raw.where(typ.isin(["3","4"]),0)
    out["migrant_5yr"]=pd.Series(pd.NA,index=out.index,dtype="boolean")
    out["internal_migrant_5yr"]=pd.Series(pd.NA,index=out.index,dtype="boolean")
    if year==1991:
        o=out["origin_5yr_uf_code"]
        valid=~o.isin([pd.NA,"","0","99"]); out.loc[valid,"migrant_5yr"]=o[valid]!="70"
        out.loc[valid,"internal_migrant_5yr"]=~o[valid].isin(["70","80"])
    elif year==2000:
        s=out.pop("status_5yr_code")
        out.loc[s.isin(["1","2"]),["migrant_5yr","internal_migrant_5yr"]]=False
        out.loc[s.isin(["3","4"]),["migrant_5yr","internal_migrant_5yr"]]=True
        out.loc[s=="5","migrant_5yr"]=True; out.loc[s=="5","internal_migrant_5yr"]=False
        same=s.isna() & out["age_years"].ge(5)
        out.loc[same,["migrant_5yr","internal_migrant_5yr"]]=False
        out.loc[same,"origin_5yr_uf_code"]=out.loc[same,"current_uf_code"]
        out.loc[same,"origin_5yr_muni_code"]=out.loc[same,"current_municipality_code"]
    elif year==2010:
        typ=out.pop("origin_5yr_type_code"); cur=out["current_uf_code"].fillna("")+out["current_municipality_code"].fillna("")
        ori=out["origin_5yr_muni_code"].fillna(""); valid=typ.notna()
        out.loc[valid,"migrant_5yr"]=(typ[valid]=="2") | (ori[valid]!=cur[valid])
        out.loc[valid,"internal_migrant_5yr"]=(typ[valid]=="1") & (ori[valid]!=cur[valid])
        same=typ.isna() & out["age_years"].ge(5)
        out.loc[same,["migrant_5yr","internal_migrant_5yr"]]=False
        out.loc[same,"origin_5yr_uf_code"]=out.loc[same,"current_uf_code"]
        out.loc[same,"origin_5yr_muni_code"]=out.loc[same,"current_municipality_code"]
    out["five_year_origin_observed"]=out["migrant_5yr"].notna()
    return out


NUMERIC_COLUMNS={"person_weight","age","age_months","education_years","hours","income_main",
                 "income_all_jobs","income_total","income_total_sm","years_muni","years_uf",
                 "household_weight","household_size","rooms","bedrooms","bathrooms",
                 "household_income","household_income_sm","household_income_pc",
                 "household_income_pc_sm","age_years"}
BOOLEAN_COLUMNS={"usual_resident","migrant_5yr","internal_migrant_5yr",
                 "five_year_origin_observed"}


def canonical_schema(columns) -> pa.Schema:
    def dtype(column):
        if column=="census_year": return pa.int32()
        if column=="source_row_in_file": return pa.int64()
        if column in NUMERIC_COLUMNS: return pa.float64()
        if column in BOOLEAN_COLUMNS: return pa.bool_()
        return pa.large_string()
    return pa.schema([pa.field(column,dtype(column)) for column in columns])


def write_chunks(chunks: Iterable[pd.DataFrame], output: Path, year: int, uf: str, stem: str):
    dest=output/f"census_year={year}"/f"current_uf={uf}"; dest.mkdir(parents=True,exist_ok=True)
    target=dest/f"part-{stem}.parquet"; temporary=target.with_suffix(".parquet.tmp")
    temporary.unlink(missing_ok=True); writer=None; rows=0
    try:
        for df in chunks:
            if df.empty: continue
            df["source_row_in_file"]=np.arange(rows+1,rows+len(df)+1,dtype=np.int64)
            # Pandas may infer an integer column in one chunk and float in the next
            # solely because the latter contains missing values.  Pin logical types
            # before Arrow sees each chunk so a state's Parquet schema is stable.
            for column in NUMERIC_COLUMNS & set(df.columns):
                df[column]=pd.to_numeric(df[column],errors="coerce").astype("Float64")
            for column in BOOLEAN_COLUMNS & set(df.columns):
                df[column]=df[column].astype("boolean")
            for column in [c for c in df.columns if c not in NUMERIC_COLUMNS | BOOLEAN_COLUMNS |
                           {"census_year","source_row_in_file"}]:
                df[column]=df[column].astype("string")
            schema=canonical_schema(df.columns)
            table=pa.Table.from_pandas(df,schema=schema,preserve_index=False,safe=False)
            if writer is None: writer=pq.ParquetWriter(temporary,table.schema,compression="zstd")
            writer.write_table(table); rows+=len(df)
    except Exception:
        if writer is not None:
            writer.close()
            writer=None
        temporary.unlink(missing_ok=True)
        raise
    finally:
        if writer is not None: writer.close()
    temporary.replace(target)
    return rows


def normalize_existing(root: Path, batch_size: int):
    files=sorted(root.rglob("*.parquet")); changed=0; rows=0
    (root/"_common_metadata").unlink(missing_ok=True)
    for path in files:
        parquet=pq.ParquetFile(path); schema=canonical_schema(parquet.schema_arrow.names)
        file_rows=parquet.metadata.num_rows; rows+=file_rows
        if parquet.schema_arrow.remove_metadata().equals(schema):
            parquet.close(); print(f"schema: skipped {path}"); continue
        temporary=path.with_suffix(".parquet.normalize.tmp"); temporary.unlink(missing_ok=True)
        writer=pq.ParquetWriter(temporary,schema,compression="zstd")
        try:
            for batch in parquet.iter_batches(batch_size=batch_size):
                writer.write_table(pa.Table.from_batches([batch]).cast(schema,safe=False))
        except Exception:
            writer.close(); parquet.close(); temporary.unlink(missing_ok=True); raise
        else:
            writer.close(); parquet.close(); temporary.replace(path); changed+=1
            print(f"schema: normalized {path} ({file_rows:,} rows)")
    print(f"schema: {changed:,}/{len(files):,} files normalized; {rows:,} rows checked")


def build_1970(path,args,uf):
    for raw in fixed_chunks(path,P70,args.chunksize,args.sample_rows):
        d=pd.DataFrame(index=raw.index)
        d["current_uf"]=str(UF_ABBR[uf]); d["current_muni"]=raw.V002; d["micro"]=raw.V001
        d["urban"]=raw.V004; d["person_weight"]=num(raw.V054); d["sex"]=raw.V023
        d["age"]=num(raw.V027); d["age_type"]=raw.V026; d["relationship"]=raw.V025
        d["nationality"]=raw.V029; d["birth_uf"]=raw.V030; d["years_uf"]=raw.V031
        d["years_muni"]=raw.V032; d["last_origin_uf"]=raw.V033; d["last_origin_urban"]=raw.V034
        d["literacy"]=raw.V035; d["school"]=raw.V036; d["education_level"]=raw.V038
        d["education_course"]=raw.V039; d["marital_status"]=raw.V040; d["income_total"]=num(raw.V041)
        d["occupation"]=raw.V044; d["industry"]=raw.V045; d["employment_position"]=raw.V046
        d["worked"]=raw.V047; d["work_muni"]=raw.V042
        for c,v in {"household_size":"V005","dwelling_type":"V008","tenure":"V009","rooms":"V020",
                    "bedrooms":"V021","water":"V012","sanitation":"V013","electricity":"V014",
                    "radio":"V016","refrigerator":"V017","television":"V018","automobile":"V019"}.items(): d[c]=raw[v]
        out=common(d,1970,uf,path); out["age_type_code"]=source_code(d,"age_type")
        out=add_household(out,d); out["usual_resident"]=raw.V024.isin(["0","1"]); out=out[out.usual_resident]
        yield harmonize(out,1970)


def dbf_transform(path,args,year,uf):
    wanted=DBF80_P if year==1980 else DBF91
    hh80=None
    if year==1980:
        dom=next(path.parents[1].rglob(f"CD80DOM{uf}.DBF"),None)
        if dom:
            hh80=pd.concat(dbf_chunks(dom,DBF80_D,args.chunksize,None),ignore_index=True)
            hh80=hh80.drop_duplicates(["UF","MUNIC","CONTADOM"])
    for r in dbf_chunks(path,wanted,args.chunksize,args.sample_rows):
        d=pd.DataFrame(index=r.index)
        if year==1980:
            ren={"UF":"current_uf","MUNIC":"current_muni","NDOM":"household_id","SEXO":"sex",
                 "PARENDOM":"relationship","COR":"race","MINACION":"nationality","MIUFNASC":"birth_uf",
                 "MINASCMU":"born_muni","MIANTEZN":"last_origin_urban","MITEMPUF":"years_uf",
                 "MITEMPMU":"years_muni","EDSABELE":"literacy","EDCURSTP":"education_course",
                 "ESTCONJ":"marital_status","TMUNTRAB":"work_muni","TRUL12M":"worked",
                 "TSITDESO":"employment_status","TOCUPACA":"occupation","TATIVIDA":"industry",
                 "TPOSICAO":"employment_position","THORTRAB":"hours","RPRINDIN":"income_main",
                 "IDADEMES":"age_months","IDADEANO":"age","PESOP":"person_weight"}
        else:
            ren={"UFNUM":"current_uf","MESONUM":"meso","MICRONUM":"micro","MUNICNUM":"current_muni",
                 "SITSET":"urban","PESO":"person_weight","SEXO":"sex","IDADEANO":"age",
                 "IDADEMES":"age_months","RACACOR":"race","PARENDOM":"relationship","PESSOAN":"person_no",
                 "EDSABELE":"literacy","EDANOEST":"education_years","EDCURSO":"education_course",
                 "SCATUAL":"marital_status","TRUL12M":"worked","SITDESO":"employment_status",
                 "OCUPACAO":"occupation","ATIVIDAD":"industry","POSOCUP":"employment_position",
                 "HORTRAB":"hours","RPRINCIV":"income_main","RTOTALPV":"income_total",
                 "MINACION":"nationality","MINASCMU":"born_muni","MIUFPAIS":"birth_uf",
                 "MIANMOMU":"years_muni","MIANMOUF":"years_uf","MIANTEUF":"last_origin_uf",
                 "MIANTEMU":"last_origin_muni","MIANTEZN":"last_origin_urban",
                 "MIMO86UF":"origin_5yr_uf","MIMO86MU":"origin_5yr_muni"}
        for src,dst in ren.items(): d[dst]=source_code(r,src)
        for c in ["person_weight","age","age_months","education_years","hours","income_main","income_total","years_muni","years_uf"]:
            if c in d: d[c]=num(d[c])
        out=common(d,year,uf,path)
        if year==1980 and hh80 is not None:
            j=r[["UF","MUNIC","NDOM"]].merge(hh80,left_on=["UF","MUNIC","NDOM"],
                right_on=["UF","MUNIC","CONTADOM"],how="left",validate="many_to_one")
            hhmap={"household_weight":"PESOD","household_size":"TPRESID","dwelling_type":"TIPO",
                   "tenure":"CONDOCUP","rooms":"COMODOS","bedrooms":"COMODOR","water":"AGUA",
                   "sanitation":"SANESCOA","electricity":"ILUMINA","radio":"RADIO",
                   "refrigerator":"GELADEIR","television":"TV","telephone":"TELEFONE",
                   "automobile":"AUTOMOVE"}
            h=pd.DataFrame(index=j.index)
            for dst,src in hhmap.items(): h[dst]=source_code(j,src)
            h["household_weight"]=num(h["household_weight"])
            out=add_household(out,h)
        elif year==1991:
            hhmap={"household_income":"RDOMICIV","rooms":"COMODOS","bedrooms":"COMODOR","water":"AGUA",
                   "sanitation":"SANESCOA","electricity":"ILUMINA","radio":"RADIO","refrigerator":"GELADEIR",
                   "telephone":"TELEFONE","dwelling_type":"ESPECIE","tenure":"CONDOCUP"}
            hh=pd.DataFrame(index=r.index)
            for dst,src in hhmap.items(): hh[dst]=source_code(r,src)
            out=add_household(out,hh)
        yield harmonize(out,year)


def fixed_modern(path,args,year,uf,dompath=None):
    pf=P00 if year==2000 else P10; df=D00 if year==2000 else D10
    hh=None; keys=["uf","muni","control"] if year==2000 else ["uf","muni","area","control"]
    if dompath:
        parts=[]
        for x in fixed_chunks(dompath,df,args.chunksize,None):
            for c,f in df.items(): x[c]=num(x[c],f.decimals) if f.decimals else source_code(x,c)
            parts.append(x)
        hh=pd.concat(parts,ignore_index=True).drop_duplicates(keys)
    for r in fixed_chunks(path,pf,args.chunksize,args.sample_rows):
        for c,f in pf.items(): r[c]=num(r[c],f.decimals) if f.decimals else source_code(r,c)
        d=r.copy(); d["current_uf"]=d.uf; d["current_muni"]=(d.muni if year==2000 else d.uf+d.muni)
        d["household_id"]=d.control
        out=common(d,year,uf,path); out["status_5yr_code" if year==2000 else "origin_5yr_type_code"]=d["status_5yr" if year==2000 else "origin_5yr_type"]
        if hh is not None:
            joined=d[keys].merge(hh,on=keys,how="left",validate="many_to_one",suffixes=("","_hh"))
            h=pd.DataFrame(index=joined.index)
            for c in df:
                if c not in keys: h[c]=joined[c]
            out=add_household(out,h)
        yield harmonize(out,year)


def find_files(root,year,ufs):
    base={1970:"Microdados_Censo_Demografico_1970_Amostra",1980:"Microdados_Censo_Demografico_1980_Amostra",
          1991:"Microdados_Censo_Demografico_1991_Amostra",2000:"Censo_Microdados_2000",2010:"Censo_Microdados_2010"}[year]
    b=root/base
    if year==1970: files=list((b/"Dados").glob("*.txt"))+list((b/"Dados").glob("*.TXT"))
    elif year==1980: files=list(b.glob("Dados/**/CD80PES*.DBF"))
    elif year==1991: files=list(b.glob("Dados/**/CD91AMOUP*.DBF"))
    elif year==2000: files=[p for p in b.glob("*/*/PES*.*") if p.suffix.lower() in {".txt",".zip"}]
    else: files=list(b.glob("*/*/Amostra_Pessoas_*.txt"))
    seen=set()
    for p in sorted(files):
        m=re.search(r"(\d{2})(?:\D|$)",p.stem); uf=(m.group(1) if m else p.stem[-2:]).upper()
        if year==1970: uf=p.stem[-2:].upper()
        elif year in (1980,1991): uf=p.stem[-2:]
        if ufs and uf not in ufs: continue
        sig=(p.stat().st_size,hashlib.sha256(p.read_bytes()[:1024*1024]).hexdigest())
        if sig in seen: continue
        seen.add(sig); yield p,uf


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--input",type=Path,default=Path("data/censo_microdados"))
    ap.add_argument("--output",type=Path,default=Path("data/processed/censo_microdados/persons"))
    ap.add_argument("--years",nargs="+",type=int,default=[1970,1980,1991,2000,2010])
    ap.add_argument("--ufs",nargs="+",help="UF codes, or abbreviations for 1970")
    ap.add_argument("--chunksize",type=int,default=100_000); ap.add_argument("--sample-rows",type=int)
    ap.add_argument("--resume",action="store_true",help="Skip complete existing state files")
    ap.add_argument("--normalize-existing",action="store_true",
                    help="Rewrite existing Parquet partitions to the canonical cross-census schema")
    args=ap.parse_args(); total=0
    if args.normalize_existing:
        normalize_existing(args.output,args.chunksize); return
    for year in args.years:
        for path,uf in find_files(args.input,year,set(args.ufs or [])):
            target=args.output/f"census_year={year}"/f"current_uf={uf}"/f"part-{path.stem.lower()}.parquet"
            if args.resume and target.exists():
                try:
                    n=pq.ParquetFile(target).metadata.num_rows; total+=n
                    print(f"{year} {uf}: skipped {n:,} existing rows from {path.name}"); continue
                except Exception:
                    target.unlink()
            if year==1970: chunks=build_1970(path,args,uf)
            elif year in (1980,1991): chunks=dbf_transform(path,args,year,uf)
            else:
                if year==2000:
                    dom=next((x for x in path.parent.glob("DOM*.*") if x.suffix.lower() in {".txt",".zip"}),None)
                else: dom=next(path.parent.glob("Amostra_Domicilios_*.txt"),None)
                chunks=fixed_modern(path,args,year,uf,dom)
            n=write_chunks(chunks,args.output,year,uf,path.stem.lower()); total+=n
            print(f"{year} {uf}: {n:,} rows from {path.name}")
    print(f"total: {total:,} rows")


if __name__=="__main__": main()
