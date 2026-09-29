from dataclasses import dataclass
from pathlib import Path
import pandas as pd

@dataclass
class Schema:
    patient_id:str|None=None; lesion_id:str|None=None; image_path:str|None=None; target:str|None=None; visit_id:str|None=None; date:str|None=None
    @classmethod
    def discover(cls, df, overrides=None):
        overrides=overrides or {}; names={c.lower().replace(' ','_'):c for c in df.columns}
        def find(keys):
            for k in keys:
                if k in overrides and overrides[k] in df: return overrides[k]
            for c in df.columns:
                s=c.lower().replace(' ','_')
                if any(k in s for k in keys): return c
            return None
        return cls(find(['patient_id','patient','subject']),find(['lesion_id','lesion','case']),find(['image_path','image','filepath','file']),find(['target','label','diagnosis','class']),find(['visit_id','visit']),find(['date','timestamp','time']))

def load_tables(root):
    root=Path(root)
    if not root.exists(): raise FileNotFoundError('Dataset path does not exist. Please update dataset.root in config.yaml.')
    tables=[]
    for p in root.rglob('*'):
        if p.suffix.lower() in {'.csv','.tsv','.parquet','.json'}:
            try: tables.append((p, pd.read_csv(p,sep='\t' if p.suffix.lower()=='.tsv' else ',') if p.suffix.lower()!='.parquet' else pd.read_parquet(p)))
            except Exception: continue
    return tables
