from collections import defaultdict

def build_mappings(df,s):
    out={k:defaultdict(list) for k in ['patient_to_lesions','lesion_to_patient','patient_to_images','lesion_to_images','patient_to_visits']}
    for _,r in df.iterrows():
        p=str(r[s.patient_id]) if s.patient_id else None; l=str(r[s.lesion_id]) if s.lesion_id else None; im=str(r[s.image_path]) if s.image_path else None; v=str(r[s.visit_id]) if s.visit_id else None
        if p and l: out['patient_to_lesions'][p].append(l); out['lesion_to_patient'][l].append(p)
        if p and im: out['patient_to_images'][p].append(im)
        if l and im: out['lesion_to_images'][l].append(im)
        if p and v: out['patient_to_visits'][p].append(v)
    return {k:{a:sorted(set(b)) for a,b in v.items()} for k,v in out.items()}
