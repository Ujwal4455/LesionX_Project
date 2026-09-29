import torch
from torch.utils.data import Dataset
from PIL import Image
class LesionDataset(Dataset):
    def __init__(self,frame,image_col,target_col=None,transform=None): self.f=frame.reset_index(drop=True); self.ic=image_col; self.tc=target_col; self.t=transform
    def __len__(self): return len(self.f)
    def __getitem__(self,i):
        r=self.f.iloc[i]; x=Image.open(r[self.ic]).convert('RGB'); x=torch.from_numpy(__import__('numpy').asarray(x.resize((384,384)),dtype='float32')).permute(2,0,1)/255
        y=r[self.tc] if self.tc else -1; return {'image':x,'target':torch.tensor(y,dtype=torch.long),'index':i}
