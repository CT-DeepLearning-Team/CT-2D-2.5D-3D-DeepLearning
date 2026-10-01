"""Final common-HU MoCo v2 3D pretraining."""
from pathlib import Path
import sys, argparse
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "3d"))
import torch
from final_common_hu.config import FINAL_RUN_ROOT, SEED
from final_common_hu.datasets import make_ssl_loader
from final_common_hu.moco import MoCo3D, MoCoConfig, fit_moco
from final_common_hu.supervised import seed_all

def main():
    p=argparse.ArgumentParser(); p.add_argument("--device",default="mps"); p.add_argument("--resume",action="store_true"); a=p.parse_args(); seed_all(SEED); device=torch.device("mps" if a.device=="mps" and torch.backends.mps.is_available() else "cpu")
    loader=make_ssl_loader(batch_size=4); out=FINAL_RUN_ROOT/"moco_pretrain_3d_resnet18"; fit_moco(MoCo3D(4096,0.999,0.07),loader,config=MoCoConfig(output_dir=out),device=device,resume=a.resume)
if __name__=="__main__": main()
