"""Final common-HU Experiment A: supervised 3-class ResNet-18."""
from pathlib import Path
import sys, argparse
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "3d"))
import torch
from final_common_hu.config import FINAL_RUN_ROOT, SEED
from final_common_hu.datasets import make_supervised_loader
from final_common_hu.model import CommonHUResNet3D18
from final_common_hu.supervised import SupervisedConfig, fit_supervised, seed_all

def main():
    p=argparse.ArgumentParser(); p.add_argument("--device",default="mps"); p.add_argument("--resume",action="store_true"); a=p.parse_args(); seed_all(SEED); device=torch.device("mps" if a.device=="mps" and torch.backends.mps.is_available() else "cpu")
    train=make_supervised_loader("train",batch_size=4,augment=False); val=make_supervised_loader("validation",batch_size=4,augment=False); out=FINAL_RUN_ROOT/"3class_A_supervised_resnet18"; cfg=SupervisedConfig(output_dir=out)
    fit_supervised(CommonHUResNet3D18(3,0.20),train,val,names=("benign","indeterminate","malignant"),config=cfg,device=device,resume=a.resume)
if __name__=="__main__": main()
