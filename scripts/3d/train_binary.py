"""Final common-HU binary augmented supervised experiment."""
from pathlib import Path
import sys, argparse
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "3d"))
import torch
from final_common_hu.config import FINAL_RUN_ROOT, SEED
from final_common_hu.datasets import make_binary_loader
from final_common_hu.model import CommonHUResNet3D18
from final_common_hu.supervised import SupervisedConfig, fit_supervised, seed_all

def main():
    p=argparse.ArgumentParser(); p.add_argument("--device",default="mps"); p.add_argument("--resume",action="store_true"); a=p.parse_args(); seed_all(SEED); device=torch.device("mps" if a.device=="mps" and torch.backends.mps.is_available() else "cpu")
    train=make_binary_loader("train",batch_size=4,augment=True); val=make_binary_loader("validation",batch_size=4,augment=False); out=FINAL_RUN_ROOT/"binary_augmented_resnet18"; fit_supervised(CommonHUResNet3D18(2,0.20),train,val,names=("benign","malignant"),config=SupervisedConfig(output_dir=out),device=device,resume=a.resume)
if __name__=="__main__": main()
