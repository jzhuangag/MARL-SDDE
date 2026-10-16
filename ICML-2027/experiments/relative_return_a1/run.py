"""A1 entry point; preserve and reuse the frozen original runner verbatim."""
import argparse
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import relative_return_estimator as estimator
import run_relative_return_qualification as runner
from stable_overlap import stable_overlap

estimator.overlap=stable_overlap
runner.PLAN=HERE/"relative_return_qualification.json"

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--output",required=True)
    ap.add_argument("--lock",required=True)
    ap.add_argument("--context",type=int)
    ap.add_argument("--qualification",action="store_true")
    runner.run(ap.parse_args())
