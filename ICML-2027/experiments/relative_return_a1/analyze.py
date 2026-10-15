"""Apply exactly the original analyzer and gates to the A1 registered plan."""
import argparse
import json
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import analyze_relative_return_qualification as analyzer

# The frozen analyzer resolves its plan relative to its declared module path.
# No implementation, result, gate, or decision formula is modified.
analyzer.__file__=str(HERE/"analyze.py")
if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",required=True)
    ap.add_argument("--output",required=True)
    ap.add_argument("--replay")
    args=ap.parse_args()
    print(json.dumps(analyzer.analyze(args.root,args.output,args.replay),indent=2))
