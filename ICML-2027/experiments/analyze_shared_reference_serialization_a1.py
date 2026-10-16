"""Serialization-only operational amendment; frozen formulas remain unchanged.

The original exact-enumeration gate yields numpy.bool_, which Python's JSON
encoder rejects. Convert NumPy scalar types to their exact Python equivalents.
No payload, seed, sample, statistic, comparison or gate is rerun or modified.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import analyze_shared_reference_qualification as analyzer


def scalar(x):
    if isinstance(x,np.generic):return x.item()
    raise TypeError("unsupported JSON type: "+str(type(x)))


def dump(path,obj):
    with Path(path).open("x",encoding="utf-8",newline="\n") as f:
        json.dump(obj,f,indent=2,sort_keys=True,allow_nan=False,default=scalar)
        f.write("\n")


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--root",required=True);ap.add_argument("--original",required=True)
    ap.add_argument("--output",required=True);ap.add_argument("--replay")
    args=ap.parse_args();analyzer.dump=dump
    print(json.dumps(analyzer.analyze(args.root,args.original,args.output,args.replay),indent=2,default=scalar))
