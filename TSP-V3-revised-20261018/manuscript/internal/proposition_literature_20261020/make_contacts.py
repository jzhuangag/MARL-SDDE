"""Pair final PDF render pages for visual QA; never modifies manuscript images."""
from pathlib import Path
from PIL import Image,ImageDraw
import hashlib
import json
HERE=Path(__file__).resolve().parent
(HERE/'contacts').mkdir(exist_ok=True)
records=[]
for stem in ('main','supp'):
    pages=sorted((HERE/'render').glob(stem+'-*.png'))
    for first in range(0,len(pages),2):
        selected=pages[first:first+2]
        ims=[Image.open(p).convert('RGB') for p in selected]
        canvas=Image.new('RGB',(sum(i.width for i in ims),max(i.height for i in ims)+35),'#e0e0e0')
        draw=ImageDraw.Draw(canvas);x=0
        for p,im in zip(selected,ims):
            draw.text((x+10,5),p.name,fill='black');canvas.paste(im,(x,35));x+=im.width
        output=HERE/'contacts'/f'{stem}_{first+1:02d}.png';canvas.save(output)
        records.append(dict(file=str(output.relative_to(HERE)),pages=[p.name for p in selected],
            sha256=hashlib.sha256(output.read_bytes()).hexdigest()))
with (HERE/'contact_inventory.json').open('x',encoding='utf-8') as f:json.dump(records,f,indent=2)
print('Contact sheets:',len(records))
