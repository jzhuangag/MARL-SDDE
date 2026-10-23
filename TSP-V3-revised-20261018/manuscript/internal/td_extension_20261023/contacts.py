from pathlib import Path
from PIL import Image,ImageDraw
root=Path(__file__).resolve().parent
for stem in ('main','supp'):
    pages=sorted(root.glob(stem+'-*.png'))
    for start in range(0,len(pages),4):
        group=pages[start:start+4]
        ims=[Image.open(p).convert('RGB') for p in group]
        w,h=ims[0].size
        sheet=Image.new('RGB',(2*w,2*(h+22)),'#dddddd')
        draw=ImageDraw.Draw(sheet)
        for i,(p,im) in enumerate(zip(group,ims)):
            x=(i%2)*w;y=(i//2)*(h+22)
            draw.text((x+3,y+3),p.name,fill='black');sheet.paste(im,(x,y+22))
        sheet.save(root/f'sheet_{stem}_{start+1:02}.png')
