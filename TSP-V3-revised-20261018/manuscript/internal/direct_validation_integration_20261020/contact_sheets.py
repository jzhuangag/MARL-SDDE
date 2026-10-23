from PIL import Image, ImageDraw
from pathlib import Path

HERE=Path(__file__).resolve().parent
for name,count in (("main",13),("supplement",10)):
    paths=sorted(HERE.glob(name+"-*.png"))
    assert len(paths)==count
    for start in range(0,count,4):
        sheet=Image.new("RGB",(1700,2260),"#cccccc")
        draw=ImageDraw.Draw(sheet)
        for j,path in enumerate(paths[start:start+4]):
            img=Image.open(path).convert("RGB")
            img.thumbnail((840,1090))
            x=(j%2)*850;y=(j//2)*1130
            sheet.paste(img,(x,y+30));draw.text((x+12,y+8),path.name,fill="black")
        sheet.save(HERE/(name+"_sheet_"+str(start//4+1)+".png"))
