import json
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

W, H = 1200, 630
GREEN = (23, 122, 71)
INK = (22, 24, 29)
MUTED = (91, 96, 107)
LINE = (232, 230, 225)
BG = (252, 252, 250)

FONT_PATH = "/System/Library/Fonts/SFNS.ttf"
MONO_PATH = "/System/Library/Fonts/SFNSMono.ttf"


def font(size, mono=False):
    try:
        return ImageFont.truetype(MONO_PATH if mono else FONT_PATH, size)
    except OSError:
        return ImageFont.load_default()


def eur(v):
    if v is None:
        return "-"
    return f"{int(v):,}".replace(",", ".") + " €"


def build(data, out_path):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    d.rectangle([0, 0, W, 12], fill=GREEN)

    name = f"{data['brand']} {data['model']}"
    ap = data.get("asking_price") or {}
    dts = (data.get("days_to_sell") or {}).get("median_days")
    dist = {x.get("key"): x for x in (data.get("by_district") or [])}
    lis = (dist.get("lisboa") or {}).get("asking_price", {}).get("median")
    por = (dist.get("porto") or {}).get("asking_price", {}).get("median")

    d.text((70, 70), "PREÇO PEDIDO NO OLX PORTUGAL", font=font(26), fill=MUTED)
    d.text((70, 112), name, font=font(74), fill=INK)

    d.text((70, 218), eur(ap.get("median")), font=font(132), fill=GREEN)
    d.text((70, 372), f"metade dos anúncios entre {eur(ap.get('p25'))} e {eur(ap.get('p75'))}",
           font=font(31), fill=MUTED)

    d.line([(70, 445), (W - 70, 445)], fill=LINE, width=2)

    col = 70
    cells = []
    if dts:
        cells.append(("DIAS ATÉ SAIR", f"{dts} dias"))
    if lis:
        cells.append(("LISBOA", eur(lis)))
    if por:
        cells.append(("PORTO", eur(por)))
    if data.get("sample_size"):
        cells.append(("ANÚNCIOS", str(data["sample_size"])))

    step = (W - 140) // max(len(cells), 1)
    for label, value in cells:
        d.text((col, 470), label, font=font(23), fill=MUTED)
        d.text((col, 502), value, font=font(48), fill=INK)
        col += step

    d.line([(70, 572), (W - 70, 572)], fill=LINE, width=1)
    d.text((70, 586), "carsbuyer.org", font=font(27), fill=GREEN)
    d.text((W - 70, 588), f"dados do OLX, {data.get('collected_until', '')}",
           font=font(23), fill=MUTED, anchor="ra")

    img.save(out_path, "PNG")
    return out_path


def main():
    src = Path(sys.argv[1])
    out = Path(sys.argv[2])
    data = json.loads(src.read_text())
    print(build(data, out))


if __name__ == "__main__":
    raise SystemExit(main())
