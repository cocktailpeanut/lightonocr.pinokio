"""Generate non-sensitive image and PDF OCR fixtures without external downloads."""
from pathlib import Path
import os
from PIL import Image, ImageDraw, ImageFont
ROOT = Path(__file__).resolve().parent
candidates = [os.environ.get('OCR_TEST_FONT', ''), '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
              '/System/Library/Fonts/Supplemental/Arial.ttf', 'C:/Windows/Fonts/arial.ttf']
font = next((ImageFont.truetype(p, 32) for p in candidates if p and Path(p).exists()), ImageFont.load_default(size=32))
image = Image.new('RGB', (1000, 420), 'white')
draw = ImageDraw.Draw(image)
for y, line in zip([65, 125, 185, 245], ['LIGHTON OCR TEST', 'Invoice number 12345', 'Apples 3.00', 'Total 3.00']):
    draw.text((60, y), line, fill='black', font=font)
image.save(ROOT / 'fixture.png')
image.save(ROOT / 'fixture.pdf', 'PDF', resolution=150)
print(ROOT / 'fixture.png')
print(ROOT / 'fixture.pdf')
