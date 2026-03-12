from pathlib import Path

labels_dir = Path("dataset_coins/train/labels")
bad = []

for p in list(labels_dir.rglob("*.txt")):
    text = p.read_text().strip()
    if not text:
        continue
    for i, line in enumerate(text.splitlines(), 1):
        parts = line.split()
        if len(parts) != 5:
            bad.append((str(p), i, f"wrong columns: {line}"))
            continue
        try:
            cls = int(float(parts[0]))
            x, y, w, h = map(float, parts[1:])
        except Exception:
            bad.append((str(p), i, f"parse error: {line}"))
            continue

        if not (0 <= cls <= 16):
            bad.append((str(p), i, f"class out of range: {cls}"))
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
            bad.append((str(p), i, f"bbox out of range: {x} {y} {w} {h}"))

print(f"bad rows: {len(bad)}")
for row in bad[:100]:
    print(row)