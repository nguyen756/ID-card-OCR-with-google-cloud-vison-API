#layout_utils.py
import numpy as np

def easyocr_pretty(results):
    if not results:
        return ""
    words = []
    for bbox, text, conf in results:
        if not str(text).strip():
            continue
        xs = [p[0] for p in bbox]; ys = [p[1] for p in bbox]
        x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
        words.append({"t": str(text), "x": x1, "y": (y1+y2)/2.0, "h": (y2-y1) or 1.0})
    words.sort(key=lambda w: (w["y"], w["x"]))
    med = float(np.median([w["h"] for w in words])) if words else 1.0
    tol = med * 0.6
    lines, cur, last_y = [], [], None
    for w in words:
        if last_y is None or abs(w["y"] - last_y) <= tol:
            cur.append(w)
        else:
            lines.append(cur); cur = [w]
        last_y = w["y"]
    if cur:
        lines.append(cur)
    out = []
    for line in lines:
        line.sort(key=lambda w: w["x"])  # left to right
        out.append(" ".join(w["t"] for w in line))
    return "\n".join(out)