import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fugue.tx import tokens as T

def test_round_trip_wtc():
    f = json.load(open(os.path.join(os.path.dirname(__file__), "..", "data", "tx_pieces.json")))["pieces"][0]
    for shift in (0, 5, -4):
        ids = T.encode(f, shift); d = T.decode(ids)
        lead = (round(f["meters"][0][1] * 4 / f["meters"][0][2] * T.TPB) - f["bar_offset"]) % round(f["meters"][0][1] * 4 / f["meters"][0][2] * T.TPB)
        want = sorted((round(o * T.TPB), m + shift) for v in f["voices"] for o, _, m in v)
        got = sorted((round(n["on"] * T.TPB) - lead, n["m"]) for n in d["notes"])
        assert want == got
        assert len(d["entries"]) == len(f["entries"])
        assert d["header"]["KEY"] == f"{(f['key'][0] + shift) % 12}_{f['key'][1]}"
