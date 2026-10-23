"""List the constant tensors of each TFLite section in a .litertlm, with dtype, shape and bytes.

Usage: python inspect_litert_tensors.py <model.litertlm> <work_dir> [name-filter]
"""
import collections, pathlib, sys
from litert_torch.generative.export_hf.experimental.litertlm_bundle import litertlm_bundle as bundle
from ai_edge_quantizer.utils import tfl_flatbuffer_utils as fb

path, work = sys.argv[1], pathlib.Path(sys.argv[2]); flt = sys.argv[3] if len(sys.argv) > 3 else ""
work.mkdir(parents=True, exist_ok=True)
unpacked = bundle.unpack_litertlm(path, str(work))
for section, file in (unpacked.items() if isinstance(unpacked, dict) else []):
    if not str(file).endswith(".tflite"):
        continue
    model = fb.read_model(str(file))
    totals = collections.Counter()
    print(f"== {section}")
    for sg in model.subgraphs:
        for t in sg.tensors:
            buf = model.buffers[t.buffer]
            n = len(buf.data) if buf.data is not None else (buf.size or 0)
            if n < 1_000_000:
                continue
            name = t.name.decode() if isinstance(t.name, bytes) else t.name
            q = t.quantization
            scales = None if q is None or q.scale is None else len(q.scale)
            totals[t.type] += n
            if flt in name:
                print(f"   {name[:90]:90} type={t.type} shape={list(t.shape)} MB={n/1e6:.1f} scales={scales}")
    print("   totals by type (MB):", {k: round(v / 1e6) for k, v in totals.items()})
