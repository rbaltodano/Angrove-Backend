"""Unpack Google's Gemma 4 E4B mobile QAT checkpoint into a plain bf16 checkpoint.

google/gemma-4-E4B-it-qat-mobile-transformers stores the phone layout as packed integers with
per-channel scales (2-bit embeddings, 4-bit decoder, 8-bit per-layer parts). transformers'
gemma_quant modules dequantize them; this copies those exact values into the ordinary
architecture (taken from the q4_0-unquantized checkpoint), so the weights sit on the mobile QAT
grid and can be fine-tuned and re-exported with the matching LiteRT recipe.

Usage: python dequantize_mobile_qat.py --mobile /root/models/qat-mobile \
           --plain /root/models/qat-unq --output /root/models/qat-mobile-bf16
"""
import argparse, pathlib, shutil, torch
from transformers import AutoModelForImageTextToText
from transformers.integrations.gemma_quant import QuantizedEmbedding, QuantizedLinear

ap = argparse.ArgumentParser()
ap.add_argument("--mobile", required=True); ap.add_argument("--plain", required=True)
ap.add_argument("--output", required=True)
a = ap.parse_args()

mobile = AutoModelForImageTextToText.from_pretrained(a.mobile, dtype=torch.bfloat16)
plain = AutoModelForImageTextToText.from_pretrained(a.plain, dtype=torch.bfloat16)
plain_modules = dict(plain.named_modules())
plain_params = dict(plain.named_parameters())

copied, quantized = 0, 0
with torch.no_grad():
    for name, module in mobile.named_modules():
        if isinstance(module, (QuantizedLinear, QuantizedEmbedding)):
            weight = (module._dequantize_weights(torch.bfloat16) if isinstance(module, QuantizedLinear)
                      else module.weight.to(torch.bfloat16))
            target = plain_modules.get(name) or plain_modules.get(name.removesuffix(".linear"))
            if target is None:
                target = plain_modules[name.rsplit(".", 1)[0]]
            assert target.weight.shape == weight.shape, (name, target.weight.shape, weight.shape)
            target.weight.copy_(weight); quantized += 1
            if isinstance(module, QuantizedLinear) and module.bias is not None:
                target.bias.copy_(module.bias)
    # Unquantized parameters (norms, scalars, small projections) come from the mobile checkpoint too.
    for name, p in mobile.named_parameters():
        if p.dtype in (torch.int8, torch.uint8) or name.endswith("_scale"):
            continue
        key = name if name in plain_params else name.replace(".linear.", ".")
        if key in plain_params and plain_params[key].shape == p.shape:
            plain_params[key].copy_(p.to(torch.bfloat16)); copied += 1

print(f"dequantized {quantized} quantized modules, copied {copied} plain parameters")
out = pathlib.Path(a.output)
plain.save_pretrained(out, safe_serialization=True)
for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "processor_config.json",
          "generation_config.json"):
    if (pathlib.Path(a.plain) / f).exists(): shutil.copy(pathlib.Path(a.plain) / f, out / f)
print("saved", out)
