"""Export the fused Aquinas Gemma 4 checkpoint to a LiteRT-LM package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from litert_torch.generative.export_hf import export
from litert_torch.generative.export_hf.core.exportable_module_config import (
    ExportTask,
)


MINIMUM_FREE_BYTES = 40 * 1024**3
EXPECTED_ARCHITECTURE = "Gemma4ForConditionalGeneration"
SUPPORTED_QUANTIZATION_RECIPES = (
    "aquinas_mixed48_b32",
    "aquinas_mixed48_hr",
    "aquinas_mixed48_c",
    "aquinas_mixed48_b64",
    "dynamic_wi4_afp32",
    "dynamic_wi8_emb4_afp32",
    "dynamic_wi8_afp32",
)


def register_mixed_recipes() -> None:
    """Flatten ai_edge_quantizer's per-component gemma4_mixed48 recipes into single recipes.

    The exporter applies one recipe string to every component, so it can't take the
    per-component dicts. Each flat recipe quantizes embeddings and fully connected layers to
    4 bits and keeps the per-layer-embedding projections at 8 bits, which Google's recipe
    notes they need. Without that, the plain INT4 export scored 24/40 on sealed-1 versus
    38/40 for Google's package.
    """
    from ai_edge_quantizer import recipe as recipe_lib
    from litert_torch.generative.export_hf.core import export_lib

    names = recipe_lib.TFLOperationName

    def mixed(four_bit):
        return (
            four_bit(operation_name=names.EMBEDDING_LOOKUP)
            + four_bit(operation_name=names.FULLY_CONNECTED)
            + recipe_lib.dynamic_wi8c_afp32(regex="per_layer", operation_name=names.FULLY_CONNECTED)
        )

    export_lib._LOCAL_QUANTIZATION_RECIPES.update({
        "aquinas_mixed48_b32": lambda: mixed(recipe_lib.dynamic_wi4b32_afp32),
        "aquinas_mixed48_hr": lambda: mixed(recipe_lib.dynamic_wi4c_hr_afp32),
        "aquinas_mixed48_c": lambda: mixed(recipe_lib.dynamic_wi4c_afp32),
        "aquinas_mixed48_b64": lambda: mixed(recipe_lib.dynamic_wi4b64_afp32),
    })


def validate_source(source: Path, output: Path) -> None:
    required_files = (
        source / "config.json",
        source / "tokenizer.json",
        source / "tokenizer_config.json",
        source / "chat_template.jinja",
    )
    missing = [path for path in required_files if not path.is_file()]
    if missing:
        formatted = "\n".join(f"- {path}" for path in missing)
        raise SystemExit(f"Source checkpoint is incomplete:\n{formatted}")
    if not (
        (source / "model.safetensors").is_file()
        or (source / "model.safetensors.index.json").is_file()
    ):
        raise SystemExit(
            "Source checkpoint has neither model.safetensors nor a sharded "
            "model index."
        )

    config = json.loads((source / "config.json").read_text())
    architectures = config.get("architectures", [])
    if EXPECTED_ARCHITECTURE not in architectures:
        raise SystemExit(
            f"Expected {EXPECTED_ARCHITECTURE}; found {architectures!r}."
        )
    if config.get("model_type") != "gemma4":
        raise SystemExit(
            f"Expected model_type 'gemma4'; found {config.get('model_type')!r}."
        )

    output.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(output).free
    if free_bytes < MINIMUM_FREE_BYTES:
        free_gib = free_bytes / 1024**3
        raise SystemExit(
            "LiteRT conversion needs at least 40 GiB of free working space; "
            f"only {free_gib:.1f} GiB is available."
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("models/Aquinas-Final-HF"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/Aquinas-Final-LiteRT"),
    )
    parser.add_argument(
        "--quantization-recipe",
        choices=SUPPORTED_QUANTIZATION_RECIPES,
        default="dynamic_wi4_afp32",
        help=(
            "Decoder quantization. dynamic_wi8_emb4_afp32 is the higher-precision "
            "phone candidate: 8-bit fully connected weights with 4-bit embeddings."
        ),
    )
    parser.add_argument(
        "--skip-vision",
        action="store_true",
        help=(
            "Skip exporting the vision tower/adapter. On-device multimodal input "
            "is post-launch; the vision_adapter compile step is also the single "
            "largest memory/disk consumer in the export. Produces a text-only "
            "package."
        ),
    )
    parser.add_argument(
        "--chat-template",
        type=Path,
        help=(
            "Jinja chat template to bundle. LiteRT-LM's template engine rejects the Hugging "
            "Face Gemma 4 template (it calls dict.get), so pass the template extracted from "
            "Google's stock .litertlm package."
        ),
    )
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    validate_source(source, output)
    register_mixed_recipes()

    export_kwargs = dict(
        model=str(source),
        output_dir=str(output),
        prefill_lengths=[128],
        cache_length=4_096,
        quantization_recipe=args.quantization_recipe,
        externalize_embedder=True,
        use_jinja_template=True,
        jinja_chat_template_override=str(args.chat_template or source / "chat_template.jinja"),
        bundle_litert_lm=True,
        experimental_lightweight_conversion=True,
    )

    if args.skip_vision:
        export_kwargs.update(
            task=ExportTask.TEXT_GENERATION,
            export_vision_encoder=False,
        )
    else:
        export_kwargs.update(
            task=ExportTask.IMAGE_TEXT_TO_TEXT,
            export_vision_encoder=True,
            vision_encoder_quantization_recipe="dynamic_wi8_afp32",
        )

    export.export(**export_kwargs)


if __name__ == "__main__":
    main()
