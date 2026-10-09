from pathlib import Path
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
IGNORED_DIRECTORY_NAMES = {
    ".venv",
    "aquinas_env",
    "litert_conversion_env",
    "models",
    "tests",
    "venv",
}
# Offline research tools that template fine-tuning data or sample a checkpoint. They never build
# runtime prompts, which only main.py assembles.
OFFLINE_TEMPLATING_TOOLS = {
    "scripts/sample_voice.py",
    "scripts/train_voice_lora.py",
}


class PromptEntrypointTests(unittest.TestCase):
    def test_only_main_assembles_tokenizer_chat_prompts(self) -> None:
        prompt_assemblers = []
        for path in REPOSITORY_ROOT.rglob("*.py"):
            relative_path = path.relative_to(REPOSITORY_ROOT)
            if any(part in IGNORED_DIRECTORY_NAMES for part in relative_path.parts):
                continue
            if relative_path.as_posix() in OFFLINE_TEMPLATING_TOOLS:
                continue
            if "apply_chat_template" in path.read_text(encoding="utf-8"):
                prompt_assemblers.append(relative_path.as_posix())

        self.assertEqual(prompt_assemblers, ["main.py"])


if __name__ == "__main__":
    unittest.main()
