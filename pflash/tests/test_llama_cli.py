from __future__ import annotations

import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pflash.llama_cli import LlamaCliRunner


class LlamaCliRunnerTest(unittest.TestCase):
    def test_build_command_includes_expected_flags(self):
        runner = LlamaCliRunner(
            binary="/tmp/llama-cli",
            model="/tmp/model.gguf",
            ctx_size=16384,
            n_predict=32,
            n_gpu_layers="all",
            flash_attn="on",
            threads=8,
            threads_batch=16,
            batch_size=1024,
            ubatch_size=256,
            cache_type_k="q8_0",
            cache_type_v="q4_0",
        )
        command = runner.build_command("hello world")
        self.assertIn("/tmp/llama-cli", command)
        self.assertIn("/tmp/model.gguf", command)
        self.assertIn("--simple-io", command)
        self.assertIn("--no-display-prompt", command)
        self.assertIn("-fa", command)
        self.assertIn("-ngl", command)
        self.assertEqual(command[-1], "--log-disable")
        self.assertIn("hello world", command)


if __name__ == "__main__":
    unittest.main()
