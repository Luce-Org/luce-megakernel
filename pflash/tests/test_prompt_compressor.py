from __future__ import annotations

import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pflash.prompt_compressor import build_middle_chunk_ranges, merge_ranges, select_scored_ranges


class PromptCompressorHelpersTest(unittest.TestCase):
    def test_build_middle_chunk_ranges_respects_head_and_tail(self):
        self.assertEqual(
            build_middle_chunk_ranges(total_tokens=2000, head_tokens=256, tail_tokens=512, chunk_tokens=384),
            [(256, 640), (640, 1024), (1024, 1408), (1408, 1488)],
        )

    def test_merge_ranges_coalesces_touching_spans(self):
        self.assertEqual(merge_ranges([(0, 10), (10, 20), (25, 30)]), [(0, 20), (25, 30)])

    def test_select_scored_ranges_prefers_high_scores(self):
        ranges = [(0, 10), (10, 20), (20, 30), (30, 40)]
        scores = [0.2, 0.9, 0.5, 0.8]
        selected = select_scored_ranges(ranges, scores, token_budget=20, min_chunks=2)
        self.assertEqual([(chunk.start, chunk.end) for chunk in selected], [(10, 20), (30, 40)])

    def test_select_scored_ranges_honors_min_chunks(self):
        ranges = [(0, 10), (10, 20), (20, 30)]
        scores = [0.9, 0.8, 0.1]
        selected = select_scored_ranges(ranges, scores, token_budget=10, min_chunks=2)
        self.assertEqual(len(selected), 2)


if __name__ == "__main__":
    unittest.main()
