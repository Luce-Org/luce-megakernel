import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "test" / "test_dflash.cpp"


class DaemonResetMergeResolutionTest(unittest.TestCase):
    def test_daemon_reset_reuses_cache_and_frees_both_transient_graphs(self):
        source = SOURCE.read_text()
        match = re.search(
            r"if \(!daemon_first_iter\) \{\n(?P<body>.*?)\n\s+\}\n\s+daemon_first_iter = false;",
            source,
            re.S,
        )
        self.assertIsNotNone(match, "daemon reset block not found")

        body = match.group("body")
        self.assertIn("step_graph_free(target_sg);", body)
        self.assertIn("step_graph_free(draft_sg);", body)
        self.assertIn("reset_target_cache(cache);", body)
        self.assertNotIn("step_graph_destroy(target_sg);", body)
        self.assertNotIn("step_graph_destroy(draft_sg);", body)
        self.assertNotIn("free_target_cache(cache);", body)
        self.assertNotIn("create_target_cache(", body)


if __name__ == "__main__":
    unittest.main()
