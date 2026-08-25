import tempfile
import unittest
from pathlib import Path

from evaluation.harness import (
    DEFAULT_MAP_SIZES,
    AgentSpec,
    MatchResult,
    RunnerStats,
    build_paired_schedule,
    paired_repeat_mismatches,
    summarize_results,
    wilson_interval,
)


def fake_spec(name):
    return AgentSpec(
        name=name,
        directory=Path("/tmp") / name,
        model_config=Path("config.yaml"),
        agent_config=Path("rl_agent_config.yaml"),
        checkpoint=Path("weights.pt"),
        content_hash=name + "-hash",
    )


def fake_result(match_id, candidate_seat, winner, map_size=12, seed=1):
    return MatchResult(
        match_id=match_id,
        seed=seed,
        map_size=map_size,
        agent_0="a",
        agent_1="b",
        agent_0_hash="a-hash",
        agent_1_hash="b-hash",
        candidate_seat=candidate_seat,
        winner=winner,
        final_city_tiles=(4, 2),
        final_units=(3, 1),
        ending_turn=100,
        wall_seconds=10.0,
        replay_path=None,
        diagnostics_path="diagnostics.json",
        cache_key="cache",
    )


class Phase1HarnessTests(unittest.TestCase):
    def test_full_schedule_is_balanced(self):
        schedule = build_paired_schedule(fake_spec("candidate"), fake_spec("baseline"))
        self.assertEqual(len(schedule), 400)
        for map_size in DEFAULT_MAP_SIZES:
            matches = [request for request in schedule if request.map_size == map_size]
            self.assertEqual(len(matches), 100)
            self.assertEqual(sum(request.candidate_seat == 0 for request in matches), 50)
            self.assertEqual(sum(request.candidate_seat == 1 for request in matches), 50)

    def test_candidate_mapping_respects_seat(self):
        player_0 = fake_result("p0", candidate_seat=0, winner=0)
        player_1 = fake_result("p1", candidate_seat=1, winner=0)
        self.assertEqual(player_0.candidate_score, 1.0)
        self.assertEqual(player_1.candidate_score, 0.0)
        self.assertEqual(player_0.candidate_city_margin, 2)
        self.assertEqual(player_1.candidate_city_margin, -2)

    def test_wilson_interval_matches_four_hundred_game_rule_of_thumb(self):
        low, high = wilson_interval(200, 400)
        self.assertAlmostEqual(low, 0.451, places=3)
        self.assertAlmostEqual(high, 0.549, places=3)

    def test_summary_reports_seat_and_map_splits(self):
        results = [
            fake_result("a", 0, 0, 12, 1),
            fake_result("b", 1, 0, 12, 1),
            fake_result("c", 0, None, 16, 2),
            fake_result("d", 1, None, 16, 2),
        ]
        summary = summarize_results(results, RunnerStats(wall_seconds=20, total_turns=400))
        self.assertEqual(summary["aggregate"]["win_rate"], 0.5)
        self.assertEqual(set(summary["by_seat"]), {"player_0", "player_1"})
        self.assertEqual(set(summary["by_map_size"]), {"12", "16"})

    def test_repeat_check_detects_nondeterminism(self):
        deterministic = [fake_result("a", 0, 0), fake_result("b", 1, 0)]
        self.assertEqual(paired_repeat_mismatches(deterministic), [])
        nondeterministic = [fake_result("a", 0, 0), fake_result("b", 1, 1)]
        self.assertEqual(len(paired_repeat_mismatches(nondeterministic)), 1)


if __name__ == "__main__":
    unittest.main()
