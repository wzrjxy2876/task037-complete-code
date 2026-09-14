import unittest

try:
    from lgfr_runtime.task042_phase_g_functional_coverage import (
        action_conditioned_median,
        average_rank_preference,
        build_class_video_positions,
        build_incidence,
        calibration_subsets,
        classify_owner_types,
        exact_argmax_owner_set,
        minimum_set_covers,
        pareto_front,
    )
except ModuleNotFoundError:
    from task042_phase_g_functional_coverage import (
        action_conditioned_median,
        average_rank_preference,
        build_class_video_positions,
        build_incidence,
        calibration_subsets,
        classify_owner_types,
        exact_argmax_owner_set,
        minimum_set_covers,
        pareto_front,
    )


def unit(global_index, unit_type="attention_head"):
    return (str(global_index), "271", f"layers.{global_index}", "0", unit_type, str(global_index))


class PhaseGTests(unittest.TestCase):
    def test_80_condition_rank_semantics(self):
        values = [float(i) for i in range(80)]
        ranks, preferences = average_rank_preference(values)
        self.assertEqual(len(ranks), 80)
        self.assertEqual(preferences[0], 0.0)
        self.assertEqual(preferences[-1], 1.0)
        self.assertEqual(preferences[39], 39 / 79)

    def test_average_rank_exact_ties(self):
        ranks, preferences = average_rank_preference([1.0, 2.0, 2.0, 4.0])
        self.assertEqual(ranks, [1.0, 2.5, 2.5, 4.0])
        self.assertEqual(preferences, [0.0, 0.5, 0.5, 1.0])
        near_ranks, _ = average_rank_preference([1.0, 2.0, 2.0 + 1e-14, 4.0])
        self.assertNotEqual(near_ranks[1], near_ranks[2])

    def test_class_video_identity_and_subsets(self):
        videos = []
        for label in range(10):
            for position in range(3):
                videos.append({"video_index": str(label * 3 + position), "label": str(label), "video_id": f"v{label}_{position}"})
        positions = build_class_video_positions(videos)
        self.assertEqual(positions["0"], (0, 1, 2))
        subsets = calibration_subsets(positions)
        self.assertEqual(subsets["P1"], tuple(range(0, 30, 3)))
        self.assertEqual(subsets["P13"], tuple(i for label in range(10) for i in (label * 3, label * 3 + 2)))
        self.assertEqual(subsets["FULL"], tuple(range(30)))
        with self.assertRaises(ValueError):
            build_class_video_positions(videos[:-1])

    def test_action_conditioned_median(self):
        self.assertEqual(action_conditioned_median([0.1, 0.9]), 0.5)
        self.assertEqual(action_conditioned_median([0.1]), 0.1)

    def test_exact_argmax_and_multi_owner_exact_ties(self):
        a, b = unit(1), unit(2)
        self.assertEqual(exact_argmax_owner_set({a: 0.7, b: 0.7}), frozenset({a, b}))
        self.assertEqual(exact_argmax_owner_set({a: 0.7, b: 0.7 + 1e-14}), frozenset({b}))

    def test_pareto_dominance_strict(self):
        a, b, c = unit(1), unit(2), unit(3)
        front = pareto_front({a: [0.1, 0.2, 0.3], b: [0.2, 0.2, 0.4], c: [0.1, 0.3, 0.2]})
        self.assertEqual(front, frozenset({b, c}))
        tied = pareto_front({a: [0.5, 0.5, 0.5], b: [0.5, 0.5, 0.5]})
        self.assertEqual(tied, frozenset({a, b}))

    def test_incidence_construction(self):
        a, b = unit(1), unit(2)
        atom1 = ("16", (1, 0, 0, 1))
        atom2 = ("17", (1, 1, 1, 2))
        incidence = build_incidence({atom1: {a, b}, atom2: {b}})
        self.assertEqual(incidence[a], {atom1})
        self.assertEqual(incidence[b], {atom1, atom2})

    def test_exact_set_cover_and_multiplicity(self):
        a, b, c = unit(1), unit(2), unit(3)
        size, covers = minimum_set_covers([{a, b}, {b, c}], [a, b, c])
        self.assertEqual(size, 1)
        self.assertEqual(covers, (frozenset({b}),))
        size, covers = minimum_set_covers([{a, b}, {b, c}, {a, c}], [a, b, c])
        self.assertEqual(size, 2)
        self.assertEqual(len(covers), 3)

    def test_subset_determinism(self):
        positions = {str(label): (label * 3, label * 3 + 1, label * 3 + 2) for label in range(10)}
        self.assertEqual(calibration_subsets(positions), calibration_subsets(positions))

    def test_mixed_type_owner_classification(self):
        attention = unit(1, "attention_head")
        ffn = unit(2, "ffn_neuron")
        self.assertEqual(classify_owner_types({attention}, {attention: "attention_head", ffn: "ffn_neuron"}), "Attention-only")
        self.assertEqual(classify_owner_types({ffn}, {attention: "attention_head", ffn: "ffn_neuron"}), "FFN-only")
        self.assertEqual(classify_owner_types({attention, ffn}, {attention: "attention_head", ffn: "ffn_neuron"}), "mixed Attention+FFN")


if __name__ == "__main__":
    unittest.main()
