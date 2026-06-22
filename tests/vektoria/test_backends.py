"""Unit tests for the vector backends behind the VectorBackend seam.

BruteForceBackend is exercised directly here; TurboVecBackend is covered through
the Index in test_ann.py (it needs the optional turbovec engine)."""

import numpy as np

from vektoria.backends import BruteForceBackend


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def _matrix(*rows):
    return np.array([_unit(r) for r in rows], dtype=np.float32)


def test_search_returns_nearest_first():
    b = BruteForceBackend(_matrix([1, 0, 0], [0, 1, 0], [1, 1, 0]))
    res = b.search(_unit([1, 0, 0]), top_k=2, filtered=False)
    assert [row for row, _ in res] == [0, 2]   # exact match, then the [1,1,0] row


def test_search_overfetches_when_filtered():
    b = BruteForceBackend(_matrix([1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]))
    # filtered=True fetches top_k*4, so a small top_k still surfaces every row
    res = b.search(_unit([1, 0, 0]), top_k=1, filtered=True)
    assert len(res) == 4


def test_candidate_scores_covers_all_rows():
    b = BruteForceBackend(_matrix([1, 0, 0], [0, 1, 0]))
    scores = b.candidate_scores(_unit([1, 0, 0]), candidate_k=1)
    assert set(scores) == {0, 1}               # exact backend scores every row
    assert scores[0] > scores[1]


def test_add_appends_contiguous_rows():
    b = BruteForceBackend(None)
    b.add(_matrix([1, 0, 0]), [0])
    b.add(_matrix([0, 1, 0]), [1])
    scores = b.candidate_scores(_unit([0, 1, 0]), candidate_k=2)
    assert set(scores) == {0, 1} and scores[1] > scores[0]


def test_replace_updates_a_row_in_place():
    b = BruteForceBackend(_matrix([1, 0, 0], [0, 1, 0]))
    b.replace(0, _unit([0, 1, 0]))
    scores = b.candidate_scores(_unit([0, 1, 0]), candidate_k=2)
    assert scores[0] > 0.99                     # row 0 now matches [0,1,0]


def test_keep_rows_compacts_in_memory_without_reload():
    b = BruteForceBackend(_matrix([1, 0, 0], [0, 1, 0], [0, 0, 1]))

    def _forbidden_reload():
        raise AssertionError("brute-force must compact in memory, never reload")

    b.keep_rows([0, 2], _forbidden_reload)
    res = b.search(_unit([0, 0, 1]), top_k=2, filtered=False)
    assert res[0][0] == 1                        # old row 2 is now row 1
