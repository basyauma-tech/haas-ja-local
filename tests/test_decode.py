from dataclasses import replace

import numpy as np

from haas_ja_local.decode import TimedToken, greedy_tdt


class ScriptedBackend:
    """台本どおりのlogitsを返す。台本は(t, 入力token)→(出力token, duration)。"""

    def __init__(self, config, script, n_frames, default=None):
        self.config = config
        self.script = script
        self.n_frames = n_frames
        self.default = default
        self.calls: list[tuple[int, int, int]] = []  # (t, token, 渡されたstate)

    def num_frames(self, encoded):
        return self.n_frames

    def init_state(self):
        return 0

    def step(self, encoded, t, token, state):
        self.calls.append((t, token, state))
        out, d = self.script.get((t, token), self.default)
        c = self.config
        logits = np.zeros(c.vocab_size + 1 + len(c.durations), dtype=np.float32)
        logits[out] = 5.0
        logits[c.vocab_size + 1 + c.durations.index(d)] = 5.0
        return logits, state + 1


def test_blank_with_duration_zero_advances_one_frame(tiny_config):
    b = tiny_config.blank_id
    be = ScriptedBackend(tiny_config, {}, 3, default=(b, 0))
    assert greedy_tdt(be, None) == []
    assert [c[0] for c in be.calls] == [0, 1, 2]


def test_token_with_duration_skips_frames(tiny_config):
    b = tiny_config.blank_id
    be = ScriptedBackend(tiny_config, {(0, b): (5, 3)}, 4, default=(b, 1))
    assert greedy_tdt(be, None) == [TimedToken(5, 0, 3)]
    assert [c[0] for c in be.calls] == [0, 3]


def test_two_tokens_same_frame_then_blank(tiny_config):
    b = tiny_config.blank_id
    script = {(0, b): (5, 0), (0, 5): (6, 0), (0, 6): (b, 1)}
    be = ScriptedBackend(tiny_config, script, 1)
    assert greedy_tdt(be, None) == [TimedToken(5, 0, 0), TimedToken(6, 0, 0)]


def test_max_symbols_forces_advance(tiny_config):
    cfg = replace(tiny_config, max_symbols=3)
    be = ScriptedBackend(cfg, {}, 2, default=(7, 0))
    out = greedy_tdt(be, None)
    assert [x.frame for x in out] == [0, 0, 0, 1, 1, 1]
    assert len(be.calls) == 6


def test_state_not_updated_on_blank(tiny_config):
    b = tiny_config.blank_id
    script = {(0, b): (5, 1), (1, 5): (b, 0), (2, 5): (b, 1)}
    be = ScriptedBackend(tiny_config, script, 3)
    greedy_tdt(be, None)
    # t=0でtoken 5を出力（state 0→1）。blank後の2回のstepは同じtokenとstateを受け取る。
    assert be.calls == [(0, b, 0), (1, 5, 1), (2, 5, 1)]


def test_empty_encoder_output_returns_no_tokens(tiny_config):
    be = ScriptedBackend(tiny_config, {}, 0)
    assert greedy_tdt(be, None) == []
    assert be.calls == []
