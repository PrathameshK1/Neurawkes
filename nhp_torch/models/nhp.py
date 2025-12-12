from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from nhp_torch.models.intensity import IntensityHead


@dataclass(frozen=True)
class NHPState:
    c: torch.Tensor       # (hidden_dim,)
    c_bar: torch.Tensor   # (hidden_dim,) long-term cell target
    delta: torch.Tensor   # (hidden_dim,) positive decay
    o: torch.Tensor       # (hidden_dim,) output gate


class CTLSTMCell(nn.Module):
    """Continuous-Time LSTM cell (Mei & Eisner style).

    We keep it minimal for this research pipeline:
    - event embedding drives gates
    - between events, cell state decays toward c_bar with rate delta
    """

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        self.lin = nn.Linear(input_dim + hidden_dim, 7 * hidden_dim)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()

    def init_state(self, device: torch.device) -> NHPState:
        z = torch.zeros(self.hidden_dim, device=device)
        return NHPState(c=z, c_bar=z, delta=torch.ones_like(z), o=z)

    def event_update(self, x: torch.Tensor, h_prev: torch.Tensor, state: NHPState) -> NHPState:
        # x: (input_dim,), h_prev: (hidden_dim,)
        inp = torch.cat([x, h_prev], dim=0)
        out = self.lin(inp)
        (i, f, z, i_bar, f_bar, delta_raw, o) = torch.chunk(out, 7, dim=0)

        i = self.sigmoid(i)
        f = self.sigmoid(f)
        z = self.tanh(z)
        i_bar = self.sigmoid(i_bar)
        f_bar = self.sigmoid(f_bar)
        delta = self.softplus(delta_raw) + 1e-4
        o = self.sigmoid(o)

        c = f * state.c + i * z
        c_bar = f_bar * state.c_bar + i_bar * z

        return NHPState(c=c, c_bar=c_bar, delta=delta, o=o)

    def decay(self, state: NHPState, dt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # dt: scalar tensor >= 0
        # c(t) = c_bar + (c - c_bar) * exp(-delta * dt)
        decay = torch.exp(-state.delta * dt)
        c_t = state.c_bar + (state.c - state.c_bar) * decay
        h_t = state.o * torch.tanh(c_t)
        return c_t, h_t


class NeuralHawkes(nn.Module):
    def __init__(self, num_types: int, embed_dim: int, hidden_dim: int, use_marks: bool) -> None:
        super().__init__()
        self.num_types = num_types
        self.use_marks = use_marks

        self.type_embed = nn.Embedding(num_types, embed_dim)
        input_dim = embed_dim + (1 if use_marks else 0)
        self.cell = CTLSTMCell(input_dim=input_dim, hidden_dim=hidden_dim)
        self.intensity = IntensityHead(hidden_dim=hidden_dim, num_types=num_types)

    def forward_intensity_path(
        self,
        times: torch.Tensor,  # (N,)
        types: torch.Tensor,  # (N,)
        marks: torch.Tensor,  # (N,)
        query_times: torch.Tensor,  # (M,) increasing, within [0, T]
    ) -> torch.Tensor:
        """Compute λ(query_times) following the history of events up to each query time.

        Returns: (M, K) intensities.
        """
        device = times.device
        state = self.cell.init_state(device=device)
        h_prev = torch.zeros(self.cell.hidden_dim, device=device)

        lam_out = []
        ev_i = 0
        t_prev = torch.tensor(0.0, device=device)

        for tq in query_times:
            # Advance through events strictly before tq
            while ev_i < times.numel() and times[ev_i] < tq:
                # decay to event time
                dt = times[ev_i] - t_prev
                _, h_prev = self.cell.decay(state, dt)
                # event update at that time
                emb = self.type_embed(types[ev_i])
                if self.use_marks:
                    x = torch.cat([emb, marks[ev_i : ev_i + 1]], dim=0)
                else:
                    x = emb
                state = self.cell.event_update(x, h_prev, state)
                # set t_prev = event time (state stored at event)
                t_prev = times[ev_i]
                ev_i += 1

            # decay to query time
            dtq = tq - t_prev
            _, hq = self.cell.decay(state, dtq)
            lam_out.append(self.intensity(hq))

        return torch.stack(lam_out, dim=0)


