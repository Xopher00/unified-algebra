"""
Backend abstraction for differential testing.

Each backend owns the full tensor lifecycle. No runtime crossing.
"""

from abc import ABC, abstractmethod
from hypothesis import strategies as st
from hydra.dsl.python import Left, Right


SCALAR = "scalar"
VECTOR = "vector"
MATRIX = "matrix"

VECTOR_DIMS = [1, 2, 4]
MATRIX_DIMS = [2, 3]

_floats = st.floats(min_value=-2, max_value=2,
                    allow_nan=False, allow_infinity=False)


class Backend(ABC):
    name: str

    @abstractmethod
    def random_vector(self, draw, dim):
        """Random vector of shape (dim,)."""

    @abstractmethod
    def random_matrix(self, draw, rows, cols):
        """Random matrix of shape (rows, cols)."""

    @abstractmethod
    def zeros_vector(self, dim):
        """Zero vector of shape (dim,)."""

    @abstractmethod
    def allclose(self, a, b, atol=1e-5):
        """Native comparison."""

    @abstractmethod
    def load_fold_seq(self):
        """Import the generated fold_seq for this backend."""

    @abstractmethod
    def load_fold_tree(self):
        """Import the generated fold_tree for this backend."""

    @abstractmethod
    def run_reference_rnn(self, wIn, wRec, b, s0, elements, input_dim, hidden_dim):
        """Run the library-native RNN. All inputs native. Returns native."""

    def __repr__(self):
        return self.name


# ── TensorFlow ───────────────────────────────────────────────────────────────

class TFBackend(Backend):
    name = "tf"

    def random_vector(self, draw, dim):
        import numpy as np
        return np.array([draw(_floats) for _ in range(dim)], dtype=np.float64)

    def random_matrix(self, draw, rows, cols):
        import numpy as np
        return np.array([[draw(_floats) for _ in range(cols)]
                         for _ in range(rows)], dtype=np.float64)

    def zeros_vector(self, dim):
        import numpy as np
        return np.zeros(dim, dtype=np.float64)

    def allclose(self, a, b, atol=1e-5):
        import numpy as np
        return np.allclose(a, b, atol=atol)

    def load_fold_seq(self):
        from seed.seq import fold_seq
        return fold_seq

    def load_fold_tree(self):
        from seed.tree import fold_tree
        return fold_tree

    def run_reference_rnn(self, wIn, wRec, b, s0, elements, input_dim, hidden_dim):
        """SimpleRNN(activation='linear', use_bias=True).

        Direct weight copy — no diagonal adapter needed.
        kernel = wIn.T  (SimpleRNN expects [input, hidden])
        recurrent_kernel = wRec.T  (SimpleRNN expects [hidden, hidden])
        bias = b

        Reversed element order: cata is right-fold, SimpleRNN is left-fold.
        """
        import numpy as np
        import tensorflow as tf

        rev = list(reversed(elements))
        rnn = tf.keras.layers.SimpleRNN(
            units=hidden_dim, activation='linear', use_bias=True,
            return_sequences=False, dtype='float64')
        x_tf = np.stack(rev, dtype=np.float64)[np.newaxis]
        rnn(x_tf)
        rnn.set_weights([
            wIn.T,   # kernel: (input, hidden) — SimpleRNN convention
            wRec.T,  # recurrent_kernel: (hidden, hidden)
            b,       # bias
        ])
        init = s0.reshape(1, hidden_dim).astype(np.float64)
        return rnn(x_tf, initial_state=tf.constant(init)).numpy()[0]


# ── PyTorch ──────────────────────────────────────────────────────────────────

class TorchBackend(Backend):
    name = "torch"

    def random_vector(self, draw, dim):
        import torch
        return torch.tensor([draw(_floats) for _ in range(dim)],
                            dtype=torch.float64)

    def random_matrix(self, draw, rows, cols):
        import torch
        return torch.tensor([[draw(_floats) for _ in range(cols)]
                             for _ in range(rows)], dtype=torch.float64)

    def zeros_vector(self, dim):
        import torch
        return torch.zeros(dim, dtype=torch.float64)

    def allclose(self, a, b, atol=1e-5):
        import torch
        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a, dtype=torch.float64)
        if not isinstance(b, torch.Tensor):
            b = torch.tensor(b, dtype=torch.float64)
        return torch.allclose(a.detach(), b.detach(), atol=atol)

    def load_fold_seq(self):
        from seed.seq_tanh import fold_seq_tanh
        return fold_seq_tanh

    def load_fold_tree(self):
        from seed.tree import fold_tree
        return fold_tree

    def run_reference_rnn(self, wIn, wRec, b, s0, elements, input_dim, hidden_dim):
        """torch.nn.RNN(nonlinearity='tanh', bias=True).

        Direct weight copy — no diagonal adapter needed.
        weight_ih = wIn   (torch expects [hidden, input])
        weight_hh = wRec  (torch expects [hidden, hidden])
        bias_ih = b, bias_hh = zeros

        Reversed element order: cata is right-fold, RNN is left-fold.
        left-fold(reversed) == right-fold(original).
        """
        import torch

        rev = list(reversed(elements))
        x_t = torch.stack(rev).unsqueeze(1)  # (steps, batch=1, input)
        h0 = s0.reshape(1, 1, hidden_dim)    # (layers=1, batch=1, hidden)

        rnn = torch.nn.RNN(input_size=input_dim, hidden_size=hidden_dim,
                           num_layers=1, nonlinearity='tanh', bias=True,
                           batch_first=False, dtype=torch.float64)
        with torch.no_grad():
            rnn.weight_ih_l0.copy_(wIn)
            rnn.weight_hh_l0.copy_(wRec)
            rnn.bias_ih_l0.copy_(b)
            rnn.bias_hh_l0.zero_()
            _, hn = rnn(x_t, h0)

        return hn.squeeze()
