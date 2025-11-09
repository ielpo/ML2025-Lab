import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List


# Generate 5x5 input representations for digits 0-9
def create_digit_inputs() -> Dict[int, torch.Tensor]:
    digits: Dict[int, torch.Tensor] = {}

    # Digit 0
    digits[0] = torch.tensor(
        [
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
        ],
        dtype=torch.float32,
    )

    # Digit 1
    digits[1] = torch.tensor(
        [
            [0, 0, 1, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0],
        ],
        dtype=torch.float32,
    )

    # Digit 2
    digits[2] = torch.tensor(
        [
            [1, 1, 1, 1, 1],
            [0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
        ],
        dtype=torch.float32,
    )

    # Digit 3
    digits[3] = torch.tensor(
        [
            [1, 1, 1, 1, 1],
            [0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
            [0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
        ],
        dtype=torch.float32,
    )

    # Digit 4
    digits[4] = torch.tensor(
        [
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
        ],
        dtype=torch.float32,
    )

    # Digit 5
    digits[5] = torch.tensor(
        [
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
        ],
        dtype=torch.float32,
    )

    # Digit 6
    digits[6] = torch.tensor(
        [
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
        ],
        dtype=torch.float32,
    )

    # Digit 7
    digits[7] = torch.tensor(
        [
            [1, 1, 1, 1, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
        ],
        dtype=torch.float32,
    )

    # Digit 8
    digits[8] = torch.tensor(
        [
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
        ],
        dtype=torch.float32,
    )

    # Digit 9
    digits[9] = torch.tensor(
        [
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
            [0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
        ],
        dtype=torch.float32,
    )

    return digits


class RNNModel(nn.Module):
    def __init__(
        self, W_xh: torch.Tensor, W_hh: torch.Tensor, W_hy: torch.Tensor
    ) -> None:
        super(RNNModel, self).__init__()

        self.hidden_size = 2
        input_size = 5 * 5
        output_size = 1

        self.linear_xh = nn.Linear(input_size, self.hidden_size, bias=False)
        self.linear_hh = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.linear_hy = nn.Linear(self.hidden_size, output_size, bias=False)

        self.linear_xh.weight.data = W_xh
        self.linear_hh.weight.data = W_hh
        self.linear_hy.weight.data = W_hy

        self.relu_h = nn.ReLU(inplace=True)

    def forward(
        self, x: torch.Tensor, h_prev: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: input
        h_prev: previous hidden state
        Returns: output, new hidden state
        """
        # Initialize h to zero
        if h_prev is None:
            h_prev = torch.zeros(self.hidden_size)

        h: torch.Tensor = self.relu_h(self.linear_hh(h_prev) + self.linear_xh(x))
        output: torch.Tensor = self.linear_hy(h)

        return output, h


if __name__ == "__main__":
    # Init digits representations
    digits = create_digit_inputs()

    # Load the RNN weights
    weights = torch.load("exercise_5/ex5_q1-3_rnn_weights.pt")

    # Initialize the model
    model = RNNModel(
        W_xh=weights["Wxh.weight"],
        W_hh=weights["Whh.weight"],
        W_hy=weights["Why.weight"],
    )

    test_sequences = {
        "a": [0, 1, 3, 5],
        "b": [1, 2, 4, 6],
        "c": [2, 5, 7, 8],
        "d": [2, 8, 2, 0],
        "e": [3, 4, 7, 9],
        "f": [3, 5, 8, 9],
        "g": [4, 1, 0, 0],
        "h": [4, 5, 7, 9],
        "i": [5, 3, 3, 3],
        "j": [5, 6, 7, 8],
    }

    model.eval()
    with torch.no_grad():
        results: Dict[str, List[float]] = {}

        for name, sequence in test_sequences.items():
            h = None
            output = []
            for digit in sequence:
                x = digits[digit].flatten()
                y, h = model.forward(x, h)
                output.append(y.item())

            results[name] = output

    for name, values in results.items():
        formatted = "".join([f"{val:8d}" for val in test_sequences[name]])
        print(f"{name}:\t{formatted}")
        formatted = "".join([f"{val:8.4f}" for val in values])
        print(f"{name}:\t{formatted}")
        mse = (
            sum([((results[name][i] - test_sequences[name][i]) ** 2) for i in range(4)])
            / 4
        )
        print(f"MSE:\t{mse:8.4f}")
