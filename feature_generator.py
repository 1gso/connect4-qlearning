import numpy as np
import scipy.signal as sig


class FeatureGenerator:
    """
    Generates convolutional features from board states using a fixed filter bank.
    """

    def __init__(self):
        self.filter_bank = self._create_filter_bank()

    @staticmethod
    def _create_filter_bank() -> list:
        diag = np.eye(4, dtype=float)  # main diagonal of ones
        return [
            np.ones((1, 4)),  # horizontal
            np.ones((4, 1)),  # vertical
            diag,
            np.flipud(diag),  # anti-diagonal
        ]

    def convolution_feature_gen(self, state_list: list) -> tuple:
        """
        Turn a per-column state_list into flattened feature vectors.

        Returns:
            board_matrix: 2D numpy array of shape (6, 7)
            features: 1D numpy array of concatenated convolution outputs.
        """
        # build board matrix
        board = np.zeros((6, 7), dtype=int)
        for col_idx in range(7):
            for row_idx, val in enumerate(state_list[col_idx]):
                board[row_idx, col_idx] = val

        conv_outputs = []
        # apply each filter and flatten its features
        for one_filter in self.filter_bank:
            conv = sig.convolve(board, one_filter, mode="valid")
            abs_conv = sig.convolve(np.abs(board), one_filter, mode="valid")
            mask = abs_conv == np.abs(conv)
            filtered = conv * mask
            conv_outputs.append(filtered.flatten())

        features = np.concatenate(conv_outputs)
        return board, features

    def convolution_feature_group_matrices(self, state_list: list) -> tuple:
        """
        Turn a per-column state_list into feature matrices per filter.

        Returns:
            board_matrix: 2D numpy array of shape (6, 7)
            group_matrices: List of 2D numpy arrays, one per filter
        """
        # build board matrix
        board = np.zeros((6, 7), dtype=int)
        for col_idx in range(7):
            for row_idx, val in enumerate(state_list[col_idx]):
                board[row_idx, col_idx] = val

        group_matrices = []
        # apply each filter and collect its 2D feature map
        for one_filter in self.filter_bank:
            conv = sig.convolve(board, one_filter, mode="valid")
            abs_conv = sig.convolve(np.abs(board), one_filter, mode="valid")
            mask = abs_conv == np.abs(conv)
            filtered = conv * mask
            group_matrices.append(filtered)

        return board, group_matrices

    def print_feature_group_matrices(self, state_list: list) -> None:
        """
        Print each feature group matrix with zeros replaced by '.'.
        The top row is printed first (right-side-up orientation).
        """
        _, group_matrices = self.convolution_feature_group_matrices(state_list)
        for grp_idx, mat in enumerate(group_matrices):
            print(f"Feature group {grp_idx}:")
            # print from top row down to bottom
            for row_idx in range(mat.shape[0] - 1, -1, -1):
                row = mat[row_idx]
                row_str = " ".join(f"{val:.0f}" if val != 0 else "." for val in row)
                print(row_str)
            print()
