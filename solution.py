"""Stereo matching."""
import numpy as np
from scipy.signal import convolve2d



class Solution:
    def __init__(self):
        pass

    @staticmethod
    def ssd_distance(left_image: np.ndarray,
                     right_image: np.ndarray,
                     win_size: int,
                     dsp_range: int) -> np.ndarray:
        """Compute the SSDD distances tensor.

        Args:
            left_image: Left image of shape: HxWx3, and type np.double64.
            right_image: Right image of shape: HxWx3, and type np.double64.
            win_size: Window size odd integer.
            dsp_range: Half of the disparity range. The actual range is
            -dsp_range, -dsp_range + 1, ..., 0, 1, ..., dsp_range.

        Returns:
            A tensor of the sum of squared differences for every pixel in a
            window of size win_size X win_size, for the 2*dsp_range + 1
            possible disparity values. The tensor shape should be:
            HxWx(2*dsp_range+1).
        """
        print('ssd_distance')
        num_of_rows, num_of_cols = left_image.shape[0], left_image.shape[1]
        disparity_values = range(-dsp_range, dsp_range + 1)
        ssdd_tensor = np.zeros((num_of_rows,num_of_cols,len(disparity_values)))
        right_image_mean = right_image.mean(2)  # RGB to grayscale
        left_image_mean = left_image.mean(2)  # RGB to grayscale
        half_window = int(np.floor(win_size / 2))
        padding_size_left_img = ((half_window, half_window), (half_window, half_window))
        left_image_padded = np.pad(left_image_mean, padding_size_left_img, 'constant')
        padding_size_right_img = ((half_window, half_window), (half_window + dsp_range, half_window + dsp_range))
        right_image_padded = np.pad(right_image_mean, padding_size_right_img, 'constant')

        for disparity_ind in range(dsp_range * 2 + 1):
            cropped_padded_right_image = right_image_padded[:, disparity_ind:left_image_padded.shape[1] + disparity_ind]
            mat_diff = np.square(left_image_padded - cropped_padded_right_image)
            kernel = np.ones((win_size, win_size))
            ssdd_tensor[:, :, disparity_ind] = convolve2d(mat_diff, kernel, mode='valid')

        ssdd_tensor -= ssdd_tensor.min()
        ssdd_tensor /= ssdd_tensor.max()
        ssdd_tensor *= 255.0
        return ssdd_tensor

    @staticmethod
    def naive_labeling(ssdd_tensor: np.ndarray) -> np.ndarray:
        """Estimate a naive depth estimation from the SSDD tensor.

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for every
            pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.

        Evaluate the labels in a naive approach. Each value in the
        result tensor should contain the disparity matching minimal ssd (sum of
        squared difference).

        Returns:
            Naive labels HxW matrix.
        """
        # you can erase the label_no_smooth initialization.
        # label_no_smooth = np.zeros((ssdd_tensor.shape[0], ssdd_tensor.shape[1]))
        label_no_smooth = np.argmin(ssdd_tensor, axis=2)

        return label_no_smooth

    @staticmethod
    def dp_grade_slice(c_slice: np.ndarray, p1: float, p2: float) -> np.ndarray:
        """Calculate the scores matrix for slice c_slice.

        Calculate the scores slice which for each column and disparity value
        states the score of the best route. The scores slice is of shape:
        (2*dsp_range + 1)xW.

        Args:
            c_slice: A slice of the ssdd tensor.
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.
        Returns:
            Scores slice which for each column and disparity value states the
            score of the best route.
        """
        c_slice = c_slice.T
        num_labels, num_of_cols = c_slice.shape[0], c_slice.shape[1]
        l_slice = np.zeros((num_labels, num_of_cols))

        l_slice[:, 0] = c_slice[:, 0]
        for col_slice in range(1, num_of_cols):
            for d_slice in range(0, num_labels):
                if d_slice - 2 < 0:
                    m = np.min(l_slice[d_slice + 2:num_labels, col_slice - 1])
                else:
                    m = np.min(np.hstack((l_slice[d_slice + 2:num_labels, col_slice - 1], l_slice[0:d_slice - 2, col_slice - 1])))
                if d_slice + 1 > num_labels - 1:
                    ld_next = np.inf
                else:
                    ld_next = l_slice[d_slice + 1, col_slice - 1]
                if d_slice - 1 < 0:
                    ld_prev = np.inf
                else:
                    ld_prev = l_slice[d_slice - 1, col_slice - 1]
                m_f = np.min((l_slice[d_slice, col_slice - 1], p1 + np.min((ld_prev, ld_next)), p2 + m))
                l_slice[d_slice, col_slice] = c_slice[d_slice, col_slice] + m_f - np.min(l_slice[:, col_slice - 1])

        return l_slice

    def dp_labeling(self,
                    ssdd_tensor: np.ndarray,
                    p1: float,
                    p2: float) -> np.ndarray:
        """Estimate a depth map using Dynamic Programming.

        (1) Call dp_grade_slice on each row slice of the ssdd tensor.
        (2) Store each slice in a corresponding l tensor (of shape as ssdd).
        (3) Finally, for each pixel in l (along each row and column), choose
        the best disparity value. That is the disparity value which
        corresponds to the lowest l value in that pixel.

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for every
            pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.
        Returns:
            Dynamic Programming depth estimation matrix of shape HxW.
        """
        l = np.zeros_like(ssdd_tensor)
        for c_slice_indx in range(0, ssdd_tensor.shape[0]):
            c_slice = ssdd_tensor[c_slice_indx, :, :]
            l_slice = self.dp_grade_slice(c_slice, p1, p2)
            l[c_slice_indx, :, :] = l_slice.T

        return self.naive_labeling(l)

    @staticmethod
    def slices_per_dir(ssdd_tensor, direction):
        number_of_direction = 8
        start = -(ssdd_tensor.shape[0] - 1)
        end = ssdd_tensor.shape[1] - 1
        dict_dir = {}
        dict_dir_idx = {}
        idx_mat = np.arange(0, (ssdd_tensor.shape[0]) * (ssdd_tensor.shape[1])).reshape(ssdd_tensor.shape[0],
                                                                                            ssdd_tensor.shape[1])
        # symmetry:
        if direction == 0:
            dict_dir[str(direction + 4)], dict_dir_idx[str(direction + 4)] = [], []
            dict_dir[str(direction + 8)], dict_dir_idx[str(direction + 8)] = [], []
        else:
            dict_dir[str(direction)], dict_dir[str(direction + 4)] = [], []
            dict_dir_idx[str(direction)], dict_dir_idx[str(direction + 4)] = [], []

        if (direction == 1) or (direction == 3):  # 1 and 5 = east and west, 3 and 7 = south and north
            if direction == 1:  # east or west
                range_tensor = ssdd_tensor.shape[0]
                ssdd_tensor = ssdd_tensor
                idx_mat = idx_mat
            if direction == 3:  # south or north
                range_tensor = ssdd_tensor.shape[1]
                ssdd_tensor = np.rot90(ssdd_tensor)
                idx_mat = np.rot90(idx_mat)
            for i in range(range_tensor):
                dict_dir[str(direction)].append(ssdd_tensor[i])
                dict_dir_idx[str(direction)].append(idx_mat[i])
                dict_dir[str(direction + 4)].append((np.fliplr(ssdd_tensor))[i])  # reversed
                dict_dir_idx[str(direction + 4)].append((np.fliplr(idx_mat))[i])  # reversed

        if (direction == 2) or (direction == 0):  # 2 and 6 = south east and north west, 0 and 4 = north east and south west
            if direction == 2:
                for offset in range(start, end):
                    dict_dir[str(direction)].append(ssdd_tensor.diagonal(offset=offset))
                    dict_dir_idx[str(direction)].append(idx_mat.diagonal(offset=offset))
                    dict_dir[str(direction + 4)].append(np.fliplr(ssdd_tensor.diagonal(offset=offset)))
                    imat = idx_mat.diagonal(offset=offset)
                    dict_dir_idx[str(direction + 4)].append(np.fliplr(imat.reshape((1, len(imat)))))
            else:
                for offset in range(start, end):
                    new_tensor = np.fliplr(ssdd_tensor)
                    new_idx_mat = np.fliplr(idx_mat)
                    dict_dir[str(direction + 4)].append((new_tensor.diagonal(offset=offset)))
                    dict_dir_idx[str(direction + 4)].append(new_idx_mat.diagonal(offset=offset))
                    dict_dir[str(direction + 8)].append(np.fliplr(new_tensor.diagonal(offset)))
                    imat = new_idx_mat.diagonal(offset)
                    dict_dir_idx[str(direction + 8)].append(np.fliplr(imat.reshape((1, len(imat)))))

        return dict_dir[str(direction)], dict_dir_idx[str(direction)], dict_dir[str(direction + 4)], dict_dir_idx[str(direction + 4)]

    def dp_labeling_per_direction(self,
                                  ssdd_tensor: np.ndarray,
                                  p1: float,
                                  p2: float) -> dict:
        """Return a dictionary of directions to a Dynamic Programming
        etimation of depth.

        For each direction in 1, ..., 8, calculate scores tensors
        according to dp_grade_slice and the method which allows you to
        extract slices along each direction.

        You may use helper methods (functions) that you write on your own.
        We found `np.diagonal` to be very helpful to extract diagonal slices.
        `np.unravel_index` might be helpful if you're thinking in MATLAB
        notations: it's the ind2sub equivalent.

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for
            every pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.

        Returns:
            Dictionary int->np.ndarray which maps each direction to the
            corresponding dynamic programming estimation of depth based on
            that direction.
        """
        num_of_directions = 8
        l = np.zeros_like(ssdd_tensor)
        direction_to_slice = {}
        op_l = np.zeros_like(ssdd_tensor)
        for direction in range(1, int(num_of_directions / 2) + 1):
            slices, slices_idx, op_slices, op_slices_idx = self.slices_per_dir(ssdd_tensor, direction)
            for c_slice_idx in range(0, len(slices)):
                c_slice = slices[c_slice_idx]
                if c_slice.shape[0] == 41:
                    c_slice = c_slice.T
                if np.shape(c_slice) == (1, 41):
                    l_slice = c_slice.T
                else:
                    l_slice = self.dp_grade_slice(c_slice, p1, p2)
                row, col = np.unravel_index(slices_idx[c_slice_idx], (ssdd_tensor.shape[0], ssdd_tensor.shape[1]))
                l[row, col, :] = l_slice.T

                op_c_slice = op_slices[c_slice_idx]
                if op_c_slice.shape[0] == 41:
                    op_c_slice = op_c_slice.T
                if np.shape(op_c_slice) == (1, 41):
                    op_l_slice = op_c_slice.T
                else:
                    op_l_slice = self.dp_grade_slice(op_c_slice, p1, p2)

                op_row, op_col = np.unravel_index(op_slices_idx[c_slice_idx], (ssdd_tensor.shape[0], ssdd_tensor.shape[1]))
                op_l[op_row, op_col, :] = op_l_slice.T

            direction_to_slice[direction], direction_to_slice[direction + 4] = self.naive_labeling(l), self.naive_labeling(op_l)

        return direction_to_slice

    def sgm_labeling(self, ssdd_tensor: np.ndarray, p1: float, p2: float):

        """Estimate the depth map according to the SGM algorithm.

        For each direction in 1, ..., 8, calculate scores tensors
        according to dp_grade_slice and the method which allows you to
        extract slices along each direction.

        You may use helper methods (functions) that you write on your own.
        We found `np.diagonal` to be very helpful to extract diagonal slices.
        `np.unravel_index` might be helpful if you're thinking in MATLAB
        notations: it's the ind2sub equivalent.

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for
            every pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.

        Returns:
            Semi-Global Mapping depth estimation matrix of shape HxW.
        """

        num_of_directions = 8
        l = np.zeros_like(ssdd_tensor)
        op_l = np.zeros_like(ssdd_tensor)
        l_final = np.zeros_like(ssdd_tensor)
        """INSERT YOUR CODE HERE"""

        for direction in range(1, int(num_of_directions / 2) + 1):
            slices, slices_idx, op_slices, op_slices_idx = self.slices_per_dir(ssdd_tensor, direction)

            for c_slice_idx in range(0, len(slices)):
                c_slice = slices[c_slice_idx]
                if c_slice.shape[0] == 41:
                    c_slice = c_slice.T
                if np.shape(c_slice) == (1, 41):
                    l_slice = c_slice.T
                else:
                    l_slice = self.dp_grade_slice(c_slice, p1, p2)
                row, col = np.unravel_index(slices_idx[c_slice_idx], (ssdd_tensor.shape[0], ssdd_tensor.shape[1]))
                l[row, col, :] = l_slice.T

                op_c_slice = op_slices[c_slice_idx]
                if op_c_slice.shape[0] == 41:
                    op_c_slice = op_c_slice.T
                if np.shape(op_c_slice) == (1, 41):
                    op_l_slice = op_c_slice.T
                else:
                    op_l_slice = self.dp_grade_slice(op_c_slice, p1, p2)

                op_row, op_col = np.unravel_index(op_slices_idx[c_slice_idx], (ssdd_tensor.shape[0], ssdd_tensor.shape[1]))
                op_l[op_row, op_col, :] = op_l_slice.T

            l_final = (l_final + l + op_l)/8

        return self.naive_labeling(l_final)
