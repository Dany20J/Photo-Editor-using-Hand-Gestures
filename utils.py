import cv2 as cv
import numpy as np


def skeleton_distance_transform_based(img):
    if img.ndim == 3:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    else:
        gray = img
    _, thresh = cv.threshold(
        gray, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    thresh = 255 - thresh
    kernel = np.ones((3, 3), np.uint8)
    opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)
    closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel, iterations=2)

    dist_transform = cv.distanceTransform(closing, cv.DIST_L2, 5)

    return dist_transform


def find_center_point_skeleton(skeleton):
    # skeleton is 2d array
    mx_v = np.max(skeleton)
    mx = np.argmax(skeleton, axis=0)
    row = skeleton[mx, np.arange(skeleton.shape[1])]
    mx_col_ind = np.argmax(row)
    mx_row_ind = np.squeeze(mx[mx_col_ind])

    return np.array([mx_row_ind, mx_col_ind], dtype=np.int32), mx_v


def center_to_bb_skeleton(center_point, mx_v):
    top_left_row = int(center_point[0] - mx_v * 2 - mx_v / 5)
    top_left_col = int(center_point[1] - mx_v - mx_v / 5)

    bottom_right_row = int(center_point[0] + mx_v + mx_v / 5)
    bottom_right_col = int(center_point[1] + mx_v + mx_v / 5)

    return np.array([top_left_row, top_left_col], dtype=np.int32), np.array([bottom_right_row, bottom_right_col], dtype=np.int32)


def homomorphic_filter(img):
    height, width, _ = img.shape

    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    img_gray_log = np.log1p(img_gray.astype(np.float64))

    fft2 = np.fft.fft2(img_gray_log)
    fft2_shifted = np.fft.fftshift(fft2)

    sigma = 100
    gaussian_filter_1d = cv.getGaussianKernel(max(
        height, width) + 1 if max(height, width) % 2 == 0 else max(height, width), sigma=sigma)

    gaussian_filter_2d = gaussian_filter_1d.T * gaussian_filter_1d

    excess_height, excess_width = gaussian_filter_2d.shape[0] - \
        height, gaussian_filter_2d.shape[1] - width

    resized_gf2d = gaussian_filter_2d[excess_height // 2: gaussian_filter_2d.shape[0] -
                                      excess_height // 2 - (excess_height % 2), excess_width // 2: gaussian_filter_2d.shape[1] - excess_width // 2 - (excess_width % 2)]

    inv_resized_gf2d = np.add(1, -resized_gf2d)

    alpha = 1.1
    beta = -.6
    modified_inv_resized_gf2d = alpha * inv_resized_gf2d + beta

    fft2_shifted_filtered = np.multiply(
        modified_inv_resized_gf2d, fft2_shifted)

    ifft2_filtered_after_shifting = np.fft.ifftshift(fft2_shifted_filtered)

    ifft2_img_filtered_after_shifting = np.exp(
        np.real(np.fft.ifft2(ifft2_filtered_after_shifting))) - 1

    return ifft2_img_filtered_after_shifting



def keep_biggest_blobs(img, blobs_to_keep=1, connectivity=4, keep_original_values=False):
    if img.ndim == 3:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    else:
        gray = img

    _, binary = cv.threshold(gray, 1, 255, cv.THRESH_BINARY)

    [_, blobs_labeled_img] = cv.connectedComponents(
        binary, connectivity=connectivity)

    unique, counts = np.unique(blobs_labeled_img, return_counts=True)

    label_to_count = list(zip(counts, unique))
    label_to_count = sorted(label_to_count, reverse=True)
    labels_to_keep = []
    blobs_sizes = []
    for (count, label) in label_to_count:
        if np.max(binary[label == blobs_labeled_img]) != 0:
            labels_to_keep.append(label)
            blobs_sizes.append(count)
        if len(labels_to_keep) == blobs_to_keep:
            break

    labels_to_keep = np.array(labels_to_keep, dtype=np.int32)

    small_blobs_removed = np.where(np.any(
        (blobs_labeled_img == labels_to_keep.reshape(-1, 1, 1)), axis=0), binary if not keep_original_values else gray, 0)

    return np.array(small_blobs_removed, dtype=np.uint8), np.array(blobs_sizes, dtype=np.int32)
