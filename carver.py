import sys
from tqdm import trange # Used to display a progress bar for loops
import numpy as np
from imageio import imread, imwrite # For reading and writing image files
from scipy.ndimage.filters import convolve # For applying convolution to the image

def calc_energy(img):
    # Define horizontal (du) and vertical (dv) gradient filters for edge detection
    filter_du = np.array([
        [1.0, 2.0, 1.0],
        [0.0, 0.0, 0.0],
        [-1.0, -2.0, -1.0],
    ])
     # Extend the filter to apply it to all color channels (R, G, B)
    filter_du = np.stack([filter_du] * 3, axis=2)

    filter_dv = np.array([
        [1.0, 0.0, -1.0],
        [2.0, 0.0, -2.0],
        [1.0, 0.0, -1.0],
    ])
    # Extend the filter to apply it to all color channels (R, G, B)
    filter_dv = np.stack([filter_dv] * 3, axis=2)

     # Convert the image to float32 for processing
    img = img.astype('float32')

      # Compute the gradient magnitudes by convolving the image with both filters
    convolved = np.absolute(convolve(img, filter_du)) + np.absolute(convolve(img, filter_dv))

    # Compute the energy map by summing the gradients across all color channels
    energy_map = convolved.sum(axis=2)

    return energy_map

def crop_c(img, scale_c):
    # Get the current image dimensions and compute the new width
    r, c, _ = img.shape
    new_c = int(scale_c * c)

    # Iteratively remove columns until the desired width is achieved
    for i in trange(c - new_c):
        img = carve_column(img)

    return img

def crop_r(img, scale_r):
    # Rotate the image 90 degrees counterclockwise to treat rows as columns
    img = np.rot90(img, 1, (0, 1))
    # Use the column cropping function to remove rows
    img = crop_c(img, scale_r)
    # Rotate the image back to its original orientation
    img = np.rot90(img, 3, (0, 1))
    return img

def carve_column(img):\
    # Get the image dimensions
    r, c, _ = img.shape

    # Compute the minimum seam and its backtracking path
    M, backtrack = minimum_seam(img)

    # Create a mask to remove the seam from the image
    mask = np.ones((r, c), dtype=np.bool)

    # Find the column index of the minimum energy pixel in the last row
    j = np.argmin(M[-1])
    # Backtrack through the seam and update the mask
    for i in reversed(range(r)):
        mask[i, j] = False
        j = backtrack[i, j]
     # Apply the mask to remove the seam and reshape the image
    mask = np.stack([mask] * 3, axis=2)
    img = img[mask].reshape((r, c - 1, 3))
    return img

def minimum_seam(img):
    # Get the image dimensions
    r, c, _ = img.shape
    # Calculate the energy map of the image
    energy_map = calc_energy(img)
    # Initialize the dynamic programming table (M) and backtracking table
    M = energy_map.copy()
    backtrack = np.zeros_like(M, dtype=int)

    # Populate the DP table
    for i in range(1, r):
        for j in range(0, c):
            # Handle the left edge of the image, to ensure we don't index a - 1 (no left diagonal)
            if j == 0:
                idx = np.argmin(M[i-1, j:j + 2])
                backtrack[i, j] = idx + j
                min_energy = M[i-1, idx + j]
            else:
                # General case: consider left diagonal, top, and right diagonal
                idx = np.argmin(M[i - 1, j - 1:j + 2])
                backtrack[i, j] = idx + j - 1
                min_energy = M[i - 1, idx + j - 1]
            # Update the DP table with the minimum energy path
            M[i, j] += min_energy

    return M, backtrack

def main():
    if len(sys.argv) != 5:
        print('usage: carver.py <r/c> <scale> <image_in> <image_out>', file=sys.stderr)
        sys.exit(1)

    # Parse command-line arguments
    which_axis = sys.argv[1]
    scale = float(sys.argv[2])
    in_filename = sys.argv[3]
    out_filename = sys.argv[4]

    # Read the input image
    img = imread(in_filename)

    # Perform cropping based on the specified axis 
    if which_axis == 'r':
        out = crop_r(img, scale)
    elif which_axis == 'c':
        out = crop_c(img, scale)
    else:
        print('usage: carver.py <r/c> <scale> <image_in> <image_out>', file=sys.stderr)
        sys.exit(1)

    # Write the cropped image to the output file
    imwrite(out_filename, out)

# This is the entire point of this script
if __name__ == '__main__':
    main()