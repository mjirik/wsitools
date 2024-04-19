import requests
import matplotlib.pyplot as plt
import numpy as np

from wsitools.tile_image import ImageSplitterMerger

def process_tile_test(tile: np.array) -> np.array:
    """
    Process a tile by drawing a red square on it.

    Parameters:
    - tile (np.array): Input tile image (assumed to be in RGB format).

    Returns:
    - np.array: Processed tile with a red square drawn on it.
    """
    if tile.shape[2] != 3:
        raise ValueError("Image ndarray must have 3 channels for RGB.")

    # Create a copy of the image to avoid modifying the original array
    result_tile = np.copy(tile)

    # Set the red color (assuming RGB format)
    red_color = [255, 0, 0]

    # Draw the red square
    result_tile[5:5 + 5, 5:5 + 5, :] = red_color

    return result_tile


if __name__ == '__main__':
    # Define the path to the CZI file
    # Possible filenames: J7_5_a.czi, J7_25_a_ann0004.czi, J8_8_a.czi
    filename = "J7_5_a.czi"

    # URL of the file on GitHub
    url_path = "https://github.com/janburian/Masters_thesis/raw/main/data_czi/" + filename

    # Fetch the file
    response = requests.get(url_path)

    # Check if the request was successful
    if response.status_code == 200:
        # Save the content to a local file
        with open(filename, "wb") as file:
            file.write(response.content)
    else:
        print("Failed to fetch the file from GitHub")

    # Create an ImageSplitterMerger instance with the specified parameters
    image = ImageSplitterMerger(filename, tilesize_px=200, overlap_px=0, pixelsize_mm=[0.01, 0.01],
                                fcn=process_tile_test)

    # Split and merge the image, applying the specified tile processing function
    merged_image = image.split_and_merge_image()

    # Display the input and merged images using Matplotlib
    plt.imshow(merged_image)
    plt.title("Merged picture")
    plt.show()

    # Save the merged image as a PNG file
    # plt.imsave("output.png", merged_image)
