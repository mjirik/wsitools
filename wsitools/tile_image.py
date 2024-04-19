import numpy as np
import cv2
import os
from pathlib import Path
import tqdm

from . import image


class ImageSplitterMerger(object):
    """Class which represents splitter and merger of the image"""

    def __init__(self, img_path: Path, tilesize_px: int, overlap_px: int, pixelsize_mm: list, fcn=None):
        """Initialize the ImageSplitterMerger object with the specified parameters."""
        self.img_path = img_path
        self.tilesize_px = tilesize_px
        self.overlap_px = overlap_px
        self.pixelsize_mm = pixelsize_mm
        self.fcn = fcn

        # Load image and initialize properties
        anim = self.load_image(img_path)
        view = anim.get_full_view(pixelsize_mm=pixelsize_mm[0])
        shape = view.get_size_on_pixelsize_mm()  # returns rows and cols
        shape = np.append(shape, 3)  # Added image channel = 3

        setattr(self, "view", view)
        setattr(self, "anim", anim)
        setattr(self, "img_shape", shape)  # 1st element num_columns, 2nd element num_rows (probably)

    def get_num_cols_rows(self, img_shape: list) -> tuple:
        """Calculate the number of rows and columns based on image shape."""
        tilesize_px = self.tilesize_px
        overlap_px = self.overlap_px

        # img_shape: 1st element num_columns, 2nd element num_rows (probably)
        num_rows = int(np.ceil(img_shape[1] / (tilesize_px - overlap_px)))
        num_cols = int(np.ceil(img_shape[0] / (tilesize_px - overlap_px)))

        return num_rows, num_cols

    @staticmethod
    def load_image(img_path: Path) -> image.AnnotatedImage:
        """Load an .czi image using the specified path."""
        img_path_str = str(img_path)
        print(os.path.exists(img_path_str))
        anim = image.AnnotatedImage(path=img_path_str)

        return anim

    def get_number_tiles(self, shape: list) -> int:
        """Calculate the total number of tiles."""
        nx, ny = self.get_num_cols_rows(shape)
        return nx * ny

    def split_iterator(self) -> np.array:
        """Split image into tiles and yield each tile."""
        img_shape = self.img_shape  # 1st element num_columns, 2nd element num_rows (probably)
        tilesize_px = self.tilesize_px
        overlap_px = self.overlap_px

        (num_rows, num_cols) = self.get_num_cols_rows(img_shape)

        for i in range(num_rows):
            for j in range(num_cols):
                row_start = i * (tilesize_px - overlap_px)
                row_end = row_start + tilesize_px

                col_start = j * (tilesize_px - overlap_px)
                col_end = col_start + tilesize_px

                # New tile with overlap for each iteration
                overlapped_tile = self.get_tile_overlap(col_end, col_start, overlap_px, row_end,
                                                        row_start, img_shape)

                yield overlapped_tile

    def get_tile_overlap(self, col_end: int, col_start: int, overlap_px: int, row_end: int, row_start: int,
                         img_shape: np.array) -> np.array:
        """Returns a tile with the specified overlap."""
        pixelsize_mm = self.pixelsize_mm

        # Calculate the valid region to copy from the original image
        img_row_start = max(0, row_start - overlap_px)
        img_row_end = min(img_shape[0], row_end + overlap_px)
        img_col_start = max(0, col_start - overlap_px)
        img_col_end = min(img_shape[1], col_end + overlap_px)

        # Get the corresponding region in the tile with overlap
        view = self.anim.get_view(
            location_mm=(img_col_start * pixelsize_mm[0], img_row_start * pixelsize_mm[1]),
            pixelsize_mm=pixelsize_mm,
            size_on_level=(self.tilesize_px, self.tilesize_px)
        )

        overlapped_tile = view.get_region_image(as_gray=False)

        return overlapped_tile

    def merge_tiles_to_image(self, tiles: list) -> np.array:
        """Merge tiles into an image and remove overlap."""
        tilesize_px = self.tilesize_px
        overlap_px = self.overlap_px
        img_shape = self.img_shape  # 1st element num_columns, 2nd element num_rows (probably)
        (num_rows, num_cols) = self.get_num_cols_rows(img_shape)

        # Initialize the merged image
        merged_image = np.zeros(img_shape, dtype="uint8").transpose(1, 0, 2)  # to get num_rows on index 0 and num_columns on index 1

        with tqdm.tqdm(total=num_rows * num_cols, desc="Merging Tiles") as pbar:
            for i in range(num_rows):
                for j in range(num_cols):
                    idx = i * num_cols + j

                    row_start = i * (tilesize_px - overlap_px)
                    row_end = min(row_start + tilesize_px, merged_image.shape[0])

                    col_start = j * (tilesize_px - overlap_px)
                    col_end = min(col_start + tilesize_px, merged_image.shape[1])

                    # Remove overlap from all sides of the tile
                    tile_no_overlap = tiles[idx][overlap_px:(overlap_px + row_end - row_start),
                                      overlap_px:(overlap_px + col_end - col_start)]

                    # Calculate the corresponding region in the merged image
                    merged_row_start = i * (tilesize_px - overlap_px)
                    merged_row_end = merged_row_start + row_end - row_start
                    merged_col_start = j * (tilesize_px - overlap_px)
                    merged_col_end = merged_col_start + col_end - col_start

                    # Copy the tile without overlap to the merged image
                    merged_image[merged_row_start:merged_row_end, merged_col_start:merged_col_end] = tile_no_overlap
                    pbar.update(1)

        return merged_image

    def split_and_merge_image(self) -> np.array:
        """Split and merge image, process each tile."""
        processed_tiles = []
        total_tiles = self.get_number_tiles(self.img_shape)

        for tile in tqdm.tqdm(self.split_iterator(), total=total_tiles, desc="Splitting and Processing Tiles"):
            if self.fcn is not None:
                processed_tile = self.fcn(tile)

                if processed_tile.dtype != np.uint8:
                    processed_tile_normalized = normalize_tile(processed_tile)
                    processed_tiles.append(processed_tile_normalized)

                else:
                    processed_tiles.append(processed_tile)

            else:
                if tile.dtype != np.uint8:
                    tile_normalized = normalize_tile(tile)
                    processed_tiles.append(tile_normalized)
                else:
                    processed_tiles.append(tile)

        merged_img = self.merge_tiles_to_image(processed_tiles)

        return merged_img


def normalize_tile(tile: np.array) -> np.array:
    # Check if the image is mostly white
    mean_intensity = np.mean(tile)
    std_dev = np.std(tile)
    is_white = mean_intensity >= 0.99 and std_dev <= 0.01  # Adjust thresholds as needed
    tile_normalized = np.full_like(tile, 255.0, dtype=np.uint8)

    if not is_white:
        # Normalize to 0-255 range
        tile_normalized = cv2.normalize(tile, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_32F)

        # Convert to uint8 for compatibility (optional)
        tile_normalized = tile_normalized.astype(np.uint8)

    return tile_normalized


def load_image(img_path: Path) -> np.array:
    """Load an image using OpenCV."""
    img = cv2.imread(str(img_path))
    return img


def create_test_image() -> np.array:
    """Create a test image (2D array) for testing purposes."""
    x, y = np.indices([300, 500])
    center1 = (256, 256)
    mask = (x - center1[0]) ** 2 + (y - center1[1]) ** 2

    return mask
