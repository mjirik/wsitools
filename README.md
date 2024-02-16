# wsitools
GitHub repository for wsitools project. WSI stands for Whole Slide Image.

## Description
The main goal is to process and afterwards convert medical histological images in .czi and .ndpi formats on standard picture
formats such as .jpg or .png.

## Google Colab notebook
You can try demo in Google Colab: https://colab.research.google.com/drive/1kDjsbPCEFS65AdRftMJ5JoS8pr350pbW?usp=sharing

## Installation
```commandline
pip install openslide-python imagecodecs tensorflow loguru read_roi czifile
pip install git+https://github.com/mjirik/imma.git

apt-get install openslide-tools
pip install openslide-python
```

```commandline
pip install git+https://github.com/mjirik/wsitools.git
```

## Usage
```commandline
import wsitools.image
```

```commandline
anim = wsitools.image.AnnotatedImage(filename)
view = anim.get_full_view(
    pixelsize_mm=[0.0003, 0.0003]
    # pixelsize_mm=[0.01, 0.01]
)  # wanted pixelsize in mm in view
img = view.get_raster_image()
plt.imshow(img)
plt.show()
```

## Output
The output is picture in .png format.
![alt text](https://github.com/mjirik/wsitools/blob/main/graphics/cell_nuclei.png?raw=true)
![alt text](https://github.com/mjirik/wsitools/blob/main/graphics/cell_nuclei_2.png?raw=true)



