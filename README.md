# wsitools
GitHub repository for wsitools project. WSI stands for Whole Slide Image.

## Description
The main goal is to process and afterwards convert medical histological images in .czi and .ndpi formats on standard picture
formats such as .jpg or .png. Thanks to this module it is also possible to split large .czi file into tiles.
Each tile (np.array format) can be processed separately. It is possible to write own function for their processing. This attitude is good mainly because the size of .czi file could be
even several gigabytes, so it can happen that the file does not fit into memory. After tile processing these tiles are merged together.
Merged tiles create picture in .png format.


## Google Colab notebooks
You can try several demos:
* Basic introduction to wsitools module: https://colab.research.google.com/drive/1kDjsbPCEFS65AdRftMJ5JoS8pr350pbW?usp=sharing
* Split and merge medical image with tile processing: https://colab.research.google.com/drive/1x9fY0MBUAcPELJuKJl31Hh6i_LYV89Qs?usp=sharing
* Split and merge with [livergan](https://github.com/VaJavorek/livergan) processing: https://colab.research.google.com/drive/1DDdi8x68rShRWBMQPJ9aq9zxtGmUsFqM?usp=sharing

## Installation
```commandline
pip install openslide-python openslide-bin imagecodecs loguru read_roi czifile
pip install git+https://github.com/mjirik/imma.git
[//]: # (apt-get install openslide-tools)

[//]: # (pip install openslide-python)
```


[](https://github.com/VaJavorek/livergan/blob/main/img/comparison/Comparison_PIG-002_J-18-0092_HE__-1_split_1200.png?raw=true)




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
![alt text](https://github.com/mjirik/wsitools/blob/main/graphics/cell_nuclei.png?raw=true)
![alt text](https://github.com/mjirik/wsitools/blob/main/graphics/cell_nuclei_2.png?raw=true)



