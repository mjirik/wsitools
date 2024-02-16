import requests
import image
import matplotlib.pyplot as plt

# Possible filenames: J7_5_a.czi, J7_25_a_ann0004.czi, J8_8_a.czi
filename = "J7_25_a_ann0004.czi"

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

# Now you can use the local file in your code
anim = image.AnnotatedImage(filename)

# Annotations
# print(anim.annotations)

view = anim.get_full_view(
    pixelsize_mm=[0.0003, 0.0003]
    # pixelsize_mm=[0.01, 0.01]
)  # wanted pixelsize in mm in view
img = view.get_raster_image()
plt.imshow(img)
plt.show()
