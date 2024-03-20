# Thanks to https://community.openai.com/t/how-to-load-a-local-image-to-gpt4-vision-using-api/533090/5
import io
from PIL import Image
import base64

def _resize_image(image, max_dimension):
    width, height = image.size

    # Check if the image has a palette and convert it to true color mode
    if image.mode == "P":
        if "transparency" in image.info:
            image = image.convert("RGBA")
        else:
            image = image.convert("RGB")

    if width > max_dimension or height > max_dimension:
        if width > height:
            new_width = max_dimension
            new_height = int(height * (max_dimension / width))
        else:
            new_height = max_dimension
            new_width = int(width * (max_dimension / height))
        image = image.resize((new_width, new_height), Image.LANCZOS)
        
    return image

def _convert_to_png(image):
    with io.BytesIO() as output:
        image.save(output, format="PNG")
        return output.getvalue()

def process_image(image, max_size=None):
    if max_size is not None:
        image = _resize_image(image, max_size)
    png_image = _convert_to_png(image)
    return base64.b64encode(png_image).decode('utf-8')
