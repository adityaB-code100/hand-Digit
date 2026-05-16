import base64
import io
from PIL import Image, ImageOps
from torchvision import transforms

def base64_to_tensor(base64_string, target_size=(64, 64)):
    """
    Decodes a base64 string from the canvas, crops/centers the drawn digit,
    and applies the exact validation transforms used during training.
    """
    # Remove header if present
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]

    # Decode image
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))

    # Convert to grayscale first (canvas is usually RGBA, we want to extract the drawing)
    # The drawing is white on transparent or white on black depending on frontend, 
    # but we assume white/light drawing on transparent background for canvas.
    if image.mode == 'RGBA':
        # Create a black background image
        background = Image.new('RGB', image.size, (0, 0, 0))
        # Paste the image using alpha channel as mask
        background.paste(image, mask=image.split()[3])
        image = background

    # The training used RGB images, so ensure we have RGB
    image = image.convert('RGB')
    
    # We want to crop to the bounding box of the drawn digit to mimic MNIST/custom dataset style
    # Convert to grayscale to find bounding box
    gray = image.convert('L')
    bbox = gray.getbbox()
    
    if bbox:
        # Crop to the bounding box
        image = image.crop(bbox)
        # Pad slightly to keep aspect ratio and not touch borders
        padding = 20
        image = ImageOps.expand(image, border=padding, fill='black')

    # Apply exactly the same transforms as during training
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor
