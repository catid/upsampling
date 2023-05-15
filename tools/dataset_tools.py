import random, time, os

from PIL import Image

def create_super_random_threadsafe_generator():
    current_time_ns = int(time.time_ns())

    # Get random bytes from the operating system
    random_bytes = os.urandom(8)  # Read 8 bytes of random data
    random_int = int.from_bytes(random_bytes, 'big')  # Convert the random bytes to an integer

    # Combine the current time and random bytes to create a seed
    seed = current_time_ns ^ random_int

    rng = random.Random()
    rng.seed(seed)
    return rng

def check_overlap(rect1, rect2):
    (x1, y1), (x2, y2) = rect1
    (a1, b1), (a2, b2) = rect2
    return not (x2 <= a1 or a2 <= x1 or y2 <= b1 or b2 <= y1)

def generate_non_overlapping_rectangles(rng, image_width, image_height, crop_width=256, crop_height=256, max_attempts=10000):
    rectangles = []

    if crop_width > image_width or crop_height > image_height:
        raise ValueError(f"The crop dimensions {crop_width}x{crop_height} cannot be larger than the image dimensions {image_height}x{image_height}.")

    attempts = 0
    while attempts < max_attempts:
        x = rng.randint(0, image_width - crop_width)
        y = rng.randint(0, image_height - crop_height)

        new_rect = ((x, y), (x + crop_width, y + crop_height))
        overlap = any(check_overlap(new_rect, rect) for rect in rectangles)

        if not overlap:
            rectangles.append(new_rect)

        attempts += 1

    return rectangles

def save_random_image_crops_to_disk(img, image_name, label_output_dir, crop_width=256, crop_height=256, downsample_first=True):
    w, h = img.size

    if w < crop_width or h < crop_height:
        raise ValueError(f"Input image {image_name} was too small to import")

    scale = 1

    if downsample_first:
        w = w//2
        h = h//2
        # First downsample the frame 2x to eliminate chroma subsampling and compression artifacts
        img = img.resize((w, h), resample=Image.BICUBIC)
        scale *= 2

    rng = create_super_random_threadsafe_generator()

    while True:
        rectangles = generate_non_overlapping_rectangles(rng, w, h, crop_width, crop_height)

        for idx, rect in enumerate(rectangles):
            (x1, y1), (x2, y2) = rect
            cropped_img = img.crop((x1, y1, x2, y2))

            output_file = os.path.join(label_output_dir, f"frame_{image_name}_scale{scale}_part{idx}.png")

            cropped_img.save(output_file, 'PNG')

        # Downsample the frame 2x to eliminate chroma subsampling and compression artifacts
        w = w//2
        h = h//2
        scale *= 2

        # Stop here if we cannot make more crops of the required size
        if w < crop_width or h < crop_height:
            break

        img = img.resize((w, h), resample=Image.BICUBIC)
