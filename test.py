from bing_image_downloader import downloader
from PIL import Image
import os
import shutil


def scrape_images_for_celebrity(celebrity_name):
    output_dir = './temp_images'
    downloader.download(celebrity_name, limit=2, output_dir=output_dir,
                        adult_filter_off=True, force_replace=False, timeout=60)

    images = []
    image_dir = os.path.join(output_dir, celebrity_name) 
    for filename in os.listdir(image_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(image_dir, filename)
            image = Image.open(image_path)
            images.append(image)
    # Clean up temporary directory
    shutil.rmtree(output_dir)

    return images


# Example usage
celebrity_name = "Tom Cruise"
images = scrape_images_for_celebrity(celebrity_name)
print(images)