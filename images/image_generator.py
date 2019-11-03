"""
IMAGE GENERATOR

Takes in font files, outputs distorted image files for training

Based mostly on code from IBM/tensorflow-hangul-recognition project
"""

import argparse
import glob
import io
import os
import random

import numpy as np
import pandas as pd
from PIL import Image, ImageFont, ImageDraw
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter


SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))

# Default data paths.
DEFAULT_LABEL_FILE = os.path.join(SCRIPT_PATH,
                                  '../labels/labels_256_comma_sep.txt')
DEFAULT_FONTS_DIR = os.path.join(SCRIPT_PATH, '../fonts/all_fonts/')
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_PATH, '../images')

# Number of random distortion images to generate per font and character.
DISTORTION_COUNT = 5

# Width and height of the resulting image.
IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64

def increment_count(count):
    if count % 5000 == 0:
        print(f'{count} images generated...')
    count += 1
    return count

def get_labels(label_file):
    labels = pd.read_csv(label_file, encoding='utf-16')
    nlabels = len(labels)
    labels = zip(labels['index'], labels['char'])
    return labels, nlabels

def get_fonts(font_dir):
    fonts = glob.glob(os.path.join(font_dir, '*.[ot]tf'))
    nfonts = len(fonts)
    return fonts, nfonts

def draw_image(character, font):
    image = Image.new('L', (IMAGE_WIDTH, IMAGE_HEIGHT), color=255)
    font = ImageFont.truetype(font, 44)
    drawing = ImageDraw.Draw(image)
    w, h = drawing.textsize(character, font=font)
    drawing.text(
        ((IMAGE_WIDTH - w) / 2, (IMAGE_HEIGHT - h) / 2),
        character,
        fill=(0),
        font=font
    )
    return image

def save_image(image, count, image_dir=DEFAULT_OUTPUT_DIR):
    file_string = f'hangul_{count}.jpeg'
    file_path = os.path.join(image_dir, file_string)
    image.save(file_path, 'JPEG')
    return file_path

def generate_hangul_images(label_file, fonts_dir, output_dir):
    """Generate Hangul image files.
    This will take in the passed in labels file and will generate several
    images using the font files provided in the font directory. The font
    directory is expected to be populated with *.ttf (True Type Font) files.
    The generated images will be stored in the given output directory. Image
    paths will have their corresponding labels listed in a CSV file.
    """
    labels, nlabels = get_labels(label_file)

    image_dir = os.path.join(output_dir, 'hangul-images')
    if not os.path.exists(image_dir):
        os.makedirs(os.path.join(image_dir))

    # Get a list of the fonts.
    fonts, nfonts = get_fonts(fonts_dir)

    labels_csv = io.open(os.path.join(output_dir, 'image-label-map.csv'),
                         'w', encoding='utf-16')

    total_count = 0
    print(f'Total number of images: {nlabels*nfonts*(DISTORTION_COUNT+1)}')
    for index, character in labels:
        for font in fonts:
            # Print image count roughly every 5000 images.
            total_count = increment_count(total_count)

            image = draw_image(character, font)

            file_name = save_image(image, total_count, image_dir)
            labels_csv.write(f'{file_name},{index},{character}\n')

            for i in range(DISTORTION_COUNT):
                total_count = increment_count(total_count)

                # invert the raw array to avoid black borders on distort
                arr = np.invert(np.array(image))
                distorted_array = elastic_distort(
                    arr, alpha=random.randint(30, 36),
                    sigma=random.randint(5, 6)
                )
                # then invert it back to black text/white background
                distorted_array = np.invert(distorted_array)
                distorted_image = Image.fromarray(distorted_array)
                file_path = save_image(distorted_image, total_count, image_dir)
                labels_csv.write(f'{file_path},{index},{character}\n')

    print(f'Finished generating {total_count} images.')
    labels_csv.close()
    return

def elastic_distort(image, alpha, sigma):
    """Perform elastic distortion on an image.
    Here, alpha refers to the scaling factor that controls the intensity of the
    deformation. The sigma variable refers to the Gaussian filter standard
    deviation.
    """
    random_state = np.random.RandomState(None)
    shape = image.shape

    dx = gaussian_filter(
        (random_state.rand(*shape) * 2 - 1),
        sigma, mode="constant"
    ) * alpha
    dy = gaussian_filter(
        (random_state.rand(*shape) * 2 - 1),
        sigma, mode="constant"
    ) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))
    return map_coordinates(image, indices, order=1).reshape(shape)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--label-file', type=str, dest='label_file',
                        default=DEFAULT_LABEL_FILE,
                        help='File containing newline delimited labels.')
    parser.add_argument('--font-dir', type=str, dest='fonts_dir',
                        default=DEFAULT_FONTS_DIR,
                        help='Directory of ttf fonts to use.')
    parser.add_argument('--output-dir', type=str, dest='output_dir',
                        default=DEFAULT_OUTPUT_DIR,
                        help='Output directory to store generated images and '
                             'label CSV file.')
    args = parser.parse_args()
    generate_hangul_images(args.label_file, args.fonts_dir, args.output_dir)