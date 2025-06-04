#!/usr/bin/env python3
import hashlib
import os
import sys
import time
import sqlite3
import subprocess
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image, ExifTags
from pillow_heif import register_heif_opener

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='pygame.pkgdata')

import pygame

import exifread

import imagehash

from ipdb import iex, set_trace as db

from dotenv import load_dotenv

load_dotenv(sys.path[1] + "/.env")
DB_FILE = os.getenv("DB_FILE")
print(DB_FILE)

allowed_extensions = ['jpg', 'jpeg', 'png', 'gif', 'tif', 'bmp', 'heic']

register_heif_opener()

window_name = 'canonificator'

# canonical = 0: noncanonical
# canonical = 1: canonical
# canonical = 2: canonical 10% thumbnail
# processed = 0: unprocessed
# processed = 1: processed
# processed = 2: error
# processed = 3: deleted
# processed = 4: rescued (not implemented yet)

def initialize_database():
    """Initialize the database schema."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    # TODO add fields: file size
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            filepath TEXT NOT NULL UNIQUE,
            filesize INTEGER,
            width INTEGER,
            height INTEGER,
            md5sum TEXT,
            phash TEXT,
            camera_model TEXT,
            processed INTEGER DEFAULT 0,
            canonical INTEGER DEFAULT 0
        )
    """)
    conn.commit()
    conn.close()

def add_image_paths(directories):
    """Add image paths to the database."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    for directory, is_canonical in directories:
        print(f"Scanning directory: {directory} (Canonical: {is_canonical})")
        for root, _, files in os.walk(directory):
            for f in files:
                if '.' not in f:
                    continue
                parts = f.split('.')
                ext = parts[-1]
                if ext.lower() not in allowed_extensions:
                    continue
                file_path = os.path.join(root, f)
                try:
                    cursor.execute("""
                        INSERT OR IGNORE INTO images (filename, filepath, canonical)
                        VALUES (?, ?, ?)
                    """, (f, file_path, int(is_canonical)))
                except Exception as e:
                    print(f"Error adding {file_path}: {e}")

    conn.commit()
    conn.close()
    print("All image paths have been added to the database.")

# TODO: mark images with missing camera model as errored

def process_images(batch_size=100):
    """Process unprocessed images from the database."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    t0 = time.time()
    total_rows = 0
    while True:
        t1 = time.time()
        cursor.execute("""
            SELECT id, filepath FROM images WHERE processed == 2 LIMIT ?
        """, (batch_size,))
        rows = cursor.fetchall()

        if not rows:
            print("All images have been processed.")
            break

        for row in rows:
            image_id, file_path = row
            try:
                with Image.open(file_path) as img:
                    width, height = img.width, img.height
                    phash = str(imagehash.phash(img))
                    if file_path.lower().endswith(".heic"):
                        camera_model = get_camera_model_exiftool(file_path)
                    else:
                        camera_model = get_camera_model_pil(img)

                filesize = os.path.getsize(file_path)

                md5sum = calculate_md5(file_path)

                # Update the database with image properties
                cursor.execute("""
                    UPDATE images
                    SET filesize = ?, width = ?, height = ?, md5sum = ?, phash = ?, camera_model = ?, processed = 1
                    WHERE id = ?
                """, (filesize, width, height, md5sum, phash, camera_model, image_id))

            except FileNotFoundError as exc:
                # processed = 3 -> deleted
                cursor.execute("""
                    UPDATE images
                    SET processed = 3
                    WHERE id = ?
                """, (image_id,))
                print(f"file deleted: {file_path}")

            except Exception as exc:
                # processed = 2 -> error
                cursor.execute("""
                    UPDATE images
                    SET processed = 2
                    WHERE id = ?
                """, (image_id,))
                print(f"Error processing {file_path}: {exc}")
                #db()

        conn.commit()
        total_rows += len(rows)
        t2 = time.time()
        batch_duration = t2 - t1
        total_duration = t2 - t0
        print(f"Processed {len(rows)} images in {batch_duration:.2f}s, {total_rows} in {total_duration:.2f}s...")
        db()

    conn.close()

def calculate_md5(file_path):
    """Calculate the MD5 checksum of a file."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def get_camera_model_exiftool(filepath):
    cmd = 'exiftool -Model "%s"' % filepath
    res = subprocess.check_output(cmd, shell=True)
    str_res = res.decode('utf-8')

    #lines = str_res.split('\n')
    #for line in lines:

    parts = str_res.split(':')
    if len(parts) != 2:
        db()

    model = parts[1].strip()
    return model

def get_camera_model_exifread(filepath):
    """Extract EXIF data from an image, including HEIC files."""
    with open(filepath, "rb") as f:
        db()
        tags = exifread.process_file(f, stop_tag="Camera Model Name")
        camera = tags.get("Image Make", "Unknown").values
        return camera

def get_camera_model_pil(image):
    """Extract camera model from EXIF metadata."""
    exif_data = image._getexif()
    if exif_data:
        for tag, value in exif_data.items():
            if ExifTags.TAGS.get(tag) == "Model":
                return value

def find_duplicates():
    """Identify non-canonical images that are duplicates of canonical images."""
    
    # Find duplicates based on MD5 or perceptual hash
    #nc_search_prefix = '/home/alan/sync/meta-mac/alan-iphone/photos/'
    #MANUAL_CHECK = False
    #DELETE_ALL = True   
    
    nc_search_prefix = '/home/alan/sync-recovery/photorec/recup_dir.418'
    MANUAL_CHECK = True
    DELETE_ALL = False

    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row  # Enable row access by name
    cursor = conn.cursor()

    query = (
    "SELECT nc.id, nc.filepath  "
    "FROM images AS nc "
    f"WHERE nc.filepath like '{nc_search_prefix}%' "
    "ORDER BY nc.filepath "
    )

    t0 = time.time()
    cursor.execute(query)
    t1 = time.time()
    prefixed = cursor.fetchall()
    t2 = time.time()
    print(f"select prefixed: {t1-t0:.2f}s execute, {t2-t1:.2f}s fetch")

    print(f"found {len(prefixed)} non-canonical files with prefix {nc_search_prefix}")

    query_name = "non-canonical duplicates"
    query = (
    "SELECT "
    "nc.id AS nc_id, c.id AS c_id, "
    "nc.filepath AS nc_filepath, c.filepath AS c_filepath, "
    "nc.md5sum AS nc_md5sum, c.md5sum AS c_md5sum, "
    "nc.phash AS nc_phash, c.phash AS c_phash, "
    "nc.width AS nc_width, c.width AS c_width, "
    "nc.height AS nc_height, c.height AS c_height, "
    "nc.filesize AS nc_filesize, c.filesize AS c_filesize "
    "FROM images AS nc "
    "JOIN images AS c "
    "ON (nc.md5sum = c.md5sum OR nc.phash = c.phash) "
    "WHERE nc.canonical = 0 AND c.canonical = 1 "
    f"AND nc.filepath LIKE '{nc_search_prefix}%' "
    "ORDER BY nc.filepath"
    )    
    # AND c.md5sum LIKE 'a%'

    t0 = time.time()
    cursor.execute(query)
    t1 = time.time()
    pairs = cursor.fetchall()
    t2 = time.time()
    print(f"select duplicates: {t1-t0:.2f}s execute, {t2-t1:.2f}s fetch")
    
    check_pairs(pairs, query_name, cursor, MANUAL_CHECK, DELETE_ALL)
    conn.close()

def find_rescues():
    """Identify non-canonical images that are candidates for rescue."""
    pass

def check_pairs(pairs, query_name, cursor, MANUAL_CHECK=True, DELETE_ALL=False):

    if not pairs:
        print("no pairs")
        return

    #db()
    print(f"Found {len(pairs)} pairs ({query_name}):")
    #pygame.init()
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 3000, 1000)
    for n, pair in enumerate(pairs):
        row_dict = dict(pair)
        nc_dict = {k[3:]: v for k, v in row_dict.items() if k.startswith('nc_')}
        c_dict = {k[2:]: v for k, v in row_dict.items() if k.startswith('c_')}
        nc = SimpleNamespace(**nc_dict)
        c = SimpleNamespace(**c_dict)

        md5_match = nc.md5sum == c.md5sum
        phash_match = nc.phash == c.phash
        dimension_match = (nc.width, nc.height) == (c.width, c.height)
        filesize_match = nc.filesize == c.filesize

        caption_dict = {
            'md5_match': md5_match,
            'phash_match': phash_match,
            'dimension_match': dimension_match,
            'filesize_match': filesize_match,
        }

        print('')
        print(f"{n}/{len(pairs)}")
        print(f"id md5, phash, filepath")
        print(f"{nc.id:07} {nc.md5sum} {nc.phash} {nc.width:4}x{nc.height:4} {nc.filepath}")
        print(f"{ c.id:07} { c.md5sum} { c.phash} { c.width:4}x{ c.height:4} { c.filepath}")

        if not os.path.exists(nc.filepath):
            print("File does not exist, skipping...")
            continue

        should_delete = False
        should_rescue = False
        if MANUAL_CHECK:
            response = display_images_opencv(nc.filepath, c.filepath, caption_dict)
            focus_window(window_name)
            if response == 'q':
                print("Quitting manual check.")
                break
            elif response == 'd':
                print('delete non-canonical item')
                should_delete = True
            elif response == 's':
                print('skip pair')
            elif response == 'r':
                print('rescue non-canonical item')
                should_rescue = True

        if should_rescue:
            # use the tba/photos-10 location to figure out what the original tba/photos should be,
            # then copy the recovered nc file to the canonical location
            print("Rescue operation not implemented yet.")

            #cursor.execute("""
            #    UPDATE images
            #    SET processed = 4
            #    WHERE id = ?
            # """, (nc.id,))

        if should_delete or DELETE_ALL:
            # processed = 3 -> deleted

            if os.path.exists(nc.filepath):
                os.remove(nc.filepath)

            cursor.execute("""
                UPDATE images
                SET processed = 3
                WHERE id = ?
                """, (nc.id,))

            print(f"file deleted: {nc.filepath}")


def resize_image(image, max_width, max_height):
    """Resize an image while maintaining aspect ratio to fit within max_width and max_height."""
    img_w, img_h = image.get_size()
    scale = min(max_width / img_w, max_height / img_h)
    new_size = (int(img_w * scale), int(img_h * scale))
    return pygame.transform.scale(image, new_size)

def compute_difference_opencv(img1, img2):
    """Compute absolute difference image, padded to match size."""
    h = max(img1.shape[0], img2.shape[0])
    w = max(img1.shape[1], img2.shape[1])

    def pad_to_shape(img, h, w):
        return cv2.copyMakeBorder(img, 0, h - img.shape[0], 0, w - img.shape[1], cv2.BORDER_CONSTANT, value=0)

    img1_p = pad_to_shape(img1, h, w)
    img2_p = pad_to_shape(img2, h, w)
    diff = cv2.absdiff(img1_p, img2_p)
    return diff

def compute_difference_pygame(image1_path, image2_path):
    """Compute the absolute difference between two images using Pillow + NumPy."""
    img1 = Image.open(image1_path).convert("RGB")
    img2 = Image.open(image2_path).convert("RGB")

    # Convert images to NumPy arrays
    img1_array = np.array(img1)
    img2_array = np.array(img2)

    # Ensure both images are the same size
    if img1_array.shape != img2_array.shape:
        img2 = img2.resize(img1.size, Image.BILINEAR)
        img2_array = np.array(img2)

    # Compute absolute difference
    diff_array = np.abs(img1_array.astype(int) - img2_array.astype(int)).astype(np.uint8)

    # Convert difference array back to a Pygame Surface
    diff_image = Image.fromarray(diff_array)
    return pygame.image.fromstring(diff_image.tobytes(), diff_image.size, diff_image.mode)

def load_image(image_path):
    """Load an image file (supports HEIC) and return a Pygame Surface."""
    if image_path.lower().endswith(".heic"):
        image = Image.open(image_path).convert("RGB")  # Convert HEIC to RGB
    else:
        image = Image.open(image_path)

    return pygame.image.fromstring(image.tobytes(), image.size, image.mode)


screen_w, screen_h = 1920*2, 1080*2


def load_image_as_cv2(path, scale=4):
    """Load any image using Pillow and convert to OpenCV format with downsampling."""
    img = Image.open(path)
    if scale > 1:
        img = img.resize((img.width // scale, img.height // scale), Image.LANCZOS)
    img = img.convert('RGB')  # Ensure 3 channels
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def compute_difference(img1, img2):
    """Compute absolute difference image, padded to match size."""
    h = max(img1.shape[0], img2.shape[0])
    w = max(img1.shape[1], img2.shape[1])

    def pad_to_shape(img, h, w):
        return cv2.copyMakeBorder(img, 0, h - img.shape[0], 0, w - img.shape[1], cv2.BORDER_CONSTANT, value=0)

    img1_p = pad_to_shape(img1, h, w)
    img2_p = pad_to_shape(img2, h, w)
    diff = cv2.absdiff(img1_p, img2_p)
    return img1_p, img2_p, diff

def focus_window(window_title):
    time.sleep(0.2)  # give time for the window to appear
    try:
        subprocess.run(['wmctrl', '-a', window_title])
    except FileNotFoundError:
        print("wmctrl not found")

def add_caption(image, caption_dict):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    color = (255, 255, 255)  # white text
    margin = 10

    color_map = {
        False: (0, 0, 255),  # red for False
        True: (0, 255, 0),   # green for True
    }
    
    x, y = 10, 30
    image = image.copy()

    # Draw background rectangle for caption
    for caption, val in caption_dict.items():
        color = color_map[val]
        (text_w, text_h), _ = cv2.getTextSize(caption, font, font_scale, thickness)

        #cv2.rectangle(image, (x - 5, y - 5), (x + text_w + 5, y + text_h + 5), (0, 0, 0), -1)
        cv2.putText(image, caption, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
        
        y += text_h + margin
    return image

def display_images_opencv(path1, path2, caption="", scale=4):
    """Load images, compute diff, and display side-by-side in OpenCV with caption."""
    img1 = load_image_as_cv2(path1, scale)
    img2 = load_image_as_cv2(path2, scale)
    img1_p, img2_p, diff = compute_difference(img1, img2)

    # Join images horizontally
    combined = cv2.hconcat([img1_p, img2_p, diff])

    img_h, img_w = combined.shape[:2]
    scale_factor = min(screen_w / img_w, screen_h / img_h, 1.0)
    if scale_factor < 1.0:
        new_size = (int(img_w * scale_factor), int(img_h * scale_factor))
        combined = cv2.resize(combined, new_size, interpolation=cv2.INTER_AREA)
        
    captioned = add_caption(combined, caption)

    cv2.imshow(window_name, captioned)
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(f"Key pressed: {chr(key) if 0 < key < 256 else key}")
    return chr(key) if 0 < key < 256 else key

def display_images_pygame(image1_path, image2_path, caption):
    pygame.init()

    # Set screen size
    screen_width, screen_height = 1200, 600  # Adjust as needed
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption(caption)

    # Load images
    img1 = load_image(image1_path)
    img2 = load_image(image2_path)

    # Compute the difference image
    diff_img = compute_difference_pygame(image1_path, image2_path)

    # Resize images to fit within a third of the screen width
    max_width, max_height = screen_width // 3, screen_height
    img1 = resize_image(img1, max_width, max_height)
    img2 = resize_image(img2, max_width, max_height)
    diff_img = resize_image(diff_img, max_width, max_height)

    # Main loop
    running = True
    while running:
        screen.fill((0, 0, 0))  # Black background
        screen.blit(img1, (0, (screen_height - img1.get_height()) // 2))  # Left
        screen.blit(img2, (max_width, (screen_height - img2.get_height()) // 2))  # Center
        screen.blit(diff_img, (max_width * 2, (screen_height - diff_img.get_height()) // 2))  # Right

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                # print(f"Key pressed: {pygame.key.name(event.key)}")
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == ord('1'):
                    valid = True
                    running = False
                if event.key == ord('2'):
                    valid = False
                    running = False
    pygame.quit()
    return valid


def query_count(cursor, query):
    cursor.execute(query)
    return cursor.fetchall()[0][0]

def make_indexes():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute("CREATE INDEX idx_md5sum ON images(md5sum);")
    res = cursor.fetchall()
    print(res)
    cursor.execute("CREATE INDEX idx_canonical ON images(canonical);")
    res = cursor.fetchall()
    print(res)
    cursor.execute("CREATE INDEX idx_phash ON images(phash);")
    res = cursor.fetchall()
    print(res)
    conn.close()

def report():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    num_total = query_count(cursor, "SELECT count(*) from images;")
    print(f"{num_total:7d} images")

    num_non_canonical = query_count(cursor, "SELECT count(*) from images as c where c.canonical = 0;")
    num_canonical = query_count(cursor, "SELECT count(*) from images as c where c.canonical = 1;")
    num_noncan_deleted = query_count(cursor, "SELECT count(*) from images as c where c.canonical = 1 and c.processed = 3;")
    print('---')
    print(f"{num_canonical:7d} canonical")
    print(f"{num_non_canonical:7d} non-canonical")
    print(f"{num_noncan_deleted:7d} non-canonical deleted")

    num_non_processed = query_count(cursor, "SELECT count(*) from images as c where c.processed = 0;")
    num_processed = query_count(cursor, "SELECT count(*) from images as c where c.processed = 1;")
    num_error = query_count(cursor, "SELECT count(*) from images as c where c.processed = 2;")
    num_deleted = query_count(cursor, "SELECT count(*) from images as c where c.processed = 3;")
    print('---')
    print(f"{num_processed:7d} processed")
    print(f"{num_non_processed:7d} unprocessed")
    print(f"{num_error:7d} errors")
    print(f"{num_deleted:7d} deleted")

    if 0:
        query = """
        SELECT count(*)
        FROM images AS nc
        JOIN images AS c
        ON (nc.md5sum = c.md5sum OR nc.phash = c.phash)
        WHERE nc.canonical = 0 AND c.canonical = 1
        """
        num_dup_candidates = query_count(cursor, query)
        print('---')
        print(f"{num_dup_candidates:7d} duplicate candidates")

    if 0:
        query = """
        EXPLAIN QUERY PLAN
        SELECT non_canonical.filepath 
        FROM images AS canonical
        JOIN images AS non_canonical
        ON canonical.md5sum = non_canonical.md5sum
        WHERE canonical.canonical = 1
        AND non_canonical.canonical = 0;
        """
        cursor.execute(query)
        res = cursor.fetchall()
        print(res)
    conn.close()

directories = [
    # full_path_string, canonical_boolean
    ['/media/tba/photos/', 1],
    ['/home/alan/Downloads/', 0],
    ['/home/alan/sync/', 0],
    ['/home/alan/sync-recovery/', 0],
    ['/home/alan/sync-recovery-photos/', 0],
    ['/home/alan/scratch', 0],
]

@iex
def main():
    if len(sys.argv) < 2 or sys.argv[1] == 'init':
        add_image_paths(directories)
    elif sys.argv[1] == 'help':
        print("Usage: python canonificator.py [init|scan|dedup|report|errors|index]")
        print("  init   - ")
        print("  scan   - ")
        print("  dedup  - ")
        print("  report - ")
        print("  errors - ")
        print("  index  - ")
    elif sys.argv[1] == 'scan':
        process_images()
    elif sys.argv[1] == 'dedup':
        find_duplicates()
    elif sys.argv[1] == 'report':
        report()
    elif sys.argv[1] == 'errors':
        process_images(1500)
    elif sys.argv[1] == 'index':
        make_indexes()

if __name__ == "__main__":
    initialize_database()
    main()
