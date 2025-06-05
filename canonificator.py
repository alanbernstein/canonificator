#!/usr/bin/env python3
import hashlib
import os
import sys
import time
import sqlite3
import subprocess
from types import SimpleNamespace

from PIL import Image, ExifTags
from pillow_heif import register_heif_opener

import exifread

import imagehash

import tkinter as tk
from tkinter import PhotoImage
from PIL import Image, ImageTk, ImageChops
import sys

from ipdb import iex, set_trace as db

from dotenv import load_dotenv

load_dotenv(sys.path[1] + "/.env")
DB_FILE = os.getenv("DB_FILE")
print(DB_FILE)

allowed_extensions = ['jpg', 'jpeg', 'png', 'gif', 'tif', 'bmp', 'heic']

register_heif_opener()

window_name = 'canonificator-review'
screen_w, screen_h = 1920*2, 1080*2

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
    
    nc_search_prefix = '/home/alan/sync-recovery/photorec/recup_dir.419'
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
    # TODO
    pass

def check_pairs(pairs, query_name, cursor, MANUAL_CHECK=True, DELETE_ALL=False):

    if not pairs:
        print("no pairs")
        return

    #db()
    print(f"Found {len(pairs)} pairs ({query_name}):")
    #pygame.init()
    #cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    #cv2.resizeWindow(window_name, 3000, 1000)
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

        def jpeg_quality(pth):
            # TODO
            return 0

        nc_quality = jpeg_quality(nc.filepath)
        c_quality = jpeg_quality(c.filepath)

        print(f"{nc.id:07} {nc.md5sum} {nc.phash} {nc.width:4}x{nc.height:4} {nc.filepath} {nc_quality}")
        print(f"{ c.id:07} { c.md5sum} { c.phash} { c.width:4}x{ c.height:4} { c.filepath} { c_quality}")

        if not os.path.exists(nc.filepath):
            print("File does not exist, skipping...")
            continue

        should_delete = False
        should_rescue = False
        if MANUAL_CHECK:
            #key = display_images_opencv(nc.filepath, c.filepath, caption_dict)
            key = display_images_tk(nc.filepath, c.filepath, caption_dict)

            focus_window(window_name)
            if key == 'q':
                print("Quitting manual check.")
                break
            elif key == 'd':
                print('delete non-canonical item')
                should_delete = True
            elif key == 's':
                print('skip pair')
            elif key == 'r':
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

def load_image_tk(path, max_width=1000):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    if w > max_width:
        ratio = max_width / w
        image = image.resize((int(w * ratio), int(h * ratio)), Image.Resampling.LANCZOS)
    return image, (w, h)

def display_images_tk(path1, path2, caption_dict):
    img1, size1 = load_image_tk(path1)
    img2, size2 = load_image_tk(path2)
    diff = ImageChops.difference(img1, img2)

    root = tk.Tk()
    root.title("canonificator-review")
    root.configure(bg="black")

    # Convert to Tkinter-compatible images
    img1_tk = ImageTk.PhotoImage(img1)
    img2_tk = ImageTk.PhotoImage(img2)

    # Layout
    label1 = tk.Label(root, image=img1_tk, bg="black")
    label2 = tk.Label(root, image=img2_tk, bg="black")
    label1.grid(row=0, column=0)
    label2.grid(row=0, column=1)
    if img1.size == img2.size:
        diff_tk = ImageTk.PhotoImage(diff)
        label_diff = tk.Label(root, image=diff_tk, bg="black")
        label_diff.grid(row=1, column=1)
    else:
        # TODO downsample to compute diff?
        pass

    caption_label1 = tk.Label(root, text=f"{path1} {size1}", justify="left", anchor="w", bg="black", fg="white", font=("Helvetica", 12))
    caption_label1.grid(row=0, column=0, sticky="nw")
    caption_label2 = tk.Label(root, text=f"{path2} {size2}", justify="left", anchor="w", bg="black", fg="white", font=("Helvetica", 12))
    caption_label2.grid(row=0, column=1, sticky="ne")

    color_map = {
        False: "red",
        True: "green",
    }

    i = 0
    for caption, val in caption_dict.items():
        color = color_map[val]
        label = tk.Label(root, text=caption, fg=color, bg="black", font=("Helvetica", 12))
        label.grid(row=2 + i, column=0, columnspan=2, sticky="w", padx=10)
        i += 1

    key_pressed = {}

    def on_key(event):
        key_pressed["key"] = event.char
        root.destroy()

    root.bind("<Key>", on_key)
    root.focus_force()
    root.mainloop()

    return key_pressed.get("key")


def focus_window(window_title):
    time.sleep(0.2)  # give time for the window to appear
    try:
        subprocess.run(['wmctrl', '-a', window_title])
    except FileNotFoundError:
        print("wmctrl not found")

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
