import os
import shutil
import glob
import random
import xml.etree.ElementTree as ET
import yaml
from src import config


class ScientificDataProcessor:
    def __init__(self):
        self.raw_path = config.RAW_DATA_PATH
        self.processed_path = config.DATASET_DIR
        self.classes = ['person']
        self.stats = {'train': 0, 'val': 0, 'test': 0}
        self.style_map = {}
        self.xml_map = {}

    def _index_xml_files(self):
        print("[*] Indexing XML annotations...")
        # Find every single XML file recursively
        # Use set() to avoid duplicates on Windows (case-insensitive)
        xmls = set(glob.glob(os.path.join(self.raw_path, '**', '*.xml'), recursive=True))
        xmls.update(glob.glob(os.path.join(self.raw_path, '**', '*.XML'), recursive=True))

        for x in xmls:
            # Step 1: Remove .xml extension
            filename_no_xml = os.path.splitext(os.path.basename(x))[0]
            # Step 2: Check if it still has an image extension and strip it
            # e.g., "image.jpg.xml" -> "image" (matching image.jpg id)
            if filename_no_xml.lower().endswith(('.jpg', '.jpeg', '.png')):
                key = os.path.splitext(filename_no_xml)[0]
            else:
                key = filename_no_xml
            self.xml_map[key] = x

        print(f"[*] Indexed {len(self.xml_map)} unique XML keys.")

    def _load_styles_from_imagesets(self):
        print("[*] Loading style definitions...")
        imagesets_path = os.path.join(self.raw_path, 'ImageSets')
        if not os.path.exists(imagesets_path):
            found = glob.glob(os.path.join(self.raw_path, '**', 'ImageSets'), recursive=True)
            if found:
                imagesets_path = found[0]
            else:
                print("[!] 'ImageSets' folder not found. Style analysis will rely on folder names.")
                return False

        txt_files = glob.glob(os.path.join(imagesets_path, '**', '*.txt'), recursive=True)
        for txt in txt_files:
            style_name = os.path.splitext(os.path.basename(txt))[0]
            if style_name.lower() in ['train', 'test', 'val', 'trainval']: continue
            with open(txt, 'r') as f:
                for line in f:
                    img_id = line.strip().split()[0]
                    self.style_map[img_id] = style_name
        return True

    def convert_xml(self, xml_file):
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()

            size = root.find('size')
            if size is None: return []
            w = int(size.find('width').text)
            h = int(size.find('height').text)
            if w == 0 or h == 0: return []

            yolo_lines = []
            found_person = False

            for obj in root.findall('object'):
                name = obj.find('name').text.lower().strip()
                if name not in self.classes: continue

                found_person = True
                cls_id = self.classes.index(name)
                bnd = obj.find('bndbox')

                box = (float(bnd.find('xmin').text), float(bnd.find('xmax').text),
                       float(bnd.find('ymin').text), float(bnd.find('ymax').text))

                # Normalize
                bb = ((box[1] + box[0]) / 2 / w, (box[3] + box[2]) / 2 / h, (box[1] - box[0]) / w,
                      (box[3] - box[2]) / h)
                yolo_lines.append(f"{cls_id} {bb[0]:.6f} {bb[1]:.6f} {bb[2]:.6f} {bb[3]:.6f}")

            return yolo_lines if found_person else []
        except Exception:
            return []

    def prepare_data_split(self):
        # Always clean up for fresh run to ensure consistency
        if os.path.exists(self.processed_path):
            shutil.rmtree(self.processed_path)

        print(f"[*] Starting Data Ingestion from: {self.raw_path}")

        # 1. Index XMLs
        self._index_xml_files()
        has_imagesets = self._load_styles_from_imagesets()

        # 2. Find Images
        images = glob.glob(os.path.join(self.raw_path, '**', '*.jpg'), recursive=True)
        images += glob.glob(os.path.join(self.raw_path, '**', '*.JPG'), recursive=True)

        if not images:
            raise FileNotFoundError(f"No images found in {self.raw_path}")

        print(f"[*] Found {len(images)} raw images. Processing...")

        random.seed(42)
        random.shuffle(images)

        n = len(images)
        splits = {
            'train': images[:int(n * 0.7)],
            'val': images[int(n * 0.7):int(n * 0.85)],
            'test': images[int(n * 0.85):]
        }

        os.makedirs(config.STYLE_LIST_DIR, exist_ok=True)
        style_file_buffers = {}
        processed_count = 0

        for split, imgs in splits.items():
            os.makedirs(os.path.join(self.processed_path, 'images', split), exist_ok=True)
            os.makedirs(os.path.join(self.processed_path, 'labels', split), exist_ok=True)

            for img_path in imgs:
                fname = os.path.basename(img_path)
                file_id = os.path.splitext(fname)[0]

                if file_id not in self.xml_map: continue

                xml_path = self.xml_map[file_id]
                labels = self.convert_xml(xml_path)

                if not labels: continue

                # Copy
                dest_path = os.path.abspath(os.path.join(self.processed_path, 'images', split, fname))
                shutil.copy(img_path, dest_path)

                with open(os.path.join(self.processed_path, 'labels', split, file_id + '.txt'), 'w') as f:
                    f.write('\n'.join(labels))

                self.stats[split] += 1
                processed_count += 1

                # Style
                style = "Unknown"
                if has_imagesets and file_id in self.style_map:
                    style = self.style_map[file_id]
                else:
                    p_style = os.path.basename(os.path.dirname(img_path))
                    if p_style not in ['JPEGImages', 'images', 'train', 'val', 'test']:
                        style = p_style

                if split == 'test' and style != "Unknown":
                    if style not in style_file_buffers: style_file_buffers[style] = []
                    style_file_buffers[style].append(dest_path)

        for style, paths in style_file_buffers.items():
            if len(paths) > 5:
                with open(os.path.join(config.STYLE_LIST_DIR, f"{style}.txt"), 'w') as f:
                    f.write('\n'.join(paths))

        print(f"[*] Processing Complete. Total Copied: {processed_count}")
        print(f"    Breakdown: {self.stats}")

        if processed_count == 0:
            raise RuntimeError("Data Processing Failed: No valid Image+XML pairs found.")

        self.create_yaml()

    def create_yaml(self):
        conf = {
            'path': os.path.abspath(self.processed_path),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'names': {0: 'person'}
        }
        with open('peopleart_replication.yaml', 'w') as f:
            yaml.dump(conf, f)

    def get_style_files(self):
        return glob.glob(os.path.join(config.STYLE_LIST_DIR, '*.txt'))