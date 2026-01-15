import os
import yaml
import json
import shutil
from pathlib import Path
from ultralytics import YOLO, settings
from src import config

# Update YOLO settings with a relative path for portability
# This sets the download location to './datasets' inside the project
print(f"[*] Updating YOLO datasets path to: datasets")
settings.update({'datasets_dir': 'datasets'})


class ExperimentRunner:
    def __init__(self, processor):
        self.processor = processor
        self.model = YOLO(config.MODEL_TYPE)
        self.results = {}
        self.style_scores = {}

        # USE RELATIVE PATHS for portability (GitHub friendly)
        self.output_root = 'replication_results'
        os.makedirs(self.output_root, exist_ok=True)

        # Dedicated folder for the final model
        self.model_dir = 'model'
        os.makedirs(self.model_dir, exist_ok=True)

        # Construct the final path using the config name + .pt
        self.final_model_filename = f"{config.MODEL_NAME}.pt"
        self.final_model_path = os.path.join(self.model_dir, self.final_model_filename)

    def _ensure_local_coco_yaml(self):
        local_yaml_path = 'coco_local.yaml'
        if os.path.exists(local_yaml_path): return local_yaml_path
        print("[*] Creating local coco_local.yaml...")
        try:
            import ultralytics
            pkg_path = Path(ultralytics.__file__).parent
            default_yaml = pkg_path / 'cfg' / 'datasets' / 'coco.yaml'

            # FIX: Encoding set to utf-8 to prevent Windows crash
            with open(default_yaml, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)

            # Use relative path for portability
            data['path'] = 'datasets/coco'

            with open(local_yaml_path, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, sort_keys=False)
            return local_yaml_path
        except Exception as e:
            print(f"[!] Error creating local config: {e}")
            return 'coco.yaml'

    def run_phase_1_baseline(self):
        print("\n--- PHASE 1: ESTABLISHING BASELINE (COCO) ---")
        coco_yaml_file = self._ensure_local_coco_yaml()

        # Output: replication_results/01_Baseline/coco_eval
        res = self.model.val(
            data=coco_yaml_file,
            split='val',
            classes=[0],
            verbose=False,
            project=os.path.join(self.output_root, '01_Baseline'),
            name='coco_eval',
            exist_ok=True
        )
        self.results['Photo (Baseline)'] = res.box.map50
        print(f"-> Baseline mAP: {self.results['Photo (Baseline)']:.3f}")

    def run_phase_2_zero_shot(self):
        print("\n--- PHASE 2: CROSS-DEPICTION TEST (ZERO-SHOT) ---")

        # Output: replication_results/02_ZeroShot/peopleart_eval
        res = self.model.val(
            data='peopleart_replication.yaml',
            split='test',
            verbose=False,
            project=os.path.join(self.output_root, '02_ZeroShot'),
            name='peopleart_eval',
            exist_ok=True
        )
        self.results['Art (Zero-Shot)'] = res.box.map50
        print(f"-> Zero-Shot Art mAP: {self.results['Art (Zero-Shot)']:.3f}")

    def run_phase_3_adaptation(self):
        print("\n--- PHASE 3: ADAPTATION (FINE-TUNING) ---")

        # Check if we already have a trained model
        if os.path.exists(self.final_model_path):
            print(f"[*] Found persistent model at: {self.final_model_path}")
            print("[*] Skipping training and loading existing weights...")
            self.model = YOLO(self.final_model_path)
        else:
            print(f"[*] Training for {config.EPOCHS} epochs (Patience={config.PATIENCE})...")

            project_path = os.path.join(self.output_root, '03_Adaptation')
            name = 'training_run'

            self.model.train(
                data='peopleart_replication.yaml',
                epochs=config.EPOCHS,
                patience=config.PATIENCE,
                imgsz=config.IMG_SIZE,
                batch=config.BATCH_SIZE,
                verbose=False,
                plots=False,
                project=project_path,
                name=name,
                exist_ok=True
            )

            # --- ROBUST PATH DETECTION ---
            # 1. Expected Path (Relative)
            expected_path = os.path.join(project_path, name, 'weights', 'best.pt')

            # 2. Fallback Path (YOLO default nesting in runs/detect)
            fallback_path = os.path.join('runs', 'detect', project_path, name, 'weights', 'best.pt')

            source_path = None
            if os.path.exists(expected_path):
                source_path = expected_path
            elif os.path.exists(fallback_path):
                print(f"[*] Note: Model found in fallback location: {fallback_path}")
                source_path = fallback_path

            # Copy to our persistent 'model' folder with the CUSTOM NAME
            if source_path:
                print(f"[*] Saving best model to persistent storage: {self.final_model_path}")
                shutil.copy(source_path, self.final_model_path)
                self.model = YOLO(self.final_model_path)
            else:
                print(f"[!] Warning: Training finished but 'best.pt' was not found.")
                print(f"    Checked: {expected_path}")
                print(f"    Checked: {fallback_path}")

        print("[*] Re-evaluating fine-tuned model...")

        res = self.model.val(
            data='peopleart_replication.yaml',
            split='test',
            verbose=False,
            project=os.path.join(self.output_root, '03_Adaptation'),
            name='finetuned_eval',
            exist_ok=True
        )
        self.results['Art (Fine-Tuned)'] = res.box.map50
        print(f"-> Fine-Tuned Art mAP: {self.results['Art (Fine-Tuned)']:.3f}")

    def run_phase_4_style_analysis(self):
        print("\n--- PHASE 4: STYLE-SPECIFIC ANALYSIS ---")
        style_files = self.processor.get_style_files()
        if not style_files:
            print("[!] No style data available.")
            return

        temp_yaml = 'temp_style_eval.yaml'
        self.results['Style Breakdown'] = {}

        for sf in style_files:
            style_name = os.path.splitext(os.path.basename(sf))[0]
            conf = {
                'path': os.path.abspath(config.DATASET_DIR),
                'train': sf, 'val': sf, 'names': {0: 'person'}
            }
            with open(temp_yaml, 'w') as f:
                yaml.dump(conf, f)
            try:
                res = self.model.val(
                    data=temp_yaml,
                    split='val',
                    verbose=False,
                    project=os.path.join(self.output_root, '04_StyleAnalysis'),
                    name=style_name,
                    exist_ok=True
                )
                score = res.box.map50
                self.style_scores[style_name] = score
                self.results['Style Breakdown'][style_name] = score
            except:
                pass
        if os.path.exists(temp_yaml): os.remove(temp_yaml)

    def run_phase_5_divergence_hypothesis(self):
        print("\n--- PHASE 5: TESTING 'STATISTICAL DIVERGENCE' CLAIM ---")
        low_abs = []
        high_abs = []
        for style, score in self.style_scores.items():
            is_high = any(x.lower() in style.lower() for x in config.ABSTRACTION_MAP['High_Abstraction'])
            is_low = any(x.lower() in style.lower() for x in config.ABSTRACTION_MAP['Low_Abstraction'])
            if is_high:
                high_abs.append(score)
            elif is_low:
                low_abs.append(score)

        avg_h = sum(high_abs) / len(high_abs) if high_abs else 0
        avg_l = sum(low_abs) / len(low_abs) if low_abs else 0
        self.results['Abstract Art'] = avg_h
        self.results['Realistic Art'] = avg_l

        print(f"-> Low Abstraction (Realistic) mAP:  {avg_l:.3f}")
        print(f"-> High Abstraction (Abstract) mAP:  {avg_h:.3f}")

    def print_conclusion(self):
        print("\n=== FINAL REPLICATION REPORT ===")
        print(f"1. Algorithmic Evolution:")
        print(f"   Westlake (2016): {config.WESTLAKE_2016_BASELINE}")
        print(f"   YOLOv8 (2025):   {self.results.get('Art (Zero-Shot)', 0):.3f}")

        # Save complete detailed data to JSON
        json_path = os.path.join(self.output_root, 'final_report.json')
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=4)
        print(f"[*] Detailed results saved to {json_path}")

        # Save Text Report
        txt_path = os.path.join(self.output_root, 'final_report.txt')
        with open(txt_path, 'w') as f:
            f.write("=== REPLICATION STUDY RESULTS ===\n")
            for k, v in self.results.items():
                if isinstance(v, dict):
                    f.write(f"\n{k}:\n")
                    for sk, sv in v.items():
                        f.write(f"  {sk}: {sv:.4f}\n")
                else:
                    f.write(f"{k}: {v:.4f}\n")
        print(f"[*] Text summary saved to {txt_path}")

        # Filter results for plotting (Numbers only)
        plot_data = {k: v for k, v in self.results.items() if isinstance(v, (int, float))}
        return plot_data