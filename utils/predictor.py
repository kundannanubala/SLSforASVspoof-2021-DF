import json
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.dataset import InputFolderDataset, collect_audio_files


class InferenceEngine:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def predict_batch(self, tensors):
        with torch.no_grad():
            batch_out = self.model(tensors)
            scores = batch_out[:, 1].data.cpu().numpy().ravel()
        return scores.tolist()

    def predict_folder(
        self,
        input_dir,
        output_dir,
        batch_size=8,
        threshold=None,
        write_json=True,
    ):
        audio_files = collect_audio_files(input_dir)
        if not audio_files:
            raise FileNotFoundError(f"No audio files found in {input_dir}")

        os.makedirs(output_dir, exist_ok=True)
        scores_path = os.path.join(output_dir, "scores.txt")
        json_path = os.path.join(output_dir, "results.json")

        dataset = InputFolderDataset(input_dir)
        data_loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, drop_last=False
        )

        results = []
        with open(scores_path, "w") as scores_file:
            for batch_x, filenames in tqdm(data_loader, desc="Inferencing"):
                batch_x = batch_x.to(self.device)
                score_list = self.predict_batch(batch_x)

                for filename, score in zip(filenames, score_list):
                    scores_file.write(f"{filename} {score}\n")
                    entry = {"file": filename, "score": score}
                    if threshold is not None:
                        entry["label"] = (
                            "bonafide" if score >= threshold else "spoof"
                        )
                    results.append(entry)

        if write_json:
            with open(json_path, "w") as json_file:
                json.dump(results, json_file, indent=2)

        self._print_summary(results, scores_path, json_path if write_json else None)
        return results

    def _print_summary(self, results, scores_path, json_path=None):
        print(f"\nScores saved to {scores_path}")
        if json_path:
            print(f"Results saved to {json_path}")
        print(f"\n{'File':<30} {'Score':>10} {'Label':>10}")
        print("-" * 52)
        for entry in results:
            label = entry.get("label", "-")
            print(f"{entry['file']:<30} {entry['score']:>10.4f} {label:>10}")
