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
            probs = torch.exp(batch_out)

        prob_spoof = probs[:, 0].cpu().numpy().ravel()
        prob_bonafide = probs[:, 1].cpu().numpy().ravel()
        log_scores = batch_out[:, 1].cpu().numpy().ravel()

        return [
            {
                "log_score": float(log_scores[i]),
                "prob_spoof": float(prob_spoof[i]),
                "prob_bonafide": float(prob_bonafide[i]),
            }
            for i in range(len(log_scores))
        ]

    def predict_folder(
        self,
        input_dir,
        output_dir,
        batch_size=8,
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
                batch_results = self.predict_batch(batch_x)

                for filename, pred in zip(filenames, batch_results):
                    scores_file.write(f"{filename} {pred['log_score']}\n")
                    results.append({"file": filename, **pred})

        if write_json:
            with open(json_path, "w") as json_file:
                json.dump(results, json_file, indent=2)

        self._print_summary(results, scores_path, json_path if write_json else None)
        return results

    def _print_summary(self, results, scores_path, json_path=None):
        print(f"\nScores saved to {scores_path}")
        if json_path:
            print(f"Results saved to {json_path}")
        print(f"\n{'File':<30} {'P(bonafide)':>12} {'P(spoof)':>12}")
        print("-" * 56)
        for entry in results:
            print(
                f"{entry['file']:<30} "
                f"{entry['prob_bonafide']:>12.4f} "
                f"{entry['prob_spoof']:>12.4f}"
            )
