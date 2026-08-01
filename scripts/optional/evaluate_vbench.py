#!/usr/bin/env python3
"""
Standalone VBench evaluation script for generated videos.

Usage:
    # Evaluate a single video
    python evaluate_vbench.py --video_path lp_t2v-14B_1280*720_prompt_20260301.mp4

    # Evaluate all videos in a folder
    python evaluate_vbench.py --video_path ./output_videos/

    # Specify output directory for results
    python evaluate_vbench.py --video_path video.mp4 --output_dir ./vbench_results

    # Specify which dimensions to evaluate
    python evaluate_vbench.py --video_path video.mp4 --dimensions subject_consistency imaging_quality
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(stream=sys.stdout)],
)

# Default dimensions to evaluate
DEFAULT_DIMENSIONS = [
    "subject_consistency",
    "background_consistency",
    "temporal_flickering",
    "motion_smoothness",
    "imaging_quality",
]


def run_vbench_evaluation(video_path, output_dir="./vbench_results", dimensions=None, device="cuda"):
    """
    Run VBench evaluation on a video file or directory of videos.

    Args:
        video_path: Path to a single video file or directory containing videos.
        output_dir: Directory to save evaluation results.
        dimensions: List of VBench dimensions to evaluate.
                    Defaults to DEFAULT_DIMENSIONS.
        device: Device to run evaluation on.

    Returns:
        dict: Evaluation results keyed by dimension name.
    """
    try:
        from vbench import VBench
    except ImportError:
        logging.error(
            "VBench is not installed. Please install it with:\n"
            "  pip install vbench\n"
            "For some dimensions you may also need:\n"
            "  pip install detectron2@git+https://github.com/facebookresearch/detectron2.git"
        )
        return None

    if dimensions is None:
        dimensions = DEFAULT_DIMENSIONS[:]

    os.makedirs(output_dir, exist_ok=True)

    video_path = os.path.abspath(video_path)
    if not os.path.exists(video_path):
        logging.error(f"Video path does not exist: {video_path}")
        return None

    # Determine video name for result labeling
    if os.path.isfile(video_path):
        video_name = os.path.splitext(os.path.basename(video_path))[0]
    else:
        video_name = os.path.basename(video_path.rstrip("/"))

    logging.info(f"VBench evaluation starting:")
    logging.info(f"  Video path:  {video_path}")
    logging.info(f"  Output dir:  {output_dir}")
    logging.info(f"  Dimensions:  {dimensions}")
    logging.info(f"  Device:      {device}")

    # VBench requires a full_info_dir for prompt suite; for custom mode it's optional
    # but the constructor expects it. We'll use an empty path and rely on custom_input mode.
    # Find the VBench_full_info.json from the installed package
    import vbench as vbench_pkg

    vbench_pkg_dir = os.path.dirname(vbench_pkg.__file__)
    full_info_path = os.path.join(vbench_pkg_dir, "VBench_full_info.json")
    if not os.path.exists(full_info_path):
        # Try common alternative locations
        for candidate in [
            os.path.join(os.getcwd(), "VBench_full_info.json"),
            os.path.join(os.path.dirname(__file__), "VBench_full_info.json"),
        ]:
            if os.path.exists(candidate):
                full_info_path = candidate
                break
        else:
            logging.warning(
                "VBench_full_info.json not found. "
                "Custom-input mode should still work for most dimensions."
            )
            full_info_path = os.path.join(vbench_pkg_dir, "VBench_full_info.json")

    all_results = {}

    # Evaluate each dimension separately to isolate failures
    for dim in dimensions:
        logging.info(f"\n{'='*60}")
        logging.info(f"Evaluating dimension: {dim}")
        logging.info(f"{'='*60}")
        try:
            my_vbench = VBench(device, full_info_path, output_dir)
            my_vbench.evaluate(
                videos_path=video_path,
                name=f"{video_name}_{dim}",
                dimension_list=[dim],
                mode="custom_input",
            )
            # Read the result file — search multiple candidate locations
            candidate_name = f"{video_name}_{dim}_eval_results.json"
            candidate_paths = [
                os.path.join(output_dir, "eval_results", candidate_name),
                os.path.join(output_dir, candidate_name),
            ]
            result_file = None
            for cp in candidate_paths:
                if os.path.exists(cp):
                    result_file = cp
                    break

            # Fallback: search for any file matching the dimension name
            if result_file is None:
                for search_dir in [os.path.join(output_dir, "eval_results"), output_dir]:
                    if os.path.isdir(search_dir):
                        for fname in os.listdir(search_dir):
                            if dim in fname and fname.endswith("_eval_results.json"):
                                result_file = os.path.join(search_dir, fname)
                                break
                    if result_file:
                        break

            if result_file and os.path.exists(result_file):
                with open(result_file, "r") as f:
                    result_data = json.load(f)
                # Handle both dict format {"dim": [score, [...]]} and list format
                if isinstance(result_data, dict):
                    for key, value in result_data.items():
                        if isinstance(value, list) and len(value) > 0:
                            all_results[dim] = value[0]
                        elif isinstance(value, (int, float)):
                            all_results[dim] = value
                elif isinstance(result_data, list):
                    for entry in result_data:
                        if isinstance(entry, dict):
                            for key, value_list in entry.items():
                                if isinstance(value_list, list) and len(value_list) > 0:
                                    all_results[dim] = value_list[0]
                                else:
                                    all_results[dim] = value_list
                logging.info(f"  {dim}: {all_results.get(dim, 'N/A')}")
            else:
                logging.warning(f"  {dim}: result file not found")
        except Exception as e:
            logging.error(f"  {dim}: evaluation failed with error: {e}")
            all_results[dim] = f"ERROR: {e}"

    # Print summary
    logging.info(f"\n{'='*60}")
    logging.info("VBench Evaluation Summary")
    logging.info(f"{'='*60}")
    logging.info(f"Video: {video_path}")
    for dim, score in all_results.items():
        if isinstance(score, (int, float)):
            logging.info(f"  {dim:30s}: {score:.4f}")
        else:
            logging.info(f"  {dim:30s}: {score}")
    logging.info(f"{'='*60}")

    # Save aggregated summary
    summary_path = os.path.join(output_dir, f"{video_name}_vbench_summary.json")
    summary = {
        "video_path": video_path,
        "timestamp": datetime.now().isoformat(),
        "dimensions": all_results,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logging.info(f"Summary saved to: {summary_path}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Run VBench evaluation on generated videos")
    parser.add_argument(
        "--video_path",
        type=str,
        required=True,
        help="Path to a video file or directory containing videos.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./vbench_results",
        help="Directory to save evaluation results. Default: ./vbench_results",
    )
    parser.add_argument(
        "--dimensions",
        type=str,
        nargs="+",
        default=None,
        help="VBench dimensions to evaluate. Default: "
        + ", ".join(DEFAULT_DIMENSIONS),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run evaluation on. Default: cuda",
    )
    args = parser.parse_args()

    results = run_vbench_evaluation(
        video_path=args.video_path,
        output_dir=args.output_dir,
        dimensions=args.dimensions,
        device=args.device,
    )

    if results is None:
        sys.exit(1)


if __name__ == "__main__":
    main()
