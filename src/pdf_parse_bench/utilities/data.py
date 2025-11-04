"""Data utilities for accessing benchmark datasets."""
from pathlib import Path


def _get_dataset_dir(dataset_name: str) -> Path:
    """Helper function to locate dataset directory."""
    # Try package data directory first (installed package)
    package_data_dir = Path(__file__).parent.parent / "data"
    dataset_dir = package_data_dir / dataset_name

    # If not found, try project root (development/editable install)
    if not dataset_dir.exists():
        project_root = Path(__file__).parent.parent.parent.parent
        dataset_dir = project_root / "data" / dataset_name

    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset '{dataset_name}' not found. "
            f"Searched in: {package_data_dir} and {project_root / 'data'}"
        )

    return dataset_dir


def get_benchmark_pdfs_dir(dataset_name: str = "2025-10-v1") -> Path:
    """
    Get the path to the PDFs directory of a benchmark dataset.

    Args:
        dataset_name: Name of the dataset (default: "2025-10-v1")

    Returns:
        Path to the pdfs directory

    Example:
        >>> pdfs_dir = get_benchmark_pdfs_dir("2025-10-v1")
        >>> pdf_files = list(pdfs_dir.glob("*.pdf"))
    """
    pdfs_dir = _get_dataset_dir(dataset_name) / "pdfs"

    if not pdfs_dir.exists():
        raise FileNotFoundError(f"PDFs directory not found: {pdfs_dir}")

    return pdfs_dir


def get_benchmark_ground_truth_dir(dataset_name: str = "2025-10-v1") -> Path:
    """
    Get the path to the ground truth directory of a benchmark dataset.

    Args:
        dataset_name: Name of the dataset (default: "2025-10-v1")

    Returns:
        Path to the ground_truth directory

    Example:
        >>> gt_dir = get_benchmark_ground_truth_dir("2025-10-v1")
        >>> gt_files = list(gt_dir.glob("*.json"))
    """
    gt_dir = _get_dataset_dir(dataset_name) / "ground_truth"

    if not gt_dir.exists():
        raise FileNotFoundError(f"Ground truth directory not found: {gt_dir}")

    return gt_dir
