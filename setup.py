"""
AniTune: ViT fine-tuning with LoRA for anime face recognition
"""
from setuptools import setup, find_packages

setup(
    name="anitune",
    version="0.1.0",
    description="ViT fine-tuning with LoRA for robust anime face recognition",
    author="AniTune Team",
    python_requires=">=3.8",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch",
        "torchvision",
        "pyyaml",
        "timm",
        "tqdm",
        "wandb",
        "pandas",
        "matplotlib",
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "ruff",
        ],
    },
)

