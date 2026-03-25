from setuptools import find_packages, setup

setup(
    name="real_data_ar",
    version="0.1.0",
    description="A library for autoregressive training on real diffusion-reaction data.",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy",
        "h5py",
        "omegaconf",
        "pytorch-lightning",
        "matplotlib",
        "tqdm",
        "pandas",
        "scipy",
        "einops",
        "timm",
        "scikit-image",
        "fvcore",
        "thop",
        "psutil",
        "tensorboard",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
