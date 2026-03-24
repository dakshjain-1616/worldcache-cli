from setuptools import setup, find_packages

setup(
    name="worldcache-cli",
    version="1.0.0",
    description="WorldCache CLI – Pre-process video for world models at 90% less RAM",
    packages=find_packages(include=['worldcache', 'worldcache.*']),
    install_requires=[
        "Pillow>=9.5.0",
        "numpy>=1.24.0",
        "click>=8.1.0",
        "rich>=13.4.0",
    ],
    extras_require={
        "video": ["opencv-python>=4.8.0"],
        "test": ["pytest>=7.0.0"],
    },
    entry_points={
        "console_scripts": [
            "worldcache=worldcache.cli:main",
        ]
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
