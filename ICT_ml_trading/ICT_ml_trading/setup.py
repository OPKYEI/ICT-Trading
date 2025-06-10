from setuptools import setup, find_packages

setup(
    name="ict_ml_trading",
    version="0.1.0",
    package_dir={"": "src"},               # ← map top-level to src/
    packages=find_packages(where="src"),   # ← discover under src/
    install_requires=[
        "pandas", "numpy", "scikit-learn", "pyyaml", "matplotlib"
    ],
    include_package_data=True,
)
