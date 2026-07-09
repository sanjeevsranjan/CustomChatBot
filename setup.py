from pathlib import Path

from setuptools import find_packages, setup


BASE_DIR = Path(__file__).resolve().parent


def load_requirements():
    requirements_file = BASE_DIR / "requirements.txt"
    if not requirements_file.exists():
        return []

    requirements = []
    for line in requirements_file.read_text(encoding="utf-8").splitlines():
        cleaned = line.strip()
        if not cleaned or cleaned.startswith("#"):
            continue
        if cleaned.startswith("-e ") or cleaned.startswith("--editable"):
            continue
        requirements.append(cleaned)

    return requirements


setup(
    name="custom-chatbot",
    version="0.1.0",
    author="Sanjeev Ranjan",
    author_email="sanjeevr1101@gmail.com",
    packages=find_packages(),
    install_requires=load_requirements(),
)