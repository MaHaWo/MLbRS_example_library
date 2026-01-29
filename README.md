# MLbRS_example_library

## Introduction
This is an example library for the 'Best practices in Machine Learning-based Research Software' course given in February 2026 at the University of Heidelberg.

## Usage
This repository uses [uv](https://uv.dev/) for environment and package management.

Clone this repository to your systme
```bash
git clone git@github.com:MaHaWo/MLbRS_example.git
```
and create a virtual environment:

```bash
uv venv
```
then

```bash
uv install
```

or
```bash
uv pip install .
```

For an editable install, use
```bash
uv pip install -e .
```

To run tests, use
```bash
python3 -m pytest tests/
```

from your terminal within the virtual environment.