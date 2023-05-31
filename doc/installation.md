## Prerequisites

Requirements:

- Python >= 3.7 and NumPy
- Orfeo ToolBox binaries (follow these
 [instructions](https://www.orfeo-toolbox.org/CookBook/Installation.html))
- Orfeo ToolBox python binding (follow these
 [instructions](https://www.orfeo-toolbox.org/CookBook/Installation.html))

## Install with pip

```bash
pip install pyotb --upgrade
```

For development, use the following:

```bash
git clone https://gitlab.orfeo-toolbox.org/nicolasnn/pyotb
cd pyotb
pip install -e ".[dev]"
```

## Old versions

If you need compatibility with python3.6, install  `pyotb<2.0` and for
 python3.5 use `pyotb==1.2.2`.  

For Python>=3.6, latest version available is pyotb 1.5.0.
For Python 3.5, latest version available is pyotb 1.2.2
