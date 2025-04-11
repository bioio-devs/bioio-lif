# bioio-lif

[![Build Status](https://github.com/bioio-devs/bioio-lif/actions/workflows/ci.yml/badge.svg)](https://github.com/bioio-devs/bioio-lif/actions)
[![PyPI version](https://badge.fury.io/py/bioio-lif.svg)](https://badge.fury.io/py/bioio-lif)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.10–3.13](https://img.shields.io/badge/python-3.10--3.13-blue.svg)](https://www.python.org/downloads/)

A BioIO reader plugin for reading LIF (Leica) images using `readlif`

---


## Documentation

[See the full documentation on our GitHub pages site](https://bioio-devs.github.io/bioio/OVERVIEW.html) - the generic use and installation instructions there will work for this package.

Information about the base reader this package relies on can be found in the `bioio-base` repository [here](https://github.com/bioio-devs/bioio-base)

## Installation

**Stable Release:** `pip install bioio-lif`<br>
**Development Head:** `pip install git+https://github.com/bioio-devs/bioio-lif.git`

## Example Usage (see full documentation for more examples)

Install bioio-lif alongside bioio:

`pip install bioio bioio-lif`


This example shows a simple use case for just accessing the pixel data of the image
by explicitly passing this `Reader` into the `BioImage`. Passing the `Reader` into
the `BioImage` instance is optional as `bioio` will automatically detect installed
plug-ins and auto-select the most recently installed plug-in that supports the file
passed in.
```python
from bioio import BioImage
import bioio_lif

img = BioImage("my_file.lif", reader=bioio_lif.Reader)
img.data
```

## Issues
[_Click here to view all open issues in bioio-devs organization at once_](https://github.com/search?q=user%3Abioio-devs+is%3Aissue+is%3Aopen&type=issues&ref=advsearch) or check this repository's issue tab.


## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for information related to developing the code.
