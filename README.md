# bioio-lif

[![Build Status](https://github.com/bioio-devs/bioio-lif/actions/workflows/ci.yml/badge.svg)](https://github.com/bioio-devs/bioio-lif/actions)
[![Documentation](https://github.com/bioio-devs/bioio-lif/actions/workflows/docs.yml/badge.svg)](https://bioio-devs.github.io/bioio-lif)

A BioIO reader plugin for reading LIF (Leica Image File) images.

This plugin is intended to be used in conjunction with [bioio](https://github.com/bioio-devs/bioio)
---

## Installation

**Stable Release:** `pip install bioio-lif`<br>
**Development Head:** `pip install git+https://github.com/bioio-devs/bioio-lif.git`

## Quickstart

```python
from bioio_lif import Reader 

r = Reader("my-image.ext")
r.dims
```

## Documentation

For full package documentation please visit [bioio-devs.github.io/bioio-lif](https://bioio-devs.github.io/bioio-lif).

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for information related to developing the code.

**MIT License**
