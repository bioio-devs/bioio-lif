#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for confocal channel-identity extraction and ``ome_metadata``.

These tests are data-light on purpose: they exercise the channel-naming and
OME-projection logic against a faithful confocal scene XML fixture, so they run
without the (git-LFS) sample ``.lif`` binaries that the rest of the suite needs.
"""

import pathlib
import xml.etree.ElementTree as ET
from typing import List, Tuple

import numpy as np
import pytest
from bioio_base import dimensions, types

from bioio_lif import lif_metadata
from bioio_lif.reader import Reader

# A faithful confocal scene XML fixture. Kept outside ``resources/`` (which is
# git-LFS tracked) so this stays a self-contained, LFS-free unit test.
LOCAL_DATA_DIR = pathlib.Path(__file__).parent / "data"
CONFOCAL_FIXTURE = LOCAL_DATA_DIR / "confocal_7ch_scene.xml"


# (index, dye, excitation_nm, emission_low_nm, emission_high_nm)
EXPECTED_CHANNELS = [
    (0, "DAPI (dsDNA bound)", 405, 430, 499),
    (1, "ALEXA 594", 590, 601, 640),
    (2, "ALEXA 750", 753, 768, 829),
    (3, "ALEXA 488", 499, 506, 548),
    (4, "ALEXA 647-R-PE", 653, 663, 688),
    (5, "ALEXA 555", 553, 562, 600),
    (6, "ALEXA 700", 696, 706, 749),
]


def _scene_root() -> ET.Element:
    return ET.parse(CONFOCAL_FIXTURE).getroot()


def test_extract_channels_recovers_identity() -> None:
    channels = lif_metadata.extract_channels(_scene_root())

    assert len(channels) == len(EXPECTED_CHANNELS)
    for ch, (index, dye, ex, em_lo, em_hi) in zip(channels, EXPECTED_CHANNELS):
        assert ch["index"] == index
        assert ch["dye"] == dye
        assert ch["fluor"] == dye
        assert ch["excitation_nm"] == ex
        assert ch["emission_low_nm"] == em_lo
        assert ch["emission_high_nm"] == em_hi


def test_channel_dye_name_prefers_dye_over_lutname() -> None:
    scene = _scene_root()
    descs = list(scene.iter("ChannelDescription"))
    expected_dyes = [row[1] for row in EXPECTED_CHANNELS]

    for desc, expected_dye in zip(descs, expected_dyes):
        # The display LUTName is a useless color name; the helper must not use it.
        assert desc.attrib.get("LUTName") != expected_dye
        assert lif_metadata.channel_dye_name(desc) == expected_dye


def test_channel_dye_name_falls_back_to_lutname() -> None:
    # A confocal channel with no DyeName must fall back to LUTName (old behavior).
    desc = ET.fromstring('<ChannelDescription LUTName="Gray" />')
    assert lif_metadata.channel_dye_name(desc) is None


def test_strip_leica_vendor_prefix() -> None:
    desc = ET.fromstring(
        "<ChannelDescription LUTName='Blue'>"
        "<ChannelProperty><Key>DyeName</Key><Value>Leica/ALEXA 594</Value>"
        "</ChannelProperty></ChannelDescription>"
    )
    assert lif_metadata.channel_dye_name(desc) == "ALEXA 594"


def test_extract_channels_failsoft_on_empty() -> None:
    assert lif_metadata.extract_channels(None) == []
    assert lif_metadata.extract_channels(ET.fromstring("<Image />")) == []


def test_channels_to_ome_channels_shape() -> None:
    channels = lif_metadata.extract_channels(_scene_root())
    ome_channels = lif_metadata.channels_to_ome_channels(channels)

    assert len(ome_channels) == len(EXPECTED_CHANNELS)
    for ome_ch, (index, dye, ex, em_lo, em_hi) in zip(ome_channels, EXPECTED_CHANNELS):
        assert ome_ch.id == "Channel:0:{}".format(index)
        assert ome_ch.name == dye
        assert ome_ch.fluor == dye
        assert ome_ch.excitation_wavelength == ex
        assert ome_ch.excitation_wavelength_unit.value == "nm"
        # Emission is the band center.
        assert ome_ch.emission_wavelength == round((em_lo + em_hi) / 2)
        assert ome_ch.emission_wavelength_unit.value == "nm"


def test_channels_to_ome_channels_omits_unknown_fields() -> None:
    # Only a dye, no wavelengths -> name/fluor set, wavelengths omitted.
    channels: List[lif_metadata.ChannelIdentity] = [
        lif_metadata.ChannelIdentity(
            index=0,
            dye="DAPI",
            fluor="DAPI",
            detector=None,
            excitation_nm=None,
            emission_low_nm=None,
            emission_high_nm=None,
        )
    ]
    (ome_ch,) = lif_metadata.channels_to_ome_channels(channels)
    assert ome_ch.name == "DAPI"
    assert ome_ch.excitation_wavelength is None
    assert ome_ch.emission_wavelength is None


class _StubReader(Reader):
    """A ``Reader`` stand-in that drives ``ome_metadata`` from the confocal
    fixture.

    It overrides only the properties the getter reads, with correctly typed
    values, so the real ``ome_metadata`` runs without the git-LFS sample
    ``.lif`` binaries.
    """

    def __init__(self, scene_root: ET.Element) -> None:
        self._scene_root = scene_root

    @property
    def metadata(self) -> ET.Element:
        return self._scene_root

    @property
    def dims(self) -> dimensions.Dimensions:
        return dimensions.Dimensions("TCZYX", self.shape)

    @property
    def shape(self) -> Tuple[int, ...]:
        return (1, 7, 1, 64, 64)

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(np.uint16)

    @property
    def current_scene(self) -> str:
        return "confocal-scene"

    @property
    def current_scene_index(self) -> int:
        return 0

    @property
    def physical_pixel_sizes(self) -> types.PhysicalPixelSizes:
        return types.PhysicalPixelSizes(0.5, 0.1, 0.1)


def test_ome_metadata_builds_valid_ome() -> None:
    from ome_types.model import OME

    reader = _StubReader(_scene_root())
    ome = reader.ome_metadata

    assert isinstance(ome, OME)
    assert len(ome.images) == 1

    pixels = ome.images[0].pixels
    assert pixels.size_c == 7
    assert pixels.size_x == 64
    assert pixels.size_y == 64
    assert pixels.size_t == 1
    assert pixels.size_z == 1
    assert pixels.type.value == "uint16"
    # TCZYX data order -> XYZCT OME dimension order.
    assert pixels.dimension_order.value == "XYZCT"

    # Channels carry real fluorophore identity, not display colors.
    assert len(pixels.channels) == 7
    assert [c.name for c in pixels.channels] == [row[1] for row in EXPECTED_CHANNELS]
    assert pixels.channels[1].excitation_wavelength == 590

    # And the whole thing serializes to valid OME-XML.
    xml = ome.to_xml()
    assert "ALEXA 594" in xml


def test_ome_pixel_type_mapping() -> None:
    assert Reader._ome_pixel_type(np.dtype(np.uint8)) == "uint8"
    assert Reader._ome_pixel_type(np.dtype(np.uint16)) == "uint16"
    assert Reader._ome_pixel_type(np.dtype(np.float32)) == "float"
    assert Reader._ome_pixel_type(np.dtype(np.float64)) == "double"


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
