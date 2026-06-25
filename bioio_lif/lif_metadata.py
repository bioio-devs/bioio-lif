#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Per-channel identity extraction for Leica LIF scene XML.

A single Leica ``.lif`` scene preserves its full acquisition settings inside the
LIF XML. :func:`extract_channels` turns one scene's XML element into one
identity dict per image channel -- dye, detector, excitation line, and emission
band. The core parser is deliberately standard-library only so it stays cheap
and dependency-free; the :func:`channels_to_ome_channels` projection is the only
piece that imports ``ome_types``.

The hard part is the JOIN, not the parsing. A Leica scene stores several copies
of its instrument settings, and only one set is the *real* acquisition:

* ``LDM_Block_Sequential / LDM_Block_Sequential_List`` holds the genuine
  sequential settings, in document order. A channel's ``SequentialSettingIndex``
  indexes directly into this list. This is the source of truth.
* ``LDM_Block_Sequential_Master`` and the top-level
  ``Attachment[HardwareSetting]`` copies are reference/duplicate snapshots, NOT
  real sequences. The Master copy in particular can carry a phantom laser line
  that excites nothing in the acquired data; folding it in yields wrong answers.
  We never read from them.

Within a real sequence, excitation is recovered by spectral pairing: the active
laser lines (``LaserLineSetting`` with ``IntensityDev > 0``) are matched, in
ascending wavelength order, to the active detectors (``Detector`` with
``IsActive == "1"``) in ascending channel order. A channel's excitation is the
laser paired to its own physical detector channel. Emission comes from the
``MultiBand`` whose ``Channel`` matches that same physical channel.

Everything is fail-soft: any missing structure degrades the affected field to
``None`` rather than raising, so a partial or unusual scene never breaks the
reader.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from typing import List, Optional, TypedDict, Union

# Vendor prefix stripped from dye names: "Leica/ALEXA 594" -> "ALEXA 594".
_DYE_VENDOR_PREFIX = "Leica/"

Number = Union[int, float]


class ChannelIdentity(TypedDict):
    """One image channel's resolved identity, in acquisition order."""

    index: int
    dye: Optional[str]
    fluor: Optional[str]
    detector: Optional[str]
    excitation_nm: Optional[Number]
    emission_low_nm: Optional[int]
    emission_high_nm: Optional[int]


def _cp_map(channel_desc: ET.Element) -> dict:
    """Flatten a ``ChannelDescription``'s ``ChannelProperty`` Key/Value pairs."""
    props: dict = {}
    for prop in channel_desc.iter("ChannelProperty"):
        key = prop.find("Key")
        val = prop.find("Value")
        if key is not None and key.text:
            props[key.text.strip()] = (
                (val.text or "").strip() if val is not None else ""
            )
    return props


def _to_number(text: Optional[str]) -> Optional[Number]:
    """Parse a numeric string -> int when integral, else float, else None.

    Non-finite values (``inf`` / ``nan``) are rejected as ``None`` -- they are
    never valid wavelengths and would otherwise break ``round``.
    """
    if text is None:
        return None
    try:
        value = float(text)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return int(value) if value.is_integer() else value


def _sequence_blocks(root: ET.Element) -> List[ET.Element]:
    """The real acquisition sequences, in document order.

    These are the ``ATLConfocalSettingDefinition`` children of
    ``LDM_Block_Sequential / LDM_Block_Sequential_List`` -- and ONLY those. The
    ``LDM_Block_Sequential_Master`` block and the top-level
    ``Attachment[HardwareSetting]`` copies are reference/duplicate snapshots and
    are deliberately excluded; ``SequentialSettingIndex`` indexes into this list.
    """
    blocks: List[ET.Element] = []
    # ``iter`` walks in document order, so each list's settings stay ordered and
    # multiple sequential lists (rare) concatenate in the order they appear.
    for seq_list in root.iter("LDM_Block_Sequential_List"):
        for atl in seq_list.iter("ATLConfocalSettingDefinition"):
            blocks.append(atl)
    return blocks


def _detector_channel_map(root: ET.Element) -> dict:
    """Map every ``Detector`` ``Name`` to its physical ``Channel`` number.

    The mapping (e.g. ``HyD S 1`` -> 1, ``HyD X 2`` -> 2) is consistent across
    every block, so we scan the whole scene to be robust to partial blocks.
    """
    mapping: dict = {}
    for det in root.iter("Detector"):
        name = det.attrib.get("Name")
        channel = det.attrib.get("Channel")
        if not name or channel is None:
            continue
        try:
            mapping[name] = int(channel)
        except ValueError:
            continue
    return mapping


def _active_lasers(block: ET.Element) -> List[Number]:
    """Excited laser lines (``IntensityDev > 0``) for a sequence, ascending."""
    lasers: List[Number] = []
    for laser in block.iter("LaserLineSetting"):
        intensity = _to_number(laser.attrib.get("IntensityDev"))
        line = _to_number(laser.attrib.get("LaserLine"))
        if intensity is not None and intensity > 0 and line is not None:
            lasers.append(line)
    return sorted(lasers)


def _active_detector_channels(block: ET.Element) -> List[int]:
    """Physical channels of active detectors (``IsActive == "1"``), ascending."""
    channels: List[int] = []
    for det in block.iter("Detector"):
        if det.attrib.get("IsActive") != "1":
            continue
        channel = det.attrib.get("Channel")
        if channel is None:
            continue
        try:
            channels.append(int(channel))
        except ValueError:
            continue
    return sorted(channels)


def _excitation_for(
    block: ET.Element, physical_channel: Optional[int]
) -> Optional[Number]:
    """Excitation line for ``physical_channel`` within one sequence.

    Spectral pairing: the i-th lowest active laser drives the i-th lowest active
    detector. Find this channel's position among the active detectors, then read
    the laser at the same position.
    """
    if physical_channel is None:
        return None
    lasers = _active_lasers(block)
    detectors = _active_detector_channels(block)
    if physical_channel not in detectors:
        return None
    position = detectors.index(physical_channel)
    if position >= len(lasers):
        return None
    return lasers[position]


def _emission_for(block: ET.Element, physical_channel: Optional[int]) -> tuple:
    """``(low, high)`` emission band (rounded nm) for ``physical_channel``."""
    if physical_channel is None:
        return None, None
    for band in block.iter("MultiBand"):
        channel = _to_number(band.attrib.get("Channel"))
        if channel is None or int(channel) != physical_channel:
            continue
        left = _to_number(band.attrib.get("LeftWorld"))
        right = _to_number(band.attrib.get("RightWorld"))
        low = round(left) if left is not None else None
        high = round(right) if right is not None else None
        return low, high
    return None, None


def _strip_dye_prefix(dye: Optional[str]) -> Optional[str]:
    """Drop the leading ``"Leica/"`` vendor prefix from a dye name."""
    if dye is None:
        return None
    if dye.startswith(_DYE_VENDOR_PREFIX):
        return dye[len(_DYE_VENDOR_PREFIX) :]
    return dye


def channel_dye_name(channel_desc: ET.Element) -> Optional[str]:
    """The fluorophore/dye for one ``ChannelDescription``, or ``None``.

    Reads the channel's ``ChannelProperty`` ``DyeName`` (the real fluorophore,
    e.g. "ALEXA 594") and strips any leading ``"Leica/"`` vendor prefix. Returns
    ``None`` when no dye is recorded -- callers fall back to ``LUTName``.

    This is the small, isolated helper the reader uses to name confocal channels
    without pulling in the full sequence join.
    """
    props = _cp_map(channel_desc)
    return _strip_dye_prefix(props.get("DyeName") or None)


def extract_channels(scene_root: ET.Element) -> List[ChannelIdentity]:
    """Extract per-channel identity from one Leica LIF scene XML element.

    ``scene_root`` is any element that contains the scene's ``ChannelDescription``
    elements and ``LDM_Block_Sequential`` settings as descendants (e.g. the
    ``<Image>`` node, or the enclosing scene ``<Element>``).

    Returns one dict per image channel, in acquisition (document) order, each
    with exactly the keys ``index``, ``dye``, ``fluor``, ``detector``,
    ``excitation_nm``, ``emission_low_nm``, ``emission_high_nm``. Individual
    undeterminable fields degrade to ``None`` rather than dropping the channel;
    any unexpected structural surprise yields ``[]``.
    """
    if scene_root is None:
        return []

    try:
        sequences = _sequence_blocks(scene_root)
        detector_to_channel = _detector_channel_map(scene_root)

        channels: List[ChannelIdentity] = []
        for index, channel_desc in enumerate(scene_root.iter("ChannelDescription")):
            props = _cp_map(channel_desc)

            dye = _strip_dye_prefix(props.get("DyeName") or None)
            detector_name = props.get("DetectorName") or None
            physical_channel = (
                detector_to_channel.get(detector_name) if detector_name else None
            )

            # Resolve which real sequence acquired this channel.
            block = None
            seq_index = _to_number(props.get("SequentialSettingIndex"))
            if seq_index is not None and 0 <= int(seq_index) < len(sequences):
                block = sequences[int(seq_index)]

            if block is not None:
                excitation = _excitation_for(block, physical_channel)
                emission_low, emission_high = _emission_for(block, physical_channel)
            else:
                excitation = None
                emission_low = emission_high = None

            channels.append(
                ChannelIdentity(
                    index=index,
                    dye=dye,
                    fluor=dye,
                    detector=detector_name,
                    excitation_nm=excitation,
                    emission_low_nm=emission_low,
                    emission_high_nm=emission_high,
                )
            )
        return channels
    except Exception:
        # Any unexpected structural surprise stays fail-soft.
        return []


def channels_to_ome_channels(channels: List[ChannelIdentity]) -> list:
    """Project identity dicts into ``ome_types`` ``Channel`` elements.

    One ``Channel`` per input channel: ``id=Channel:0:<index>``, ``name`` the
    dye, ``fluor`` the dye, excitation with ``nm`` units, and emission with
    ``nm`` units at the band center. Any field whose source is ``None`` is
    omitted entirely rather than set to a bogus value, so the element never
    claims wavelengths it does not have.

    Imported lazily so the rest of this module stays standard-library only on
    import.
    """
    from ome_types.model import Channel

    out = []
    for index, ch in enumerate(channels):
        # ``"...".format`` rather than an f-string: the channel-id literal has a
        # ``:<digit>`` run that pycodestyle (flake8's E231) mis-flags inside an
        # f-string on the versions this project pins.
        fields: dict = {"id": "Channel:0:{}".format(index)}

        dye = ch.get("dye")
        if dye is not None:
            fields["name"] = dye
        fluor = ch.get("fluor")
        if fluor is not None:
            fields["fluor"] = fluor

        excitation = ch.get("excitation_nm")
        if excitation is not None:
            fields["excitation_wavelength"] = excitation
            fields["excitation_wavelength_unit"] = "nm"

        low = ch.get("emission_low_nm")
        high = ch.get("emission_high_nm")
        if low is not None and high is not None:
            fields["emission_wavelength"] = round((low + high) / 2)
            fields["emission_wavelength_unit"] = "nm"

        out.append(Channel(**fields))
    return out
