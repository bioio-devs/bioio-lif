"""
Build a small synthetic FLIM LIF from the XML header of a real Leica FALCON file.

Only the header is read from the source. The named scenes are kept with their metadata,
every image is shrunk, and the pixel data is replaced with synthetic blobs.

    python scripts/make_flim_fixtures.py SOURCE.lif OUTPUT.lif SCENE [SCENE ...]
"""

import argparse
import struct
import xml.etree.ElementTree as ET

import numpy as np

XY_DIVISOR = 16
MAX_Z = 3
MAX_T = 2
# Maps longer than their parent keep this many empty planes, like a stopped acquisition
EXTRA_Z = 2
DROPPED_ELEMENTS = {"LineOffsets", "FrameOffsets", "FrameProperties"}
DIM_X, DIM_Y, DIM_Z, DIM_T = "1", "2", "3", "4"


def get_dtype(channel: ET.Element) -> np.dtype:
    resolution = int(channel.get("Resolution"))
    if channel.get("DataType") == "1":
        return np.dtype(np.float16 if resolution == 16 else np.float32)
    return np.dtype(
        {8: np.uint8, 12: np.uint16, 16: np.uint16, 32: np.uint32}[resolution]
    )


def synthesize(name, dtype, resolution, shape, empty_from_z):
    """Fill a (C, T, Z, Y, X, S) array with a blob that differs by channel and time."""
    nc, nt, nz, ny, nx, ns = shape
    c, t, z, y, x, s = np.ogrid[:nc, :nt, :nz, :ny, :nx, :ns]
    radius = np.sqrt(
        ((x + 0.5) / nx - 0.5 - 0.05 * c) ** 2
        + ((y + 0.5) / ny - 0.5) ** 2
        + 0.5 * ((z + 0.5) / nz - 0.5) ** 2
    )
    blob = np.clip(1 - radius / 0.5, 0, 1) * (0.5 + 0.5 * (t + 1) / nt)
    blob = blob * (s + 1) / ns

    if name == "Fast Flim":
        values = 1.5 + 0.5 * c + blob
    elif name == "Standard Deviation":
        values = 0.1 + 0.2 * (1 - blob)
    elif name == "Phasor Real":
        values = 0.9 - 0.5 * blob
    elif name == "Phasor Imaginary":
        values = 0.3 + 0.1 * blob
    elif name == "Phasor Mask":
        values = np.where(blob > 0.5, 0xFF0000FF, 0)
    elif name in ("Intensity", "Phasor Intensity"):
        values = 50 * (c + 1) * blob
    else:
        values = (
            (2**resolution - 1)
            * (0.25 + 0.25 * min(nc - 1, 2) * c / max(nc - 1, 1))
            * blob
        )

    values = np.where(z >= empty_from_z, 0, values)
    if dtype.kind != "f":
        values = np.round(values)
    return np.broadcast_to(values, shape).astype(dtype)


def shrink_image(element, parent_z, parent_new_z):
    """Resize an image element in place, returning its new Z size, data and XY size."""
    image = element.find("Data/Image")
    channels = image.findall("ImageDescription/Channels/ChannelDescription")
    dims = {
        d.get("DimID"): d
        for d in image.findall("ImageDescription/Dimensions/DimensionDescription")
    }
    dtype = get_dtype(channels[0])
    interleaved = (
        len(channels) > 1 and int(channels[1].get("BytesInc")) == dtype.itemsize
    )

    old_z = int(dims[DIM_Z].get("NumberOfElements")) if DIM_Z in dims else 1
    if parent_z is None:
        new_z = min(old_z, MAX_Z)
    elif old_z <= parent_z:
        new_z = min(old_z, MAX_Z)
    else:
        new_z = parent_new_z + EXTRA_Z

    counts = {}
    increment = dtype.itemsize * (len(channels) if interleaved else 1)
    for dim_id, dim in dims.items():
        old = int(dim.get("NumberOfElements"))
        if dim_id in (DIM_X, DIM_Y):
            new = max(old // XY_DIVISOR, 4)
        elif dim_id == DIM_Z:
            new = new_z
        elif dim_id == DIM_T:
            new = min(old, MAX_T)
        else:
            raise ValueError(f"Unsupported dimension {dim_id} in {element.get('Name')}")

        length = float(dim.get("Length"))
        unitless = dim.get("Unit") == "m" and abs(length - (old - 1)) < 1e-3
        dim.set("Length", f"{new - 1 if unitless else length * new / old:.6e}")
        dim.set("NumberOfElements", str(new))
        dim.set("BytesInc", str(increment))
        increment *= new
        counts[dim_id] = new

    volume = increment
    for i, channel in enumerate(channels):
        channel.set("BytesInc", str(i * (dtype.itemsize if interleaved else volume)))

    shape = (
        1 if interleaved else len(channels),
        counts.get(DIM_T, 1),
        counts.get(DIM_Z, 1),
        counts[DIM_Y],
        counts[DIM_X],
        len(channels) if interleaved else 1,
    )
    data = synthesize(
        element.get("Name"),
        dtype,
        int(channels[0].get("Resolution")),
        shape,
        empty_from_z=parent_new_z if parent_z is not None else MAX_Z + EXTRA_Z,
    )

    timestamps = image.find("TimeStampList")
    if parent_z is None and timestamps is not None:
        planes_per_stamp = max(
            old_z * int(dims[DIM_T].get("NumberOfElements") if DIM_T in dims else 1), 1
        )
        stamps = timestamps.text.split() if timestamps.text else []
        if stamps and len(stamps) % planes_per_stamp == 0:
            keep = len(stamps) // planes_per_stamp * new_z * counts.get(DIM_T, 1)
            timestamps.text = " ".join(stamps[:keep]) + " "
            timestamps.set("NumberOfTimeStamps", str(keep))

    return new_z, data, counts[DIM_X], counts[DIM_Y]


def shrink_scene(scene, blocks):
    """Shrink a scene and everything under it, collecting (memory block id, bytes)."""
    parent_z = parent_new_z = None
    parent_xy = None

    def visit(element, is_scene):
        nonlocal parent_z, parent_new_z, parent_xy
        children = element.find("Children")
        if children is not None:
            for child in children.findall("Element"):
                if child.get("Name") in DROPPED_ELEMENTS:
                    children.remove(child)

        memory = element.find("Memory")
        if element.find("Data/Image") is not None:
            if is_scene:
                parent_z = int(
                    next(
                        (
                            d.get("NumberOfElements")
                            for d in element.findall(
                                "Data/Image/ImageDescription/Dimensions"
                                "/DimensionDescription"
                            )
                            if d.get("DimID") == DIM_Z
                        ),
                        1,
                    )
                )
                new_z, data, nx, ny = shrink_image(element, None, None)
                parent_new_z, parent_xy = new_z, (nx, ny)
            else:
                new_z, data, nx, ny = shrink_image(element, parent_z, parent_new_z)
            memory.set("Size", str(data.nbytes))
            blocks.append((memory.get("MemoryBlockID"), data.tobytes()))
        elif memory is not None:
            memory.set("Size", "0")
            blocks.append((memory.get("MemoryBlockID"), b""))

        for child in children.findall("Element") if children is not None else []:
            visit(child, False)

    visit(scene, True)

    # The raw FLIM image describes the same grid as its parent and its maps
    for flim in scene.iter("Element"):
        raw = flim.find("Data/SingleMoleculeDetection/Dataset/RawData")
        if raw is None:
            continue
        maps = flim.find("Children")
        fast_flim = next(
            e for e in maps.findall("Element") if e.get("Name") == "Fast Flim"
        )
        map_dims = {
            d.get("DimID"): d.get("NumberOfElements")
            for d in fast_flim.findall(
                "Data/Image/ImageDescription/Dimensions/DimensionDescription"
            )
        }
        new_sizes = {
            "X": parent_xy[0],
            "Y": parent_xy[1],
            "Z": map_dims.get(DIM_Z, 1),
            "T": map_dims.get(DIM_T, 1),
        }
        for dimension in raw.findall("Dimensions/Dimension"):
            axis = dimension.findtext("DimensionIdentifier")
            if axis in new_sizes:
                old = int(dimension.findtext("Size"))
                dimension.find("Size").text = str(new_sizes[axis])
                if axis in ("X", "Y"):
                    voxel = raw.find(f"VoxelSize{axis}")
                    voxel.text = repr(float(voxel.text) * old / new_sizes[axis])


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    parser.add_argument("source")
    parser.add_argument("output")
    parser.add_argument("scenes", nargs="+")
    args = parser.parse_args()

    with open(args.source, "rb") as f:
        f.seek(9)
        (xml_length,) = struct.unpack("<i", f.read(4))
        header = ET.fromstring(f.read(xml_length * 2).decode("utf-16-le"))

    root = header.find("Element")
    scenes = {e.get("Name"): e for e in root.find("Children").findall("Element")}
    root.find("Children").clear()
    root.set("Name", args.output.rsplit("/", 1)[-1])
    root.find("Data/Experiment").set("Path", root.get("Name"))

    blocks = [(root.find("Memory").get("MemoryBlockID"), b"")]
    for name in args.scenes:
        root.find("Children").append(scenes[name])
        shrink_scene(scenes[name], blocks)

    xml = ET.tostring(header, encoding="unicode")
    with open(args.output, "wb") as f:
        f.write(struct.pack("<iibi", 0x70, 2 * len(xml) + 5, 0x2A, len(xml)))
        f.write(xml.encode("utf-16-le"))
        for block_id, data in blocks:
            f.write(
                struct.pack("<iibqbi", 0x70, 0x2A, 0x2A, len(data), 0x2A, len(block_id))
            )
            f.write(block_id.encode("utf-16-le") + data)


if __name__ == "__main__":
    main()
