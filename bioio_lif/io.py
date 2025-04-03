import xml.etree.ElementTree as ET
from typing import Dict, Optional


def search_for_node(
    parent: ET.Element, tag: str, attributes: Optional[Dict[str, str]] = None
) -> ET.Element:
    """
    Recursively searches the XML tree for a node with the given tag and attributes.
    Raises a ValueError if no matching node is found.

    Parameters
    ----------
    parent : ET.Element
        The element to begin the search from.
    tag : str
        The XML tag to search for.
    attributes : dict, optional
        A dictionary of attributes that the node must match.
        If provided, the node's attributes must include all the given key-value pairs.

    Returns
    -------
    ET.Element
        The first matching XML node.

    Raises
    ------
    ValueError
        If no matching node is found.
    """
    if parent.tag == tag and (
        attributes is None or parent.attrib.items() >= attributes.items()
    ):
        return parent

    for child in parent:
        try:
            return search_for_node(child, tag, attributes)
        except ValueError:
            # Continue searching in the next child if not found in the current branch.
            pass

    raise ValueError(
        f"No matching node found for tag '{tag}' with attributes {attributes}"
    )
