"""This file contins different useful functions."""

from __future__ import annotations

from urllib.request import urlretrieve


def get_file(url: str, path: str):
    """Download file from url.

    Args:
        url (str): Url to download file from.
        path (str): Path to save file to

    Returns:
        (str): Filename of saved file,
        (Any): headers
    """
    filename, msg = urlretrieve(url, path)
    return filename, msg

