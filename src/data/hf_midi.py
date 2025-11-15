"""Utilities for streaming MIDI datasets from Hugging Face."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional

import io

import requests
from datasets import Dataset, load_dataset
import pretty_midi
from urllib.parse import urlparse, urljoin


DEFAULT_URL_COLUMN = "file_name"


class MidiDownloadError(RuntimeError):
    """Raised when a MIDI file cannot be downloaded or parsed."""


@dataclass
class MidiFeatureResult:
    """Container for a single MIDI feature extraction result."""

    features: Dict[str, float]
    metadata: Dict[str, str]


def load_hf_dataset(
    dataset_name: str,
    *,
    split: str = "train",
    streaming: bool = False,
    url_column: str = DEFAULT_URL_COLUMN,
) -> Dataset:
    """Load a Hugging Face dataset that contains MIDI file URLs.

    Parameters
    ----------
    dataset_name:
        The dataset identifier on Hugging Face (e.g. ``"drengskapur/midi-classical-music"``).
    split:
        Which split to load. Defaults to ``"train"``.
    streaming:
        Whether to enable Hugging Face streaming mode to avoid local caching. Defaults to ``False``.
    url_column:
        The column in the dataset that contains the remote file URL. Used for validation only.

    Returns
    -------
    Dataset
        A Hugging Face dataset ready for iteration.
    """

    dataset = load_dataset(dataset_name, split=split, streaming=streaming)

    if url_column not in dataset.features:
        raise KeyError(
            f"Expected column '{url_column}' in dataset features {list(dataset.features)}"
        )

    return dataset


def _resolve_url(url: str, base_url: Optional[str]) -> str:
    """Return an absolute URL, using ``base_url`` if the URL is relative."""
    parsed = urlparse(url)
    if parsed.scheme:
        return url
    if base_url is None:
        raise MidiDownloadError(f"Missing scheme for URL '{url}' and no base URL provided.")
    return urljoin(base_url, url)


def load_midi_from_url(url: str, *, timeout: int = 30) -> pretty_midi.PrettyMIDI:
    """Download a MIDI file from a URL and parse it with ``pretty_midi``."""

    response = requests.get(url, timeout=timeout)
    if response.status_code != requests.codes.ok:
        raise MidiDownloadError(f"Failed to download MIDI from {url} (status {response.status_code})")

    midi_bytes = io.BytesIO(response.content)
    try:
        return pretty_midi.PrettyMIDI(midi_bytes)
    except Exception as exc:  # pylint: disable=broad-except
        raise MidiDownloadError(f"Failed to parse MIDI from {url}") from exc


def iter_dataset_urls(
    dataset: Dataset,
    *,
    url_column: str = DEFAULT_URL_COLUMN,
    max_items: Optional[int] = None,
) -> Iterable[str]:
    """Yield MIDI URLs from a Hugging Face dataset."""

    count = 0
    for row in dataset:
        if max_items is not None and count >= max_items:
            break
        if url_column not in row:
            continue
        yield row[url_column]
        count += 1


def extract_features_from_dataset(
    dataset: Dataset,
    feature_fn: Callable[[pretty_midi.PrettyMIDI], Dict[str, float]],
    *,
    url_column: str = DEFAULT_URL_COLUMN,
    max_items: Optional[int] = None,
    composer_from_url: Optional[Callable[[str], Optional[str]]] = None,
    progress_callback: Optional[Callable[[int], None]] = None,
    base_url: Optional[str] = None,
) -> Iterable[MidiFeatureResult]:
    """Stream features from a dataset by fetching each remote MIDI file."""

    processed = 0

    for row in dataset:
        if url_column not in row:
            continue

        raw_url = row[url_column]
        try:
            url = _resolve_url(raw_url, base_url)
        except MidiDownloadError:
            continue

        try:
            midi = load_midi_from_url(url)
        except MidiDownloadError:
            continue

        try:
            feature_values = feature_fn(midi)
        except MemoryError:
            continue
        except Exception:
            continue

        metadata: Dict[str, str] = {"file_url": url}
        if composer_from_url is not None:
            composer = composer_from_url(url)
            if composer:
                metadata["composer"] = composer

        yield MidiFeatureResult(features=feature_values, metadata=metadata)

        processed += 1
        if progress_callback is not None:
            progress_callback(processed)

        if max_items is not None and processed >= max_items:
            break

