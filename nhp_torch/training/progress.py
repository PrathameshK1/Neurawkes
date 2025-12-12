from __future__ import annotations

from typing import Any, Iterable, TypeVar

T = TypeVar("T")


def tqdm_wrap(iterable: Iterable[T], **kwargs: Any) -> Iterable[T]:
    """Optional tqdm wrapper.

    - If `tqdm` is installed, returns a tqdm iterator.
    - Otherwise, returns the original iterable unchanged.
    """
    try:
        from tqdm import tqdm as _tqdm  # type: ignore[import-not-found]

        return _tqdm(iterable, **kwargs)
    except Exception:
        return iterable


