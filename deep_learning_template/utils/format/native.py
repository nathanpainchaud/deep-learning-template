from typing import Any, Mapping, Sequence, TypeVar, Union

StrMappingT = TypeVar("StrMappingT", bound=Mapping[str, Any])


def prefix(map: StrMappingT, prefix: str, exclude: Union[str, Sequence[str]] = None) -> StrMappingT:
    """Prepends a prefix to the keys of a mapping with string keys.

    Args:
        map: Mapping with string keys for which to add a prefix to the keys.
        prefix: Prefix to add to the current keys in the mapping.
        exclude: Keys to exclude from the prefix addition. These will remain unchanged in the new mapping.

    Returns:
        Mapping where the keys have been prepended with `prefix`.
    """
    if exclude is None:
        exclude = []
    elif isinstance(exclude, str):
        exclude = [exclude]

    return map.__class__(**{f"{prefix}{k}" if k not in exclude else k: v for k, v in map.items()})  # type: ignore
