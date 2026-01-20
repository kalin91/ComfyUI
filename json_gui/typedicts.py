"""TypedDicts used in json_gui."""

from typing import Any, Iterable, Optional, TypeGuard, TypedDict


class CreationDict(TypedDict):
    """Dictionary for expressing creation arguments."""

    args: Iterable[Any]
    kwargs: dict[str, Any]


class SavedImagesDict(TypedDict):
    """Dictionary for saved images."""

    created_images: list[str]
    last_saved_to_temp: Optional[bool]


class BodyNode(TypedDict):
    """Dictionary representing a node in the flow body."""

    type: str
    isArray: bool
    props: dict[str, Any]


class BodyDict(TypedDict):
    """Dictionary representing the flow body."""

    props: dict[str, BodyNode]


def get_empty_creation_dict() -> CreationDict:
    """Returns a new empty CreationDict."""
    return {"args": [], "kwargs": {}}


def is_bodydict(obj: object) -> TypeGuard[BodyDict]:
    """Typeguard for BodyDict."""
    if not isinstance(obj, dict):
        return False
    if "props" not in obj:
        return False
    props = obj["props"]
    if not isinstance(props, dict):
        return False
    for v in props.values():
        if not isinstance(v, dict):
            return False
        if not all(k in v for k in ("type", "isArray", "props")):
            return False
    return True
