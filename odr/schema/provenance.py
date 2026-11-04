"""Semantic component identities without credentials or transient runtime state."""

from __future__ import annotations

import inspect
import json
from collections.abc import Mapping
from typing import Any, Protocol

from pydantic import BaseModel

IDENTITY_VERSION = "decisionrag-components-v1"
ManifestFields = tuple[tuple[str, str], ...]
PRIVATE_SETTINGS = frozenset(
    ("api_key", "password", "authorization", "token", "http_client", "cache")
)


class SemanticComponent(Protocol):
    """Explicit provenance contract for configurable user-defined components."""

    def semantic_identity(self) -> object:
        """Return JSON-compatible semantic configuration without secrets.

        :returns: Configuration and explicit semantic revision of the implementation.
        """
        ...


def component_identity(component: object) -> Any:
    """Describe constructor configuration or an explicit semantic identity hook.

    :param component: Injected implementation or serializable configuration.
    :returns: JSON-compatible identity without transient locks, logs or loaded weights.
    """
    return semantic_value(component, set())


def semantic_value(supplied: object, visited: set[int]) -> Any:
    """Normalize safe configuration values recursively.

    :param supplied: Configuration supplied under inspection.
    :param visited: Ancestor objects used to reject cyclic configurations.
    :returns: JSON-compatible semantic supplied.
    :raises ValueError: If component identities contain cycles.
    """
    if supplied is None or isinstance(supplied, (str, bool, int, float)):
        return supplied
    if isinstance(supplied, BaseModel):
        return semantic_value(supplied.model_dump(mode="json"), visited)
    if isinstance(supplied, Mapping):
        return {
            str(key): semantic_value(entry, visited)
            for key, entry in supplied.items()
            if str(key).lower() not in PRIVATE_SETTINGS
        }
    if isinstance(supplied, (list, tuple)):
        return [semantic_value(entry, visited) for entry in supplied]
    if id(supplied) in visited:
        raise ValueError("Cyclic semantic configuration; provide semantic_identity()")
    return object_identity(supplied, visited | {id(supplied)})


def object_identity(component: object, visited: set[int]) -> dict[str, Any]:
    """Inspect named constructor inputs, excluding credentials and runtime objects.

    :param component: Concrete injected component.
    :param visited: Ancestor identities used to reject cycles.
    :returns: Qualified implementation identity and normalized configuration.
    """
    implementation = type(component)
    identity: dict[str, Any] = dict(
        type=f"{implementation.__module__}.{implementation.__qualname__}",
        implementation_revision=IDENTITY_VERSION,
    )
    hook = getattr(component, "semantic_identity", None)
    if callable(hook):
        identity["configuration"] = semantic_value(hook(), visited)
    elif callable(component) and hasattr(component, "__qualname__"):
        identity["callable"] = f"{component.__module__}.{component.__qualname__}"
    else:
        identity["configuration"] = constructor_configuration(component, visited)
    return identity


def constructor_configuration(component: object, visited: set[int]) -> dict[str, Any]:
    """Capture declared constructor inputs that are stored on the component.

    :param component: Concrete component implementation.
    :param visited: Ancestor identities for cycle detection.
    :returns: Semantic constructor configuration.
    """
    return {
        name: semantic_value(getattr(component, name), visited)
        for name in inspect.signature(type(component).__init__).parameters
        if name != "self"
        and name.lower() not in PRIVATE_SETTINGS
        and hasattr(component, name)
    }


def build_manifest(components: Mapping[str, object]) -> ManifestFields:
    """Persist auditable component identities using stable JSON fields.

    :param components: Actual builder, chunker, models, settings and source content.
    :returns: Immutable manifest fields suitable for snapshot hashing and inspection.
    """
    return tuple(
        (
            name,
            json.dumps(
                component_identity(component),
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            ),
        )
        for name, component in sorted(components.items())
    )
