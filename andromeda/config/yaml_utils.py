from __future__ import annotations

from typing import Any, IO, Optional, Union

import yaml


class AndromedaYamlLoader(yaml.SafeLoader):
    """Safe YAML loader with support for a small set of extra tags."""


class AndromedaYamlDumper(yaml.SafeDumper):
    """Safe YAML dumper with support for serializing extra Python types."""


def _construct_python_tuple(loader: AndromedaYamlLoader, node: yaml.Node) -> tuple[Any, ...]:
    return tuple(loader.construct_sequence(node))


def _represent_python_tuple(dumper: AndromedaYamlDumper, data: tuple[Any, ...]) -> yaml.Node:
    return dumper.represent_sequence("tag:yaml.org,2002:python/tuple", list(data))


def _represent_multiline_str(dumper: AndromedaYamlDumper, data: str) -> yaml.Node:
    """
    Represent multi-line strings as YAML block scalars.

    PyYAML's default behavior for strings containing newlines can produce
    hard-to-read quoted scalars with apparent blank lines. Using a literal block
    scalar keeps prompts readable and preserves line breaks.
    """

    if "\n" not in data and "\r" not in data:
        return yaml.representer.SafeRepresenter.represent_str(dumper, data)

    normalized = data.replace("\r\n", "\n").replace("\r", "\n")
    # Avoid an extra blank line at the end of the YAML block output.
    normalized = normalized.rstrip("\n")
    return dumper.represent_scalar("tag:yaml.org,2002:str", normalized, style="|")


AndromedaYamlLoader.add_constructor("tag:yaml.org,2002:python/tuple", _construct_python_tuple)
AndromedaYamlDumper.add_representer(tuple, _represent_python_tuple)
AndromedaYamlDumper.add_representer(str, _represent_multiline_str)


YamlStream = Union[str, bytes, IO[str], IO[bytes]]


def yaml_load(stream: YamlStream) -> Any:
    """Like yaml.safe_load, but supports a limited set of extra tags (e.g. python/tuple)."""

    return yaml.load(stream, Loader=AndromedaYamlLoader)


def yaml_dump(data: Any, stream: Optional[IO[str]] = None, **kwargs: Any) -> Optional[str]:
    """Like yaml.safe_dump, but preserves tuples as !!python/tuple for round-trips."""

    return yaml.dump(data, stream, Dumper=AndromedaYamlDumper, **kwargs)
