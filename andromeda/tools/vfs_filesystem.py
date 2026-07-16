from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import fnmatch
import hashlib
import json
from pathlib import PurePosixPath
import re
from typing import Any, Callable, Iterator, Literal, Optional

from langchain_core.tools import tool
from pydantic import BaseModel
from andromeda.utils.ignore_rules import VFSIgnoreMatcher, manual_ignore_matches


_PATCH_FILE_HEADER_RE = re.compile(r"^\*\*\* (Add|Update|Delete) File: (.+)$")


def normalize_virtual_path(path: str) -> str:
    raw = str(path or "/").replace("\\", "/")
    pure = PurePosixPath(raw if raw.startswith("/") else f"/{raw}")
    normalized = re.sub(r"/+", "/", pure.as_posix())
    if normalized != "/" and normalized.endswith("/"):
        normalized = normalized[:-1]
    parts = [part for part in normalized.split("/") if part]
    if any(part in {"..", ""} for part in parts):
        raise ValueError("Path traversal is not allowed.")
    return normalized or "/"


def parent_path(path: str) -> str:
    normalized = normalize_virtual_path(path)
    if normalized == "/":
        return "/"
    parent = str(PurePosixPath(normalized).parent)
    return parent if parent.startswith("/") else f"/{parent}"


def basename(path: str) -> str:
    normalized = normalize_virtual_path(path)
    return "/" if normalized == "/" else PurePosixPath(normalized).name


def _escape_like_pattern(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _subtree_like_pattern(path: str) -> str:
    normalized = normalize_virtual_path(path)
    return _escape_like_pattern(normalized.rstrip("/") + "/") + "%"


@dataclass(frozen=True)
class FileEntry:
    path: str
    node_type: Literal["file", "directory"]
    size_bytes: int = 0
    updated_at: str | None = None
    metadata: dict[str, Any] | None = None

    @property
    def name(self) -> str:
        return basename(self.path)


@dataclass(frozen=True)
class FileRecord:
    path: str
    content: str
    encoding: str = "utf-8"
    revision: int = 1
    created_at: str | None = None
    updated_at: str | None = None
    metadata: dict[str, Any] | None = None


class FilesystemDriver(ABC):
    @abstractmethod
    def read(self, path: str) -> FileRecord:
        raise NotImplementedError

    @abstractmethod
    def write(
        self,
        path: str,
        content: str,
        *,
        overwrite: bool = True,
        metadata: Optional[dict[str, Any]] = None,
    ) -> FileRecord:
        raise NotImplementedError

    @abstractmethod
    def append(self, path: str, content: str) -> FileRecord:
        raise NotImplementedError

    @abstractmethod
    def mkdir(self, path: str, *, create_parents: bool = True) -> FileEntry:
        raise NotImplementedError

    @abstractmethod
    def ls(self, path: str = "/") -> list[FileEntry]:
        raise NotImplementedError

    @abstractmethod
    def delete(self, path: str, *, recursive: bool = False) -> None:
        raise NotImplementedError

    @abstractmethod
    def stat(self, path: str) -> FileEntry:
        raise NotImplementedError

    @abstractmethod
    def exists(self, path: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def walk(self, path: str = "/") -> list[FileEntry]:
        raise NotImplementedError


class InMemoryFilesystemDriver(FilesystemDriver):
    """Small VFS driver for tests and non-persistent local provider checks."""

    def __init__(self) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._nodes: dict[str, FileEntry] = {
            "/": FileEntry(path="/", node_type="directory", updated_at=now)
        }
        self._files: dict[str, FileRecord] = {}

    def _ensure_parent(self, path: str) -> None:
        parent = parent_path(path)
        if parent != "/" and parent not in self._nodes:
            self.mkdir(parent)
        if self._nodes.get(parent, FileEntry("/", "directory")).node_type != "directory":
            raise NotADirectoryError(parent)

    def read(self, path: str) -> FileRecord:
        normalized = normalize_virtual_path(path)
        record = self._files.get(normalized)
        if record is None:
            if normalized in self._nodes and self._nodes[normalized].node_type == "directory":
                raise IsADirectoryError(normalized)
            raise FileNotFoundError(normalized)
        return record

    def write(
        self,
        path: str,
        content: str,
        *,
        overwrite: bool = True,
        metadata: Optional[dict[str, Any]] = None,
    ) -> FileRecord:
        normalized = normalize_virtual_path(path)
        if normalized == "/":
            raise IsADirectoryError("Cannot write root directory.")
        if normalized in self._files and not overwrite:
            raise FileExistsError(normalized)
        self._ensure_parent(normalized)
        now = datetime.now(timezone.utc).isoformat()
        previous = self._files.get(normalized)
        revision = 1 if previous is None else previous.revision + 1
        record = FileRecord(
            path=normalized,
            content=content,
            revision=revision,
            created_at=previous.created_at if previous else now,
            updated_at=now,
            metadata=metadata or {},
        )
        self._files[normalized] = record
        self._nodes[normalized] = FileEntry(
            path=normalized,
            node_type="file",
            size_bytes=len(content.encode("utf-8")),
            updated_at=now,
            metadata=metadata or {},
        )
        return record

    def append(self, path: str, content: str) -> FileRecord:
        existing = self.read(path).content if self.exists(path) else ""
        return self.write(path, existing + content)

    def mkdir(self, path: str, *, create_parents: bool = True) -> FileEntry:
        normalized = normalize_virtual_path(path)
        if normalized in self._files:
            raise FileExistsError(normalized)
        if normalized != "/":
            parent = parent_path(normalized)
            if parent not in self._nodes:
                if not create_parents:
                    raise FileNotFoundError(parent)
                self.mkdir(parent)
        now = datetime.now(timezone.utc).isoformat()
        self._nodes[normalized] = FileEntry(
            path=normalized,
            node_type="directory",
            updated_at=now,
        )
        return self._nodes[normalized]

    def ls(self, path: str = "/") -> list[FileEntry]:
        normalized = normalize_virtual_path(path)
        if normalized not in self._nodes:
            raise FileNotFoundError(normalized)
        if self._nodes[normalized].node_type != "directory":
            raise NotADirectoryError(normalized)
        prefix = "" if normalized == "/" else normalized
        entries = []
        for candidate, entry in self._nodes.items():
            if candidate == normalized:
                continue
            if parent_path(candidate) == (prefix or "/"):
                entries.append(entry)
        return sorted(entries, key=lambda item: item.path)

    def delete(self, path: str, *, recursive: bool = False) -> None:
        normalized = normalize_virtual_path(path)
        if normalized == "/":
            raise ValueError("Cannot delete root directory.")
        if normalized in self._files:
            self._files.pop(normalized, None)
            self._nodes.pop(normalized, None)
            return
        if normalized not in self._nodes:
            raise FileNotFoundError(normalized)
        children = [entry.path for entry in self.walk(normalized) if entry.path != normalized]
        if children and not recursive:
            raise OSError(f"Directory is not empty: {normalized}")
        for child in sorted(children, key=len, reverse=True):
            self._files.pop(child, None)
            self._nodes.pop(child, None)
        self._nodes.pop(normalized, None)

    def stat(self, path: str) -> FileEntry:
        normalized = normalize_virtual_path(path)
        entry = self._nodes.get(normalized)
        if entry is None:
            raise FileNotFoundError(normalized)
        return entry

    def exists(self, path: str) -> bool:
        try:
            return normalize_virtual_path(path) in self._nodes
        except ValueError:
            return False

    def walk(self, path: str = "/") -> list[FileEntry]:
        normalized = normalize_virtual_path(path)
        if normalized not in self._nodes:
            raise FileNotFoundError(normalized)
        prefix = normalized.rstrip("/") + "/"
        entries = [self._nodes[normalized]]
        entries.extend(
            entry
            for candidate, entry in self._nodes.items()
            if candidate != normalized and candidate.startswith(prefix)
        )
        return sorted(entries, key=lambda item: item.path)


class PostgresFilesystemDriver(FilesystemDriver):
    SCHEMA_SQL = """
    create table if not exists vfs_namespaces (
        id bigserial primary key,
        namespace_key text not null unique,
        kind text not null default 'generic',
        metadata jsonb not null default '{}'::jsonb,
        created_at timestamptz not null default now(),
        updated_at timestamptz not null default now()
    );
    create table if not exists vfs_nodes (
        id bigserial primary key,
        namespace_id bigint not null references vfs_namespaces(id),
        parent_id bigint references vfs_nodes(id),
        path text not null,
        name text not null,
        node_type text not null check (node_type in ('file', 'directory')),
        size_bytes bigint not null default 0,
        current_revision int not null default 0,
        metadata jsonb not null default '{}'::jsonb,
        deleted_at timestamptz,
        created_at timestamptz not null default now(),
        updated_at timestamptz not null default now(),
        unique (namespace_id, path)
    );
    create table if not exists vfs_revisions (
        id bigserial primary key,
        node_id bigint not null references vfs_nodes(id),
        revision_no int not null,
        content_text text,
        content_sha256 text not null,
        encoding text not null default 'utf-8',
        metadata jsonb not null default '{}'::jsonb,
        created_at timestamptz not null default now(),
        unique (node_id, revision_no)
    );
    create index if not exists idx_vfs_nodes_namespace_parent
        on vfs_nodes(namespace_id, parent_id) where deleted_at is null;
    create index if not exists idx_vfs_revisions_node_revision
        on vfs_revisions(node_id, revision_no desc);
    """

    def __init__(
        self,
        *,
        connection_factory: Callable[[], Any],
        namespace_key: str,
        namespace_kind: str = "workspace",
        namespace_metadata: Optional[dict[str, Any]] = None,
        ensure_schema: bool = False,
        auto_create_namespace: bool = True,
    ) -> None:
        self.connection_factory = connection_factory
        self.namespace_key = namespace_key
        self.namespace_kind = namespace_kind
        self.namespace_metadata = namespace_metadata or {}
        if ensure_schema:
            self.ensure_schema()
        if auto_create_namespace:
            self.ensure_namespace()

    @contextmanager
    def _connection(self) -> Iterator[Any]:
        connection = self.connection_factory()
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @contextmanager
    def _cursor(self, connection: Any) -> Iterator[Any]:
        cursor = connection.cursor()
        try:
            yield cursor
        finally:
            close = getattr(cursor, "close", None)
            if callable(close):
                close()

    def ensure_schema(self) -> None:
        with self._connection() as connection:
            with self._cursor(connection) as cursor:
                cursor.execute(self.SCHEMA_SQL)

    def ensure_namespace(self) -> None:
        with self._connection() as connection:
            with self._cursor(connection) as cursor:
                cursor.execute(
                    """
                    insert into vfs_namespaces(namespace_key, kind, metadata)
                    values (%s, %s, %s::jsonb)
                    on conflict (namespace_key)
                    do update set kind = excluded.kind,
                                  metadata = vfs_namespaces.metadata || excluded.metadata,
                                  updated_at = now()
                    """,
                    (
                        self.namespace_key,
                        self.namespace_kind,
                        json.dumps(self.namespace_metadata),
                    ),
                )
            self._ensure_root_directory(connection)

    def _namespace_id(self, cursor: Any) -> int:
        cursor.execute(
            "select id from vfs_namespaces where namespace_key = %s",
            (self.namespace_key,),
        )
        row = cursor.fetchone()
        if row is None:
            raise ValueError(f"Namespace {self.namespace_key!r} does not exist.")
        return int(row[0])

    def _get_node(self, cursor: Any, path: str) -> Any:
        normalized = normalize_virtual_path(path)
        cursor.execute(
            """
            select n.id, n.parent_id, n.path, n.name, n.node_type, n.size_bytes,
                   n.current_revision, n.metadata, n.created_at, n.updated_at
            from vfs_nodes n
            join vfs_namespaces ns on ns.id = n.namespace_id
            where ns.namespace_key = %s and n.path = %s and n.deleted_at is null
            """,
            (self.namespace_key, normalized),
        )
        return cursor.fetchone()

    def _ensure_root_directory(self, connection: Any) -> None:
        with self._cursor(connection) as cursor:
            namespace_id = self._namespace_id(cursor)
            cursor.execute(
                """
                insert into vfs_nodes(namespace_id, parent_id, path, name, node_type, metadata)
                values (%s, null, '/', '/', 'directory', '{}'::jsonb)
                on conflict (namespace_id, path)
                do update set deleted_at = null, node_type = 'directory', updated_at = now()
                """,
                (namespace_id,),
            )

    def _ensure_directory(self, cursor: Any, path: str) -> Any:
        normalized = normalize_virtual_path(path)
        node = self._get_node(cursor, normalized)
        if node is not None:
            if node[4] != "directory":
                raise NotADirectoryError(normalized)
            return node
        parent = self._ensure_directory(cursor, parent_path(normalized))
        namespace_id = self._namespace_id(cursor)
        cursor.execute(
            """
            insert into vfs_nodes(namespace_id, parent_id, path, name, node_type, metadata)
            values (%s, %s, %s, %s, 'directory', '{}'::jsonb)
            on conflict (namespace_id, path)
            do update set deleted_at = null, node_type = 'directory', parent_id = excluded.parent_id,
                          updated_at = now()
            returning id, parent_id, path, name, node_type, size_bytes,
                      current_revision, metadata, created_at, updated_at
            """,
            (namespace_id, parent[0], normalized, basename(normalized)),
        )
        return cursor.fetchone()

    def _entry_from_row(self, row: Any) -> FileEntry:
        return FileEntry(
            path=row[2],
            node_type=row[4],
            size_bytes=int(row[5] or 0),
            updated_at=row[9].isoformat() if hasattr(row[9], "isoformat") else str(row[9]),
            metadata=row[7] if isinstance(row[7], dict) else {},
        )

    def read(self, path: str) -> FileRecord:
        normalized = normalize_virtual_path(path)
        with self._connection() as connection:
            with self._cursor(connection) as cursor:
                node = self._get_node(cursor, normalized)
                if node is None:
                    raise FileNotFoundError(normalized)
                if node[4] == "directory":
                    raise IsADirectoryError(normalized)
                cursor.execute(
                    """
                    select content_text, encoding, revision_no, metadata, created_at
                    from vfs_revisions
                    where node_id = %s and revision_no = %s
                    """,
                    (node[0], node[6]),
                )
                revision = cursor.fetchone()
                if revision is None:
                    raise FileNotFoundError(normalized)
                return FileRecord(
                    path=normalized,
                    content=revision[0] or "",
                    encoding=revision[1] or "utf-8",
                    revision=int(revision[2]),
                    created_at=node[8].isoformat() if hasattr(node[8], "isoformat") else str(node[8]),
                    updated_at=node[9].isoformat() if hasattr(node[9], "isoformat") else str(node[9]),
                    metadata=revision[3] if isinstance(revision[3], dict) else {},
                )

    def write(
        self,
        path: str,
        content: str,
        *,
        overwrite: bool = True,
        metadata: Optional[dict[str, Any]] = None,
    ) -> FileRecord:
        normalized = normalize_virtual_path(path)
        if normalized == "/":
            raise IsADirectoryError("Cannot write root directory.")
        with self._connection() as connection:
            with self._cursor(connection) as cursor:
                existing = self._get_node(cursor, normalized)
                if existing is not None and existing[4] == "file" and not overwrite:
                    raise FileExistsError(normalized)
                parent = self._ensure_directory(cursor, parent_path(normalized))
                namespace_id = self._namespace_id(cursor)
                next_revision = 1 if existing is None else int(existing[6]) + 1
                cursor.execute(
                    """
                    insert into vfs_nodes(namespace_id, parent_id, path, name, node_type,
                                          size_bytes, current_revision, metadata)
                    values (%s, %s, %s, %s, 'file', %s, %s, %s::jsonb)
                    on conflict (namespace_id, path)
                    do update set deleted_at = null,
                                  parent_id = excluded.parent_id,
                                  node_type = 'file',
                                  size_bytes = excluded.size_bytes,
                                  current_revision = vfs_nodes.current_revision + 1,
                                  metadata = excluded.metadata,
                                  updated_at = now()
                    returning id, current_revision
                    """,
                    (
                        namespace_id,
                        parent[0],
                        normalized,
                        basename(normalized),
                        len(content.encode("utf-8")),
                        next_revision,
                        json.dumps(metadata or {}),
                    ),
                )
                node_id, revision_no = cursor.fetchone()
                cursor.execute(
                    """
                    insert into vfs_revisions(node_id, revision_no, content_text, content_sha256,
                                              encoding, metadata)
                    values (%s, %s, %s, %s, 'utf-8', %s::jsonb)
                    """,
                    (
                        node_id,
                        revision_no,
                        content,
                        hashlib.sha256(content.encode("utf-8")).hexdigest(),
                        json.dumps(metadata or {}),
                    ),
                )
        return self.read(normalized)

    def append(self, path: str, content: str) -> FileRecord:
        existing = self.read(path).content if self.exists(path) else ""
        return self.write(path, existing + content)

    def mkdir(self, path: str, *, create_parents: bool = True) -> FileEntry:
        normalized = normalize_virtual_path(path)
        with self._connection() as connection:
            with self._cursor(connection) as cursor:
                if not create_parents and parent_path(normalized) != "/" and self._get_node(cursor, parent_path(normalized)) is None:
                    raise FileNotFoundError(parent_path(normalized))
                node = self._ensure_directory(cursor, normalized)
                return self._entry_from_row(node)

    def ls(self, path: str = "/") -> list[FileEntry]:
        normalized = normalize_virtual_path(path)
        with self._connection() as connection:
            with self._cursor(connection) as cursor:
                node = self._get_node(cursor, normalized)
                if node is None:
                    raise FileNotFoundError(normalized)
                if node[4] != "directory":
                    raise NotADirectoryError(normalized)
                cursor.execute(
                    """
                    select n.id, n.parent_id, n.path, n.name, n.node_type, n.size_bytes,
                           n.current_revision, n.metadata, n.created_at, n.updated_at
                    from vfs_nodes n
                    where n.parent_id = %s and n.deleted_at is null
                    order by n.name
                    """,
                    (node[0],),
                )
                return [self._entry_from_row(row) for row in cursor.fetchall()]

    def delete(self, path: str, *, recursive: bool = False) -> None:
        normalized = normalize_virtual_path(path)
        if normalized == "/":
            raise ValueError("Cannot delete root directory.")
        with self._connection() as connection:
            with self._cursor(connection) as cursor:
                node = self._get_node(cursor, normalized)
                if node is None:
                    raise FileNotFoundError(normalized)
                if node[4] == "directory" and not recursive and self.ls(normalized):
                    raise OSError(f"Directory is not empty: {normalized}")
                namespace_id = self._namespace_id(cursor)
                pattern = _subtree_like_pattern(normalized)
                cursor.execute(
                    """
                    update vfs_nodes
                    set deleted_at = now(), updated_at = now()
                    where namespace_id = %s and (path = %s or path like %s escape '\\')
                    """,
                    (namespace_id, normalized, pattern),
                )

    def stat(self, path: str) -> FileEntry:
        normalized = normalize_virtual_path(path)
        with self._connection() as connection:
            with self._cursor(connection) as cursor:
                node = self._get_node(cursor, normalized)
                if node is None:
                    raise FileNotFoundError(normalized)
                return self._entry_from_row(node)

    def exists(self, path: str) -> bool:
        try:
            normalized = normalize_virtual_path(path)
            with self._connection() as connection:
                with self._cursor(connection) as cursor:
                    return self._get_node(cursor, normalized) is not None
        except Exception:
            return False

    def walk(self, path: str = "/") -> list[FileEntry]:
        normalized = normalize_virtual_path(path)
        with self._connection() as connection:
            with self._cursor(connection) as cursor:
                node = self._get_node(cursor, normalized)
                if node is None:
                    raise FileNotFoundError(normalized)
                namespace_id = self._namespace_id(cursor)
                if normalized == "/":
                    cursor.execute(
                        """
                        select id, parent_id, path, name, node_type, size_bytes,
                               current_revision, metadata, created_at, updated_at
                        from vfs_nodes
                        where namespace_id = %s and deleted_at is null
                        order by path
                        """,
                        (namespace_id,),
                    )
                else:
                    cursor.execute(
                        """
                        select id, parent_id, path, name, node_type, size_bytes,
                               current_revision, metadata, created_at, updated_at
                        from vfs_nodes
                        where namespace_id = %s and deleted_at is null
                          and (path = %s or path like %s escape '\\')
                        order by path
                        """,
                        (namespace_id, normalized, _subtree_like_pattern(normalized)),
                    )
                return [self._entry_from_row(row) for row in cursor.fetchall()]


class ScopedFilesystemDriver(FilesystemDriver):
    def __init__(self, base: FilesystemDriver, root_path: str = "/", *, create_root: bool = True):
        self.base = base
        self.root_path = normalize_virtual_path(root_path)
        if create_root:
            self.base.mkdir(self.root_path)

    def _map(self, path: str) -> str:
        normalized = normalize_virtual_path(path)
        if self.root_path == "/":
            return normalized
        relative = normalized.lstrip("/")
        return normalize_virtual_path(f"{self.root_path}/{relative}")

    def _unmap_entry(self, entry: FileEntry) -> FileEntry:
        if self.root_path == "/":
            return entry
        relative = entry.path.removeprefix(self.root_path).lstrip("/")
        path = "/" if not relative else f"/{relative}"
        return FileEntry(path=path, node_type=entry.node_type, size_bytes=entry.size_bytes, updated_at=entry.updated_at, metadata=entry.metadata)

    def read(self, path: str) -> FileRecord:
        record = self.base.read(self._map(path))
        return FileRecord(path=normalize_virtual_path(path), content=record.content, encoding=record.encoding, revision=record.revision, created_at=record.created_at, updated_at=record.updated_at, metadata=record.metadata)

    def write(self, path: str, content: str, *, overwrite: bool = True, metadata: Optional[dict[str, Any]] = None) -> FileRecord:
        self.base.write(self._map(path), content, overwrite=overwrite, metadata=metadata)
        return self.read(path)

    def append(self, path: str, content: str) -> FileRecord:
        self.base.append(self._map(path), content)
        return self.read(path)

    def mkdir(self, path: str, *, create_parents: bool = True) -> FileEntry:
        return self._unmap_entry(self.base.mkdir(self._map(path), create_parents=create_parents))

    def ls(self, path: str = "/") -> list[FileEntry]:
        return [self._unmap_entry(entry) for entry in self.base.ls(self._map(path))]

    def delete(self, path: str, *, recursive: bool = False) -> None:
        self.base.delete(self._map(path), recursive=recursive)

    def stat(self, path: str) -> FileEntry:
        return self._unmap_entry(self.base.stat(self._map(path)))

    def exists(self, path: str) -> bool:
        return self.base.exists(self._map(path))

    def walk(self, path: str = "/") -> list[FileEntry]:
        return [self._unmap_entry(entry) for entry in self.base.walk(self._map(path))]


class ReadOnlyFilesystemDriver(FilesystemDriver):
    def __init__(self, base: FilesystemDriver):
        self.base = base

    def read(self, path: str) -> FileRecord:
        return self.base.read(path)

    def write(self, path: str, content: str, *, overwrite: bool = True, metadata: Optional[dict[str, Any]] = None) -> FileRecord:
        raise PermissionError("Filesystem is read-only; cannot write.")

    def append(self, path: str, content: str) -> FileRecord:
        raise PermissionError("Filesystem is read-only; cannot append.")

    def mkdir(self, path: str, *, create_parents: bool = True) -> FileEntry:
        raise PermissionError("Filesystem is read-only; cannot create directory.")

    def ls(self, path: str = "/") -> list[FileEntry]:
        return self.base.ls(path)

    def delete(self, path: str, *, recursive: bool = False) -> None:
        raise PermissionError("Filesystem is read-only; cannot delete.")

    def stat(self, path: str) -> FileEntry:
        return self.base.stat(path)

    def exists(self, path: str) -> bool:
        return self.base.exists(path)

    def walk(self, path: str = "/") -> list[FileEntry]:
        return self.base.walk(path)


class EditFileArgs(BaseModel):
    path: str
    edits: list[dict[str, str]]


def _line_range(content: str, start_line: int, end_line: int) -> str:
    lines = content.splitlines()
    if not lines:
        return ""
    start = max(0, start_line)
    end = min(len(lines), end_line)
    return "\n".join(lines[start:end])


def _tree(
    driver: FilesystemDriver,
    path: str,
    depth: int,
    max_depth: int,
    ignore: list[str] | None,
    ignore_matcher: VFSIgnoreMatcher,
) -> list[Any]:
    if depth >= max_depth:
        return [f"[MAX_DEPTH_REACHED] Make another tool call with {path} as the path if needed."]
    result = []
    for entry in driver.ls(path):
        if ignore_matcher.is_ignored(entry.path, is_dir=entry.node_type == "directory"):
            continue
        if manual_ignore_matches(entry.name, entry.path.lstrip("/"), ignore):
            continue
        item: dict[str, Any] = {"name": entry.name, "type": entry.node_type}
        if entry.node_type == "directory":
            item["children"] = _tree(
                driver,
                entry.path,
                depth + 1,
                max_depth,
                ignore,
                ignore_matcher,
            )
        result.append(item)
    return result


def _strip_patch_fence(patch: str) -> str:
    text = (patch or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _resolve_vfs_patch_path(path_text: str) -> str:
    raw_path = (path_text or "").strip()
    if not raw_path:
        raise ValueError("Patch file path is required.")
    if "\x00" in raw_path:
        raise ValueError("Patch file path contains a null byte.")
    if raw_path.startswith("-") or raw_path == "--":
        raise ValueError(f"Patch file path must not look like an option: {raw_path}")
    return normalize_virtual_path(raw_path)


def _find_unique_match(lines: list[str], needle: list[str]) -> int:
    if not needle:
        if not lines:
            return 0
        raise ValueError(
            "Update hunk has no removable/context lines and cannot be placed safely."
        )

    matches = [
        index
        for index in range(0, len(lines) - len(needle) + 1)
        if lines[index:index + len(needle)] == needle
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError("Update hunk matched multiple locations exactly.")

    normalized_needle = [line.strip() for line in needle]
    fuzzy_matches = [
        index
        for index in range(0, len(lines) - len(needle) + 1)
        if (
            [line.strip() for line in lines[index:index + len(needle)]]
            == normalized_needle
        )
    ]
    if len(fuzzy_matches) == 1:
        return fuzzy_matches[0]
    if len(fuzzy_matches) > 1:
        raise ValueError(
            "Update hunk matched multiple locations after whitespace normalization."
        )
    raise ValueError("Update hunk did not match file contents.")


def _apply_update_hunks(original: str, hunks: list[list[str]]) -> str:
    lines = original.splitlines()
    had_trailing_newline = original.endswith("\n")

    for hunk in hunks:
        old_lines: list[str] = []
        new_lines: list[str] = []
        for line in hunk:
            if not line:
                raise ValueError("Patch hunk line is missing a prefix.")
            prefix = line[0]
            text = line[1:]
            if prefix == " ":
                old_lines.append(text)
                new_lines.append(text)
            elif prefix == "-":
                old_lines.append(text)
            elif prefix == "+":
                new_lines.append(text)
            else:
                raise ValueError(f"Invalid patch hunk line prefix: {prefix!r}")

        pos = _find_unique_match(lines, old_lines)
        lines[pos:pos + len(old_lines)] = new_lines

    output = "\n".join(lines)
    has_added_final_line = lines and any(
        hunk[-1].startswith("+") for hunk in hunks if hunk
    )
    if had_trailing_newline or has_added_final_line:
        output += "\n"
    return output


def _parse_patch(patch: str) -> list[dict[str, Any]]:
    text = _strip_patch_fence(patch)
    lines = text.splitlines()
    if not lines or lines[0] != "*** Begin Patch":
        raise ValueError("Patch must start with '*** Begin Patch'.")
    if lines[-1] != "*** End Patch":
        raise ValueError("Patch must end with '*** End Patch'.")

    ops: list[dict[str, Any]] = []
    i = 1
    while i < len(lines) - 1:
        if not lines[i].strip():
            i += 1
            continue

        match = _PATCH_FILE_HEADER_RE.match(lines[i])
        if not match:
            raise ValueError(f"Expected file operation header, got: {lines[i]}")
        action, path_text = match.groups()
        i += 1

        if action == "Add":
            body: list[str] = []
            while i < len(lines) - 1 and not _PATCH_FILE_HEADER_RE.match(lines[i]):
                if not lines[i].startswith("+"):
                    raise ValueError("Add File lines must start with '+'.")
                body.append(lines[i][1:])
                i += 1
            ops.append(
                {
                    "action": action,
                    "path": path_text,
                    "content": "\n".join(body) + ("\n" if body else ""),
                }
            )
            continue

        if action == "Delete":
            while i < len(lines) - 1 and not _PATCH_FILE_HEADER_RE.match(lines[i]):
                if lines[i].strip():
                    raise ValueError(
                        "Delete File operation must not include hunk content."
                    )
                i += 1
            ops.append({"action": action, "path": path_text})
            continue

        hunks: list[list[str]] = []
        while i < len(lines) - 1 and not _PATCH_FILE_HEADER_RE.match(lines[i]):
            if not lines[i].startswith("@@"):
                raise ValueError("Update File hunks must start with '@@'.")
            i += 1
            hunk: list[str] = []
            while (
                i < len(lines) - 1
                and not lines[i].startswith("@@")
                and not _PATCH_FILE_HEADER_RE.match(lines[i])
            ):
                hunk.append(lines[i])
                i += 1
            if not hunk:
                raise ValueError("Update File hunk is empty.")
            hunks.append(hunk)
        if not hunks:
            raise ValueError("Update File operation requires at least one hunk.")
        ops.append({"action": action, "path": path_text, "hunks": hunks})

    if not ops:
        raise ValueError("Patch contains no file operations.")
    return ops


def _apply_simple_patch(
    driver: FilesystemDriver,
    patch: str,
    *,
    dry_run: bool = False,
) -> str:
    ops = _parse_patch(patch)
    planned: dict[str, str | None] = {}
    summaries: list[str] = []

    for op in ops:
        path = _resolve_vfs_patch_path(op["path"])
        action = op["action"]

        if action == "Add":
            if path in planned and planned[path] is not None:
                raise ValueError(f"Add File target already planned: {op['path']}")
            if path not in planned and driver.exists(path):
                raise ValueError(f"Add File target already exists: {op['path']}")
            planned[path] = op["content"]
            summaries.append(f"add {op['path']}")
            continue

        if action == "Delete":
            if path in planned:
                if planned[path] is None:
                    raise ValueError(
                        f"Delete File target already planned for deletion: {op['path']}"
                    )
            else:
                entry = driver.stat(path)
                if entry.node_type != "file":
                    raise ValueError(
                        "Delete File target does not exist or is not a file: "
                        f"{op['path']}"
                    )
                driver.read(path)
            planned[path] = None
            summaries.append(f"delete {op['path']}")
            continue

        if path in planned:
            base_content = planned[path]
            if base_content is None:
                raise ValueError(
                    f"Update File target is already planned for deletion: {op['path']}"
                )
        else:
            entry = driver.stat(path)
            if entry.node_type != "file":
                raise ValueError(
                    "Update File target does not exist or is not a file: "
                    f"{op['path']}"
                )
            base_content = driver.read(path).content

        updated_content = _apply_update_hunks(base_content, op["hunks"])
        planned[path] = updated_content
        summaries.append(f"update {op['path']}")

    if dry_run:
        return "Patch dry run succeeded:\n" + "\n".join(
            f"- {item}" for item in summaries
        )

    backups: dict[str, tuple[bool, str | None]] = {}
    try:
        for path in planned:
            if driver.exists(path):
                entry = driver.stat(path)
                if entry.node_type != "file":
                    raise ValueError(f"Patch target is not a file: {path}")
                backups[path] = (True, driver.read(path).content)
            else:
                backups[path] = (False, None)

        for path, content in planned.items():
            if content is None:
                driver.delete(path, recursive=False)
            else:
                driver.write(path, content)
    except Exception:
        for path, (existed, content) in backups.items():
            try:
                if existed and content is not None:
                    driver.write(path, content)
                elif not existed and driver.exists(path):
                    driver.delete(path, recursive=False)
            except Exception:
                pass
        raise

    return "Patch applied successfully:\n" + "\n".join(
        f"- {item}" for item in summaries
    )


def make_vfs_filesystem_tools(driver: FilesystemDriver) -> dict[str, object]:
    @tool
    def read_file(path: str, start_line: int = 0, end_line: int = 100) -> str:
        """
        Read the complete contents of a file from the file system.

        Use this tool when you need to examine the contents of a single file.
        Do not use this repeatedly for large files, as it may consume a lot of memory and possibly overflow your context.
        Only works within allowed directories.

        Be intelligent and proactive in making use of start_line and end_line to minimize memory and token usage.

        Args:
            path: Path to the file to read
            start_line: Line number to start reading from (0-based, optional, default is 0)
            end_line: Line number to end reading at (0-based, optional, default is 100)
        Returns:
            str: Content of the file
        """
        if path is None or str(path).strip() == "":
            return (
                "Error: read_file requires a file path. Use the given tools to "
                "inspect available directories and find a specific file before "
                "calling read_file again."
            )
        try:
            record = driver.read(path)
            if record.content == "":
                return f"File {path} is empty."
            return _line_range(record.content, start_line, end_line)
        except Exception as exc:
            return f"Error reading file: {exc}"

    @tool
    def write_file(path: str, content: str) -> str:
        """
        Create a new file or completely overwrite an existing file with new content.

        Use with caution as it will overwrite existing files without warning.
        Handles text content with proper encoding.

        Args:
            path: Path where the file should be written
            content: Content to write to the file

        Returns:
            str: Success message
        """
        try:
            driver.write(path, content)
            return f"Successfully wrote to {path}"
        except Exception as exc:
            return f"Error writing file: {exc}"

    @tool
    def append_to_file(path: str, content: str) -> str:
        """
        Append content to a file.
        If the file does not exist, it will be created.
        If the file exists, the content will be appended to the end of the file.
        If the file exists and the content is the same as the existing content, the operation will succeed silently.

        Args:
            path: Full path to the file to append to
            content: Content to append to the file

        Returns:
            str: Success message
        """
        try:
            driver.append(path, content)
            return f"Successfully appended to {path}"
        except Exception as exc:
            return f"Error appending to file: {exc}"

    @tool
    def list_directory(path: str, ignore: Optional[list[str]] = None) -> str:
        """
        Get a detailed listing of all files and directories in a specified path.

        Results clearly distinguish between files and directories with [FILE] and [DIR] prefixes.

        Args:
            path: Relative Path to the directory to list
            ignore: Optional list of glob-like syntax patterns to ignore during the tree generation. Use this to ignore unneccesary files and directories.
        Returns:
            str: Formatted list of directory contents
        """
        try:
            entries = []
            ignore_matcher = VFSIgnoreMatcher(driver)
            for entry in driver.ls(path):
                if ignore_matcher.is_ignored(entry.path, is_dir=entry.node_type == "directory"):
                    continue
                if manual_ignore_matches(entry.name, entry.path.lstrip("/"), ignore):
                    continue
                prefix = "[DIR]" if entry.node_type == "directory" else "[FILE]"
                entries.append(f"{prefix} {entry.name}")
            return "\n".join(entries)
        except Exception as exc:
            return f"Error listing directory: {exc}"

    @tool
    def list_allowed_directories() -> str:
        """
        Returns the list of directories that these tools are allowed to access.

        Use this to understand which directories are available before trying to access files.

        Returns:
            str: List of allowed directories
        """
        return "Allowed directories:\n/"

    @tool
    def directory_tree(path: str, ignore: Optional[list[str]] = None, max_depth: int = 3) -> str:
        """
        Get a recursive tree view of files and directories as a JSON structure.

        Each entry includes 'name', 'type' (file/directory), and 'children' for directories.
        Files have no children array, while directories always have a children array (which may be empty).

        Args:
            path: Path to get tree structure for. Use '/' to get the root directory.
            ignore: Optional list of glob-like syntax patterns to ignore during the tree generation. Use this to ignore unneccesary files and directories.
            max_depth: Optional maximum depth of the tree to generate. Use this to limit the depth of the tree to prevent context window issues.
        Returns:
            str: JSON formatted tree structure
        """
        try:
            ignore_matcher = VFSIgnoreMatcher(driver)
            return json.dumps(
                _tree(driver, path, 0, max_depth, ignore, ignore_matcher),
                indent=2,
            )
        except Exception as exc:
            return f"Error building directory tree: {exc}"

    @tool
    def grep_file(pattern: str, path: str, ignore_patterns: Optional[list[str]] = None) -> str:
        """
        Search for a pattern return matching lines from all files in the specified path.

        Use ignore_patterns to ignore unneccesary files and directories to minimize memory and token usage.

        Args:
            pattern: The pattern to search for in the file
            path: Path to a specific file or directory to search in
            ignore_patterns: Optional list of glob-like syntax patterns to ignore
        Returns:
            str: The first 20 matching lines with 1 line above and 1 line below the matching line if found.
        """
        try:
            ignore_matcher = VFSIgnoreMatcher(driver)
            entries = driver.walk(path)
            matches = []
            for entry in entries:
                if entry.node_type != "file":
                    continue
                if ignore_matcher.is_ignored(entry.path, is_dir=False):
                    continue
                if manual_ignore_matches(entry.name, entry.path.lstrip("/"), ignore_patterns):
                    continue
                content = driver.read(entry.path).content
                for index, line in enumerate(content.splitlines(), start=1):
                    if re.search(pattern, line, re.IGNORECASE):
                        matches.append(f"{entry.path},\nLine {index}: {line}")
                        if len(matches) >= 20:
                            return "\n".join(matches)
            return "\n".join(matches) if matches else f"No matches found for pattern '{pattern}' in '{path}'."
        except Exception as exc:
            return f"Error searching files: {exc}"

    @tool
    def search_files(path: str, pattern: str, exclude_patterns: Optional[list[str]] = None) -> str:
        """
        Recursively search for files and directories matching a pattern.

        Searches through all subdirectories from the starting path.
        The search is case-insensitive and matches partial names.

        Args:
            path: Root path to start searching from
            pattern: Pattern to search for in filenames with glob-like syntax (e.g., "*.txt" for all text files)
            exclude_patterns: Optional list of patterns to exclude

        Returns:
            str: Newline-separated list of matching files and directories
        """
        try:
            exclude_patterns = exclude_patterns or []
            ignore_matcher = VFSIgnoreMatcher(driver)
            matches = []
            for entry in driver.walk(path):
                if ignore_matcher.is_ignored(entry.path, is_dir=entry.node_type == "directory"):
                    continue
                if manual_ignore_matches(entry.name, entry.path.lstrip("/"), exclude_patterns):
                    continue
                if fnmatch.fnmatch(entry.name.lower(), pattern.lower()):
                    matches.append(entry.path)
            return "\n".join(matches) if matches else "No matches found"
        except Exception as exc:
            return f"Error searching files: {exc}"

    @tool
    def create_directory(path: str) -> str:
        """
        Create a new directory or ensure a directory exists.

        Can create multiple nested directories in one operation.
        If the directory already exists, this operation will succeed silently.

        Args:
            path: Relative Path of the directory to create

        Returns:
            str: Success message
        """
        try:
            driver.mkdir(path)
            return f"Successfully created directory {path}"
        except Exception as exc:
            return f"Error creating directory: {exc}"

    @tool
    def delete_file_or_directory(path: str) -> str:
        """
        Delete a file or directory.
        Only works within allowed directories.
        Use with caution as it will delete a file or directory without warning.

        Args:
            path: Path to the file or directory to delete

        Returns:
            str: Success message
        """
        try:
            entry = driver.stat(path)
            driver.delete(path, recursive=entry.node_type == "directory")
            return f"Successfully deleted {path}"
        except Exception as exc:
            return f"Error deleting file or directory: {exc}"

    @tool
    def search_and_replace_file_edit(path: str, search: str, replace: str) -> str:
        """
        Search and replace a string in a file.

        Args:
            path: Path to the file to edit
            search: String to search for
            replace: String to replace with

        Returns:
            str: Success message

        Use this tool when you need to search and replace a string in a file.
        Do not use this tool for large files, as it may consume a lot of memory and possibly overflow your context.
        Ensure search string is an exact match and unique in the file.
        Only works within allowed directories.
        """
        try:
            content = driver.read(path).content
            if search not in content:
                return f"'{search}' not found in {path}"
            driver.write(path, content.replace(search, replace))
            return f"Successfully searched and replaced '{search}' with '{replace}' in {path}"
        except Exception as exc:
            return f"Error editing file: {exc}"

    @tool(args_schema=EditFileArgs)
    def edit_file(path: str, edits: list[dict[str, str]], dry_run: bool = False) -> str:
        """
        Make line-based edits to a text file.

        Args:
            path: Path to the file to edit/write
            edits: List of edits, where each edit is a dict with 'oldText' and 'newText' keys. 'oldText' can be empty if writing to a new file.
        Returns:
            str: Detailed diff of the changes made
        """
        try:
            content = driver.read(path).content if driver.exists(path) else ""
            for edit in edits:
                old_text = edit.get("oldText", "")
                new_text = edit.get("newText", "")
                if old_text:
                    if old_text not in content:
                        return f"Error: oldText not found in {path}"
                    content = content.replace(old_text, new_text, 1)
                else:
                    content += new_text
            if not dry_run:
                driver.write(path, content)
            return "Changes found:\n- Exact text edits applied"
        except Exception as exc:
            return f"Error during file edit: {exc}"

    @tool
    def apply_patch(patch: str, dry_run: bool = False) -> str:
        """
        Apply a structured patch to create, update, or delete files.

        Use this tool for atomic, single or multi-file edits when you can identify the
        exact text to change. Prefer one patch containing all related edits over
        a sequence of separate write operations, unless you need iterative validation
        and changes.

        Usage guide:
        - Start with `*** Begin Patch` and end with `*** End Patch`.
        - Use `*** Add File: path` for new files. Every content line must start
          with `+`.
        - Use `*** Update File: path` for existing files. Each hunk starts with
          `@@`, then uses space-prefixed context lines, `-` lines to remove, and
          `+` lines to add.
        - Include enough unchanged context in update hunks to match exactly one
          location. If a hunk matches zero or multiple locations, the whole patch
          is rejected.
        - Use `*** Delete File: path` for text files that should be removed. Do
          not include hunk content for deletes.
        - Use `dry_run=True` only for broad or risky edits; it validates the
          patch and reports planned operations without writing files.

        Example patch:
        *** Begin Patch
        *** Add File: relative/or/absolute/path
        +new file line
        *** Update File: relative/or/absolute/path
        @@
         context line
        -old line
        +new line
        *** Delete File: relative/or/absolute/path
        *** End Patch

        Paths may be relative to the workspace root or absolute within allowed directories.
        The patch is atomic: if any operation fails, no files are changed.
        Symlinks, binary files, non-UTF-8 files, and paths outside allowed
        directories are rejected.
        """
        try:
            return _apply_simple_patch(driver, patch, dry_run=dry_run)
        except Exception as exc:
            return f"Error applying patch: {exc}"

    return {
        "read_file": read_file,
        "write_file": write_file,
        "search_and_replace_file_edit": search_and_replace_file_edit,
        "edit_file": edit_file,
        "append_to_file": append_to_file,
        "apply_patch": apply_patch,
        "list_directory": list_directory,
        "list_allowed_directories": list_allowed_directories,
        "directory_tree": directory_tree,
        "grep_file": grep_file,
        "search_files": search_files,
        "create_directory": create_directory,
        "delete_file_or_directory": delete_file_or_directory,
    }
