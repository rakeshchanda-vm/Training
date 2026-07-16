from __future__ import annotations

import json
import sys

from andromeda.workspace.availability import check_all_providers


def main(argv: list[str] | None = None) -> int:
    args = list(argv or sys.argv[1:])
    if args and args[0] != "check":
        print(f"Unknown command: {args[0]}", file=sys.stderr)
        return 2

    payload = {
        name: {
            "available": result.available,
            **({"reason": result.reason} if result.reason else {}),
            **({"details": result.details} if result.details else {}),
        }
        for name, result in check_all_providers().items()
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
