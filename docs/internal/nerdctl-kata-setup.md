# Internal: nerdctl + Kata setup for Andromeda microVM workspaces

This document explains how to set up a **local development host** for
`backend="microvm"` with `settings=NerdctlDevSettings(...)`.

It is **not** the production path. Production uses `ContainerdKataSandboxProvider`
and a separate sandbox control plane service (not shipped in this repo).
This path is currently untested and should be considered experimental.

## What Andromeda expects

`NerdctlKataDevProvider` shells out to `nerdctl` and runs roughly:

```bash
nerdctl --namespace andromeda image inspect <image>
nerdctl --namespace andromeda run -d \
  --runtime io.containerd.kata.v2 \
  --name andromeda-<session_id> \
  --workdir /workspace \
  <image> sleep infinity
```

File tools write to a local materialized workspace under agent-home. Before each
shell command, the provider copies the workspace into the microVM and copies
changes back afterward.

Required on the host:

| Component | Purpose |
|-----------|---------|
| `/dev/kvm` | Kata/Firecracker needs hardware virtualization |
| `containerd` | Container runtime socket (`/run/containerd/containerd.sock`) |
| Kata runtime | Registered as `io.containerd.kata.v2` in containerd |
| `nerdctl` | CLI used by the dev provider |
| Agent image | Must exist in the `andromeda` namespace and include `sleep` |

`NerdctlDevSettings` fields:

| Field | Default | Notes |
|-------|---------|-------|
| `image` | — | **Required** |
| `runtime` | `io.containerd.kata.v2` | Must match containerd runtime name |
| `namespace` | `andromeda` | nerdctl/containerd namespace |
| `workspace_path` | `/workspace` | Path inside the microVM |
| `container_name` | `andromeda-<session_id>` | Override for debugging |
| `nerdctl_path` | `nerdctl` | Full path if not on `PATH` |
| `create_timeout_seconds` | `60` | Container create timeout |

## Preflight on the current host

Run these before installing anything:

```bash
# OS + virtualization
uname -a
ls -l /dev/kvm

# containerd
systemctl is-active containerd
containerd --version
ctr --version

# nerdctl / kata (expect "not found" until installed)
which nerdctl kata-runtime 2>/dev/null || true
grep -i kata /etc/containerd/config.toml || echo "kata runtime not configured"
```

**Important for AWS EC2:** most standard instances do **not** expose `/dev/kvm`.
Kata will not work without it. Use one of:

- A **bare metal** instance type (for example `m5.metal`, `c5.metal`)
- An instance type with **nested virtualization** enabled
- A non-AWS dev machine (local Linux workstation with KVM)

If `ls -l /dev/kvm` fails, **you cannot run real Kata microVMs on this host**.
Installing `kata-runtime` alone is not enough.

### No `/dev/kvm` on Amazon Linux / standard EC2

This is expected on most managed cloud VMs. The hypervisor does not expose
hardware virtualization to the guest, and you usually cannot fix that from
inside the instance (no BIOS access, no instance-type change).

**Do not continue with sections 2–5 on this host** expecting microVM smoke
tests to pass. `nerdctl run --runtime io.containerd.kata.v2` will fail when Kata
tries to start a VM.

What you can still do on this machine:

| Goal | Option |
|------|--------|
| Develop Andromeda workspace features | `local_fs`, `ephemeral_fs`, `postgres_vfs` |
| Run shell commands in a sandbox without KVM | `bubblewrap_process`, `gvisor_container` (see [workspace-sandbox-providers.md](workspace-sandbox-providers.md)) |
| Test microVM provider logic | Unit tests with fake `runner` / `SandboxControlPlaneClient` (see `tests/test_workspace_session.py`) |
| Validate end-to-end Kata isolation | Use a KVM-capable host (see below) |

Where real Kata *can* run:

- Local Linux laptop/desktop with `/dev/kvm`
- EC2 **bare metal** (`*.metal`) or instances with nested virt enabled
- A dedicated internal dev/CI runner your platform team provisions
- Production via `containerd_kata` + sandbox control plane on KVM-capable nodes

There is no practical QEMU-software-emulation path for agent sandboxes; it is
unsupported, extremely slow, and still often blocked in cloud images.

## 1. Install Kata Containers

On Ubuntu 24.04 (adjust versions as needed):

```bash
# Example: install from Kata release artifacts.
# Check https://github.com/kata-containers/kata-containers/releases for the
# current stable version.
export KATA_VERSION=3.10.0
curl -fsSL -o kata-static.tar.xz \
  "https://github.com/kata-containers/kata-containers/releases/download/${KATA_VERSION}/kata-static-${KATA_VERSION}-amd64.tar.xz"
sudo tar -C / -xvf kata-static.tar.xz

# Binaries land under /opt/kata; ensure they are on PATH
echo 'export PATH="/opt/kata/bin:$PATH"' | sudo tee /etc/profile.d/kata.sh
source /etc/profile.d/kata.sh

kata-runtime --version
```

Alternative: distribution packages if your environment standardizes on them.
The critical outcome is a working `kata-runtime` (or `kata`) binary and
configuration under `/opt/kata` or `/usr/share/kata-containers`.

## 2. Register Kata in containerd

Andromeda defaults to runtime name `io.containerd.kata.v2`.

Kata ships a containerd config snippet. Merge it into `/etc/containerd/config.toml`:

```bash
# Generate or copy the kata containerd v2 runtime block from the Kata install.
# Typical location after kata-static install:
sudo cp /opt/kata/share/defaults/kata-containers/configuration.toml /etc/kata/config.toml 2>/dev/null || true

# Add a [plugins."io.containerd.grpc.v1.cri".containerd.runtimes.kata] block
# or the containerd v2 equivalent runtime registration for io.containerd.kata.v2.
# Follow the "containerd integration" section in the Kata docs for your version.
```

Minimal runtime registration pattern (containerd 1.7+/2.x style — verify against
your installed containerd version):

```toml
[plugins."io.containerd.grpc.v1.cri".containerd.runtimes.kata]
  runtime_type = "io.containerd.kata.v2"
  privileged_without_host_devices = false
  pod_annotations = []
  container_annotations = []

  [plugins."io.containerd.grpc.v1.cri".containerd.runtimes.kata.options]
    ConfigPath = "/etc/kata/config.toml"
```

Restart containerd:

```bash
sudo systemctl restart containerd
sudo systemctl status containerd
```

Verify the runtime is visible:

```bash
ctr plugins ls | grep -i kata || true
```

**Note:** this host currently has `disabled_plugins = ["cri"]` in
`/etc/containerd/config.toml`. nerdctl can still talk to containerd directly,
but you must ensure the Kata **runtime handler** is registered for the namespace
nerdctl uses. If runtime registration fails, re-enable required plugins or use
the containerd v2 runtime configuration format documented for your containerd
version.

## 3. Install nerdctl

```bash
export NERDCTL_VERSION=2.0.3
curl -fsSL -o nerdctl.tar.gz \
  "https://github.com/containerd/nerdctl/releases/download/v${NERDCTL_VERSION}/nerdctl-${NERDCTL_VERSION}-linux-amd64.tar.gz"
sudo tar -C /usr/local/bin -xzf nerdctl.tar.gz nerdctl
nerdctl --version
```

Confirm nerdctl can reach containerd:

```bash
sudo nerdctl info
```

If permission errors occur, either:

- run nerdctl with sufficient privileges during setup verification, or
- configure rootless nerdctl (more complex; not required for initial dev setup)

## 4. Build the agent image

There is no `andromeda-agent` Dockerfile in this repo yet. For smoke tests, a
minimal image is enough:

```dockerfile
# save as docs/internal/andromeda-agent.Dockerfile
FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    coreutils \
    python3 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
```

Build and load into the `andromeda` namespace:

```bash
cd /path/to/ET-Agentify
sudo nerdctl --namespace andromeda build \
  -t andromeda-agent:latest \
  -f docs/internal/andromeda-agent.Dockerfile .
sudo nerdctl --namespace andromeda image ls
```

The image must respond to `nerdctl --namespace andromeda image inspect andromeda-agent:latest`.

## 5. Manual smoke test (before Andromeda)

Run the same commands Andromeda uses:

```bash
export NS=andromeda
export IMAGE=andromeda-agent:latest
export RUNTIME=io.containerd.kata.v2
export NAME=andromeda-manual-smoke

sudo nerdctl --namespace "$NS" run -d \
  --runtime "$RUNTIME" \
  --name "$NAME" \
  --workdir /workspace \
  "$IMAGE" sleep infinity

sudo nerdctl --namespace "$NS" exec "$NAME" sh -lc 'echo kata-ok && uname -a'
sudo nerdctl --namespace "$NS" rm -f "$NAME"
```

If this fails, fix nerdctl/Kata/containerd first. Andromeda will fail the same way.

## 6. Andromeda smoke test

```python
from andromeda.workspace import FileSeed, NerdctlDevSettings, WorkspaceHomeConfig, WorkspacePolicy, WorkspaceSession

session = WorkspaceSession.create(
    backend="microvm",
    seed=[FileSeed(path="AGENTS.md", content="Work only in this workspace.")],
    policy=WorkspacePolicy(enable_shell=True),
    home=WorkspaceHomeConfig(base_dir="tmp-agent-home", session_id="run-123"),
    settings=NerdctlDevSettings(
        image="andromeda-agent:latest",
        runtime="io.containerd.kata.v2",
        namespace="andromeda",
    ),
)
tools = session.tools()
print(tools["shell"].invoke({"command": "cat", "argv": ["AGENTS.md"]}))
session.cleanup()
```

Or run the repo script:

```bash
/mnt/drive/projects/jnelsson/envs/andromeda11/bin/python test_workspace.py
```

Opt-in integration test (skips automatically if env vars are unset):

```bash
export ANDROMEDA_NERDCTL_KATA_TEST=1
export ANDROMEDA_KATA_IMAGE=andromeda-agent:latest
export ANDROMEDA_KATA_RUNTIME=io.containerd.kata.v2
pytest tests/test_workspace_session.py::test_nerdctl_kata_integration_when_configured -q
```

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `No such file or directory: 'nerdctl'` | nerdctl not installed / not on PATH | Install nerdctl (section 3) |
| `Failed to create nerdctl Kata sandbox` on image inspect | image not built in namespace | Build with `--namespace andromeda` |
| `io.containerd.kata.v2` not found | Kata not registered in containerd | Section 2 |
| Kata fails immediately, no `/dev/kvm` | Host lacks virtualization | Change instance type / enable nested virt |
| `permission denied` on containerd socket | insufficient privileges | use sudo for setup; check socket permissions |
| Provider asks for `control_plane_url` | wrong settings type for injection | Use `NerdctlDevSettings` locally; use `ContainerdKataSettings` only with control plane |
| HTTP 404 on control plane | no sandbox service running | Expected locally; use `NerdctlDevSettings` instead |

## Production vs dev

| Path | When to use | Config |
|------|-------------|--------|
| `NerdctlDevSettings` | Local dev, manual smoke, opt-in CI on a prepared host | `settings=NerdctlDevSettings(image=...)` |
| `ContainerdKataSettings` | Production | `settings=ContainerdKataSettings(control_plane_url=..., image=...)` |

The sandbox control plane is a **separate service** your platform team must deploy.
Andromeda only includes the HTTP client. There is nothing to "download" for
`control_plane_url` today.

## Security notes (dev)

- `nerdctl_dev` is not a production isolation boundary.
- The dev provider copies the full workspace into the microVM before shell exec.
- Prefer dedicated dev hosts; do not share the containerd socket on multi-tenant machines.
- Always call `session.cleanup()` to remove containers and owned agent-home directories.
