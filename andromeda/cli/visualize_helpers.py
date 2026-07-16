"""Helper functions for visualization rendering."""

import os
import subprocess
import tempfile
from pathlib import Path

from andromeda.cli.helpers import console

def mermaid_to_text(diagram_code: str) -> str:
    """Convert mermaid diagram code to a simple text representation.
    
    Args:
        diagram_code: Mermaid diagram code
        
    Returns:
        Simple text representation of the diagram
    """
    lines = []
    lines.append("Workflow Structure:")
    lines.append("=" * 50)
    
    # Parse basic graph structure
    import re
    
    # Extract node definitions (broader ID charset, leading letter)
    node_pattern = r'([A-Za-z][A-Za-z0-9_.-]*)\["([^"]+)"\]'
    nodes = {}
    for match in re.finditer(node_pattern, diagram_code):
        node_id, node_label = match.groups()
        nodes[node_id] = node_label
    
    # Extract edges (support -->, -.->, ==> styles)
    edge_pattern = r'([A-Za-z][A-Za-z0-9_.-]*)\s*(-->|-\.->|==>)\s*([A-Za-z][A-Za-z0-9_.-]*)'
    edges = []
    for match in re.finditer(edge_pattern, diagram_code):
        from_node, edge_type, to_node = match.groups()
        edges.append((from_node, to_node, edge_type))
    
    # Build text representation
    for node_id, node_label in nodes.items():
        lines.append(f"  • {node_label} ({node_id})")
    
    lines.append("\nConnections:")
    for from_node, to_node, edge_type in edges:
        arrow = "→" if edge_type == "-->" else "⇢"
        from_label = nodes.get(from_node, from_node)
        to_label = nodes.get(to_node, to_node)
        lines.append(f"  {from_label} {arrow} {to_label}")
    
    return "\n".join(lines)


def render_mermaid_to_image(diagram_code: str, output_path: Path, format: str) -> None:
    """Render mermaid diagram to SVG or PNG using mmdc CLI if available.
    
    Args:
        diagram_code: Mermaid diagram code
        output_path: Output file path
        format: Desired format ('svg' or 'png')
    """
    import subprocess
    import tempfile
    
    # Check if mmdc is available
    try:
        result = subprocess.run(
            ["mmdc", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode != 0:
            raise FileNotFoundError("mmdc not found")
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        console.print(
            "[yellow]⚠[/yellow] Mermaid CLI (mmdc) not found. "
            "Saving as mermaid code instead."
        )
        console.print(
            "Install with: npm install -g @mermaid-js/mermaid-cli"
        )
        # Fallback: save as mermaid code
        fallback_path = output_path.with_suffix(".mmd")
        fallback_path.write_text(diagram_code, encoding="utf-8")
        console.print(f"[green]✓[/green] Saved mermaid diagram to {fallback_path}")
        return
    
    # Determine output format from file extension or format parameter
    if output_path.suffix in [".svg", ".png"]:
        output_format = output_path.suffix[1:]  # Remove the dot
    else:
        output_format = format
        # Update output path extension to match format
        output_path = output_path.with_suffix(f".{output_format}")
    
    # Create temporary markdown file with mermaid code block
    # mmdc expects markdown with ```mermaid code blocks
    with tempfile.NamedTemporaryFile(
        mode="w", 
        suffix=".md", 
        delete=False,
        encoding="utf-8"
    ) as tmp:
        tmp.write("```mermaid\n")
        tmp.write(diagram_code)
        tmp.write("\n```")
        tmp_path = tmp.name
    
    try:
        # Build mmdc command
        # Use puppeteer config file if it exists to handle sandbox issues
        puppeteer_config = Path.home() / ".cache" / "puppeteer" / "puppeteer-config.json"
        mmdc_command = [
            "mmdc",
            "-i", tmp_path,
            "-o", str(output_path),
            "-b", "transparent",
        ]
        
        # Add puppeteer config if it exists
        if puppeteer_config.exists():
            mmdc_command.extend(["--puppeteerConfigFile", str(puppeteer_config)])
        
        if output_format == "png":
            mmdc_command.extend(["-w", "1920", "-H", "1080"])
        elif output_format == "svg":
            # SVG doesn't need width/height, but we can add scale
            mmdc_command.extend(["-s", "2"])
        
        result = subprocess.run(
            mmdc_command,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            # mmdc may create files with numbered suffixes (e.g., output-1.svg)
            # Check for both the exact filename and numbered variants
            output_file = None
            if output_path.exists() and output_path.stat().st_size > 0:
                output_file = output_path
            else:
                # Check for numbered variants (mmdc adds -1, -2, etc. when multiple charts)
                base_name = output_path.stem
                parent_dir = output_path.parent
                for i in range(1, 10):  # Check up to 9 numbered variants
                    variant = parent_dir / f"{base_name}-{i}{output_path.suffix}"
                    if variant.exists() and variant.stat().st_size > 0:
                        output_file = variant
                        # Rename to expected output name
                        variant.rename(output_path)
                        break
            
            if output_file or output_path.exists():
                console.print(f"[green]✓[/green] Saved {output_format.upper()} diagram to {output_path}")
            else:
                raise RuntimeError("Output file was not created or is empty")
        else:
            error_msg = result.stderr or result.stdout or "Unknown error"
            raise RuntimeError(f"mmdc failed: {error_msg}")
            
    except (subprocess.TimeoutExpired, RuntimeError, OSError) as exc:
        console.print(f"[red]✗[/red] Error rendering diagram: {exc}")
        # Fallback: save as mermaid code
        fallback_path = output_path.with_suffix(".mmd")
        fallback_path.write_text(diagram_code, encoding="utf-8")
        console.print(f"[green]✓[/green] Saved mermaid diagram to {fallback_path} instead")
    finally:
        # Clean up temp file
        try:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except OSError:
            pass
