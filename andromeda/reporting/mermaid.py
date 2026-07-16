import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Optional


def _safe_name(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(name or "report")).strip("_")
    return cleaned or "report"


def _normalize_mermaid_code(mermaid_code: str) -> str:
    # Mermaid often does not render literal "\n" in labels as expected.
    # Convert escaped newlines into explicit HTML breaks for stable wrapping.
    return mermaid_code.replace("\\n", "<br/>")


def generate_mermaid_diagram(text: str, output_path: str | Path, fname: str) -> str:
    """Generate high-resolution PNG diagrams from Mermaid code blocks in a Markdown string.

    This function searches the provided Markdown text for Mermaid diagram code blocks,
    generates PNG images for each diagram using the Mermaid CLI (`mmdc`), and replaces
    the original Mermaid code blocks in the text with Markdown image links to the generated PNGs.
    The PNG files are saved in the specified output directory, and temporary files are cleaned up
    after processing.

    Args:
        text (str): The Markdown text containing Mermaid code blocks (```mermaid ... ```).
        output_path (str or Path): The directory path where PNG files will be saved.
        fname (str): The base filename to use for generated PNG files.

    Returns:
        str: The modified Markdown text with Mermaid code blocks replaced by image links.

    Raises:
        Exception: If there is an error during file operations or diagram generation.

    Example:
        >>> md = "Here is a diagram:\n```mermaid\ngraph TD; A-->B;\n```"
        >>> generate_mermaid_diagram(md, "assets", "example")
        'Here is a diagram:\n![example Diagram 0](assets/example_diagram_0-1.png)'"""
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_filename = _safe_name(fname)

    mermaid_pattern = r"```mermaid\s*\n(.*?)```"
    matches = list(re.finditer(mermaid_pattern, text, re.DOTALL | re.IGNORECASE))
    if not matches:
        return text

    rendered_text = text
    puppeteer_cfg: Optional[str] = None
    mermaid_cfg: Optional[str] = None
    env_cfg = Path("~/.andromeda/puppeteer-config.json").expanduser()
    if env_cfg.exists():
        puppeteer_cfg = str(env_cfg)
    else:
        env_cfg.touch()
        env_cfg.write_text(json.dumps({
            "args": [
                "--no-sandbox",
                "--disable-setuid-sandbox",
            ]
        }), encoding="utf-8")
        puppeteer_cfg = str(env_cfg)

    # Keep settings robust across Markdown->PDF pipelines.
    diagram_cfg = Path("~/.andromeda/mermaid-config.json").expanduser()
    if not diagram_cfg.exists():
        diagram_cfg.parent.mkdir(parents=True, exist_ok=True)
        diagram_cfg.write_text(
            json.dumps(
                {
                    "theme": "neutral",
                    "flowchart": {"htmlLabels": False},
                    "fontFamily": "Arial",
                }
            ),
            encoding="utf-8",
        )
    mermaid_cfg = str(diagram_cfg)


    for idx, match in enumerate(reversed(matches)):
        mermaid_code = _normalize_mermaid_code(match.group(1).strip())
        start, end = match.span()
        diagram_index = len(matches) - 1 - idx

        png_filename = f"{base_filename}_diagram_{diagram_index}.png"
        png_path_abs = output_dir / png_filename
        markdown_image_link = f"![{base_filename} Diagram {diagram_index}](assets/{png_filename})"

        tmp_file = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".md",
                encoding="utf-8",
                delete=False,
                dir=str(output_dir),
            ) as f:
                # mmdc is most reliable with markdown fenced mermaid blocks.
                f.write("```mermaid\n")
                f.write(mermaid_code)
                f.write("\n```")
                tmp_file = f.name

            mmdc_command = [
                "mmdc",
                "-i",
                str(tmp_file),
                "-o",
                str(png_path_abs),
                "-t",
                "neutral",
                "-b",
                "white",
                "-s",
                "3",
                "-w",
                "2400",
                "-H",
                "1600",
            ]
            if puppeteer_cfg:
                mmdc_command.extend(["--puppeteerConfigFile", puppeteer_cfg])
            if mermaid_cfg:
                mmdc_command.extend(["-c", mermaid_cfg])

            result = subprocess.run(
                mmdc_command, capture_output=True, text=True, check=False, timeout=30
            )

            if result.returncode == 0:
                # mmdc may emit numbered variants (e.g., *_diagram_0-1.png).
                resolved_png = png_path_abs
                if not resolved_png.exists() or resolved_png.stat().st_size == 0:
                    stem = png_path_abs.stem
                    suffix = png_path_abs.suffix
                    for i in range(1, 10):
                        variant = output_dir / f"{stem}-{i}{suffix}"
                        if variant.exists() and variant.stat().st_size > 0:
                            try:
                                variant.replace(png_path_abs)
                            except Exception:
                                resolved_png = variant
                            else:
                                resolved_png = png_path_abs
                            break

                if resolved_png.exists() and resolved_png.stat().st_size > 0:
                    link_name = resolved_png.name
                    markdown_image_link = f"![{base_filename} Diagram {diagram_index}](assets/{link_name})"
                    rendered_text = rendered_text[:start] + markdown_image_link + rendered_text[end:]
        finally:
            if tmp_file:
                try:
                    Path(tmp_file).unlink(missing_ok=True)
                except Exception:
                    pass

    return rendered_text
