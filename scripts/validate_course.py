"""Validate links, bilingual structure, notation, figures, and runnable examples."""

from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
ZH = ROOT / "docs" / "zh-CN"
EN = ROOT / "docs" / "en"
LINK = re.compile(r"\]\(([^)]+)\)")


def chapter_pages() -> list[Path]:
    pages: list[Path] = []
    for folder in (ZH, EN):
        pages.extend(
            path
            for path in folder.glob("[0-9][0-9]-*.md")
            if 0 <= int(path.name[:2]) < 90
        )
        notation = folder / "appendix-a-notation.md"
        if notation.exists():
            pages.append(notation)
    return sorted(pages)


def check_local_links() -> list[str]:
    errors: list[str] = []
    for markdown in ROOT.rglob("*.md"):
        text = markdown.read_text(encoding="utf-8")
        for raw_target in LINK.findall(text):
            if raw_target.startswith(("http://", "https://", "#")):
                continue
            target = raw_target.split("#", maxsplit=1)[0]
            if target and not (markdown.parent / target).resolve().exists():
                errors.append(f"broken link: {markdown.relative_to(ROOT)} -> {raw_target}")
    return errors


def check_markdown_syntax() -> list[str]:
    """Catch structures that often make GitHub/preview rendering consume the rest of a page."""
    errors: list[str] = []
    bare_math_commands = re.compile(
        r"(?<!\\)\b(qquad|quad|times|top|rightarrow|leftarrow|geq|leq|"
        r"approx|partial|nabla|frac|mathcal|mathbb|mathbf|mathrm|operatorname|sqrt|"
        r"propto|odot|underbrace|alpha|beta|gamma|delta|epsilon|theta|lambda|"
        r"mu|sigma|phi|psi|rho|tau)\b"
    )
    math_span = re.compile(r"\$\$(.+?)\$\$|(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)", re.DOTALL)

    for path in chapter_pages():
        relative = path.relative_to(ROOT)
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        fence: tuple[str, int] | None = None
        inline_math_start: tuple[int, int] | None = None
        display_math_open = False
        outside: list[str] = []
        for line_number, line in enumerate(lines, 1):
            match = re.match(r"^\s{0,3}(`{3,}|~{3,})(.*)$", line)
            if match:
                marker = match.group(1)
                if fence is None:
                    fence = (marker[0], len(marker))
                elif marker[0] == fence[0] and len(marker) >= fence[1]:
                    fence = None
                continue
            if fence is None:
                outside.append(line)
                scan_line = re.sub(r"`+[^`]*`+", "", line)
                if re.search(
                    r"\[[^]\n]*\$(?!\$)[^$\n]+\$[^]\n]*\]\(", scan_line
                ):
                    errors.append(
                        "math delimiters inside link labels are not rendered by GitHub: "
                        f"{relative}:{line_number}"
                    )
                if scan_line.lstrip().startswith("|"):
                    for table_math in re.finditer(
                        r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)", scan_line
                    ):
                        if re.search(r"(?<!\\)\|", table_math.group(1)):
                            errors.append(
                                "unescaped table pipe inside math; use \\lvert/\\rvert: "
                                f"{relative}:{line_number}"
                            )
                column = 0
                while column < len(scan_line):
                    if scan_line.startswith("$$", column):
                        if inline_math_start is None:
                            display_math_open = not display_math_open
                        column += 2
                        continue
                    if scan_line[column] == "$" and not display_math_open:
                        if inline_math_start is None:
                            previous = scan_line[column - 1] if column else ""
                            if previous and not previous.isspace() and previous not in "([{>":
                                errors.append(
                                    "unsafe inline-math opening boundary; add a space or "
                                    f"rewrite the term: {relative}:{line_number}"
                                )
                            inline_math_start = (line_number, column + 1)
                        else:
                            inline_math_start = None
                    column += 1
        if fence is not None:
            errors.append(f"unclosed code fence: {relative}")
        if inline_math_start is not None:
            errors.append(
                "unbalanced inline-math delimiter: "
                f"{relative}:{inline_math_start[0]}"
            )
        if display_math_open:
            errors.append(f"unbalanced display-math delimiter: {relative}")

        outside_text = "\n".join(outside)
        if outside_text.count("<details") != outside_text.count("</details>"):
            errors.append(f"unbalanced details tag: {relative}")
        if outside_text.count("<summary") != outside_text.count("</summary>"):
            errors.append(f"unbalanced summary tag: {relative}")

        for match in math_span.finditer(outside_text):
            expression = match.group(1) if match.group(1) is not None else match.group(2)
            if expression is None:
                continue
            if r"\\" in expression:
                line_number = outside_text.count("\n", 0, match.start()) + 1
                errors.append(
                    "LaTeX row breaks inside dollar-delimited math are unstable on GitHub; "
                    "use a fenced ```math block: "
                    f"{relative}:{line_number}"
                )
            if re.search(r"\\operatorname\s*\{", expression):
                line_number = outside_text.count("\n", 0, match.start()) + 1
                errors.append(
                    "GitHub rejects the \\operatorname macro; use \\mathrm instead: "
                    f"{relative}:{line_number}"
                )
            if re.search(r"(?<!\\)[_^]\*", expression):
                line_number = outside_text.count("\n", 0, match.start()) + 1
                errors.append(
                    "unsafe Markdown asterisk in LaTeX script; use _{\\ast} or ^{\\ast}: "
                    f"{relative}:{line_number}"
                )
            command = bare_math_commands.search(expression)
            if command:
                line_number = outside_text.count("\n", 0, match.start()) + 1
                errors.append(
                    f"unescaped LaTeX command '{command.group(1)}': {relative}:{line_number}"
                )
            depth = 0
            escaped = False
            for character in expression:
                if escaped:
                    escaped = False
                    continue
                if character == "\\":
                    escaped = True
                elif character == "{":
                    depth += 1
                elif character == "}":
                    depth -= 1
                    if depth < 0:
                        break
            if depth != 0:
                line_number = outside_text.count("\n", 0, match.start()) + 1
                errors.append(f"unbalanced LaTeX braces: {relative}:{line_number}")
    return errors


def check_mermaid_syntax() -> list[str]:
    """Catch node labels that GitHub's Mermaid parser treats as graph syntax."""
    errors: list[str] = []
    unquoted_node = re.compile(
        r'\b[A-Za-z][A-Za-z0-9_]*(?:\[(?!")|\{(?!"))'
    )
    for path in ROOT.rglob("*.md"):
        relative = path.relative_to(ROOT)
        in_mermaid = False
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.strip() == "```mermaid":
                in_mermaid = True
                continue
            if in_mermaid and line.strip() == "```":
                in_mermaid = False
                continue
            if in_mermaid and unquoted_node.search(line):
                errors.append(
                    "Mermaid node labels must be quoted for GitHub compatibility: "
                    f"{relative}:{line_number}"
                )
    return errors


def check_bilingual_pairs() -> list[str]:
    errors: list[str] = []
    expected = [
        "00-prerequisites.md",
        "00-course-roadmap.md",
        "01-what-is-ml.md",
        "02-math-foundations.md",
        "03-linear-regression.md",
        "04-logistic-regression.md",
        "05-generative-classifiers.md",
        "06-neighbors-and-kernels.md",
        "07-support-vector-machines.md",
        "08-trees-and-ensembles.md",
        "09-learning-theory-and-model-selection.md",
        "10-clustering-em-and-elbo.md",
        "11-dimensionality-reduction.md",
        "12-neural-networks-and-backprop.md",
        "13-representation-learning.md",
        "14-sequence-vision-and-attention.md",
        "15-transformers-llms-and-finetuning.md",
        "16-vae-and-diffusion.md",
        "17-recommender-systems.md",
        "18-reinforcement-learning.md",
        "19-evaluation-causality-and-mlops.md",
        "20-machine-learning-systems.md",
        "21-interview-and-capstone.md",
        "appendix-a-notation.md",
    ]
    for name in expected:
        for folder in (ZH, EN):
            if not (folder / name).exists():
                errors.append(f"missing bilingual page: {folder.relative_to(ROOT) / name}")
    for chapter in range(6, 22):
        name = f"{chapter:02d}-solutions.md"
        for folder in (ZH / "solutions", EN / "solutions"):
            if not (folder / name).exists():
                errors.append(f"missing bilingual solution: {folder.relative_to(ROOT) / name}")
    return errors


def check_notation() -> list[str]:
    errors: list[str] = []
    forbidden = re.compile(r"(?<![A-Za-z])(x_i|y_i|e_i|\\epsilon_i)(?![A-Za-z])")
    pages = []
    for folder in (ZH, EN):
        pages.extend(path for path in folder.glob("[0-9][0-9]-*.md") if 2 <= int(path.name[:2]) < 90)
        pages.extend((folder / "solutions").glob("*.md"))
    for path in pages:
        relative = path.relative_to(ROOT)
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if forbidden.search(line):
                errors.append(
                    f"ambiguous sample index: {relative}:{line_number}; use parenthesized superscripts"
                )
    return errors


def check_latex_escapes() -> list[str]:
    """Catch control characters and bare Greek command names in new chapters."""
    errors: list[str] = []
    pages = []
    for folder in (ZH, EN):
        pages.extend(path for path in folder.glob("[0-9][0-9]-*.md") if 6 <= int(path.name[:2]) < 90)
        pages.extend((folder / "solutions").glob("*.md"))
    bare_greek = re.compile(r"\$(alpha|gamma|sigma|phi)(?=[_({\s=])")
    for path in pages:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "\t" in line or "\f" in line:
                errors.append(
                    f"control character in LaTeX: {path.relative_to(ROOT)}:{line_number}"
                )
            if bare_greek.search(line):
                errors.append(
                    f"unescaped Greek command: {path.relative_to(ROOT)}:{line_number}"
                )
    return errors


def check_visual_minimums() -> list[str]:
    errors: list[str] = []
    minimums = {
        "01-what-is-ml.md": 6,
        "02-math-foundations.md": 4,
        "03-linear-regression.md": 7,
        "04-logistic-regression.md": 5,
        "05-generative-classifiers.md": 5,
        "06-neighbors-and-kernels.md": 5,
        "07-support-vector-machines.md": 4,
        "08-trees-and-ensembles.md": 4,
        "09-learning-theory-and-model-selection.md": 4,
        "10-clustering-em-and-elbo.md": 4,
        "11-dimensionality-reduction.md": 4,
        "12-neural-networks-and-backprop.md": 4,
        "13-representation-learning.md": 4,
        "14-sequence-vision-and-attention.md": 9,
        "15-transformers-llms-and-finetuning.md": 4,
        "16-vae-and-diffusion.md": 4,
        "17-recommender-systems.md": 4,
        "18-reinforcement-learning.md": 4,
        "19-evaluation-causality-and-mlops.md": 4,
        "20-machine-learning-systems.md": 5,
        "21-interview-and-capstone.md": 4,
    }
    for name, minimum in minimums.items():
        count = len(re.findall(r"!\[[^]]+\]\([^)]+\)", (ZH / name).read_text(encoding="utf-8")))
        if count < minimum:
            errors.append(f"too few figures: docs/zh-CN/{name} has {count}, requires {minimum}")
    return errors


def check_chapter_closing_structure() -> list[str]:
    """Keep Chapter 6 and all later algorithm chapters on the standard close."""
    errors: list[str] = []
    required = {
        ZH: ["与现代 AI 的联系", "苏格拉底导师", "面试八股", "## 本章通关标准", "### 扩展阅读"],
        EN: ["## Modern AI connections", "## Socratic mentor", "## Exercises", "## Interview recap", "## Mastery checklist", "### Further reading"],
    }
    for folder, markers in required.items():
        for path in sorted(folder.glob("[0-9][0-9]-*.md")):
            chapter = int(path.name[:2])
            if chapter < 6 or chapter >= 90:
                continue
            text = path.read_text(encoding="utf-8")
            for marker in markers:
                if marker not in text:
                    errors.append(f"missing chapter close '{marker}': {path.relative_to(ROOT)}")
            if folder == ZH and not re.search(r"^## \d+\.\d+ 练习$", text, flags=re.MULTILINE):
                errors.append(f"missing standalone exercises section: {path.relative_to(ROOT)}")
    return errors


def run_examples() -> list[str]:
    errors: list[str] = []
    examples = [
        "code/chapter00/readiness_check.py",
        "code/chapter00/first_workflow.py",
        "code/generate_course_figures.py",
        "code/chapter02/gradient_descent_demo.py",
        "code/chapter03/linear_regression.py",
        "code/chapter03/regression_family_comparison.py",
        "code/chapter03/real_dataset_baseline.py",
        "code/chapter04/generate_figures.py",
        "code/chapter04/logistic_regression.py",
        "code/chapter05/generative_classifiers.py",
        "code/chapter06/neighbors_and_kernels.py",
        "code/chapter06/course_lab.py",
        "code/chapter07/support_vector_machine.py",
        "code/chapter07/course_lab.py",
        "code/chapter08/trees_and_ensembles.py",
        "code/chapter08/course_lab.py",
        "code/chapter09/learning_theory.py",
        "code/chapter10/clustering_and_em.py",
        "code/chapter11/dimensionality_reduction.py",
        "code/chapter12/miniautograd.py",
        "code/chapter12/neural_networks.py",
        "code/chapter13/representation_learning.py",
        "code/chapter14/expanded_models.py",
        "code/chapter15/transformers_and_llms.py",
        "code/chapter15/course_lab.py",
        "code/chapter16/generative_models.py",
        "code/chapter16/course_lab.py",
        "code/chapter17/recommender_systems.py",
        "code/chapter17/course_lab.py",
        "code/chapter18/reinforcement_learning.py",
        "code/chapter18/course_lab.py",
        "code/chapter19/evaluation_and_mlops.py",
        "code/chapter19/course_lab.py",
        "code/chapter20/ml_systems.py",
        "code/chapter20/course_lab.py",
        "code/chapter21/capstone_and_interviews.py",
        "code/chapter21/course_lab.py",
    ]
    # Keep Matplotlib's transient font cache inside the writable workspace.
    # On Windows a short-lived font lock can survive process exit; it must not
    # turn successful numerical examples into a false course failure.
    with tempfile.TemporaryDirectory(dir=ROOT, ignore_cleanup_errors=True) as mpl_config:
        environment = os.environ.copy()
        environment["MPLCONFIGDIR"] = mpl_config
        environment["LOKY_MAX_CPU_COUNT"] = "1"
        for relative in examples:
            result = subprocess.run(
                [sys.executable, str(ROOT / relative)],
                cwd=ROOT,
                env=environment,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            if result.returncode:
                errors.append(f"example failed: {relative}\n{result.stderr[-1000:]}")
    return errors


def main() -> None:
    checks = [
        ("links", check_local_links),
        ("Markdown and math syntax", check_markdown_syntax),
        ("Mermaid syntax", check_mermaid_syntax),
        ("bilingual pages", check_bilingual_pairs),
        ("notation", check_notation),
        ("LaTeX escapes", check_latex_escapes),
        ("visual minimums", check_visual_minimums),
        ("chapter closing structure", check_chapter_closing_structure),
        ("examples", run_examples),
    ]
    all_errors: list[str] = []
    for name, check in checks:
        errors = check()
        print(f"{'PASS' if not errors else 'FAIL'}: {name}")
        all_errors.extend(errors)
    if all_errors:
        print("\n".join(all_errors))
        raise SystemExit(1)
    print("Course validation passed.")


if __name__ == "__main__":
    main()
