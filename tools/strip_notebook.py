"""Clear Jupyter notebook outputs so that Kaggle data never reaches git.

Used as a git clean filter (see .gitattributes): git stores the notebook with empty outputs,
while the local file keeps its outputs.

  python tools/strip_notebook.py < in.ipynb > out.ipynb    # filter mode (stdin -> stdout)
  python tools/strip_notebook.py nb1.ipynb [nb2.ipynb ...]  # strip files in place

One-time setup per clone:
  git config filter.strip-nb.clean "python tools/strip_notebook.py"
  git config filter.strip-nb.smudge cat
"""
import json
import sys


def strip(text: str) -> str:
    nb = json.loads(text)
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
        meta = cell.get("metadata", {})
        for key in ("execution", "collapsed", "scrolled", "ExecuteTime"):
            meta.pop(key, None)
    nb.get("metadata", {}).pop("widgets", None)
    return json.dumps(nb, indent=1, ensure_ascii=False) + "\n"


def main() -> None:
    if len(sys.argv) > 1:
        for path in sys.argv[1:]:
            with open(path, encoding="utf-8") as f:
                out = strip(f.read())
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                f.write(out)
        return
    data = sys.stdin.buffer.read().decode("utf-8")
    sys.stdout.buffer.write(strip(data).encode("utf-8"))


if __name__ == "__main__":
    main()
