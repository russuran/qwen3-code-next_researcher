"""Autonomous data acquisition: find, download, and prepare datasets.

Fallback chain: HuggingFace Hub → Kaggle → direct URL → None.
Returns path to prepared data dir with train.jsonl / valid.jsonl
in MLX chat format, or None if nothing found.
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

from core.llm import LLM, LLMMode

logger = logging.getLogger(__name__)

DATASET_SEARCH_PROMPT = """\
You are a data engineer. Find real, downloadable datasets for: {task}

Search for:
1. HuggingFace datasets (provide dataset ID like "org/dataset")
2. Kaggle datasets (provide kaggle slug like "user/dataset-name")
3. GitHub repos with sample data (provide URL)

IMPORTANT: Only suggest datasets that ACTUALLY EXIST and can be downloaded.
Prefer HuggingFace datasets (easiest to download programmatically).

Respond with ONLY valid JSON:
{{
  "datasets": [
    {{
      "name": "dataset name",
      "source": "huggingface|kaggle|github|url",
      "identifier": "download path or URL",
      "description": "what it contains",
      "relevance": 1-10,
      "columns_hint": "which columns contain input text and labels"
    }}
  ]
}}
"""

CONVERT_PROMPT = """\
You are a data engineer. Convert a dataset to MLX chat training format.

Dataset columns: {columns}
Sample rows (first 3):
{sample}

Task: {task}

Write a Python function `convert(rows: list[dict]) -> list[dict]` that converts
each row to MLX chat format: {{"messages": [{{"role": "user", "content": "..."}}, {{"role": "assistant", "content": "..."}}]}}

The user message should contain the input text/document.
The assistant message should contain the label/answer.

Output ONLY the Python function in ```python ... ```
"""


class DataAcquisition:
    def __init__(self, llm: LLM, workspace: Path) -> None:
        self.llm = llm
        self.workspace = workspace
        self.data_dir = workspace / "acquired_data"

    async def acquire(self, topic: str, repo_context: dict | None = None) -> str | None:
        """Find and download a real dataset. Returns path or None."""
        logger.info("Searching for datasets for: %s", topic[:80])

        # Ask LLM for dataset suggestions
        prompt = DATASET_SEARCH_PROMPT.format(task=topic)
        raw = await self.llm.generate(prompt, mode=LLMMode.FAST)
        datasets = self._parse_suggestions(raw)

        if not datasets:
            logger.warning("No dataset suggestions from LLM")
            return None

        # Sort by relevance and try top 3
        datasets.sort(key=lambda d: -d.get("relevance", 0))
        for ds in datasets[:3]:
            result = await self._try_download(ds, topic)
            if result:
                return result

        logger.warning("All dataset download attempts failed")
        return None

    async def _try_download(self, ds: dict, topic: str) -> str | None:
        source = ds.get("source", "")
        identifier = ds.get("identifier", "")
        name = ds.get("name", "unknown")

        logger.info("Trying %s: %s (%s)", source, name, identifier[:60])

        if source == "huggingface":
            return await self._download_huggingface(identifier, ds, topic)
        elif source == "kaggle":
            return await self._download_kaggle(identifier, ds, topic)
        elif source in ("github", "url"):
            return await self._download_url(identifier, ds, topic)
        return None

    async def _download_huggingface(
        self, identifier: str, ds: dict, topic: str
    ) -> str | None:
        """Download from HuggingFace and convert to chat format."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        raw_dir = self.data_dir / "raw"
        raw_dir.mkdir(exist_ok=True)

        script = f"""\
import json, sys
try:
    from datasets import load_dataset
    ds = load_dataset("{identifier}", split="train[:500]", trust_remote_code=True)
    # Save as JSONL
    rows = [dict(row) for row in ds]
    with open("{raw_dir}/raw.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row, default=str, ensure_ascii=False) + "\\n")
    # Save columns info
    with open("{raw_dir}/columns.json", "w") as f:
        json.dump({{"columns": list(ds.column_names), "num_rows": len(rows)}}, f)
    print("OK:" + str(len(rows)))
except Exception as e:
    print("FAIL:" + str(e), file=sys.stderr)
    sys.exit(1)
"""
        script_path = self.data_dir / "download_hf.py"
        script_path.write_text(script)

        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True, text=True, timeout=300,
            )
            if result.returncode != 0:
                logger.warning("HF download failed: %s", result.stderr[:200])
                return None
        except (subprocess.TimeoutExpired, Exception) as e:
            logger.warning("HF download error: %s", e)
            return None

        # Check we got data
        raw_file = raw_dir / "raw.jsonl"
        if not raw_file.exists() or raw_file.stat().st_size < 100:
            return None

        # Convert to MLX chat format
        return await self._convert_to_chat(raw_dir, topic, ds)

    async def _download_kaggle(
        self, identifier: str, ds: dict, topic: str
    ) -> str | None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        raw_dir = self.data_dir / "raw"
        raw_dir.mkdir(exist_ok=True)

        try:
            result = subprocess.run(
                [sys.executable, "-m", "kaggle", "datasets", "download",
                 "-d", identifier, "-p", str(raw_dir), "--unzip"],
                capture_output=True, text=True, timeout=300,
            )
            if result.returncode != 0:
                logger.warning("Kaggle download failed: %s", result.stderr[:200])
                return None
        except Exception as e:
            logger.warning("Kaggle error: %s", e)
            return None

        if not list(raw_dir.rglob("*.*")):
            return None
        return await self._convert_to_chat(raw_dir, topic, ds)

    async def _download_url(
        self, url: str, ds: dict, topic: str
    ) -> str | None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        raw_dir = self.data_dir / "raw"
        raw_dir.mkdir(exist_ok=True)

        try:
            import httpx
            async with httpx.AsyncClient(timeout=120, follow_redirects=True) as client:
                resp = await client.get(url)
                if resp.status_code != 200:
                    return None
                fname = url.split("/")[-1] or "data.zip"
                fpath = raw_dir / fname
                fpath.write_bytes(resp.content)
                if fname.endswith(".zip"):
                    import zipfile
                    with zipfile.ZipFile(fpath) as zf:
                        zf.extractall(raw_dir)
        except Exception as e:
            logger.warning("URL download error: %s", e)
            return None

        return await self._convert_to_chat(raw_dir, topic, ds)

    async def _convert_to_chat(
        self, raw_dir: Path, topic: str, ds_info: dict
    ) -> str | None:
        """Convert downloaded data to MLX chat format (train.jsonl + valid.jsonl)."""
        out_dir = self.data_dir / "prepared"
        out_dir.mkdir(exist_ok=True)

        # Load raw data
        rows: list[dict] = []
        for fpath in list(raw_dir.rglob("*.jsonl")) + list(raw_dir.rglob("*.json")):
            try:
                for line in open(fpath):
                    if line.strip():
                        rows.append(json.loads(line))
            except Exception:
                continue

        # Also try CSV
        if not rows:
            for fpath in raw_dir.rglob("*.csv"):
                try:
                    import csv
                    with open(fpath) as f:
                        reader = csv.DictReader(f)
                        rows = list(reader)[:500]
                    break
                except Exception:
                    continue

        if len(rows) < 5:
            logger.warning("Too few rows: %d", len(rows))
            return None

        # Check if already in messages format
        if "messages" in rows[0]:
            chat_rows = rows
        else:
            # Ask LLM to write converter
            columns = list(rows[0].keys())
            sample = json.dumps(rows[:3], default=str, ensure_ascii=False)[:2000]

            conv_prompt = CONVERT_PROMPT.format(
                columns=columns, sample=sample, task=topic,
            )
            raw_code = await self.llm.generate(conv_prompt, mode=LLMMode.FAST)
            code = self._extract_code(raw_code)

            if len(code.strip()) < 30:
                # Fallback: use first text-like column as input, label-like as output
                chat_rows = self._auto_convert(rows, topic)
            else:
                chat_rows = self._exec_converter(code, rows)

        if not chat_rows or len(chat_rows) < 5:
            chat_rows = self._auto_convert(rows, topic)

        if len(chat_rows) < 5:
            return None

        # Split train/valid (90/10)
        split = int(len(chat_rows) * 0.9)
        train = chat_rows[:split]
        valid = chat_rows[split:]

        (out_dir / "train.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in train),
            encoding="utf-8",
        )
        (out_dir / "valid.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in valid),
            encoding="utf-8",
        )

        logger.info("Prepared %d train + %d valid samples at %s", len(train), len(valid), out_dir)
        return str(out_dir)

    def _auto_convert(self, rows: list[dict], topic: str) -> list[dict]:
        """Heuristic: find text-like and label-like columns."""
        if not rows:
            return []
        keys = list(rows[0].keys())

        # Find text column (longest average value)
        text_col = max(keys, key=lambda k: sum(len(str(r.get(k, ""))) for r in rows[:10]))
        # Find label column (shortest average value, different from text_col)
        label_candidates = [k for k in keys if k != text_col]
        if not label_candidates:
            return []
        label_col = min(label_candidates, key=lambda k: sum(len(str(r.get(k, ""))) for r in rows[:10]))

        chat_rows = []
        for row in rows:
            text = str(row.get(text_col, ""))[:500]
            label = str(row.get(label_col, ""))
            if text and label:
                chat_rows.append({"messages": [
                    {"role": "user", "content": text},
                    {"role": "assistant", "content": label},
                ]})
        return chat_rows

    def _exec_converter(self, code: str, rows: list[dict]) -> list[dict]:
        """Execute LLM-generated converter function safely."""
        try:
            ns: dict[str, Any] = {}
            exec(code, ns)
            convert_fn = ns.get("convert")
            if not convert_fn:
                return []
            result = convert_fn(rows)
            if isinstance(result, list) and result and "messages" in result[0]:
                return result
        except Exception as e:
            logger.warning("Converter exec failed: %s", e)
        return []

    def _parse_suggestions(self, raw: str) -> list[dict]:
        try:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(raw[start:end])
                return data.get("datasets", [])
        except Exception:
            pass
        return []

    @staticmethod
    def _extract_code(text: str) -> str:
        if "```python" in text:
            start = text.index("```python") + 9
            end = text.index("```", start)
            return text[start:end].strip()
        if "```" in text:
            parts = text.split("```")
            if len(parts) >= 3:
                code = parts[1].strip()
                if code.startswith("python"):
                    code = code[6:].strip()
                return code
        return ""
