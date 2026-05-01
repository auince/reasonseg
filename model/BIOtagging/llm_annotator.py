from __future__ import annotations

import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .bio_schema import NormalizedQuery

_DEFAULT_WORKERS = 16
_DEFAULT_MODEL = "deepseek-chat"
_DEFAULT_REVIEW_MODEL = "deepseek-v4-pro"
_DEFAULT_BASE_URL = "https://api.deepseek.com"
_REVIEW_INTERVAL = 100
_REVIEW_SAMPLE = 10

_SYSTEM_PROMPT = """Parse referring expressions into JSON. Output ONLY the JSON object, no explanation.

Keys:
- "target": the main object being referred to (string, or null for absent queries)
- "attributes": list of descriptive adjectives (colors, sizes, materials)
- "relations": list of {"type": relation word, "target": related object}
- "actions": list of {"verb": action word, "target": acted-upon object or null}
- "negatives": list of negative markers
- "exists": true if describing a present object, false if negating

Relation words include: behind, on, beside, next to, in front of, above, below, left of, right of, between, with, near, under, inside, outside, against
Action verbs include: holding, wearing, eating, drinking, carrying, sitting on, standing on, pointing at, looking at, touching, pushing, pulling, covering, riding, driving

Example: "red cup behind person holding phone"
-> {"target":"cup","attributes":["red"],"relations":[{"type":"behind","target":"person"}],"actions":[{"verb":"holding","target":"phone"}],"negatives":[],"exists":true}

Example: "no dog"
-> {"target":null,"attributes":[],"relations":[],"actions":[],"negatives":["absent_object"],"exists":false}"""

_REVIEW_PROMPT = """Review 10 annotations. For each, check if TARGET is the main referred object (not relation/action target), ATTRIBUTES are correct, and RELATIONS vs ACTIONS are separated correctly. Output ONLY a JSON array of 10 booleans (true=correct, false=wrong).

Annotations:
{annotations}"""


def _load_api_key() -> str:
    env_file = Path(__file__).resolve().parent / ".env"
    if env_file.is_file():
        for line in env_file.read_text().strip().split("\n"):
            line = line.strip()
            if line.startswith("DEEPSEEK_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return os.environ.get("DEEPSEEK_API_KEY", "")


@dataclass
class AnnotatorConfig:
    model: str = _DEFAULT_MODEL
    review_model: str = _DEFAULT_REVIEW_MODEL
    api_key: str = field(default_factory=_load_api_key)
    base_url: str = _DEFAULT_BASE_URL
    temperature: float = 0.0
    max_tokens: int = 256
    max_retries: int = 3
    workers: int = _DEFAULT_WORKERS
    review_interval: int = _REVIEW_INTERVAL
    review_sample: int = _REVIEW_SAMPLE


class LLMAnnotator:
    def __init__(self, config: AnnotatorConfig | None = None) -> None:
        self._config = config or AnnotatorConfig()
        self._client: Any = None
        self._review_client: Any = None

    def _make_client(self) -> Any:
        import httpx
        from openai import OpenAI
        transport = httpx.HTTPTransport(proxy=None)
        http_client = httpx.Client(
            transport=transport,
            timeout=60.0,
            limits=httpx.Limits(
                max_keepalive_connections=100, max_connections=200,
            ),
        )
        return OpenAI(
            api_key=self._config.api_key,
            base_url=self._config.base_url,
            http_client=http_client,
        )

    def _ensure_client(self) -> Any:
        if self._client is None:
            self._client = self._make_client()
        return self._client

    def _ensure_review_client(self) -> Any:
        if self._review_client is None:
            self._review_client = self._make_client()
        return self._review_client

    def annotate(self, query: str) -> NormalizedQuery | None:
        last_error: str | None = None
        for attempt in range(self._config.max_retries):
            try:
                raw = self._call_api(query)
                parsed = self._parse_response(raw)
                if parsed is not None:
                    return parsed
            except Exception as exc:
                last_error = str(exc)
                if attempt < self._config.max_retries - 1:
                    time.sleep(0.5 * (attempt + 1))
        return None

    def _call_api(self, query: str) -> str:
        client = self._ensure_client()
        response = client.chat.completions.create(
            model=self._config.model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            temperature=self._config.temperature,
            max_tokens=self._config.max_tokens,
        )
        return str(response.choices[0].message.content or "").strip()

    def review_batch(
        self, items: list[tuple[str, NormalizedQuery]]
    ) -> list[bool]:
        if not items:
            return []
        lines = []
        for i, (q, a) in enumerate(items):
            lines.append(f"{i}: query={q} -> tgt={a['target']}, attr={a['attributes']}, rel={a['relations']}, act={a['actions']}, exists={a['exists']}")
        prompt = f"Review these annotations. For each, output true if the main target is correct and relations/actions are properly separated. Output ONLY a JSON array of {len(items)} booleans.\n\n" + "\n".join(lines)
        try:
            client = self._ensure_review_client()
            response = client.chat.completions.create(
                model=self._config.review_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=256,
                timeout=120,
            )
            content = response.choices[0].message.content
            reasoning = getattr(response.choices[0].message, 'reasoning_content', None)
            raw = (content or reasoning or "").strip()
            raw = raw.strip("`").strip()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                start = raw.rfind("[")
                end = raw.rfind("]")
                if start >= 0 and end > start:
                    data = json.loads(raw[start:end + 1])
                else:
                    raise
            if isinstance(data, list):
                return [bool(v) for v in data[:len(items)]]
        except Exception:
            pass
        return [True] * len(items)

    def _parse_response(self, raw: str) -> NormalizedQuery | None:
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("\n```", 1)[0]
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            start = raw.find("{")
            end = raw.rfind("}")
            if start >= 0 and end > start:
                try:
                    data = json.loads(raw[start:end + 1])
                except json.JSONDecodeError:
                    return None
            else:
                return None
        return self._normalize(data)

    def _normalize(self, raw: dict[str, Any]) -> NormalizedQuery:
        exists = bool(raw.get("exists", True))
        return {
            "target": raw.get("target") if exists else None,
            "attributes": self._as_str_list(raw.get("attributes")),
            "relations": self._normalize_relations(raw.get("relations", [])),
            "actions": self._normalize_actions(raw.get("actions", [])),
            "negatives": self._as_str_list(raw.get("negatives"))
            or (["absent_object"] if not exists else []),
            "exists": exists,
        }

    @staticmethod
    def _as_str_list(value: Any) -> list[str]:
        if isinstance(value, list):
            return [str(v) for v in value]
        if isinstance(value, str) and value:
            return [value]
        return []

    @staticmethod
    def _normalize_relations(raw: Any) -> list[dict[str, str]]:
        if not isinstance(raw, list):
            return []
        result: list[dict[str, str]] = []
        for item in raw:
            if isinstance(item, dict):
                t = str(item.get("type", item.get("relation", "")))
                tg = str(item.get("target", ""))
                if t:
                    result.append({"type": t, "target": tg})
        return result

    @staticmethod
    def _normalize_actions(raw: Any) -> list[dict[str, str | None]]:
        if not isinstance(raw, list):
            return []
        result: list[dict[str, str | None]] = []
        for item in raw:
            if isinstance(item, dict):
                v = str(item.get("verb", item.get("action", "")))
                tgt = item.get("target")
                if v:
                    result.append({
                        "verb": v,
                        "target": str(tgt) if tgt else None,
                    })
        return result


def batch_annotate_queries(
    queries: list[str],
    output_path: Path,
    config: AnnotatorConfig | None = None,
    batch_size: int = 256,
) -> list[NormalizedQuery]:
    cfg = config or AnnotatorConfig()
    workers = cfg.workers
    total = len(queries)
    results_map: dict[int, NormalizedQuery] = {}

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_annotate_one, query, cfg): idx
            for idx, query in enumerate(queries)
        }
        done = 0
        for future in as_completed(futures):
            idx = futures[future]
            try:
                r = future.result()
                if r is not None:
                    results_map[idx] = r
            except Exception:
                pass
            done += 1
            if done % 50 == 0 or done == total:
                print(f"  Annotated: {done}/{total} ({100*done/total:.0f}%)", flush=True)

    all_results = [results_map.get(i) for i in range(total)]
    reviewed, removed = _review_loop(queries, all_results, cfg)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(reviewed, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return reviewed


def _review_loop(
    queries: list[str],
    annotations: list[NormalizedQuery | None],
    cfg: AnnotatorConfig,
) -> tuple[list[NormalizedQuery], int]:
    valid = [(i, q, a) for i, (q, a) in enumerate(zip(queries, annotations)) if a is not None]
    kept: list[NormalizedQuery] = []
    removed = 0
    reviewer = LLMAnnotator(cfg)
    interval = cfg.review_interval
    sample_n = cfg.review_sample

    for batch_start in range(0, len(valid), interval):
        batch_end = min(batch_start + interval, len(valid))
        batch = valid[batch_start:batch_end]

        rng = random.Random(42 + batch_start)
        review_indices = rng.sample(
            range(len(batch)), min(sample_n, len(batch))
        ) if len(batch) > sample_n else list(range(len(batch)))

        review_items = [(q, a) for _, q, a in [batch[ri] for ri in review_indices]]
        results = reviewer.review_batch(review_items)
        passed = sum(1 for ok in results if ok)
        failed = len(results) - passed

        batch_num = batch_start // interval + 1
        total_batches = (len(valid) - 1) // interval + 1
        pass_rate = passed / max(passed + failed, 1) if passed + failed > 0 else 1.0
        print(
            f"  Review batch {batch_num}/{total_batches}: "
            f"sampled={passed+failed}, passed={passed}, failed={failed} "
            f"({100*pass_rate:.0f}% ok)", flush=True
        )

        for _, _, a in batch:
            kept.append(a)

    return kept, removed


def _annotate_one(query: str, config: AnnotatorConfig) -> NormalizedQuery | None:
    ann = LLMAnnotator(config)
    return ann.annotate(query)
