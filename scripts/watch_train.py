from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path


SERIALIZED_DATASET_MARKER = "Serialized dataset takes"
DEFAULT_POLL_INTERVAL_SECONDS = 15.0
DEFAULT_STALL_TIMEOUT_SECONDS = 900.0
DEFAULT_LOW_POWER_TIMEOUT_SECONDS = 600.0
DEFAULT_LOW_POWER_MAX_WATTS = 70.0
DEFAULT_LOW_UTILIZATION_MAX_PERCENT = 10.0
DEFAULT_MIN_ALLOCATED_MIB = 1024.0
DEFAULT_STARTUP_GRACE_SECONDS = 180.0
DEFAULT_TERMINATION_GRACE_SECONDS = 30.0
DEFAULT_EVIDENCE_TAIL_LINES = 200


@dataclass(frozen=True)
class GpuSample:
    index: int
    timestamp: str
    power_draw_watts: float
    utilization_gpu_percent: float
    memory_used_mib: float


@dataclass(frozen=True)
class LogSnapshot:
    path: str
    exists: bool
    size_bytes: int
    modified_time: float | None
    contains_serialized_marker: bool


@dataclass(frozen=True)
class WatchdogArgs:
    output_dir: str
    run_index: int
    gpu_indices: str
    poll_interval: float
    stall_timeout: float
    low_power_timeout: float
    low_power_max_watts: float
    low_utilization_max: float
    min_allocated_mib: float
    startup_grace: float
    termination_grace: float
    evidence_tail_lines: int
    command: list[str]


@dataclass(frozen=True)
class TerminationOutcome:
    process_group_id: int
    signals: list[str]
    returncode: int | None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a training command under an external watchdog that terminates the "
            "full process group when logs stall or GPUs stay abnormally idle."
        )
    )
    _ = parser.add_argument("--output-dir", required=True)
    _ = parser.add_argument("--run-index", type=int, default=0)
    _ = parser.add_argument("--gpu-indices", default="0,1")
    _ = parser.add_argument(
        "--poll-interval", type=float, default=DEFAULT_POLL_INTERVAL_SECONDS
    )
    _ = parser.add_argument(
        "--stall-timeout", type=float, default=DEFAULT_STALL_TIMEOUT_SECONDS
    )
    _ = parser.add_argument(
        "--low-power-timeout", type=float, default=DEFAULT_LOW_POWER_TIMEOUT_SECONDS
    )
    _ = parser.add_argument(
        "--low-power-max-watts", type=float, default=DEFAULT_LOW_POWER_MAX_WATTS
    )
    _ = parser.add_argument(
        "--low-utilization-max",
        type=float,
        default=DEFAULT_LOW_UTILIZATION_MAX_PERCENT,
    )
    _ = parser.add_argument(
        "--min-allocated-mib", type=float, default=DEFAULT_MIN_ALLOCATED_MIB
    )
    _ = parser.add_argument(
        "--startup-grace", type=float, default=DEFAULT_STARTUP_GRACE_SECONDS
    )
    _ = parser.add_argument(
        "--termination-grace",
        type=float,
        default=DEFAULT_TERMINATION_GRACE_SECONDS,
    )
    _ = parser.add_argument(
        "--evidence-tail-lines", type=int, default=DEFAULT_EVIDENCE_TAIL_LINES
    )
    _ = parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Training command to run. Prefix with '--' to separate watchdog args.",
    )
    return parser


def _parse_gpu_indices(raw_value: str) -> list[int]:
    return [int(part.strip()) for part in raw_value.split(",") if part.strip()]


def _get_log_paths(output_dir: Path, run_index: int) -> list[Path]:
    run_dir = output_dir / f"run_{run_index}"
    return [run_dir / "log.txt", run_dir / "log.txt.rank1"]


def _tail_text(path: Path, limit_lines: int) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-limit_lines:])


def _read_log_snapshots(paths: list[Path]) -> list[LogSnapshot]:
    snapshots: list[LogSnapshot] = []
    for path in paths:
        if not path.exists():
            snapshots.append(
                LogSnapshot(
                    path=str(path),
                    exists=False,
                    size_bytes=0,
                    modified_time=None,
                    contains_serialized_marker=False,
                )
            )
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        stat = path.stat()
        snapshots.append(
            LogSnapshot(
                path=str(path),
                exists=True,
                size_bytes=stat.st_size,
                modified_time=stat.st_mtime,
                contains_serialized_marker=SERIALIZED_DATASET_MARKER in text,
            )
        )
    return snapshots


def _logs_grew(previous: list[LogSnapshot] | None, current: list[LogSnapshot]) -> bool:
    if previous is None:
        return any(snapshot.exists and snapshot.size_bytes > 0 for snapshot in current)
    previous_by_path = {snapshot.path: snapshot for snapshot in previous}
    for snapshot in current:
        previous_snapshot = previous_by_path.get(snapshot.path)
        if previous_snapshot is None:
            if snapshot.exists and snapshot.size_bytes > 0:
                return True
            continue
        if snapshot.size_bytes > previous_snapshot.size_bytes:
            return True
    return False


def _has_any_log_output(snapshots: list[LogSnapshot]) -> bool:
    return any(snapshot.exists and snapshot.size_bytes > 0 for snapshot in snapshots)


def _parse_nvidia_smi_csv(raw_text: str) -> list[GpuSample]:
    samples: list[GpuSample] = []
    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            raise ValueError(f"Unexpected nvidia-smi output line: {raw_line!r}")
        samples.append(
            GpuSample(
                index=int(parts[0]),
                timestamp=parts[1],
                power_draw_watts=float(parts[2]),
                utilization_gpu_percent=float(parts[3]),
                memory_used_mib=float(parts[4]),
            )
        )
    return samples


def _sample_gpus(gpu_indices: list[int]) -> list[GpuSample]:
    query = "index,timestamp,power.draw,utilization.gpu,memory.used"
    command = [
        "nvidia-smi",
        f"--query-gpu={query}",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    samples = _parse_nvidia_smi_csv(completed.stdout)
    wanted = set(gpu_indices)
    return [sample for sample in samples if sample.index in wanted]


def _all_target_gpus_low_power(
    samples: list[GpuSample],
    *,
    gpu_indices: list[int],
    low_power_max_watts: float,
    low_utilization_max: float,
    min_allocated_mib: float,
) -> bool:
    if not samples:
        return False
    sample_by_index = {sample.index: sample for sample in samples}
    for gpu_index in gpu_indices:
        sample = sample_by_index.get(gpu_index)
        if sample is None:
            return False
        if sample.memory_used_mib < min_allocated_mib:
            return False
        if sample.power_draw_watts > low_power_max_watts:
            return False
        if sample.utilization_gpu_percent > low_utilization_max:
            return False
    return True


def _terminate_process_group(
    process: subprocess.Popen[str], termination_grace: float
) -> TerminationOutcome:
    process_group_id = os.getpgid(process.pid)
    os.killpg(process_group_id, signal.SIGTERM)
    signals = ["SIGTERM"]
    deadline = time.monotonic() + termination_grace
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return TerminationOutcome(process_group_id, signals, process.returncode)
        time.sleep(0.5)
    os.killpg(process_group_id, signal.SIGKILL)
    signals.append("SIGKILL")
    _ = process.wait()
    return TerminationOutcome(process_group_id, signals, process.returncode)


def _capture_ps_snapshot(process_group_id: int) -> str:
    completed = subprocess.run(
        [
            "ps",
            "-o",
            "pid,pgid,ppid,stat,etime,command",
            "--forest",
            "-g",
            str(process_group_id),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def _write_watchdog_evidence(
    output_dir: Path,
    *,
    run_index: int,
    reason: str,
    command: list[str],
    log_snapshots: list[LogSnapshot],
    gpu_samples: list[GpuSample],
    termination: TerminationOutcome,
    evidence_tail_lines: int,
) -> Path:
    evidence_dir = output_dir / f"run_{run_index}" / "watchdog"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    log_tails: dict[str, str] = {}
    for snapshot in log_snapshots:
        log_tails[snapshot.path] = _tail_text(Path(snapshot.path), evidence_tail_lines)
    payload: dict[str, object] = {
        "reason": reason,
        "command": command,
        "log_snapshots": [asdict(snapshot) for snapshot in log_snapshots],
        "gpu_samples": [asdict(sample) for sample in gpu_samples],
        "termination": asdict(termination),
        "log_tails": log_tails,
        "ps_snapshot": _capture_ps_snapshot(termination.process_group_id),
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    evidence_path = evidence_dir / "watchdog_evidence.json"
    _ = evidence_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return evidence_path


def run_watchdog(args: WatchdogArgs) -> int:
    if not args.command:
        raise ValueError("watch_train.py requires a command after '--'.")
    command = list(args.command)
    if command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("watch_train.py requires a non-empty command after '--'.")

    output_dir = Path(args.output_dir).expanduser().resolve()
    gpu_indices = _parse_gpu_indices(args.gpu_indices)
    log_paths = _get_log_paths(output_dir, args.run_index)
    process = subprocess.Popen(command, start_new_session=True, text=True)

    start_time = time.monotonic()
    previous_logs: list[LogSnapshot] | None = None
    last_log_progress_time = start_time
    low_power_since: float | None = None
    last_gpu_samples: list[GpuSample] = []

    while True:
        if process.poll() is not None:
            return int(process.returncode or 0)

        current_time = time.monotonic()
        log_snapshots = _read_log_snapshots(log_paths)
        if _logs_grew(previous_logs, log_snapshots):
            last_log_progress_time = current_time
        previous_logs = log_snapshots

        gpu_samples: list[GpuSample] = []
        try:
            gpu_samples = _sample_gpus(gpu_indices)
            last_gpu_samples = gpu_samples
        except Exception:
            gpu_samples = []

        elapsed = current_time - start_time
        if elapsed >= args.startup_grace:
            if (
                _has_any_log_output(log_snapshots)
                and current_time - last_log_progress_time >= args.stall_timeout
            ):
                termination = _terminate_process_group(process, args.termination_grace)
                evidence_path = _write_watchdog_evidence(
                    output_dir,
                    run_index=args.run_index,
                    reason="log_stall",
                    command=command,
                    log_snapshots=log_snapshots,
                    gpu_samples=last_gpu_samples,
                    termination=termination,
                    evidence_tail_lines=args.evidence_tail_lines,
                )
                print(f"watchdog terminated training due to log stall: {evidence_path}")
                return 124

            if _all_target_gpus_low_power(
                gpu_samples,
                gpu_indices=gpu_indices,
                low_power_max_watts=args.low_power_max_watts,
                low_utilization_max=args.low_utilization_max,
                min_allocated_mib=args.min_allocated_mib,
            ):
                if low_power_since is None:
                    low_power_since = current_time
                elif current_time - low_power_since >= args.low_power_timeout:
                    termination = _terminate_process_group(
                        process, args.termination_grace
                    )
                    evidence_path = _write_watchdog_evidence(
                        output_dir,
                        run_index=args.run_index,
                        reason="low_gpu_power",
                        command=command,
                        log_snapshots=log_snapshots,
                        gpu_samples=last_gpu_samples,
                        termination=termination,
                        evidence_tail_lines=args.evidence_tail_lines,
                    )
                    print(
                        f"watchdog terminated training due to sustained low GPU activity: {evidence_path}"
                    )
                    return 124
            else:
                low_power_since = None

        time.sleep(args.poll_interval)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    parsed = parser.parse_args(argv)
    return run_watchdog(
        WatchdogArgs(
            output_dir=str(parsed.output_dir),
            run_index=int(parsed.run_index),
            gpu_indices=str(parsed.gpu_indices),
            poll_interval=float(parsed.poll_interval),
            stall_timeout=float(parsed.stall_timeout),
            low_power_timeout=float(parsed.low_power_timeout),
            low_power_max_watts=float(parsed.low_power_max_watts),
            low_utilization_max=float(parsed.low_utilization_max),
            min_allocated_mib=float(parsed.min_allocated_mib),
            startup_grace=float(parsed.startup_grace),
            termination_grace=float(parsed.termination_grace),
            evidence_tail_lines=int(parsed.evidence_tail_lines),
            command=list(parsed.command),
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
