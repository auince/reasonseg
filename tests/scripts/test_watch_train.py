from __future__ import annotations

import importlib.util
import signal
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


def _load_watch_train_module():
    module_path = ROOT / "scripts" / "watch_train.py"
    spec = importlib.util.spec_from_file_location("reasonseg_watch_train", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_nvidia_smi_csv_parses_expected_fields() -> None:
    watch_train = _load_watch_train_module()

    samples = watch_train._parse_nvidia_smi_csv(
        "0, 2026/04/28 00:00:00.000, 45.5, 3, 2048\n1, 2026/04/28 00:00:00.000, 46.0, 4, 2050\n"
    )

    assert [sample.index for sample in samples] == [0, 1]
    assert samples[0].power_draw_watts == 45.5
    assert samples[1].memory_used_mib == 2050


def test_all_target_gpus_low_power_requires_allocated_and_idle() -> None:
    watch_train = _load_watch_train_module()

    low_power = [
        watch_train.GpuSample(0, "ts", 40.0, 2.0, 2048.0),
        watch_train.GpuSample(1, "ts", 41.0, 3.0, 3072.0),
    ]
    assert watch_train._all_target_gpus_low_power(
        low_power,
        gpu_indices=[0, 1],
        low_power_max_watts=70.0,
        low_utilization_max=10.0,
        min_allocated_mib=1024.0,
    )

    one_busy = [
        watch_train.GpuSample(0, "ts", 40.0, 2.0, 2048.0),
        watch_train.GpuSample(1, "ts", 120.0, 45.0, 3072.0),
    ]
    assert not watch_train._all_target_gpus_low_power(
        one_busy,
        gpu_indices=[0, 1],
        low_power_max_watts=70.0,
        low_utilization_max=10.0,
        min_allocated_mib=1024.0,
    )


def test_logs_grew_detects_size_increase(tmp_path: Path) -> None:
    watch_train = _load_watch_train_module()
    log_path = tmp_path / "log.txt"
    _ = log_path.write_text("first\n", encoding="utf-8")

    first = watch_train._read_log_snapshots([log_path])
    _ = log_path.write_text("first\nsecond\n", encoding="utf-8")
    second = watch_train._read_log_snapshots([log_path])

    assert watch_train._logs_grew(first, second)


def test_terminate_process_group_escalates_after_grace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    watch_train = _load_watch_train_module()
    signals: list[tuple[int, int]] = []

    class _FakeProcess:
        pid: int
        returncode: int | None
        poll_calls: int

        def __init__(self) -> None:
            self.pid = 321
            self.returncode = -9
            self.poll_calls = 0

        def poll(self):
            self.poll_calls += 1
            return None

        def wait(self) -> int:
            assert self.returncode is not None
            return self.returncode

    process = _FakeProcess()
    monotonic_values = iter([0.0, 0.0, 0.3, 0.6])
    monkeypatch.setattr(watch_train.os, "getpgid", lambda pid: 999)
    monkeypatch.setattr(
        watch_train.os, "killpg", lambda pgid, sig: signals.append((pgid, sig))
    )
    monkeypatch.setattr(watch_train.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(watch_train.time, "sleep", lambda _: None)

    outcome = watch_train._terminate_process_group(process, termination_grace=0.5)

    assert signals == [(999, signal.SIGTERM), (999, signal.SIGKILL)]
    assert outcome.returncode == -9


def test_run_watchdog_terminates_on_log_stall(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    watch_train = _load_watch_train_module()
    output_dir = tmp_path / "outputs"
    run_dir = output_dir / "run_0"
    run_dir.mkdir(parents=True)
    log_path = run_dir / "log.txt"
    _ = log_path.write_text("boot\n", encoding="utf-8")

    class _FakeProcess:
        pid: int
        returncode: int | None

        def __init__(self) -> None:
            self.pid = 123
            self.returncode = None

        def poll(self):
            return None

        def wait(self) -> int:
            self.returncode = -15
            return -15

    fake_process = _FakeProcess()
    monotonic_values = iter([0.0, 1.0, 10.0, 11.0])
    monkeypatch.setattr(
        watch_train.subprocess,
        "Popen",
        lambda *args, **kwargs: fake_process,
    )
    monkeypatch.setattr(watch_train.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(watch_train.time, "sleep", lambda _: None)
    monkeypatch.setattr(watch_train, "_sample_gpus", lambda indices: [])
    monkeypatch.setattr(
        watch_train,
        "_terminate_process_group",
        lambda process, termination_grace: watch_train.TerminationOutcome(
            process_group_id=123,
            signals=["SIGTERM"],
            returncode=-15,
        ),
    )
    monkeypatch.setattr(watch_train, "_capture_ps_snapshot", lambda pgid: "ps")

    result = watch_train.run_watchdog(
        watch_train.WatchdogArgs(
            command=["python", "train.py"],
            output_dir=str(output_dir),
            run_index=0,
            gpu_indices="0,1",
            poll_interval=0.1,
            stall_timeout=5.0,
            low_power_timeout=5.0,
            low_power_max_watts=70.0,
            low_utilization_max=10.0,
            min_allocated_mib=1024.0,
            startup_grace=1.0,
            termination_grace=0.1,
            evidence_tail_lines=20,
        )
    )

    assert result == 124
    evidence_path = run_dir / "watchdog" / "watchdog_evidence.json"
    assert evidence_path.is_file()
    assert "log_stall" in evidence_path.read_text(encoding="utf-8")
