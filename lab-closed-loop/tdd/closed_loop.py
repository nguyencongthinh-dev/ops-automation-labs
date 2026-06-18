#!/usr/bin/env python3
"""
closed_loop.py — Ronki closed-loop auto-remediation orchestrator.

Usage: uv run python closed_loop.py --config config.yaml [--dry-run]

5 sub-checkpoints (all mandatory): dry-run / blast-radius / verify /
auto-rollback / circuit-breaker.  See engine/ for helper modules.

Stress extensions:
  - Per-service mutex: serializes runbook execution per service name.
  - Transactional action history: tracks completed steps; on failure
    invokes rollbacks in reverse order.
  - Decision validation: rejects runbook names absent from registry.
"""

import argparse
import json
import subprocess
import threading
import time
from pathlib import Path

import requests
import yaml

from engine.logger import JsonLogger
from engine.metrics import (
    action_counter,
    blast_radius_gauge,
    circuit_breaker_gauge,
    mutex_gauge,
    start_metrics_server,
    verify_status_gauge,
)
from engine.safety import BlastRadiusGuard, CircuitBreaker
from engine.verify import verify_service

log = JsonLogger("orchestrator")

# ── Per-service mutex map ─────────────────────────────────────────────────────
# Serializes runbook execution per service; different services run in parallel.
_service_locks: dict[str, threading.Lock] = {}
_locks_meta = threading.Lock()


def get_service_lock(service: str) -> threading.Lock:
    with _locks_meta:
        if service not in _service_locks:
            _service_locks[service] = threading.Lock()
        return _service_locks[service]


# ── Config ────────────────────────────────────────────────────────────────────


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Alertmanager polling ──────────────────────────────────────────────────────


def fetch_active_alerts(alertmanager_url: str) -> list[dict]:
    """Return active, non-silenced, non-inhibited alerts."""
    try:
        resp = requests.get(
            f"{alertmanager_url}/api/v2/alerts",
            params={"active": "true", "silenced": "false", "inhibited": "false"},
            timeout=5,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        log.error("ALERTMANAGER_FETCH_ERROR", error=str(exc))
        return []


# ── Runbook execution ─────────────────────────────────────────────────────────


def run_runbook(script: str, service: str, dry_run: bool, timeout_s: int = 30) -> bool:
    """Execute runbook script. Returns True on exit code 0."""
    # Split the script string in case there are arguments passed (e.g. for multi-step)
    parts = script.split()
    script_path = parts[0]
    script_args = parts[1:]

    import os

    bash_exec = "bash" if os.name == "nt" else "/bin/bash"
    cmd = [bash_exec, script_path, "--service", service]
    cmd.extend(script_args)
    if dry_run:
        cmd.append("--dry-run")
    log.info("RUNBOOK_EXEC", script=script, service=service, dry_run=dry_run)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        log.info(
            "RUNBOOK_RESULT",
            script=script,
            service=service,
            returncode=result.returncode,
            stdout=result.stdout.strip(),
            stderr=result.stderr.strip(),
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        log.error(
            "RUNBOOK_TIMEOUT", script=script, service=service, timeout_s=timeout_s
        )
        return False
    except Exception as exc:
        log.error("RUNBOOK_ERROR", script=script, service=service, error=str(exc))
        return False


# ── Decide ────────────────────────────────────────────────────────────────────


def extract_service(alert: dict) -> str:
    labels = alert.get("labels", {})
    return labels.get("service") or labels.get("job") or "unknown"


def validate_runbook(
    runbook: str, cfg: dict, alertname: str, raw_decision: str
) -> bool:
    """Stress 3: reject runbook names absent from the runbook registry."""
    registry: list[str] = cfg.get(
        "runbook_registry", list(cfg.get("runbook_map", {}).values())
    )

    # Extract the base script path to validate against the registry
    base_runbook = runbook.split()[0] if runbook else ""
    if base_runbook in registry:
        return True
    log.error(
        "DECISION_VALIDATION_FAILED",
        bad_runbook=runbook,
        alertname=alertname,
        raw_decision=raw_decision,
        action="escalate_no_auto_action",
    )
    return False


def run_transactional_steps(
    steps: list[str],
    service: str,
    dry_run: bool,
    timeout_s: int,
) -> tuple[bool, list[str]]:
    """Stress 1: execute steps A→B→C; return (success, completed_steps).

    completed_steps lists every step that exited 0, in execution order.
    Caller uses this list to drive reverse-order rollback.
    """
    completed: list[str] = []
    for step in steps:
        if not run_runbook(step, service, dry_run=dry_run, timeout_s=timeout_s):
            log.error(
                "TRANSACTIONAL_STEP_FAIL",
                step=step,
                service=service,
                completed_before_failure=completed,
            )
            return False, completed
        completed.append(step)
    return True, completed


# ── Per-alert processing (all 5 checkpoints) ─────────────────────────────────


def process_alert(
    alert: dict,
    cfg: dict,
    baseline: dict,
    guard: BlastRadiusGuard,
    cb: CircuitBreaker,
    global_dry_run: bool,
):
    alertname = alert.get("labels", {}).get("alertname", "")
    service = extract_service(alert)

    log.info(
        "ALERT_DETECTED",
        alertname=alertname,
        service=service,
        severity=alert.get("labels", {}).get("severity", ""),
    )

    # 1. Decide — map alert → runbook
    runbook = cfg["runbook_map"].get(alertname)
    if not runbook:
        log.warning("NO_RUNBOOK", alertname=alertname, service=service)
        return

    # Stress 3: decision validation — reject hallucinated runbook names
    if not validate_runbook(runbook, cfg, alertname, raw_decision=runbook):
        return
    log.info("DECIDE_RUNBOOK", alertname=alertname, service=service, runbook=runbook)

    # 2. Blast-radius check
    ok, reason = guard.check(service)
    if not ok:
        log.warning("BLAST_RADIUS_EXCEEDED", service=service, reason=reason)
        return
    log.info("BLAST_RADIUS_OK", service=service)

    # Stress 2: per-service lock — serialize actions on same service
    svc_lock = get_service_lock(service)
    acquired = svc_lock.acquire(blocking=False)
    if not acquired:
        log.warning(
            "SERVICE_LOCK_BUSY",
            service=service,
            message="Another runbook is executing for this service; skipping duplicate",
        )
        return
    mutex_gauge.labels(service=service).set(1)
    try:
        _process_alert_locked(
            alert, alertname, service, runbook, cfg, baseline, guard, cb, global_dry_run
        )
    finally:
        mutex_gauge.labels(service=service).set(0)
        svc_lock.release()


def _process_alert_locked(
    alert: dict,
    alertname: str,
    service: str,
    runbook: str,
    cfg: dict,
    baseline: dict,
    guard: BlastRadiusGuard,
    cb: CircuitBreaker,
    global_dry_run: bool,
):
    timeout_s = cfg["runbook_timeout_seconds"]

    # 3a. Dry-run (always, regardless of global --dry-run flag)
    # For multi-step, dry-run the first step or the general script
    dry_run_target = runbook
    multi_steps: list[str] = cfg.get("multi_step_map", {}).get(alertname, [])
    if multi_steps:
        dry_run_target = multi_steps[0]

    if not run_runbook(dry_run_target, service, dry_run=True, timeout_s=timeout_s):
        log.error("DRY_RUN_FAIL", runbook=dry_run_target, service=service)
        return
    log.info("DRY_RUN_PASS", runbook=dry_run_target, service=service)

    # Short-circuit: global --dry-run stops here
    if global_dry_run:
        action_counter.labels(service=service, runbook=runbook, outcome="dry_run").inc()
        log.info(
            "GLOBAL_DRY_RUN_SKIP", message="--dry-run flag set; skipping real action"
        )
        return

    # 3b. Execute action — optionally as a transactional multi-step chain
    guard.record(service)
    remaining = cfg["blast_radius"]["max_actions_per_minute"] - len(
        guard._global_window
    )
    blast_radius_gauge.labels(service=service).set(max(0, remaining))

    if multi_steps:
        # Stress 1: transactional execution
        success, completed = run_transactional_steps(
            multi_steps, service, False, timeout_s
        )
        if not success:
            # Rollback in reverse order
            rollback_steps: list[str] = cfg.get("multi_step_rollback_map", {}).get(
                alertname, []
            )
            # Only rollback the steps that were actually completed
            for rb_step in reversed(rollback_steps[: len(completed)]):
                log.warning(
                    "TRANSACTIONAL_ROLLBACK_STEP", step=rb_step, service=service
                )
                run_runbook(rb_step, service, dry_run=False, timeout_s=timeout_s)
            log.info(
                "TRANSACTIONAL_ROLLBACK_COMPLETE",
                service=service,
                rolled_back=list(reversed(rollback_steps[: len(completed)])),
            )
            cb.record_failure()
            return
    else:
        if not run_runbook(runbook, service, dry_run=False, timeout_s=timeout_s):
            log.error("ACTION_EXEC_FAIL", runbook=runbook, service=service)
            cb.record_failure()
            return
    log.info("ACTION_EXECUTED", runbook=runbook, service=service)

    # 4. Verify post-action
    t = baseline["verify_thresholds"]
    verify_status_gauge.labels(service=service, runbook=runbook).set(2)  # in_progress
    verify_ok = verify_service(
        prometheus_url=cfg["prometheus_url"],
        service=service,
        baseline=baseline,
        timeout_s=t["verify_timeout_seconds"],
        poll_interval_s=t["verify_poll_interval_seconds"],
        min_samples=t["verify_min_samples"],
    )

    if verify_ok:
        verify_status_gauge.labels(service=service, runbook=runbook).set(1)  # pass
        action_counter.labels(service=service, runbook=runbook, outcome="success").inc()
        log.info(
            "ACTION_SUCCESS", alertname=alertname, service=service, runbook=runbook
        )
        cb.record_success()
        circuit_breaker_gauge.labels(service=service).set(0)
        return

    # 5. Auto-rollback on verify failure
    verify_status_gauge.labels(service=service, runbook=runbook).set(0)  # fail
    action_counter.labels(service=service, runbook=runbook, outcome="rollback").inc()
    rollback = cfg.get("rollback_map", {}).get(alertname, runbook)
    log.warning("ROLLBACK_TRIGGERED", service=service, rollback_runbook=rollback)
    rollback_ok = run_runbook(rollback, service, dry_run=False, timeout_s=timeout_s)

    if rollback_ok:
        log.info("ROLLBACK_EXECUTED", service=service, rollback_runbook=rollback)
        # Verify rollback result using Prometheus to see if the service has recovered
        log.info("ROLLBACK_VERIFY_START", service=service)
        rollback_verify_ok = verify_service(
            prometheus_url=cfg["prometheus_url"],
            service=service,
            baseline=baseline,
            timeout_s=t["verify_timeout_seconds"],
            poll_interval_s=t["verify_poll_interval_seconds"],
            min_samples=t["verify_min_samples"],
        )
        if rollback_verify_ok:
            log.info(
                "ROLLBACK_VERIFY_SUCCESS", service=service, rollback_runbook=rollback
            )
        else:
            log.error(
                "ROLLBACK_VERIFY_FAIL", service=service, rollback_runbook=rollback
            )
    else:
        log.error("ROLLBACK_FAILED", service=service, rollback_runbook=rollback)

    cb.record_failure()
    circuit_breaker_gauge.labels(service=service).set(1 if cb.is_open() else 0)


# ── Main loop ─────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Ronki closed-loop orchestrator")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Detect + decide only; do not execute real actions",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    baseline_path = Path(args.config).parent / cfg["baseline_path"]
    with open(baseline_path) as f:
        baseline = json.load(f)

    guard = BlastRadiusGuard(
        max_per_minute=cfg["blast_radius"]["max_actions_per_minute"],
        max_restarts_per_hour=cfg["blast_radius"]["max_restarts_per_service_per_hour"],
    )
    cb = CircuitBreaker(
        threshold=cfg["circuit_breaker"]["consecutive_failure_threshold"]
    )
    seen: set[str] = set()

    start_metrics_server()
    log.info(
        "ORCHESTRATOR_START",
        config=args.config,
        dry_run=args.dry_run,
        poll_interval_s=cfg["poll_interval_seconds"],
    )

    while True:
        if cb.is_open():
            log.error(
                "CIRCUIT_BREAKER_HALT", message="Circuit open — polling suspended."
            )
            time.sleep(cfg["poll_interval_seconds"])
            continue

        for alert in fetch_active_alerts(cfg["alertmanager_url"]):
            fp = alert.get("fingerprint", "")
            if fp and fp in seen:
                continue
            if fp:
                seen.add(fp)
            t = threading.Thread(
                target=process_alert,
                args=(alert, cfg, baseline, guard, cb, args.dry_run),
            )
            t.start()

        if len(seen) > 500:
            seen.clear()

        time.sleep(cfg["poll_interval_seconds"])


if __name__ == "__main__":
    main()
