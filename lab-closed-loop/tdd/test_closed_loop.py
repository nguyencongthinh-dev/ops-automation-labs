import sys
import unittest
from unittest.mock import patch, MagicMock
import time
import threading
from pathlib import Path

# Add the tdd directory to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import closed_loop
from engine.safety import BlastRadiusGuard, CircuitBreaker


class TestBlastRadiusGuard(unittest.TestCase):
    def test_global_limit(self):
        guard = BlastRadiusGuard(max_per_minute=2, max_restarts_per_hour=5)
        # First action
        ok, reason = guard.check("svc-a")
        self.assertTrue(ok)
        guard.record("svc-a")

        # Second action
        ok, reason = guard.check("svc-b")
        self.assertTrue(ok)
        guard.record("svc-b")

        # Third action should fail
        ok, reason = guard.check("svc-c")
        self.assertFalse(ok)
        self.assertIn("global actions/min limit", reason)

    def test_service_limit(self):
        guard = BlastRadiusGuard(max_per_minute=10, max_restarts_per_hour=2)
        # First restart
        ok, reason = guard.check("svc-a")
        self.assertTrue(ok)
        guard.record("svc-a")

        # Second restart
        ok, reason = guard.check("svc-a")
        self.assertTrue(ok)
        guard.record("svc-a")

        # Third restart for svc-a should fail
        ok, reason = guard.check("svc-a")
        self.assertFalse(ok)
        self.assertIn("restarts/hour limit", reason)

        # But another service is unaffected
        ok, reason = guard.check("svc-b")
        self.assertTrue(ok)


class TestCircuitBreaker(unittest.TestCase):
    def test_circuit_breaker_flow(self):
        cb = CircuitBreaker(threshold=3)
        self.assertFalse(cb.is_open())

        cb.record_failure()
        self.assertFalse(cb.is_open())

        cb.record_failure()
        self.assertFalse(cb.is_open())

        cb.record_failure()
        self.assertTrue(cb.is_open())

        # Success should reset failures (if we reset it, though config reset is manual,
        # the class supports reset_success if called, but main loop handles manual restart).
        cb.record_success()
        self.assertEqual(cb._failures, 0)


class TestDecisionValidation(unittest.TestCase):
    def test_runbook_validation(self):
        cfg = {
            "runbook_registry": [
                "runbooks/restart_service.sh",
                "runbooks/clear_cache.sh",
            ]
        }
        # Valid runbooks
        self.assertTrue(
            closed_loop.validate_runbook(
                "runbooks/restart_service.sh",
                cfg,
                "AlertA",
                "runbooks/restart_service.sh",
            )
        )
        self.assertTrue(
            closed_loop.validate_runbook(
                "runbooks/clear_cache.sh", cfg, "AlertB", "runbooks/clear_cache.sh"
            )
        )

        # Valid runbook with arguments
        self.assertTrue(
            closed_loop.validate_runbook(
                "runbooks/restart_service.sh --step-a",
                cfg,
                "AlertA",
                "runbooks/restart_service.sh --step-a",
            )
        )

        # Invalid runbook (hallucination defense)
        with patch.object(closed_loop.log, "error") as mock_log:
            self.assertFalse(
                closed_loop.validate_runbook(
                    "runbooks/nonexistent.sh", cfg, "AlertC", "runbooks/nonexistent.sh"
                )
            )
            mock_log.assert_called_once()
            self.assertEqual(mock_log.call_args[0][0], "DECISION_VALIDATION_FAILED")


class TestTransactionalSteps(unittest.TestCase):
    @patch("closed_loop.run_runbook")
    def test_transactional_steps_success(self, mock_run):
        mock_run.return_value = True
        steps = ["step-a", "step-b", "step-c"]
        success, completed = closed_loop.run_transactional_steps(
            steps, "svc-a", False, 30
        )
        self.assertTrue(success)
        self.assertEqual(completed, steps)
        self.assertEqual(mock_run.call_count, 3)

    @patch("closed_loop.run_runbook")
    def test_transactional_steps_failure(self, mock_run):
        # First step succeeds, second fails, third not executed
        mock_run.side_effect = [True, False]
        steps = ["step-a", "step-b", "step-c"]
        success, completed = closed_loop.run_transactional_steps(
            steps, "svc-a", False, 30
        )
        self.assertFalse(success)
        self.assertEqual(completed, ["step-a"])
        self.assertEqual(mock_run.call_count, 2)


class TestServiceMutex(unittest.TestCase):
    @patch("closed_loop._process_alert_locked")
    @patch("closed_loop.log")
    def test_concurrent_mutex(self, mock_log, mock_locked):
        # We simulate a slow execution of _process_alert_locked to trigger lock contention
        def slow_execution(*args, **kwargs):
            time.sleep(0.5)

        mock_locked.side_effect = slow_execution

        cfg = {
            "runbook_map": {"HighLatency": "runbooks/restart_service.sh"},
            "runbook_registry": ["runbooks/restart_service.sh"],
        }
        baseline = {}
        guard = MagicMock()
        guard.check.return_value = (True, "ok")
        cb = MagicMock()

        alert = {"labels": {"alertname": "HighLatency", "service": "payment-svc"}}

        # Start thread 1
        t1 = threading.Thread(
            target=closed_loop.process_alert,
            args=(alert, cfg, baseline, guard, cb, False),
        )
        t1.start()
        time.sleep(0.1)  # Allow thread 1 to acquire lock

        # Run thread 2 (same service → should fail to acquire lock and log SERVICE_LOCK_BUSY)
        closed_loop.process_alert(alert, cfg, baseline, guard, cb, False)

        t1.join()

        # Assert SERVICE_LOCK_BUSY was logged
        busy_calls = [
            c for c in mock_log.warning.call_args_list if c[0][0] == "SERVICE_LOCK_BUSY"
        ]
        self.assertEqual(len(busy_calls), 1)

    @patch("closed_loop._process_alert_locked")
    @patch("closed_loop.log")
    def test_independent_services_do_not_block(self, mock_log, mock_locked):
        cfg = {
            "runbook_map": {"HighLatency": "runbooks/restart_service.sh"},
            "runbook_registry": ["runbooks/restart_service.sh"],
        }
        baseline = {}
        guard = MagicMock()
        guard.check.return_value = (True, "ok")
        cb = MagicMock()

        alert1 = {"labels": {"alertname": "HighLatency", "service": "payment-svc"}}
        alert2 = {"labels": {"alertname": "HighLatency", "service": "inventory-svc"}}

        # Both should succeed in acquiring their respective locks without any busy logs
        closed_loop.process_alert(alert1, cfg, baseline, guard, cb, False)
        closed_loop.process_alert(alert2, cfg, baseline, guard, cb, False)

        busy_calls = [
            c for c in mock_log.warning.call_args_list if c[0][0] == "SERVICE_LOCK_BUSY"
        ]
        self.assertEqual(len(busy_calls), 0)


class TestVerifyAndRollback(unittest.TestCase):
    @patch("closed_loop.verify_service")
    @patch("closed_loop.run_runbook")
    @patch("closed_loop.log")
    def test_verify_fail_trigger_rollback(self, mock_log, mock_run, mock_verify):
        # Action execution succeeds, but verification fails
        mock_run.return_value = True
        mock_verify.return_value = False

        cfg = {
            "runbook_map": {"HighLatency": "runbooks/restart_service.sh"},
            "runbook_registry": ["runbooks/restart_service.sh"],
            "rollback_map": {"HighLatency": "runbooks/restart_service.sh"},
            "runbook_timeout_seconds": 30,
            "prometheus_url": "http://localhost:9090",
            "blast_radius": {
                "max_actions_per_minute": 3,
                "max_restarts_per_service_per_hour": 5,
            },
        }
        baseline = {
            "verify_thresholds": {
                "verify_timeout_seconds": 10,
                "verify_poll_interval_seconds": 2,
                "verify_min_samples": 2,
            }
        }
        guard = MagicMock()
        cb = MagicMock()

        closed_loop._process_alert_locked(
            alert={"labels": {"alertname": "HighLatency", "service": "payment-svc"}},
            alertname="HighLatency",
            service="payment-svc",
            runbook="runbooks/restart_service.sh",
            cfg=cfg,
            baseline=baseline,
            guard=guard,
            cb=cb,
            global_dry_run=False,
        )

        # Should log ROLLBACK_TRIGGERED and execute rollback runbook
        rb_triggered = [
            c
            for c in mock_log.warning.call_args_list
            if c[0][0] == "ROLLBACK_TRIGGERED"
        ]
        self.assertEqual(len(rb_triggered), 1)
        cb.record_failure.assert_called_once()


if __name__ == "__main__":
    unittest.main()
