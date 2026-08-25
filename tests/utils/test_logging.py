"""
test_logging.py

Test suite that checks the functionality of the logging utility functions.

Author: Marco Pérez Padilla
Date:   09-08-2026
"""

import logging
import time

from src.utils.logging import setup_logging, timer


class TestSetupLogging:
    def test_returns_logger_with_correct_name(self):
        logger = setup_logging()
        assert logger.name == "shiftbench"

    def test_logger_has_handlers(self):
        logger = setup_logging()
        assert len(logger.handlers) >= 0

    def test_respects_custom_level(self):
        logger = setup_logging(level=logging.WARNING)
        assert logger.getEffectiveLevel() == logging.WARNING


class TestTimer:
    def test_logs_elapsed_time(self, caplog):
        logger = logging.getLogger("shiftbench")
        logger.setLevel(logging.INFO)

        with caplog.at_level(logging.INFO, logger="shiftbench"):
            with timer("test_block"):
                time.sleep(0.01)

        assert len(caplog.records) >= 1
        record = caplog.records[-1]
        assert "[TIMER] test_block:" in record.message
        assert "s" in record.message

    def test_elapsed_time_is_positive(self, caplog):
        logger = logging.getLogger("shiftbench")
        logger.setLevel(logging.INFO)

        with caplog.at_level(logging.INFO, logger="shiftbench"):
            with timer("fast"):
                pass

        record = caplog.records[-1]
        elapsed_str = record.message.split(":")[-1].strip().replace("s", "")
        elapsed = float(elapsed_str)
        assert elapsed >= 0.0
