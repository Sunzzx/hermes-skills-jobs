"""
Comprehensive Unit and Integration Test Suite for io_uring High-Performance Asynchronous I/O Engine.
"""

import subprocess
import sys
import unittest
from pathlib import Path
import types
from importlib.machinery import SourceFileLoader

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "bin" / "io-uring"
sys.path.insert(0, str(ROOT / "bin"))

loader = SourceFileLoader("io_uring", str(CLI))
mod = types.ModuleType(loader.name)
loader.exec_module(mod)

IOUring = mod.IOUring
SubmissionQueueEntry = mod.SubmissionQueueEntry
CompletionQueueEntry = mod.CompletionQueueEntry
IOUringBenchmark = mod.IOUringBenchmark
IORING_OP_NOP = mod.IORING_OP_NOP
IORING_OP_READ = mod.IORING_OP_READ
IORING_OP_WRITE = mod.IORING_OP_WRITE
IORING_OP_FSYNC = mod.IORING_OP_FSYNC
IORING_OP_READ_FIXED = mod.IORING_OP_READ_FIXED
IOSQE_IO_LINK = mod.IOSQE_IO_LINK
IORING_SETUP_SQPOLL = mod.IORING_SETUP_SQPOLL
ECANCELED = mod.ECANCELED


class TestIOUringCore(unittest.TestCase):
    """Unit tests for core ring buffer mechanisms and ABI compliance."""

    def test_ring_buffer_initialization(self):
        """Verifies power-of-two sizing, mask calculation, and initial pointer state."""
        ring = IOUring(entries=64, flags=0)
        self.assertEqual(ring.sq_ring.entries, 64)
        self.assertEqual(ring.sq_ring.ring_mask, 63)
        self.assertEqual(ring.sq_ring.head, 0)
        self.assertEqual(ring.sq_ring.tail, 0)

        # CQ size is doubled in io_uring architecture
        self.assertEqual(ring.cq_ring.entries, 128)
        self.assertEqual(ring.cq_ring.ring_mask, 127)
        self.assertEqual(ring.cq_ring.head, 0)
        self.assertEqual(ring.cq_ring.tail, 0)
        ring.close()

    def test_sqe_submission_and_cqe_harvest(self):
        """Tests standard SQE write, ring submission, and CQE harvesting."""
        ring = IOUring(entries=16, flags=0)
        sqe = ring.get_sqe()
        self.assertIsNotNone(sqe)
        sqe.opcode = IORING_OP_WRITE
        sqe.fd = 10
        sqe.data_buffer = b'HermesEngine'
        sqe.user_data = 777

        submitted = ring.submit()
        self.assertEqual(submitted, 1)

        completions = ring.wait_cqe_count(1)
        self.assertEqual(len(completions), 1)
        cqe = completions[0]
        self.assertEqual(cqe.user_data, 777)
        self.assertEqual(cqe.res, len(b'HermesEngine'))
        ring.close()

    def test_ring_wrap_around_lock_free_math(self):
        """Verifies seamless circular ring wrapping over multiple cycles."""
        ring = IOUring(entries=8, flags=0)
        num_cycles = 40
        for i in range(num_cycles):
            sqe = ring.get_sqe()
            sqe.opcode = IORING_OP_NOP
            sqe.user_data = i
            ring.submit()
            cqes = ring.wait_cqe_count(1)
            self.assertEqual(len(cqes), 1)
            self.assertEqual(cqes[0].user_data, i)
            self.assertEqual(cqes[0].res, 0)

        self.assertEqual(ring.sq_ring.head, num_cycles)
        self.assertEqual(ring.sq_ring.tail, num_cycles)
        self.assertEqual(ring.cq_ring.head, num_cycles)
        self.assertEqual(ring.cq_ring.tail, num_cycles)
        ring.close()

    def test_linked_pipeline_happy_path(self):
        """Tests IOSQE_IO_LINK chained pipeline execution."""
        ring = IOUring(entries=16, flags=0)
        sqe1 = ring.get_sqe()
        sqe1.opcode = IORING_OP_WRITE
        sqe1.fd = 1
        sqe1.data_buffer = b'Alpha'
        sqe1.flags = IOSQE_IO_LINK
        sqe1.user_data = 1

        sqe2 = ring.get_sqe()
        sqe2.opcode = IORING_OP_FSYNC
        sqe2.fd = 1
        sqe2.flags = IOSQE_IO_LINK
        sqe2.user_data = 2

        sqe3 = ring.get_sqe()
        sqe3.opcode = IORING_OP_READ
        sqe3.fd = 1
        sqe3.len = 5
        sqe3.flags = 0
        sqe3.user_data = 3

        ring.submit()
        cqes = ring.wait_cqe_count(3)
        self.assertEqual(len(cqes), 3)
        self.assertTrue(all(c.res >= 0 for c in cqes))
        ring.close()

    def test_linked_pipeline_error_cancellation(self):
        """Tests atomic dependency failure and downstream cancellation with -ECANCELED."""
        ring = IOUring(entries=16, flags=0)
        sqe1 = ring.get_sqe()
        sqe1.opcode = IORING_OP_READ_FIXED
        sqe1.fd = 999
        sqe1.buf_index = 88
        sqe1.flags = IOSQE_IO_LINK
        sqe1.user_data = 10

        sqe2 = ring.get_sqe()
        sqe2.opcode = IORING_OP_WRITE
        sqe2.fd = 1
        sqe2.flags = 0
        sqe2.user_data = 20

        ring.submit()
        cqes = ring.wait_cqe_count(2)
        self.assertEqual(len(cqes), 2)
        self.assertLess(cqes[0].res, 0)
        self.assertEqual(cqes[1].res, -ECANCELED)
        ring.close()

    def test_sqpoll_mode(self):
        """Verifies background kernel poller thread execution with zero manual syscalls."""
        ring = IOUring(entries=32, flags=IORING_SETUP_SQPOLL)
        self.assertTrue(bool(ring.flags & IORING_SETUP_SQPOLL))
        self.assertTrue(ring._sqpoll_running)
        for i in range(10):
            sqe = ring.get_sqe()
            sqe.opcode = IORING_OP_NOP
            sqe.user_data = 500 + i
            ring.sq_ring.tail += 1

        cqes = ring.wait_cqe_count(10, timeout=2.0)
        self.assertEqual(len(cqes), 10)
        self.assertGreater(ring.metrics["syscalls_avoided"], 0)
        ring.close()

    def test_benchmark_engine(self):
        """Verifies benchmark suite runs and returns metrics dictionary."""
        res = IOUringBenchmark.run_benchmark(num_ops=200, batch_size=32)
        self.assertIn('sync', res)
        self.assertIn('epoll', res)
        self.assertIn('io_uring_batched', res)
        self.assertIn('io_uring_sqpoll', res)
        self.assertEqual(res['io_uring_sqpoll']['syscalls'], 0)


class TestIOUringCLI(unittest.TestCase):
    """End-to-end integration tests for command-line tool."""

    def test_cli_info(self):
        proc = subprocess.run([str(CLI), 'info'], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0)
        self.assertIn('Dual Circular Ring Buffer', proc.stdout)

    def test_cli_ring(self):
        proc = subprocess.run([str(CLI), 'ring'], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0)
        self.assertIn('SUBMISSION QUEUE (SQ)', proc.stdout)
        self.assertIn('COMPLETION QUEUE (CQ)', proc.stdout)

    def test_cli_chain(self):
        proc = subprocess.run([str(CLI), 'chain'], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0)
        self.assertIn('Atomic dependency ordering verified', proc.stdout)

    def test_cli_benchmark(self):
        proc = subprocess.run([str(CLI), 'benchmark', '--ops', '200', '--batch', '16'], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0)
        self.assertIn('io_uring SQPOLL', proc.stdout)


if __name__ == '__main__':
    unittest.main()
