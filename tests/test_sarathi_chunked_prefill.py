"""Unit tests for sarathi-chunked-prefill: Chunked-Prefill & Piggyback Batching Engine (OSDI 2024)."""

import os
import sys
from importlib.machinery import SourceFileLoader

bin_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../bin/sarathi-chunked-prefill"))
sarathi_mod = SourceFileLoader("sarathi_chunked_prefill", bin_path).load_module()


def test_hardware_roofline_estimation():
    hw = sarathi_mod.HardwareProfile()
    # Prefill compute time should grow with tokens
    time_small = hw.estimate_iteration_time_ms(prefill_tokens=128, decode_tokens=4, total_kv_tokens=1024)
    time_large = hw.estimate_iteration_time_ms(prefill_tokens=1024, decode_tokens=4, total_kv_tokens=1024)
    assert time_large > time_small
    assert time_small > 0.0


def test_request_lifecycle():
    req = sarathi_mod.Request(request_id="r1", prompt_len=1024, output_len=10, arrival_time=0.0)
    assert not req.is_prefill_done
    assert not req.is_finished

    req.remaining_prompt = 0
    assert req.is_prefill_done
    assert not req.is_finished

    req.generated_tokens = 10
    assert req.is_finished


def test_scheduler_piggyback_budgeting():
    scheduler = sarathi_mod.SarathiScheduler(chunk_size=512, max_batch_tokens=1024)
    req = sarathi_mod.Request(request_id="r1", prompt_len=1024, output_len=5, arrival_time=0.0)
    scheduler.add_request(req)

    # First step should schedule chunk_size (512)
    prefill_slices, decode_batch, lat = scheduler.schedule_iteration(0.0)
    assert len(prefill_slices) == 1
    assert prefill_slices[0][1] == 512
    assert len(decode_batch) == 0


def test_simulation_run():
    res = sarathi_mod.simulate_serving(num_requests=5, chunk_size=256, max_batch_tokens=1024)
    assert res["num_requests"] == 5
    assert res["total_steps"] > 0
    assert res["median_itl_ms"] > 0
    assert res["p99_itl_ms"] >= res["median_itl_ms"]


def test_chunk_size_impact():
    # Chunked prefill should have much tighter ITL p99 than huge monolithic prefill
    res_chunked = sarathi_mod.simulate_serving(num_requests=10, chunk_size=256, max_batch_tokens=1024, seed=123)
    res_monolithic = sarathi_mod.simulate_serving(num_requests=10, chunk_size=4096, max_batch_tokens=4096, seed=123)
    assert res_chunked["p99_itl_ms"] < res_monolithic["p99_itl_ms"]
