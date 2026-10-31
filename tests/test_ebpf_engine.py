import math
import os
import struct
import subprocess
import sys
import types
import unittest
from importlib.machinery import SourceFileLoader
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "bin" / "ebpf-engine"

loader = SourceFileLoader("ebpf_engine", str(CLI))
ebpf = types.ModuleType(loader.name)
loader.exec_module(ebpf)


class TestEbpfEngine(unittest.TestCase):
    def test_instruction_encoding_decoding(self):
        # Encode r1 = 42
        insn = ebpf.mov64_imm(ebpf.Reg.R1, 42)
        raw = insn.encode()
        self.assertEqual(len(raw), 8)

        decoded = ebpf.Insn.decode(raw)
        self.assertEqual(decoded.opcode, insn.opcode)
        self.assertEqual(decoded.dst, ebpf.Reg.R1)
        self.assertEqual(decoded.imm, 42)

        # Memory load instruction encoding
        ld = ebpf.ldx_mem(ebpf.BpfSize.DW, ebpf.Reg.R0, ebpf.Reg.R10, -16)
        raw_ld = ld.encode()
        dec_ld = ebpf.Insn.decode(raw_ld)
        self.assertEqual(dec_ld.dst, ebpf.Reg.R0)
        self.assertEqual(dec_ld.src, ebpf.Reg.R10)
        self.assertEqual(dec_ld.off, -16)

    def test_alu_64bit_operations(self):
        prog = [
            ebpf.mov64_imm(ebpf.Reg.R0, 100),
            ebpf.add64_imm(ebpf.Reg.R0, 50),     # R0 = 150
            ebpf.mov64_imm(ebpf.Reg.R1, 20),
            ebpf.sub64_reg(ebpf.Reg.R0, ebpf.Reg.R1), # R0 = 130
            ebpf.mov64_imm(ebpf.Reg.R2, 3),
            ebpf.mul64_reg(ebpf.Reg.R0, ebpf.Reg.R2), # R0 = 390
            ebpf.exit_insn(),
        ]
        vm = ebpf.BPFVirtualMachine(prog)
        res, _ = vm.run()
        self.assertEqual(res, 390)

    def test_stack_memory_and_atomic_xadd(self):
        prog = [
            ebpf.mov64_imm(ebpf.Reg.R1, 1000),
            ebpf.stx_mem(ebpf.BpfSize.DW, ebpf.Reg.R10, ebpf.Reg.R1, -16), # stack[-16] = 1000
            ebpf.mov64_imm(ebpf.Reg.R2, 250),
            ebpf.xadd64(ebpf.BpfSize.DW, ebpf.Reg.R10, ebpf.Reg.R2, -16),  # lock stack[-16] += 250
            ebpf.ldx_mem(ebpf.BpfSize.DW, ebpf.Reg.R0, ebpf.Reg.R10, -16), # R0 = 1250
            ebpf.exit_insn(),
        ]
        vm = ebpf.BPFVirtualMachine(prog)
        res, _ = vm.run()
        self.assertEqual(res, 1250)

    def test_bpf_maps(self):
        # Hash map
        hmap = ebpf.BPFHashMap(map_id=1, key_size=4, value_size=8, max_entries=10)
        key = struct.pack("<I", 1337)
        val = struct.pack("<Q", 987654321)
        self.assertEqual(hmap.update(key, val), 0)
        ret = hmap.lookup(key)
        self.assertIsNotNone(ret)
        self.assertEqual(struct.unpack("<Q", bytes(ret))[0], 987654321)
        self.assertEqual(hmap.delete(key), 0)
        self.assertIsNone(hmap.lookup(key))

        # Array map
        amap = ebpf.BPFArrayMap(map_id=2, value_size=4, max_entries=16)
        self.assertEqual(amap.update(5, struct.pack("<I", 42)), 0)
        arr_ret = amap.lookup(5)
        self.assertIsNotNone(arr_ret)
        self.assertEqual(struct.unpack("<I", bytes(arr_ret))[0], 42)

        # Ring buffer
        ring = ebpf.BPFRingBuffer(map_id=3, capacity=1024)
        self.assertEqual(ring.output(b"ALERT:SYN_FLOOD"), 0)
        events = ring.consume()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0], b"ALERT:SYN_FLOOD")
        self.assertEqual(len(ring.consume()), 0)

    def test_static_verifier_pass(self):
        firewall = ebpf.build_xdp_syn_firewall()
        verifier = ebpf.Verifier(firewall)
        safe, err, _ = verifier.verify()
        self.assertTrue(safe, f"Firewall should verify safe: {err}")

        fib = ebpf.build_fibonacci()
        safe_fib, err_fib, _ = ebpf.Verifier(fib).verify()
        self.assertTrue(safe_fib, f"Fibonacci should verify safe: {err_fib}")

    def test_static_verifier_rejections(self):
        # Division by zero
        div0 = ebpf.build_div_by_zero_prog()
        safe, err, _ = ebpf.Verifier(div0).verify()
        self.assertFalse(safe)
        self.assertIn("division/modulo by zero", err)

        # Out-of-bounds packet access
        oob = ebpf.build_oob_packet_prog()
        safe, err, _ = ebpf.Verifier(oob).verify()
        self.assertFalse(safe)
        self.assertIn("packet read offset", err)

        # Stack overflow
        stack = ebpf.build_stack_overflow_prog()
        safe, err, _ = ebpf.Verifier(stack).verify()
        self.assertFalse(safe)
        self.assertIn("stack write out of bounds", err)

        # Uninitialized register
        uninit = ebpf.build_uninit_reg_prog()
        safe, err, _ = ebpf.Verifier(uninit).verify()
        self.assertFalse(safe)
        self.assertIn("uninitialized register", err)

        # Infinite loop / dead code
        loop = ebpf.build_infinite_loop_prog()
        safe, err, _ = ebpf.Verifier(loop).verify()
        self.assertFalse(safe)

    def test_xdp_syn_firewall_verdicts(self):
        insns = ebpf.build_xdp_syn_firewall()
        vm = ebpf.BPFVirtualMachine(insns)

        # SYN attack packet (flags=0x02) -> DROP
        syn_pkt = ebpf.PacketBuilder.build_tcp_packet(flags=0x02)
        verdict, _ = vm.run(syn_pkt)
        self.assertEqual(verdict, int(ebpf.XdpAction.XDP_DROP))

        # ACK established packet (flags=0x10) -> PASS
        ack_pkt = ebpf.PacketBuilder.build_tcp_packet(flags=0x10)
        verdict, _ = vm.run(ack_pkt)
        self.assertEqual(verdict, int(ebpf.XdpAction.XDP_PASS))

        # UDP packet -> PASS
        udp_pkt = ebpf.PacketBuilder.build_udp_packet()
        verdict, _ = vm.run(udp_pkt)
        self.assertEqual(verdict, int(ebpf.XdpAction.XDP_PASS))

    def test_cli_subcommands(self):
        # info
        res = subprocess.run([sys.executable, str(CLI), "info"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        self.assertIn("Extended Berkeley Packet Filter", res.stdout)

        # disasm
        res = subprocess.run([sys.executable, str(CLI), "disasm", "firewall"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        self.assertIn("ASSEMBLY DISASSEMBLY", res.stdout)

        # verify
        res = subprocess.run([sys.executable, str(CLI), "verify", "firewall"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        self.assertIn("VERIFIER SUCCESS", res.stdout)

        # demo
        res = subprocess.run([sys.executable, str(CLI), "demo"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        self.assertIn("Production XDP SYN Firewall validated successfully", res.stdout)


if __name__ == '__main__':
    unittest.main()
