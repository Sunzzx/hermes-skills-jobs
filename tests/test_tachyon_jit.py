# Comprehensive Test Suite for Tachyon Register VM and Tracing JIT
import pytest, sys, types
from pathlib import Path
from importlib.machinery import SourceFileLoader

ROOT = Path(__file__).resolve().parent.parent
BIN_PATH = ROOT / 'bin' / 'tachyon-jit'
loader = SourceFileLoader('tachyon_jit_mod', str(BIN_PATH))
tachyon_mod = types.ModuleType(loader.name)
sys.modules[loader.name] = tachyon_mod
loader.exec_module(tachyon_mod)

OpCode = tachyon_mod.OpCode
Instruction = tachyon_mod.Instruction
FunctionBytecode = tachyon_mod.FunctionBytecode
Register = tachyon_mod.Register
reg = tachyon_mod.reg
VirtualMachine = tachyon_mod.VirtualMachine
VMStatus = tachyon_mod.VMStatus
ControlFlowGraph = tachyon_mod.ControlFlowGraph
TracingJITCompiler = tachyon_mod.TracingJITCompiler
assemble = tachyon_mod.assemble
disassemble = tachyon_mod.disassemble
optimize_bytecode = tachyon_mod.optimize_bytecode

"""Tests for Tachyon Assembler and Disassembler."""



def test_assembler_labels_and_comments():
    source = """
    # Setup code
    start: LOAD_CONST R0, 100 ; initialize R0
           LOAD_CONST R1, 200 # comment with hash
    step:  ADD R2, R0, R1
           JMP done
    done:  RET R2
    """
    fn = assemble(source, "test_fn")
    assert fn.name == "test_fn"
    assert "start" in fn.labels
    assert "step" in fn.labels
    assert "done" in fn.labels
    assert len(fn.instructions) == 5
    assert fn.instructions[0].opcode == OpCode.LOAD_CONST
    assert fn.instructions[3].opcode == OpCode.JMP
    assert fn.instructions[3].operands[0] == fn.labels["done"]


def test_disassemble_formatting():
    source = """
    LOAD_CONST R0, 42
    RET R0
    """
    fn = assemble(source, "demo")
    dis = disassemble(fn)
    assert ".function demo" in dis
    assert "LOAD_CONST" in dis
    assert "RET" in dis
    assert ".end" in dis

"""Tests for Control Flow Graph and Dominator Tree Analysis."""



def test_basic_block_splitting():
    asm = """
    LOAD_CONST R0, 10
    LOAD_CONST R1, 20
    CMP_LT R2, R0, R1
    JMP_IF R2, branch_true
    LOAD_CONST R3, 100
    JMP branch_done
branch_true:
    LOAD_CONST R3, 200
branch_done:
    RET R3
    """
    fn = assemble(asm, "branch_func")
    cfg = ControlFlowGraph(fn)

    assert len(cfg.blocks) >= 4
    assert cfg.entry_block_id == 0
    # Entry block must branch to branch_true and fallthrough block
    entry = cfg.blocks[0]
    assert len(entry.successors) == 2


def test_dominator_tree_properties():
    # Every block must dominate itself
    # Entry block must dominate all reachable blocks
    asm = """
start:
    LOAD_CONST R0, 0
loop:
    CMP_GE R1, R0, 10
    JMP_IF R1, exit
    ADD R0, R0, 1
    JMP loop
exit:
    RET R0
    """
    fn = assemble(asm, "loop_func")
    cfg = ControlFlowGraph(fn)

    for b_id, block in cfg.blocks.items():
        assert b_id in block.dominators
        assert 0 in block.dominators

    # Natural loop check
    assert len(cfg.loops) >= 1
    lp = cfg.loops[0]
    assert lp["header"] in cfg.blocks
    assert lp["latch"] in cfg.blocks


def test_cfg_render_ascii():
    asm = """
    LOAD_CONST R0, 42
    RET R0
    """
    fn = assemble(asm, "simple")
    cfg = ControlFlowGraph(fn)
    rendered = cfg.render_ascii()
    assert "Block B0" in rendered
    assert "Preds" in rendered
    assert "Succs" in rendered

"""Tests for Optimizer Passes in Tachyon."""



def test_constant_folding():
    asm = """
    LOAD_CONST R0, 10
    LOAD_CONST R1, 20
    ADD R2, R0, R1
    RET R2
    """
    fn = assemble(asm, "fold_test")
    opt_fn = optimize_bytecode(fn)

    # Verify that ADD was folded into LOAD_CONST 30
    has_add = any(i.opcode == OpCode.ADD for i in opt_fn.instructions)
    assert not has_add
    assert any(i.opcode == OpCode.LOAD_CONST and i.operands[1] == 30 for i in opt_fn.instructions)

    vm = VirtualMachine()
    vm.register_function(opt_fn)
    vm.call_function("fold_test", [])
    assert vm.run() == 30


def test_algebraic_identities():
    asm = """
    LOAD_CONST R0, 42
    ADD R1, R0, 0
    MUL R2, R1, 1
    SUB R3, R2, 0
    RET R3
    """
    fn = assemble(asm, "alg_test")
    opt_fn = optimize_bytecode(fn)

    # None of ADD, MUL, SUB with identity values should remain as ALU operations
    assert not any(i.opcode in (OpCode.ADD, OpCode.MUL, OpCode.SUB) for i in opt_fn.instructions)

    vm = VirtualMachine()
    vm.register_function(opt_fn)
    vm.call_function("alg_test", [])
    assert vm.run() == 42


def test_dead_code_elimination():
    asm = """
    LOAD_CONST R0, 100  ; Dead write
    LOAD_CONST R1, 500  ; Dead write
    LOAD_CONST R2, 99
    RET R2
    """
    fn = assemble(asm, "dce_test")
    opt_fn = optimize_bytecode(fn)

    # Instructions writing to dead R0, R1 should be turned to NOP
    active_loads = [i for i in opt_fn.instructions if i.opcode == OpCode.LOAD_CONST]
    assert len(active_loads) == 1
    assert active_loads[0].operands[1] == 99

"""Tests for Tracing JIT Compiler."""



def test_jit_loop_execution():
    # Simple loop summing numbers 0 to 99
    asm = """
    LOAD_CONST R0, 0    ; sum
    LOAD_CONST R1, 0    ; i
    LOAD_CONST R2, 100  ; limit

loop:
    LOOP_HEADER 10
    CMP_GE R3, R1, R2
    JMP_IF R3, done
    ADD R0, R0, R1
    ADD R1, R1, 1
    JMP loop

done:
    RET R0
    """
    fn = assemble(asm, "sum_loop")

    # Run interpreted
    vm_interp = VirtualMachine(enable_jit=False)
    vm_interp.register_function(fn)
    vm_interp.call_function("sum_loop", [])
    res_interp = vm_interp.run()
    assert res_interp == 4950

    # Run with JIT (low threshold)
    jit = TracingJITCompiler()
    vm_jit = VirtualMachine(enable_jit=True, jit_threshold=5)
    vm_jit.jit_compiler = jit
    vm_jit.register_function(fn)
    vm_jit.call_function("sum_loop", [])
    res_jit = vm_jit.run()

    assert res_jit == 4950
    assert jit.total_jit_executions >= 1


def test_jit_nested_operations():
    # Compute 2^10 using repeated multiplication
    asm = """
    LOAD_CONST R0, 1    ; acc
    LOAD_CONST R1, 0    ; i
    LOAD_CONST R2, 10   ; count

pow_loop:
    LOOP_HEADER 99
    CMP_GE R3, R1, R2
    JMP_IF R3, pow_done
    MUL R0, R0, 2
    ADD R1, R1, 1
    JMP pow_loop

pow_done:
    RET R0
    """
    fn = assemble(asm, "pow_test")
    jit = TracingJITCompiler()
    vm = VirtualMachine(enable_jit=True, jit_threshold=2)
    vm.jit_compiler = jit
    vm.register_function(fn)
    vm.call_function("pow_test", [])
    res = vm.run()
    assert res == 1024

"""Unit and integration tests for Tachyon Virtual Machine."""

import pytest


def test_arithmetic_operations():
    vm = VirtualMachine()
    fn = FunctionBytecode(name="math_test")
    fn.add_instruction(OpCode.LOAD_CONST, reg(0), 15)
    fn.add_instruction(OpCode.LOAD_CONST, reg(1), 4)
    fn.add_instruction(OpCode.ADD, reg(2), reg(0), reg(1))   # 19
    fn.add_instruction(OpCode.SUB, reg(3), reg(0), reg(1))   # 11
    fn.add_instruction(OpCode.MUL, reg(4), reg(0), reg(1))   # 60
    fn.add_instruction(OpCode.DIV, reg(5), reg(0), reg(1))   # 3.75
    fn.add_instruction(OpCode.MOD, reg(6), reg(0), reg(1))   # 3
    fn.add_instruction(OpCode.POW, reg(7), reg(1), 2)        # 16
    fn.add_instruction(OpCode.NEG, reg(8), reg(0))           # -15
    fn.add_instruction(OpCode.RET, reg(2))

    vm.register_function(fn)
    vm.call_function("math_test", [])
    res = vm.run()

    assert res == 19
    # Inspect registers
    frame = vm.call_stack[0] if vm.call_stack else None
    # Run completed, verify last return value
    assert vm.last_return_value == 19


def test_bitwise_operations():
    vm = VirtualMachine()
    fn = FunctionBytecode(name="bit_test")
    fn.add_instruction(OpCode.LOAD_CONST, reg(0), 0b1100)
    fn.add_instruction(OpCode.LOAD_CONST, reg(1), 0b1010)
    fn.add_instruction(OpCode.AND, reg(2), reg(0), reg(1))  # 0b1000 = 8
    fn.add_instruction(OpCode.OR, reg(3), reg(0), reg(1))   # 0b1110 = 14
    fn.add_instruction(OpCode.XOR, reg(4), reg(0), reg(1))  # 0b0110 = 6
    fn.add_instruction(OpCode.SHL, reg(5), reg(0), 2)       # 0b110000 = 48
    fn.add_instruction(OpCode.SHR, reg(6), reg(0), 1)       # 0b0110 = 6
    fn.add_instruction(OpCode.RET, reg(2))

    vm.register_function(fn)
    vm.call_function("bit_test", [])
    assert vm.run() == 8


def test_comparisons():
    asm = """
    LOAD_CONST R0, 10
    LOAD_CONST R1, 20
    CMP_LT R2, R0, R1   ; 1
    CMP_GT R3, R0, R1   ; 0
    CMP_EQ R4, R0, 10   ; 1
    CMP_NE R5, R1, 20   ; 0
    CMP_LE R6, R0, 10   ; 1
    CMP_GE R7, R1, 20   ; 1
    RET R2
    """
    fn = assemble(asm, "cmp_test")
    vm = VirtualMachine()
    vm.register_function(fn)
    vm.call_function("cmp_test", [])
    assert vm.run() == 1


def test_heap_memory():
    asm = """
    LOAD_CONST R0, 5    ; size 5
    ALLOC R1, R0        ; heap base ptr
    LOAD_CONST R2, 42   ; val
    STORE_MEM R1, 2, R2 ; heap[base + 2] = 42
    LOAD_MEM R3, R1, 2  ; load back
    RET R3
    """
    fn = assemble(asm, "heap_test")
    vm = VirtualMachine()
    vm.register_function(fn)
    vm.call_function("heap_test", [])
    assert vm.run() == 42
    assert vm.heap[2] == 42


def test_recursive_factorial():
    asm = """
    # R0 = n
    CMP_LE R1, R0, 1
    JMP_IF_NOT R1, recurse
    LOAD_CONST R2, 1
    RET R2

recurse:
    SUB R3, R0, 1
    CALL R4, fact, R3
    MUL R5, R0, R4
    RET R5
    """
    fn = assemble(asm, "fact")
    vm = VirtualMachine()
    vm.register_function(fn)
    vm.call_function("fact", [5])
    res = vm.run()
    assert res == 120


def test_division_by_zero_error():
    asm = """
    LOAD_CONST R0, 10
    LOAD_CONST R1, 0
    DIV R2, R0, R1
    RET R2
    """
    fn = assemble(asm, "div_zero")
    vm = VirtualMachine()
    vm.register_function(fn)
    vm.call_function("div_zero", [])
    with pytest.raises(ZeroDivisionError):
        vm.run()


def test_print_and_globals():
    asm = """
    LOAD_CONST R0, 999
    STORE_GLOBAL count, R0
    LOAD_GLOBAL R1, count
    PRINT R1
    RET R1
    """
    fn = assemble(asm, "globals_test")
    vm = VirtualMachine()
    vm.register_function(fn)
    vm.call_function("globals_test", [])
    assert vm.run() == 999
    assert vm.globals["count"] == 999
    assert vm.output_buffer == ["999"]
