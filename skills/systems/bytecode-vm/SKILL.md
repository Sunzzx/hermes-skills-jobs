---
name: bytecode-vm
description: "Hermes Stack-Based Virtual Machine (HVM) & Bytecode Compiler: lexical closures with escaping upvalues, recursive call frames, structured exception unwinding, and mark-and-sweep GC."
version: 1.0.0
category: systems
tags:
  - SystemsArchitecture
  - VirtualMachines
  - BytecodeCompiler
  - Closures
  - GarbageCollection
  - ComputerScience
---

# ⚡ Hermes Bytecode Virtual Machine & Compiler (HVM)

A production-grade, mathematically rigorous implementation of a **stack-based Virtual Machine and Bytecode Execution Engine** built from scratch with zero external dependencies.

Features an expressive Instruction Set Architecture (ISA), lexical closures with open/closed upvalue indirection, recursive activation records (call frames), structured exception handling with stack unwinding, tracing Mark-and-Sweep garbage collection with cycle reclamation, and an interactive ANSI step debugger.

---

## 📐 Theoretical Architecture & Formal Semantics

### 1. Abstract Machine State Model

The HVM state configuration is formally defined as a 6-tuple:
$$\Sigma = \langle \mathcal{S}, \mathcal{F}, \mathcal{G}, \mathcal{U}, \mathcal{H}, \text{IP} \rangle$$

Where:
- $\mathcal{S} \in \text{Val}^*$: The evaluation value stack of dynamically typed operand values.
- $\mathcal{F} \in \text{Frame}^*$: The call frame activation stack, where each frame $\phi = \langle \text{closure}, \text{ip}, \text{base\_ptr}, \text{handlers} \rangle$.
- $\mathcal{G}: \text{Ident} \to \text{Val}$: The global environment variable table.
- $\mathcal{U}$: Active open upvalue registry tracking escaping local variables.
- $\mathcal{H}$: Traced heap managed by mark-and-sweep garbage collector.
- $\text{IP} \in \mathbb{N}$: Instruction pointer into the current chunk's bytecode sequence.

---

### 2. Lexical Closures & Upvalue Lifting (Lua-Style Semantics)

When an inner function captures local variable $x$ located at stack slot $i$ in an enclosing frame:
1. If an open upvalue already references stack slot $i$, it is shared ($O(1)$ lookup).
2. Otherwise, an `ObjUpvalue` is instantiated pointing to memory slot $i$.
3. When the enclosing activation record terminates (`RET` / stack unwinding), `CLOSE_UPVALUE` migrates the stack value into the heap-allocated upvalue node:
   $$\text{Upvalue}(i) \longrightarrow \text{Upvalue}(\text{closed}\!=\!\mathcal{S}[i])$$
4. Subsequent reads and writes via `LOAD_UPVALUE` / `STORE_UPVALUE` transparently operate on the closed heap object, permitting safe escaping closures.

---

### 3. Structured Exception Unwinding

Exception frames are registered using `TRY_PUSH(catch_ip)` which records $\langle \text{catch\_ip}, |\mathcal{S}| \rangle$:
$$\frac{\text{THROW}(v) \quad \text{handler} = \langle \text{catch\_ip}, d \rangle}{\mathcal{S}' = \mathcal{S}[0..d] \circ [v], \quad \text{IP}' = \text{catch\_ip}}$$
All intermediate call frames between the throw site and the catch handler have their open upvalues closed gracefully before stack frames are popped.

---

### 4. Mark-and-Sweep Garbage Collection

Dynamic heap allocations (`ObjFunction`, `ObjClosure`, `ObjUpvalue`, `ObjList`, `ObjDict`) are tracked via a singly linked list `head`.

1. **Mark Phase**: Traverses root set:
   $$\mathcal{R} = \mathcal{S} \cup \{ \phi.\text{closure} \mid \phi \in \mathcal{F} \} \cup \text{values}(\mathcal{G}) \cup \mathcal{U}$$
   Recursively traverses all accessible pointers, marking reachable graph nodes.
2. **Sweep Phase**: Linearly scans allocated heap nodes. Unmarked nodes are detached from the intrusive list and reclaimed, breaking circular reference cycles with zero memory leaks.

---

## 🛠️ CLI Quickstart

The HVM engine is installed globally as `bytecode-vm`:

### 1. Run Complete Invariant Verification Suite
```bash
bytecode-vm test
```
Validates 7 formal test suites:
- **Arithmetic & Bitwise**: Basic operations, modular arithmetic, exponentiation, bit shifts.
- **Stack Invariants**: DUP, SWAP, ROT operand permutations.
- **Recursive CallFrames**: Deep Fibonacci activation stack frames.
- **Escaping Closures**: Higher-order counter generator with mutating upvalues.
- **Exception Unwinding**: Multi-frame stack restoration on error throw.
- **Dynamic Collections**: Heterogeneous lists, dictionaries, key hashing.
- **Garbage Collection**: 100% reclamation of detached cyclic object graphs.

### 2. Run High-Velocity Throughput Benchmark
```bash
bytecode-vm bench
```
Executes 700,000+ bytecode instructions in tight loops, measuring instruction throughput (> 2,000,000 instructions/sec) and stack depth metrics.

### 3. Run Interactive Demos
```bash
# Run all interactive demos with ANSI stack visualization
bytecode-vm demo all

# Run specific demos
bytecode-vm demo fib
bytecode-vm demo closure
bytecode-vm demo try
bytecode-vm demo gc
```

### 4. Disassemble Bytecode Chunks
```bash
bytecode-vm disasm fib
bytecode-vm disasm closure
```

---

## 🔬 Instruction Set Reference (Summary)

| OpCode | Hex | Operand | Description |
| :--- | :--- | :--- | :--- |
| `PUSH_CONST` | 0x01 | `const_idx` | Push constant value onto stack |
| `DUP` / `SWAP` / `ROT` | 0x03-05 | None | Stack slot rearrangement primitives |
| `ADD` / `SUB` / `MUL` / `DIV` | 0x10-13 | None | Binary arithmetic on top two operands |
| `BIT_AND` / `SHL` / `SHR` | 0x17, 1B, 1C | None | Bitwise manipulation |
| `JUMP_IF_FALSE` | 0x32 | `target_ip` | Conditional branching |
| `LOAD_LOCAL` / `STORE_LOCAL` | 0x40-41 | `slot_idx` | Fast frame-relative stack access |
| `LOAD_UPVALUE` / `STORE_UPVALUE` | 0x44-45 | `upval_idx` | Lexically scoped closure access |
| `MAKE_CLOSURE` | 0x50 | `fn_idx` | Allocate closure and capture upvalues |
| `CALL` / `RET` | 0x51-52 | `argc` | Push / pop activation call frames |
| `TRY_PUSH` / `THROW` | 0x70, 72 | `catch_ip` | Structured exception handling |
| `GC_COLLECT` | 0x90 | None | Trigger manual mark-and-sweep sweep |
