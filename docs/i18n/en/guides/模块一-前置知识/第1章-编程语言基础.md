---
title: "Chapter 1: Programming language foundations"
description: "The Python advanced topics, C/C++ core ideas, and Linux development environment you need for AI Infra, so later CUDA programming and systems work have a solid base"
tags: ["Python", "C++", "Linux", "Development environment", "Prerequisites"]
sourceHash: "7ed9625e6f7ddf16"
---

AI Infra sits at the boundary of algorithms, frameworks, compilers, operating systems, and hardware. The same problem often means jumping across languages and tools: Python for models and experiments, C++ for performance-sensitive paths, CUDA to drive the GPU, and Linux tooling to chase down process, memory, dynamic library, and networking issues.

This chapter does not try to teach every language's full syntax. It builds a usable spine: **read it, change it, debug it, and know where the performance cost comes from**.

<!-- more -->

## 📑 Table of contents

- [1. Learning goals and the development landscape](#1-learning-goals-and-the-development-landscape)
- [2. Python: from scripts to engineering](#2-python-from-scripts-to-engineering)
- [3. Python concurrency and profiling](#3-python-concurrency-and-profiling)
- [4. C/C++: memory, lifetime, and resource management](#4-cc-memory-lifetime-and-resource-management)
- [5. C++ compilation, linking, templates, and build systems](#5-c-compilation-linking-templates-and-build-systems)
- [6. Python and C++ interoperability](#6-python-and-c-interoperability)
- [7. Linux development basics](#7-linux-development-basics)
- [8. Environment isolation and the GPU software stack](#8-environment-isolation-and-the-gpu-software-stack)
- [9. Git collaboration and a method for locating problems](#9-git-collaboration-and-a-method-for-locating-problems)
- [10. Capstone: add a C++ operator to Python](#10-capstone-add-a-c-operator-to-python)
- [Summary](#summary)
- [Self-check list](#self-check-list)
- [Exercises](#exercises)
- [References](#references)

---

## 1. Learning goals and the development landscape

After this chapter, you should be able to:

1. Explain how Python names, objects, references, mutability, and scope relate;
2. Comfortably use iterators, generators, decorators, context managers, and type annotations;
3. Choose threads, processes, or async I/O based on the workload, and find Python hotspots with tools;
4. Read pointers, references, object lifetime, RAII, and smart pointers in C/C++;
5. Explain the full path of a C++ source file from preprocessing to dynamic loading;
6. Make sense of common templates, STL containers, and CMake projects;
7. Use `pybind11` to understand the basic boundary when Python calls a C++ extension;
8. Manage files, processes, permissions, environment variables, dynamic libraries, and remote jobs on Linux;
9. Tell apart the GPU driver, CUDA Toolkit, CUDA Runtime, and CUDA components shipped with a framework;
10. Locate, with reproducible steps, whether the bug is in the code, the environment, or performance.

### 1.1 What each of the three language layers owns

| Layer | Common tech | Strengths | Typical work |
|------|----------|------|----------|
| Orchestration | Python, Shell | Fast to write, rich ecosystem | Model definition, training scripts, job orchestration, data processing |
| System | C/C++ | Controllable memory and performance; easy to wire to low-level APIs | Framework core, operator scheduling, communication libraries, Python extensions |
| Device | CUDA C++, Triton | Explicit GPU parallelism | Kernels, operator fusion, memory-access optimization |

The three layers do not replace each other. A single `torch.matmul(x, w)` looks like one line of Python, but it may go through:

```text
Python API
  └─ Operator dispatcher (C++)
      └─ cuBLAS / custom CUDA Kernel
          └─ GPU instructions and device-memory access
```

So the language skill that matters most for an AI Infra engineer is not "memorize syntax," but **tracing a problem across layers along the call chain**.

### 1.2 A minimal toolbox before you start

You do not need to master every command below right now, but you should know which question each one answers:

```bash
python --version       # which Python is in use
which python           # where this executable comes from
python -m pip --version

g++ --version          # C++ compiler version
cmake --version        # build-system version

uname -a               # kernel and system info
ps aux                 # which processes are running
free -h                # host memory usage
df -h                   # filesystem capacity

nvidia-smi             # GPUs and processes visible to the driver
nvcc --version         # compiler version from the CUDA Toolkit
```

> **Build a diagnosis habit first**: when an environment problem shows up, record the command, output, working directory, and versions before you reinstall over and over. The more complete the observables, the faster you can locate the issue.

---

## 2. Python: from scripts to engineering

### 2.1 Names, objects, and references

A more accurate way to talk about Python variables is "**a name bound to an object**." Assignment usually does not copy the object; it creates a new reference:

```python
a = [1, 2]
b = a
b.append(3)

print(a)        # [1, 2, 3]
print(a is b)   # True: same object
print(a == b)   # True: equal values
```

Distinguish two operations:

- `is` compares object identity; use it for `x is None`;
- `==` runs value comparison and asks whether the "contents" are equal.

Function parameters are name bindings too. The function receives an object reference, not either of the classic labels "pass by reference" or "pass by value":

```python
def mutate(xs: list[int]) -> None:
    xs.append(1)          # mutates the same list the caller sees

def rebind(xs: list[int]) -> None:
    xs = [1,2]              # only rebinds the local name xs to a new list

data: list[int] = []
mutate(data)
print(data)               # [1]

rebind(data)
print(data)               # still [1]
```

This model explains a lot of common bugs: mutable default arguments, shallow copies, cache pollution, and shared state across threads.

### 2.2 Mutable objects, immutable objects, and copying

Common immutable objects include `int`, `float`, `bool`, `str`, `bytes`, and a `tuple` whose elements are all immutable; common mutable objects include `list`, `dict`, `set`, and most custom instances.

#### Default arguments are created once

```python
# Wrong: the same list is shared across calls
def collect_bad(x: int, result: list[int] = []) -> list[int]:
    result.append(x)
    return result

# Right: use None to mean "not provided"
def collect(x: int, result: list[int] | None = None) -> list[int]:
    if result is None:
        result = []
    result.append(x)
    return result
```

Default arguments are evaluated when the function is defined, not on each call.

#### Shallow copy vs deep copy

```python
import copy

src = [[1], [2]]
shallow = src.copy()           # copies only the outermost list
deep = copy.deepcopy(src)      # recursively copies nested objects

shallow[0].append(9)
print(src)                     # [[1, 9], [2]]
print(deep)                    # [[1], [2]]
```

Deep copy is not the default answer. Unintentional copies of large tensors or model state can cost huge memory and time. A better first step is to decide which data must be exclusive and which can be shared read-only.

### 2.3 Scope, closures, and LEGB

Python looks up names in **LEGB** order: Local → Enclosing → Global (module) → Builtins.

A closure keeps variables from the enclosing scope:

```python
from collections.abc import Callable

def make_multiplier(scale: float) -> Callable[[float], float]:
    def multiply(x: float) -> float:
        return x * scale
    return multiply

double = make_multiplier(2.0)
print(double(3.0))  # 6.0
```

When you create closures in a loop, watch for **late binding**: the closure reads the enclosing name at call time.

```python
# All three functions end up reading the same i, so all return 2
bad = [lambda: i for i in range(3)]

# Capture the current value with a default argument at each definition
good = [lambda i=i: i for i in range(3)]
print([fn() for fn in good])  # [0, 1, 2]
```

If an inner function needs to rebind a name from the enclosing function, use `nonlocal`; to rebind a module-global name, use `global`. Prefer passing state through parameters and return values; that is usually easier to test.

### 2.4 Classes, the data model, and protocols

Python special methods hook objects into language protocols. For example, implementing `__len__` lets you call `len(obj)`, and implementing `__iter__` lets a `for` loop iterate over it.

```python
from dataclasses import dataclass
from collections.abc import Iterator

@dataclass
class Batch:
    samples: list[list[float]]

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[list[float]]:
        return iter(self.samples)

batch = Batch([[1.0, 2.0], [3.0, 4.0]])
print(len(batch))
for sample in batch:
    print(sample)
```

Common special methods and what they are for:

| Method | Triggered by | Typical use |
|------|----------|----------|
| `__init__` | After instance creation | Initialize instance state |
| `__repr__` | `repr(obj)`, debuggers | Unambiguous debug representation |
| `__call__` | `obj(...)` | Make an instance behave like a function |
| `__enter__` / `__exit__` | `with` | Manage resource lifetime |
| `__iter__` / `__next__` | `for`, `next` | Define the iteration protocol |
| `__getitem__` | `obj[key]` | Indexing, slicing, or mapping access |

#### Instance methods, class methods, and static methods

- An instance method's first parameter is `self` and operates on a concrete object;
- `@classmethod`'s first parameter is `cls`, often used for alternate constructors;
- `@staticmethod` receives no implicit object; it mainly groups related helpers in the class namespace.

```python
@dataclass
class Device:
    kind: str
    index: int

    @classmethod
    def parse(cls, value: str) -> "Device":
        kind, index = value.split(":")
        return cls(kind=kind, index=int(index))

device = Device.parse("cuda:0")
```

### 2.5 Iterators and generators: produce data on demand

An **iterable** can return an iterator; an **iterator** holds traversal state and returns elements one by one via `__next__`. Generator functions are a concise way to write iterators:

```python
from collections.abc import Iterator

def read_batches(path: str, batch_size: int) -> Iterator[list[str]]:
    batch: list[str] = []
    with open(path, encoding="utf-8") as file:
        for line in file:
            batch.append(line.rstrip("\n"))
            if len(batch) == batch_size:
                yield batch
                batch = []
    if batch:
        yield batch
```

`yield` returns a value and pauses the function; the next iteration resumes from that pause. Compared with building a full list up front, a generator's peak memory usually tracks one batch, which fits streaming data pipelines.

Remember:

- A generator is usually consumable only once;
- Lazy evaluation defers exceptions until iteration time;
- Turning a generator into a `list` materializes every element again and loses the memory win.

### 2.6 Decorators: attach behavior at the call boundary

A decorator takes a callable and returns another callable. The timer below keeps the original function's metadata:

```python
from collections.abc import Callable
from functools import wraps
from time import perf_counter
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")

def timed(fn: Callable[P, R]) -> Callable[P, R]:
    @wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        start = perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            elapsed_ms = (perf_counter() - start) * 1_000
            print(f"{fn.__name__}: {elapsed_ms:.3f} ms")
    return wrapper

@timed
def preprocess(values: list[float]) -> list[float]:
    return [value * 2 for value in values]
```

`@timed` is equivalent to `preprocess = timed(preprocess)`. `functools.wraps` keeps the function name, docstring, and similar attributes, which matters for logging, tests, and framework reflection.

Decorators fit logging, caching, permission checks, and performance observation. If only one call site needs timing, a plain `with timer():` is often clearer; you do not need a decorator just because it looks advanced.

### 2.7 Context managers: make resource release certain

`with` puts acquire and release in the same lexical scope, so cleanup still runs if an exception is raised:

```python
from contextlib import contextmanager
from collections.abc import Iterator
from time import perf_counter

@contextmanager
def timer(name: str) -> Iterator[None]:
    start = perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (perf_counter() - start) * 1_000
        print(f"{name}: {elapsed_ms:.3f} ms")

with timer("load data"):
    data = [value for value in range(100_000)]
```

Files, locks, database connections, temporary directories, and automatic mixed-precision contexts all fit this pattern. It solves the same problem as RAII in C++: **resource lifetime needs a clear boundary**.

### 2.8 Exception handling: catch only what you can handle

```python
def parse_world_size(raw: str) -> int:
    try:
        world_size = int(raw)
    except ValueError as exc:
        raise ValueError(f"WORLD_SIZE must be an integer, got {raw!r}") from exc

    if world_size <= 0:
        raise ValueError("WORLD_SIZE must be greater than 0")
    return world_size
```

Follow three rules:

1. Catch specific exceptions; do not use a bare `except:` that also swallows `KeyboardInterrupt` and similar signals;
2. Catch only when you can recover, add context, or translate across an abstraction boundary;
3. Use `raise ... from exc` to keep the root-cause chain.

Do not use exceptions for normal high-frequency control flow, especially in performance-sensitive loops.

### 2.9 Type annotations and dataclasses

Type annotations mainly serve static checking, IDEs, and readers; they do not automatically validate types at runtime:

```python
from dataclasses import dataclass
from collections.abc import Sequence

@dataclass(frozen=True)
class ShardSpec:
    rank: int
    world_size: int

def shard(values: Sequence[int], spec: ShardSpec) -> Sequence[int]:
    return values[spec.rank::spec.world_size]
```

Common choices:

- When a parameter only needs "iterable" or "sequence" capability, annotate abstract protocols such as `Iterable[T]` or `Sequence[T]`;
- When returning a concrete container, you can annotate `list[T]` or `dict[K, V]`;
- `T | None` means the value may be missing;
- `Protocol` can describe a structural interface and cut unnecessary inheritance;
- `dataclass` fits objects that mostly carry data with simple rules.

The type system's goal is to surface interface misuse early, not to force a dynamic language into Java. Prefer clear annotations at public boundaries; local variables that are obvious do not need every name annotated.

---

## 3. Python concurrency and profiling

### 3.1 Separate concurrency from parallelism first

- **Concurrency**: multiple tasks make progress over the same time span;
- **Parallelism**: multiple tasks truly execute at the same moment.

Before choosing a concurrency model, decide where the bottleneck is:

| Workload type | Typical examples | Common choice |
|----------|----------|----------|
| I/O bound | Calling services, reading many small files | Threads or `asyncio` |
| Python CPU bound | Pure-Python parsing, complex loops | Multiprocessing or a native extension |
| Native-operator bound | NumPy / PyTorch / CUDA work | Underlying thread pools / GPU parallelism; Python schedules |
| Large data across processes | Data loading, shared caches | Multiprocessing + shared memory; serialize carefully |

### 3.2 What the GIL actually limits

Common CPython builds use a Global Interpreter Lock (GIL) to protect interpreter-internal state. In one process, usually only one thread can execute Python bytecode at a time. So two threads running pure-Python CPU loops generally will not get 2× speed.

That does not mean "Python threads are useless":

- The interpreter releases the GIL during blocking I/O;
- Native extensions such as NumPy and PyTorch can release the GIL during heavy computation;
- Background logging, prefetch, and light control tasks often fit threads well.

The GIL is a CPython implementation detail; do not generalize it into an eternal rule for every Python implementation or every build. When judging performance, measure the interpreter and dependencies you actually run, instead of only memorizing a conclusion.

### 3.3 Threads: shared memory, shared risk

```python
from concurrent.futures import ThreadPoolExecutor
from urllib.request import urlopen

def fetch_size(url: str) -> int:
    with urlopen(url, timeout=10) as response:
        return len(response.read())

urls = ["https://example.com"] * 4
with ThreadPoolExecutor(max_workers=4) as pool:
    sizes = list(pool.map(fetch_size, urls))
```

Threads share process memory, so passing data is cheap, but a read-modify-write on the same mutable object can race. When you need a lock, keep the critical section as small as possible:

```python
from threading import Lock

lock = Lock()
counter = 0

def increment() -> None:
    global counter
    with lock:
        counter += 1
```

Do not design correctness around "this builtin looks atomic"; interpreter version, object implementation, or a compound operation can all break that assumption.

### 3.4 Multiprocessing: bypass the GIL, pay for data movement

```python
from concurrent.futures import ProcessPoolExecutor

def count_primes(limit: int) -> int:
    def is_prime(value: int) -> bool:
        if value < 2:
            return False
        return all(value % divisor for divisor in range(2, int(value**0.5) + 1))
    return sum(is_prime(value) for value in range(limit))

if __name__ == "__main__":
    with ProcessPoolExecutor() as pool:
        results = list(pool.map(count_primes, [50_000] * 4))
```

`if __name__ == "__main__":` matters especially on platforms that start child processes with `spawn`; it avoids recursively creating new processes when a child re-imports the module.

The main costs of multiprocessing include:

- Creating processes and separate interpreters;
- Serializing and deserializing arguments and return values;
- Copying or mapping memory;
- Inter-process communication and synchronization.

If each task does only tens of microseconds of work, scheduling cost can exceed compute. Increase task granularity, and avoid shuttling large arrays or tensors back and forth.

> **GPU note**: a bad mix of CUDA contexts and how multiprocessing is started can cause init errors or extra device-memory use. When you use a framework's data loading and distributed features, follow its recommended launcher and start method; do not casually `fork` after the parent has already initialized CUDA.

### 3.5 `asyncio`: cooperative I/O concurrency

`asyncio` schedules coroutines with an event loop. A coroutine yields control only at `await`:

```python
import asyncio

async def worker(name: str, delay: float) -> str:
    await asyncio.sleep(delay)  # simulate non-blocking I/O
    return name

async def main() -> None:
    results = await asyncio.gather(
        worker("a", 0.2),
        worker("b", 0.1),
    )
    print(results)

asyncio.run(main())
```

Running blocking I/O or long CPU work directly on the event loop stalls every coroutine. Switch to async libraries, or move the blocking work to a thread/process executor.

### 3.6 Benchmark correctly

The most common performance-test mistakes are measuring once, counting initialization, or forgetting that GPU execution is asynchronous.

For small pure-CPU snippets, `timeit` works:

```bash
python -m timeit -s 'xs = list(range(1000))' 'sum(xs)'
```

A hand-rolled test should at least do this:

```python
from statistics import median
from time import perf_counter

def benchmark(fn, warmup: int = 5, repeat: int = 20) -> float:
    for _ in range(warmup):
        fn()

    samples = []
    for _ in range(repeat):
        start = perf_counter()
        fn()
        samples.append(perf_counter() - start)
    return median(samples)
```

On GPU, kernel launches are usually asynchronous. A CPU timer often only measures "submit the work"; you need a sync at the boundary, or GPU Events from the framework. Sync changes pipeline behavior, so measure end-to-end throughput and single-operator latency separately.

### 3.7 From "it's slow" to a specific line of code

Work coarse to fine:

1. **End-to-end timing**: confirm the regression is real;
2. **Stage timing**: separate data, compute, communication, and saving;
3. **Function-level profiling**: use `cProfile` to find functions with high cumulative time;
4. **Line-level / sampling profiling**: narrow to hot lines or native stacks;
5. **Memory profiling**: use tools like `tracemalloc` to find Python allocation growth;
6. **GPU profiling**: move to dedicated tools such as PyTorch Profiler and Nsight Systems/Compute.

```bash
python -m cProfile -o profile.out train.py
python -m pstats profile.out
```

In `pstats`, sort by cumulative time:

```text
sort cumulative
stats 30
```

Check Python memory allocations:

```python
import tracemalloc

tracemalloc.start()
# run the code under inspection
snapshot = tracemalloc.take_snapshot()
for stat in snapshot.statistics("lineno")[:10]:
    print(stat)
```

### 3.8 Priority order for Python performance work

Recommended order:

1. Fix algorithmic complexity and data structures first;
2. Cut unnecessary I/O, copies, and serialization;
3. Turn fine-grained Python loops into batched / vectorized native operators;
4. Reduce frequent round-trips across the Python and C++/GPU boundary;
5. Only after you confirm the hotspot, write a C++, CUDA, or Triton extension.

For example, a million tiny operator calls can pile up scheduling overhead even when each call is fast. Batching the data and calling one large operator once usually beats micro-tuning Python syntax.

---

## 4. C/C++: memory, lifetime, and resource management

### 4.1 Values, addresses, pointers, and references

```cpp
#include <iostream>

int main() {
    int value = 42;
    int* ptr = &value;  // ptr holds the address of value
    int& ref = value;   // ref is an alias for value

    *ptr = 7;           // dereference the pointer and write
    ref += 1;

    std::cout << value << '\n';  // 8
}
```

Use this table to build intuition first:

| Concept | Can be null? | Can retarget? | Common uses |
|------|----------|------------|----------|
| Value `T` | No | N/A | Own an independent object |
| Pointer `T*` | Can be `nullptr` | Yes | Optional objects, arrays, low-level APIs, non-owning observation |
| Reference `T&` | Must bind an object in language semantics | Cannot rebind | Required alias, function parameters |
| Const reference `const T&` | Must bind an object | No | Avoid copies and disallow mutation |

Function parameters should make interface intent as clear as possible:

```cpp
void consume(T value);          // by value: the function gets its own object
void update(T& value);          // mutable borrow: the function will modify the caller's object
void inspect(const T& value);   // read-only borrow: avoid copying
void optional(T* value);        // pointer may be null; usually does not own the object
```

For small scalars (`int`, `float`, pointers), passing by value is usually clearest; do not mechanically turn every parameter into a reference.

### 4.2 Arrays, pointers, and contiguous memory

In most expressions a native array decays to a pointer to its first element, so a bare `T*` does not carry a length:

```cpp
void scale(float* data, std::size_t size, float factor) {
    for (std::size_t i = 0; i < size; ++i) {
        data[i] *= factor;
    }
}
```

In modern C++, prefer types that express length:

```cpp
#include <span>

void scale(std::span<float> values, float factor) {
    for (float& value : values) {
        value *= factor;
    }
}
```

`std::span` does not own the data; it is a lightweight "pointer + length" view. The underlying memory must stay valid for the duration of the call.

2D matrices are usually stored linearly. For a row-major matrix `A[M][N]`, the offset of `(row, col)` is:

$$
\text{offset} = row \times N + col
$$

You will see this formula again and again in CUDA. Whether access is contiguous directly affects CPU cache behavior and GPU coalesced memory access.

### 4.3 Object storage duration and lifetime

People often say "stack" and "heap," but what matters more accurately is **an object's storage duration and ownership**:

```cpp
void example() {
    int local = 1;                  // automatic storage duration; destroyed when leaving scope
    auto buffer = new float[1024];  // dynamic allocation; must delete[] explicitly
    delete[] buffer;
}
```

Hand-written `new`/`delete` easily produces:

- Memory leaks: forgetting to free;
- Double free: freeing the same address twice;
- Dangling pointers: the object is gone, the pointer is still used;
- Mismatched allocation/deallocation forms: `new[]` paired with `delete`;
- Leaks on exception paths: the function exits before the free.

So modern C++'s default rule is: **let objects manage resources, and let scopes manage objects.**

### 4.4 RAII: bind resources to object lifetime

RAII (Resource Acquisition Is Initialization) acquires a resource in the constructor and releases it in the destructor:

```cpp
#include <cstdio>
#include <stdexcept>

class File {
public:
    explicit File(const char* path)
        : handle_(std::fopen(path, "rb")) {
        if (handle_ == nullptr) {
            throw std::runtime_error("failed to open file");
        }
    }

    ~File() {
        std::fclose(handle_);
    }

    File(const File&) = delete;
    File& operator=(const File&) = delete;

private:
    std::FILE* handle_;
};
```

Whether the function returns normally or throws, local objects' destructors still run. Files, locks, host memory, GPU memory, and CUDA streams can all be managed with the same idea.

In real projects, prefer the standard library's existing RAII types such as `std::vector`, `std::string`, `std::fstream`, `std::lock_guard`, and smart pointers; do not reinvent them.

### 4.5 Smart pointers and ownership

```cpp
#include <memory>

auto unique = std::make_unique<Buffer>(1024);  // unique ownership
auto shared = std::make_shared<Buffer>(1024);  // shared ownership
```

| Type | Semantics | Cost and caveats |
|------|------|--------------|
| `std::unique_ptr<T>` | Unique ownership; movable, not copyable | Cost near a raw pointer; default first choice |
| `std::shared_ptr<T>` | Reference-counted shared ownership | Count and control-block cost; cycles need `weak_ptr` |
| `std::weak_ptr<T>` | Observer that does not bump the count | Call `lock()` before use to check the object still exists |
| `T*` / `T&` | Usually non-owning access | The pointed-to object must live long enough |

If ownership does not need to be shared, do not use `shared_ptr`. "Avoid thinking about ownership" is not a good reason for shared ownership.

### 4.6 Copy, move, and the Rule of Zero

Containers such as `std::vector` own resources; copy duplicates contents, while move transfers the internal resources:

```cpp
#include <utility>
#include <vector>

std::vector<float> make_buffer() {
    std::vector<float> values(1'000'000, 0.0f);
    return values;  // the compiler usually elides the copy; otherwise it can move
}

auto a = make_buffer();
auto b = std::move(a);  // a remains valid, but its contents are in an unspecified state
```

If a class is composed only of standard-library RAII members, you usually do not need to hand-write destructor, copy, or move operations; that is the **Rule of Zero**. Only when you directly manage a special resource should you customize these operations and define ownership clearly.

Do not casually write `std::move` on return values; it can inhibit return-value optimization. After a move, do not assume the source still holds its original contents.

### 4.7 `const`: put immutability into the interface

```cpp
const float* p1;        // points to const float: cannot change the value through p1
float* const p2 = ptr;  // const pointer: cannot retarget p2
const float* const p3 = ptr;
```

A trailing `const` on a member function means it does not modify the object's observable state:

```cpp
class TensorView {
public:
    std::size_t size() const { return size_; }
private:
    std::size_t size_{};
};
```

`const` helps the compiler and the reader check the interface; it does not automatically solve concurrency safety or low-level aliasing issues.

### 4.8 Why undefined behavior is dangerous

Out-of-bounds access, use of dangling references, signed integer overflow, data races, and similar issues can trigger undefined behavior (UB). Compilers may assume UB never happens and optimize accordingly; the result is not always "an error." A debug build can look fine while an optimized build silently computes the wrong answer.

```cpp
std::vector<int> values(4);
values[4] = 1;  // out of bounds; undefined behavior
```

When debugging, enable warnings and sanitizers:

```bash
g++ -std=c++20 -Wall -Wextra -Wpedantic \
    -fsanitize=address,undefined -g main.cpp -o app
./app
```

AddressSanitizer is often used to find out-of-bounds access and use-after-free; UndefinedBehaviorSanitizer checks many kinds of UB; ThreadSanitizer can locate data races, but usually cannot be enabled together with AddressSanitizer.

---

## 5. C++ compilation, linking, templates, and build systems

### 5.1 From source to executable

```text
Source file .cpp
  └─ Preprocess: expand #include, macros, and conditional compilation
      └─ Compile: syntax/type checks and generate assembly
          └─ Assemble: produce object file .o
              └─ Link: resolve symbols and produce an executable or library
```

You can watch the stages:

```bash
g++ -E main.cpp -o main.ii       # preprocess only
g++ -S main.ii -o main.s         # generate assembly
g++ -c main.cpp -o main.o        # generate object file
g++ main.o -o app                # link
```

Headers usually hold declarations; source files hold definitions:

```cpp
// add.h
#pragma once
int add(int lhs, int rhs);
```

```cpp
// add.cpp
#include "add.h"
int add(int lhs, int rhs) {
    return lhs + rhs;
}
```

### 5.2 Compile errors, link errors, and runtime load errors

Do not mix the three kinds of failure:

| Stage | Typical message | Common causes |
|------|----------|----------|
| Compile | `not declared`, type mismatch | Missing declaration, template instantiation failure, syntax error |
| Link | `undefined reference` | Declared but not defined, missing library link, ABI/signature mismatch |
| Runtime load | `cannot open shared object file` | Dynamic library not on the search path, architecture mismatch |

Useful commands for symbols and dependencies:

```bash
nm -C build/app | less          # inspect and demangle C++ symbols
ldd build/app                   # Linux: dynamic library dependencies
readelf -d build/app            # inspect the ELF dynamic section
```

On macOS the corresponding tool is usually `otool -L`.

### 5.3 Static libraries and dynamic libraries

- A static library (commonly `.a` on Linux) folds needed code into the target at link time; deployment is direct, but the artifact may be larger;
- A dynamic library (Linux `.so`, macOS `.dylib`) resolves at load/run time; multiple programs can share it, but you must handle search paths and ABI compatibility.

Linux dynamic-library search is affected by the binary's RPATH/RUNPATH, `LD_LIBRARY_PATH`, the system cache, and default directories. Temporarily setting `LD_LIBRARY_PATH` is fine for diagnosis; it should not become a long-term way to hide a broken install layout.

### 5.4 ABI: source compatibility is not binary compatibility

ABI (Application Binary Interface) covers calling conventions, name mangling, object layout, and how standard-library types are implemented at the binary level. Any of the following can break ABI compatibility:

- Inconsistent compilers or standard libraries;
- Compile options that change object layout or the C++ ABI;
- Exported symbols changing after a library upgrade;
- Mixing Debug/Release runtimes;
- Different CPU architectures.

When a Python C++ extension "compiles but fails to import," check the extension and its dependency libraries for ABI, architecture, and symbols, not only the Python path.

### 5.5 Templates: generate type-specialized code at compile time

```cpp
template <typename T>
T add(T lhs, T rhs) {
    return lhs + rhs;
}

auto i = add(1, 2);        // instantiate add<int>
auto f = add(1.0f, 2.0f);  // instantiate add<float>
```

Template definitions usually live in headers, because the compiler needs the full definition at the use site to instantiate. CUDA and high-performance code use templates heavily to express:

- Data types: `float`, `half`, `bfloat16`;
- Tile sizes and thread-block configurations;
- Layout and transpose options;
- Compile-time branches and specialized implementations.

Non-type template parameters can turn sizes into compile-time constants:

```cpp
template <typename T, std::size_t N>
T sum(const std::array<T, N>& values) {
    T result{};
    for (const T& value : values) {
        result += value;
    }
    return result;
}
```

Template errors are often very long. Start from the earliest "user-code instantiation site" and the deepest real constraint failure; do not get lost in dozens of intermediate expansions.

### 5.6 STL containers and iterators

| Type | Memory / access traits | Common uses |
|------|---------------|----------|
| `std::vector<T>` | Contiguous memory, random access O(1) | Numeric buffers; default sequence container |
| `std::array<T, N>` | Fixed size, contiguous memory | Small arrays known at compile time |
| `std::deque<T>` | Segmented storage; efficient inserts at both ends | Queues, sliding windows |
| `std::unordered_map<K,V>` | Hash table; average O(1) lookup | Unordered key-value index |
| `std::map<K,V>` | Ordered tree; O(log n) | Ordered traversal / range queries |

Growing a `std::vector` may reallocate and invalidate existing pointers, references, and iterators:

```cpp
std::vector<int> values;
values.reserve(1024);  // when the size is known, reduce repeated growth
```

In performance-sensitive code, look beyond big-O at contiguity, allocation count, cache locality, and element size. In many cases a linear scan of a contiguous `vector` beats a structure with better theoretical lookup complexity but frequent pointer chasing.

### 5.7 A minimal CMake project

Layout:

```text
vector_ops/
├── CMakeLists.txt
├── include/vector_ops.h
└── src/vector_ops.cpp
```

`CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.20)
project(vector_ops LANGUAGES CXX)

add_library(vector_ops src/vector_ops.cpp)
target_include_directories(vector_ops PUBLIC include)
target_compile_features(vector_ops PUBLIC cxx_std_20)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(vector_ops PRIVATE -Wall -Wextra -Wpedantic)
endif()
```

Build:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

Modern CMake centers on **targets**. Describe dependencies with `target_include_directories`, `target_link_libraries`, and `target_compile_features`; avoid global `include_directories` and hand-stitched compile commands.

### 5.8 Minimal awareness of C++ concurrency

If multiple threads concurrently access the same memory, at least one writes, and there is no synchronization, you may have a data race, which is undefined behavior.

```cpp
#include <mutex>

std::mutex mutex;
int counter = 0;

void increment() {
    std::lock_guard<std::mutex> guard(mutex);
    ++counter;
}
```

Mutexes enforce mutual exclusion in a critical section; atomics fit simple shared state; condition variables wait for state changes. Do not treat `volatile` as a thread-synchronization tool; it does not provide atomicity or inter-thread happens-before.

Before high-performance concurrency optimization, get correctness first, and use tools such as ThreadSanitizer to check for data races.

---

## 6. Python and C++ interoperability

### 6.1 Why cross the language boundary

Common reasons include:

- Reusing existing C/C++ libraries;
- Pushing hot Python loops down into native code;
- Wiring drivers, communication libraries, or custom hardware interfaces;
- Adding custom operators to frameworks such as PyTorch.

Crossing languages does not automatically make things faster. One call does argument checking, type/layout conversion, possible data copies, and exception translation. If you cross the boundary once per element, binding overhead can erase the native compute win. A good interface usually passes a batch or a contiguous buffer.

### 6.2 `ctypes`: call a stable C ABI

Export a C interface from C++ first:

```cpp
// scale.cpp
extern "C" void scale(float* data, int size, float factor) {
    for (int i = 0; i < size; ++i) {
        data[i] *= factor;
    }
}
```

Build the dynamic library:

```bash
g++ -O3 -shared -fPIC scale.cpp -o libscale.so
```

Call it from Python:

```python
import ctypes
from array import array

library = ctypes.CDLL("./libscale.so")
library.scale.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_float,
]
library.scale.restype = None

values = array("f", [1.0, 2.0, 3.0])
pointer = (ctypes.c_float * len(values)).from_buffer(values)
library.scale(pointer, len(values), 2.0)
print(values)
```

`ctypes` is simple and direct, but types, lengths, and lifetimes are mostly the caller's responsibility. A wrong signature can crash, not only raise an ordinary Python exception.

### 6.3 `pybind11`: bind C++ interfaces naturally

`pybind11` can map C++ functions, classes, and exceptions to Python objects:

```cpp
#include <pybind11/pybind11.h>

int add(int lhs, int rhs) {
    return lhs + rhs;
}

PYBIND11_MODULE(vector_ops, module) {
    module.doc() = "minimal vector ops extension";
    module.def("add", &add, pybind11::arg("lhs"), pybind11::arg("rhs"));
}
```

When binding large arrays, double-check:

- Whether dtypes match;
- Whether shape and stride match the algorithm's assumptions;
- Whether the data is contiguous;
- Whether memory lives on CPU or GPU;
- Who owns the underlying storage, and whether it remains valid after return;
- Whether the GIL should be released during compute;
- How native exceptions become Python exceptions.

### 6.4 Zero-copy is not "unconditional"

Zero-copy usually means sharing the same underlying memory. Both sides must agree on this contract:

```text
address + element type + shape + stride + device + lifetime + mutability
```

Any mismatch can cause an implicit copy, wrong results, or out-of-bounds access. For example, a transposed NumPy array may not be C-contiguous; taking only the base address and reading it as a contiguous array yields wrong data.

> **Core principle**: treat data layout and ownership as part of the interface, not as an implementation detail.

---

## 7. Linux development basics

### 7.1 Files, directories, and text streams

```bash
pwd                         # current directory
ls -lah                     # detailed listing including hidden files
mkdir -p runs/exp01/logs     # create directories recursively
cp -a configs runs/exp01/   # copy while preserving attributes
mv old_name new_name
rm -i file                  # interactive delete; confirm the path first for dangerous commands

find runs -type f -name '*.log'
rg 'CUDA error|out of memory' runs
```

Shell composition power comes from stdin, stdout, and stderr:

```bash
python train.py >train.log 2>train.err       # redirect separately
python train.py >train.all.log 2>&1           # merge streams
rg 'loss=' train.log | tail -n 20
```

A pipe connects the left command's stdout to the right command's stdin. Each stage should do one clear job; complex flows that need maintenance should become scripts under version control.

### 7.2 Quotes, variables, and exit codes

```bash
name='experiment 01'
printf '%s\n' "$name"   # double quotes keep it as one argument
printf '%s\n' '$name'   # single quotes do not expand variables
```

Unquoted variables can undergo word splitting and glob expansion. In shell scripts you should almost always write `"$variable"`.

Every command has an exit code: `0` means success; non-zero means failure.

```bash
python check_data.py
status=$?
printf 'exit code: %s\n' "$status"
```

Robust Bash scripts often start with:

```bash
#!/usr/bin/env bash
set -euo pipefail
```

- `-e`: an unhandled failure exits the script;
- `-u`: using an undefined variable is an error;
- `pipefail`: if any command in a pipeline fails, the pipeline fails.

These options do not replace clear error handling, especially the exception semantics inside conditionals and pipelines.

### 7.3 The permission model

```bash
ls -l script.sh
# -rwxr-x---  owner group ... script.sh

chmod u+x script.sh          # add execute permission for the owner
chmod 640 config.yaml        # owner: rw, group: r, other: none
```

For files, `r/w/x` mean read, write, and execute; for directories they mean list names, modify directory entries, and traverse. For permission problems, start with `ls -l`, `id`, and parent-directory permissions; do not habitually `chmod 777`.

### 7.4 Processes, jobs, and signals

```bash
ps -ef | rg train.py
pgrep -af train.py
top                         # or htop

python train.py &           # run in the background
jobs
fg %1                       # bring back to the foreground
```

Send signals:

```bash
kill -TERM <pid>            # ask the process to exit gracefully
kill -INT <pid>             # similar to Ctrl-C
kill -KILL <pid>            # kernel terminates immediately; no cleanup; last resort
```

Prefer `SIGTERM` so the program can save a checkpoint, release shared resources, and flush logs. `SIGKILL` cannot be caught or cleaned up after.

For long remote jobs, use `tmux` or a cluster scheduler instead of relying only on an SSH terminal:

```bash
tmux new -s train
# Ctrl-b d: detach but keep the session
tmux attach -t train
```

### 7.5 Inspect CPU, memory, disk, and network

| Question | Common commands |
|------|----------|
| CPU / load | `top`, `uptime`, `mpstat` |
| Process memory | `ps`, `top`, `pmap -x PID` |
| System memory | `free -h`, `vmstat` |
| Disk capacity | `df -h` |
| Directory usage | `du -sh PATH`, `du -h --max-depth=1` |
| File I/O | `iostat`, `iotop` |
| Listening ports | `ss -lntp` |
| Connectivity | `curl -v URL`, `nc -vz HOST PORT` |

"Out of memory" can mean host RAM, GPU device memory, shared memory `/dev/shm`, a process address-space limit, or a full disk. Confirm which resource before you act.

### 7.6 Environment variables and dynamic library paths

```bash
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

env | sort
which nvcc
type -a python
```

Environment variables are inherited from parent to child. Editing `~/.bashrc` does not magically change terminals, IDEs, or services that are already running; reload the config or restart the relevant process.

`env -i` can reproduce a problem in a nearly empty environment, but you must supply necessary variables explicitly:

```bash
env -i HOME="$HOME" PATH=/usr/bin:/bin bash --noprofile --norc
```

### 7.7 SSH and file transfer

Generate a key and set up login:

```bash
ssh-keygen -t ed25519 -C "your-name"
ssh-copy-id user@server
ssh user@server
```

`~/.ssh/config` can store aliases:

```text
Host gpu-dev
    HostName 192.0.2.10
    User lxy
    IdentityFile ~/.ssh/id_ed25519
    ServerAliveInterval 60
```

Then just `ssh gpu-dev`. For syncing large directories, prefer `rsync` with incremental transfer:

```bash
rsync -av --progress ./project/ gpu-dev:~/project/
```

Before syncing, carefully check whether the source path ends with `/`; that decides whether you copy "the directory itself" or "the directory's contents." Do not put sensitive data or private keys in the repo, and do not casually copy them onto shared machines.

### 7.8 Debugging native programs

Compile with debug info:

```bash
g++ -g -O0 main.cpp -o app
gdb ./app
```

Basic GDB operations:

```text
break main       # set a breakpoint
run              # start
next / step      # single-step; whether to enter functions
print variable   # inspect a variable
bt               # backtrace
continue         # continue running
```

Crashes in optimized builds should still keep symbols, for example `-O2 -g`, so a core dump has a readable stack. A debugger answers "where it crashed"; sanitizers are better at answering "which illegal memory operation caused it."

For system-call-level issues, watch with `strace` (Linux):

```bash
strace -f -o trace.log ./app
rg 'ENOENT|EACCES' trace.log
```

This is very effective for locating "which config file / dynamic library the program is actually looking for."

---

## 8. Environment isolation and the GPU software stack

### 8.1 `venv`, conda, and containers solve different layers

| Tool | Isolation scope | Good fit |
|------|----------|----------|
| `venv` | Python packages and interpreter environment | Pure-Python projects, simple dependencies |
| conda | Python + some native libraries and toolchains | Scientific computing, complex binary dependencies |
| Docker / containers | User-space filesystem, dependencies, and process environment | Deployment, CI, cross-machine reproducibility |

`venv` example:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Always use `python -m pip` to reduce mix-ups like "running Python from environment A while calling pip from environment B."

conda example:

```bash
conda create -n ai-infra python=3.12
conda activate ai-infra
conda env export --from-history > environment.yml
```

Pinning direct dependency versions helps reproducibility, but the full environment also includes the OS, driver, compiler, hardware architecture, and launch arguments. Do not treat one `requirements.txt` as the whole environment description.

### 8.2 A container is not a VM

Containers share the host kernel and only isolate user-space filesystem, processes, networking, and similar resources. GPU containers usually still use the **host NVIDIA driver**, and the NVIDIA Container Toolkit exposes devices and driver libraries into the container.

```bash
docker run --rm -it \
  --gpus all \
  -v "$PWD:/workspace" \
  -w /workspace \
  nvidia/cuda:12.4.1-runtime-ubuntu22.04 \
  nvidia-smi
```

Pin image tags to a version specific enough. Volume mounts let processes inside the container modify host directories directly; understand mounts and user permissions before running unfamiliar commands.

### 8.3 The GPU software stack: four easy-to-confuse pieces

```text
Application / PyTorch / custom extensions
        │
CUDA Runtime and math libraries (cudart, cuBLAS, cuDNN, ...)
        │
User-space driver library (libcuda)
        │
Kernel-space NVIDIA Driver
        │
GPU
```

- **NVIDIA Driver**: the foundation for OS ↔ GPU communication; `nvidia-smi` mainly reflects driver state;
- **CUDA Toolkit**: a developer toolkit that includes `nvcc`, headers, debugging/profiling tools, and libraries;
- **CUDA Runtime / math libraries**: user-space components loaded when applications run;
- **cuDNN**: a separate acceleration library aimed at deep-learning operators; it is not all of CUDA.

The "CUDA Version" shown by `nvidia-smi` usually means the highest CUDA compatibility level the driver supports; it does not prove that the same version of `nvcc` is installed. Check the compile toolchain with `nvcc --version`, and check which runtime the framework actually uses via the framework's own info.

### 8.4 A fixed order for diagnosing the GPU environment

```bash
# 1. Does the driver see the device?
nvidia-smi

# 2. Where does the compile toolchain come from?
which nvcc
nvcc --version

# 3. Where do Python and the framework come from?
which python
python -c 'import sys; print(sys.executable)'

# 4. What does the framework see?
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("built with CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
PY
```

What common symptoms usually mean:

| Symptom | Check first |
|------|----------|
| `nvidia-smi` fails | Driver, device mount, host state |
| `nvidia-smi` works but `nvcc` is missing | Toolkit not installed or `PATH` wrong; a prebuilt framework may not need `nvcc` at runtime |
| Framework imports but CUDA is unavailable | CPU build installed, driver compatibility, container without GPU passthrough, devices masked |
| Custom extension fails to compile | Compiler, Toolkit, headers, ABI, target GPU architecture |
| Runtime missing a `.so` | Dynamic library search path, wheel/conda dependencies, architecture mismatch |

Change only one variable at a time, and keep the original error. Most environment problems can be located along this chain without "delete everything and reinstall."

---

## 9. Git collaboration and a method for locating problems

### 9.1 A clean contribution flow

```bash
git clone https://github.com/owner/project.git
cd project
git switch -c docs/improve-foundations

# edit and verify
git status --short
git diff --check
git diff

git add path/to/changed-file
git commit -m "docs: expand programming foundations"
```

A commit should cover one clear topic. Do not mix whole-repo formatting, editor config, and the change you care about.

### 9.2 Read the working-tree state

Git distinguishes three layers:

```text
commit in HEAD  ← git commit ← staging area ← git add ← working tree
```

```bash
git diff                 # working tree vs staging area
git diff --staged        # staging area vs HEAD
git status --short
git log --oneline --decorate -10
```

Before committing, look at both `git diff` and `git diff --staged` so you do not miss files or drag unrelated ones in.

### 9.3 Merge, rebase, and conflicts

- `merge` keeps both branch histories and creates a merge commit;
- `rebase` replays this branch's commits onto a new base; history stays linear, but commit IDs change.

You can rebase a personal feature branch that has not been shared; do not casually rewrite history on public branches. Resolving a conflict is not mechanically deleting markers; understand both sides' intent, keep the correct combination, and re-run tests.

### 9.4 A reusable localization framework

When something breaks, record in this order:

1. **Expected behavior**: what should happen;
2. **Actual behavior**: full error, exit code, stderr;
3. **Minimal reproduction**: shortest command, smallest input;
4. **Environment**: code commit, dependency versions, OS, hardware;
5. **Boundary localization**: which layer among data, Python, C++, CUDA, driver, network;
6. **Single-variable experiments**: change one thing at a time and record the result.

"Sometimes OOM" is not an actionable problem statement; "commit abc123, batch size 8, sequence length 4096, CUDA OOM at step 23 while allocating 1.2 GiB; batch size 4 does not reproduce" is.

---

## 10. Capstone: add a C++ operator to Python

Below we implement element-wise scaling with `pybind11` and walk the full path of source, build, import, and verification. The example only shows the cross-language chain; real numeric work should prefer mature vectorized libraries.

### 10.1 Project layout

```text
fast_scale/
├── CMakeLists.txt
├── scale.cpp
└── test_scale.py
```

### 10.2 C++ implementation and binding

```cpp
// scale.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <vector>

std::vector<float> scale(const std::vector<float>& input, float factor) {
    std::vector<float> output;
    output.reserve(input.size());
    for (float value : input) {
        output.push_back(value * factor);
    }
    return output;
}

PYBIND11_MODULE(fast_scale, module) {
    module.def("scale", &scale,
               pybind11::arg("input"),
               pybind11::arg("factor"));
}
```

Using `const std::vector<float>&` avoids copying the input again inside the function, but converting a Python list into a `std::vector` still does a boundary conversion; the return value is also converted into a Python list. This is not zero-copy.

### 10.3 CMake build

```cmake
cmake_minimum_required(VERSION 3.20)
project(fast_scale LANGUAGES CXX)

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(fast_scale scale.cpp)
target_compile_features(fast_scale PRIVATE cxx_std_17)
```

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install pybind11

cmake -S . -B build \
  -Dpybind11_DIR="$(python -m pybind11 --cmakedir)" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

### 10.4 Verify correctness

```python
# test_scale.py
import sys

sys.path.insert(0, "build")
import fast_scale

actual = fast_scale.scale([1.0, -2.0, 3.5], 2.0)
expected = [2.0, -4.0, 7.0]
assert actual == expected
print(actual)
```

```bash
python test_scale.py
```

### 10.5 What a teaching example still lacks for a high-performance interface

This example has no advantage over a Python list comprehension, because the boundary does element-wise conversion. A real high-performance extension also needs to:

1. Accept contiguous NumPy/PyTorch buffers instead of Python lists;
2. Validate dtype, shape, stride, and device;
3. Avoid unnecessary copies and make ownership explicit;
4. Release the GIL during heavy compute;
5. Provide batched CPU-vectorized or CUDA implementations;
6. Compare against a trusted reference and test edge inputs;
7. Measure boundary conversion and core compute separately.

Those are exactly the problems later chapters on PyTorch extensions, CUDA kernels, and operator optimization keep solving.

---

## Summary

The core of this chapter is not three isolated syntaxes, but a cross-layer way of thinking:

- In Python, see object references, data copies, concurrency models, and the interpreter/native boundary clearly;
- In C++, see memory layout, lifetime, ownership, compile/link, and ABI clearly;
- On Linux, see processes, resources, paths, permissions, dynamic libraries, and environment inheritance clearly;
- For performance, measure before optimizing; for environment issues, verify layer by layer along the software stack;
- A good interface describes not only "what values are passed," but also type, shape, layout, device, ownership, and lifetime.

When you later learn CUDA, PyTorch, distributed training, and inference frameworks, you will keep meeting these ideas; they just show up as GPU memory, streams, tensors, communication processes, and model-serving forms.

## Self-check list

- [ ] Can explain the difference between `is` and `==`, and between shallow and deep copy
- [ ] Can explain why mutable default arguments are shared across calls
- [ ] Can write a decorator that preserves function metadata and a context manager
- [ ] Can choose threads, processes, or async I/O for I/O-bound vs CPU-bound work
- [ ] Knows why asynchronous GPU execution can distort ordinary CPU timing
- [ ] Can distinguish the interface semantics of C++ pointers, references, values, and `const` references
- [ ] Can explain what RAII, `unique_ptr`, and `shared_ptr` each solve
- [ ] Can explain the preprocess, compile, assemble, link, and dynamic-load stages
- [ ] Can build a minimal C++ library with CMake
- [ ] Can explain why shape, stride, dtype, device, and lifetime belong in a cross-language interface
- [ ] Can inspect processes, resource use, ports, and dynamic-library dependencies on Linux
- [ ] Can distinguish NVIDIA Driver, CUDA Toolkit, CUDA Runtime, and cuDNN
- [ ] Can create a small, clear Git branch and commit, and check the diff before committing
- [ ] Can locate problems with "minimal reproduction + environment snapshot + single-variable experiments"

## Exercises

1. Write code that reproduces the shallow-copy problem of sharing an inner list, then fix it with data restructuring and with deep copy.
2. Implement a parameterized `@retry(max_attempts=3)` decorator that only catches exception types specified by the caller.
3. Run I/O-wait tasks and pure-Python CPU tasks with a thread pool and a process pool; record and explain the timing differences.
4. Profile a script with `cProfile`, find the three functions with the highest cumulative time; optimize one and remeasure.
5. Write a C++ RAII class that holds a file handle, disables copying, and supports moving.
6. Deliberately cause a heap out-of-bounds access; compare ordinary runs with AddressSanitizer error reports.
7. Split a C++ library into header, source, and executable; build it with target-based CMake.
8. Use `ldd`/`otool -L` and `nm -C` to analyze a Python native extension's dependencies and exported symbols.
9. Install PyTorch in a clean virtual environment; record `sys.executable`, framework version, build CUDA version, and device visibility.
10. Extend the capstone: accept a NumPy array, check dtype and contiguity, and compare timings of a copying version vs a shared-buffer version.

## References

- [Python Tutorial](https://docs.python.org/3/tutorial/)
- [Python Data Model](https://docs.python.org/3/reference/datamodel.html)
- [Python `concurrent.futures`](https://docs.python.org/3/library/concurrent.futures.html)
- [Python `asyncio`](https://docs.python.org/3/library/asyncio.html)
- [Python Profilers](https://docs.python.org/3/library/profile.html)
- [C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines)
- [CMake Tutorial](https://cmake.org/cmake/help/latest/guide/tutorial/)
- [pybind11 documentation](https://pybind11.readthedocs.io/)
- [GNU GDB documentation](https://sourceware.org/gdb/documentation/)
- [Git documentation](https://git-scm.com/doc)
- [Docker documentation](https://docs.docker.com/)
- [NVIDIA CUDA Installation Guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
