---
title: "Chương 1: Nền tảng ngôn ngữ lập trình"
description: "Python nâng cao, C/C++ cốt lõi và môi trường Linux cần cho AI Infra, để lập trình CUDA và phát triển hệ thống sau này có nền vững"
tags: ["Python", "C++", "Linux", "Môi trường phát triển", "Kiến thức nền"]
sourceHash: "7ed9625e6f7ddf16"
---

AI Infra nằm ở giao điểm của thuật toán, framework, trình biên dịch, hệ điều hành và phần cứng. Cùng một vấn đề thường phải nhảy giữa nhiều ngôn ngữ và công cụ: Python để mô tả mô hình và thí nghiệm, C++ cho đường đi nhạy hiệu năng, CUDA để điều khiển GPU, rồi công cụ Linux để lần theo tiến trình, bộ nhớ, thư viện động và mạng.

Chương này không cố dạy hết cú pháp từng ngôn ngữ. Nó dựng một trục đủ dùng: **đọc được, sửa được, gỡ lỗi được, và biết chi phí hiệu năng nằm ở đâu**.

<!-- more -->

## 📑 Mục lục

- [1. Mục tiêu học và toàn cảnh phát triển](#1-mục-tiêu-học-và-toàn-cảnh-phát-triển)
- [2. Python: từ script đến kỹ thuật phần mềm](#2-python-từ-script-đến-kỹ-thuật-phần-mềm)
- [3. Đồng thời Python và phân tích hiệu năng](#3-đồng-thời-python-và-phân-tích-hiệu-năng)
- [4. C/C++: bộ nhớ, vòng đời và quản lý tài nguyên](#4-cc-bộ-nhớ-vòng-đời-và-quản-lý-tài-nguyên)
- [5. Biên dịch C++, liên kết, template và hệ thống build](#5-biên-dịch-c-liên-kết-template-và-hệ-thống-build)
- [6. Tương tác Python và C++](#6-tương-tác-python-và-c)
- [7. Kỹ năng Linux cơ bản](#7-kỹ-năng-linux-cơ-bản)
- [8. Cô lập môi trường và phần mềm GPU](#8-cô-lập-môi-trường-và-phần-mềm-gpu)
- [9. Cộng tác Git và cách định vị vấn đề](#9-cộng-tác-git-và-cách-định-vị-vấn-đề)
- [10. Thực chiến: thêm một toán tử C++ vào Python](#10-thực-chiến-thêm-một-toán-tử-c-vào-python)
- [Tóm tắt](#tóm-tắt)
- [Checklist tự kiểm](#checklist-tự-kiểm)
- [Bài tập](#bài-tập)
- [Tài liệu tham khảo](#tài-liệu-tham-khảo)

---

## 1. Mục tiêu học và toàn cảnh phát triển

Sau chương này, bạn nên:

1. Giải thích quan hệ giữa tên, đối tượng, tham chiếu, tính biến đổi và phạm vi trong Python;
2. Thành thạo iterator, generator, decorator, context manager và chú thích kiểu;
3. Chọn thread, process hoặc I/O bất đồng bộ theo bản chất tác vụ, và dùng công cụ tìm hotspot Python;
4. Đọc được pointer, reference, vòng đời đối tượng, RAII và smart pointer trong C/C++;
5. Giải thích toàn bộ đường đi của một file nguồn C++ từ tiền xử lý đến nạp động;
6. Đọc được template thường gặp, container STL và dự án CMake;
7. Dùng `pybind11` để hiểu biên giới cơ bản khi Python gọi extension C++;
8. Trên Linux, quản lý file, tiến trình, quyền, biến môi trường, thư viện động và tác vụ từ xa;
9. Phân biệt GPU driver, CUDA Toolkit, CUDA Runtime và thành phần CUDA đi kèm framework;
10. Dùng bước tái lập được để định vị lỗi thuộc code, môi trường hay hiệu năng.

### 1.1 Ba lớp ngôn ngữ mỗi lớp làm gì

| Lớp | Công nghệ thường dùng | Điểm mạnh | Công việc điển hình |
|------|----------|------|----------|
| Orchestration | Python, Shell | Viết nhanh, hệ sinh thái phong phú | Định nghĩa mô hình, script huấn luyện, điều phối tác vụ, xử lý dữ liệu |
| System | C/C++ | Bộ nhớ và hiệu năng kiểm soát được; dễ nối API tầng thấp | Lõi framework, lập lịch toán tử, thư viện giao tiếp, extension Python |
| Device | CUDA C++, Triton | Biểu đạt tường minh song song GPU | Kernel, fusion toán tử, tối ưu truy cập bộ nhớ |

Ba lớp không thay thế nhau. Một dòng `torch.matmul(x, w)` trông như chỉ là Python, nhưng có thể đi qua:

```text
Python API
  └─ Bộ phân phối toán tử (C++)
      └─ cuBLAS / CUDA Kernel tùy chỉnh
          └─ Lệnh GPU và truy cập bộ nhớ thiết bị
```

Vì vậy, năng lực ngôn ngữ quan trọng nhất với kỹ sư AI Infra không phải "học thuộc cú pháp", mà là **lần theo vấn đề xuyên lớp dọc chuỗi gọi**.

### 1.2 Hộp công cụ tối thiểu trước khi bắt đầu

Không cần nắm hết các lệnh dưới đây ngay, nhưng nên biết mỗi lệnh trả lời câu hỏi gì:

```bash
python --version       # đang dùng Python nào
which python           # executable này đến từ đâu
python -m pip --version

g++ --version          # phiên bản trình biên dịch C++
cmake --version        # phiên bản hệ thống build

uname -a               # thông tin kernel và hệ thống
ps aux                 # có những tiến trình nào
free -h                # bộ nhớ máy chủ
df -h                   # dung lượng hệ thống file

nvidia-smi             # GPU và tiến trình mà driver thấy
nvcc --version         # phiên bản trình biên dịch trong CUDA Toolkit
```

> **Xây thói quen chẩn đoán trước**: khi gặp lỗi môi trường, hãy ghi lại lệnh, output, thư mục làm việc và phiên bản trước khi cài lại liên tục. Quan sát càng đủ, định vị càng nhanh.

---

## 2. Python: từ script đến kỹ thuật phần mềm

### 2.1 Tên, đối tượng và tham chiếu

Cách nói chính xác hơn về biến Python là "**một tên được gắn (bind) với một đối tượng**". Gán thường không sao chép đối tượng; nó tạo một tham chiếu mới:

```python
a = [1, 2]
b = a
b.append(3)

print(a)        # [1, 2, 3]
print(a is b)   # True: cùng một đối tượng
print(a == b)   # True: giá trị bằng nhau
```

Cần phân biệt hai phép:

- `is` so sánh danh tính đối tượng; phù hợp với `x is None`;
- `==` gọi logic so sánh giá trị, hỏi "nội dung có bằng nhau không".

Tham số hàm cũng là gắn tên. Hàm nhận tham chiếu tới đối tượng, không phải nhãn truyền thống "pass by reference" hay "pass by value":

```python
def mutate(xs: list[int]) -> None:
    xs.append(1)          # sửa cùng list mà phía gọi thấy

def rebind(xs: list[int]) -> None:
    xs = [1,2]              # chỉ gắn lại tên cục bộ xs sang list mới

data: list[int] = []
mutate(data)
print(data)               # [1]

rebind(data)
print(data)               # vẫn là [1]
```

Mô hình này giải thích nhiều bug thường gặp: tham số mặc định khả biến, shallow copy, làm bẩn cache, và trạng thái dùng chung giữa các thread.

### 2.2 Đối tượng khả biến, bất biến và sao chép

Đối tượng bất biến thường gặp gồm `int`, `float`, `bool`, `str`, `bytes` và `tuple` khi mọi phần tử đều bất biến; đối tượng khả biến thường gặp gồm `list`, `dict`, `set` và hầu hết instance tùy chỉnh.

#### Tham số mặc định chỉ được tạo một lần

```python
# Sai: cùng một list bị nhiều lần gọi dùng chung
def collect_bad(x: int, result: list[int] = []) -> list[int]:
    result.append(x)
    return result

# Đúng: dùng None để nghĩa là "chưa cung cấp"
def collect(x: int, result: list[int] | None = None) -> list[int]:
    if result is None:
        result = []
    result.append(x)
    return result
```

Tham số mặc định được tính khi định nghĩa hàm, không phải mỗi lần gọi.

#### Shallow copy và deep copy

```python
import copy

src = [[1], [2]]
shallow = src.copy()           # chỉ sao chép list ngoài cùng
deep = copy.deepcopy(src)      # sao chép đệ quy các đối tượng bên trong

shallow[0].append(9)
print(src)                     # [[1, 9], [2]]
print(deep)                    # [[1], [2]]
```

Deep copy không phải câu trả lời mặc định. Sao chép vô tình tensor lớn hoặc trạng thái mô hình có thể tốn bộ nhớ và thời gian khổng lồ. Cách tốt hơn là xác định trước dữ liệu nào cần độc quyền, dữ liệu nào có thể chia sẻ chỉ-đọc.

### 2.3 Phạm vi, closure và LEGB

Python tìm tên theo thứ tự **LEGB**: Local → Enclosing → Global (module) → Builtins.

Closure giữ biến từ phạm vi bao ngoài:

```python
from collections.abc import Callable

def make_multiplier(scale: float) -> Callable[[float], float]:
    def multiply(x: float) -> float:
        return x * scale
    return multiply

double = make_multiplier(2.0)
print(double(3.0))  # 6.0
```

Khi tạo closure trong vòng lặp, chú ý **late binding**: closure chỉ đọc tên bao ngoài lúc được gọi.

```python
# Cả ba hàm cuối cùng đều đọc cùng một i, kết quả đều là 2
bad = [lambda: i for i in range(3)]

# Dùng tham số mặc định để lưu giá trị hiện tại mỗi vòng định nghĩa
good = [lambda i=i: i for i in range(3)]
print([fn() for fn in good])  # [0, 1, 2]
```

Nếu hàm trong cần gắn lại tên của hàm bao ngoài, dùng `nonlocal`; gắn lại tên global của module thì dùng `global`. Ưu tiên truyền trạng thái qua tham số và giá trị trả về; thường dễ kiểm thử hơn.

### 2.4 Class, data model và protocol

Special method của Python gắn đối tượng vào protocol của ngôn ngữ. Ví dụ, triển khai `__len__` thì gọi được `len(obj)`; triển khai `__iter__` thì `for` duyệt được.

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

Special method thường gặp và công dụng:

| Method | Kích hoạt bởi | Công dụng điển hình |
|------|----------|----------|
| `__init__` | Sau khi tạo instance | Khởi tạo trạng thái instance |
| `__repr__` | `repr(obj)`, debugger | Biểu diễn gỡ lỗi không mơ hồ |
| `__call__` | `obj(...)` | Cho instance hành xử như hàm |
| `__enter__` / `__exit__` | `with` | Quản lý vòng đời tài nguyên |
| `__iter__` / `__next__` | `for`, `next` | Định nghĩa protocol duyệt |
| `__getitem__` | `obj[key]` | Index, slice hoặc truy cập mapping |

#### Instance method, class method và static method

- Tham số đầu của instance method là `self`, thao tác trên đối tượng cụ thể;
- Tham số đầu của `@classmethod` là `cls`, thường dùng cho constructor thay thế;
- `@staticmethod` không nhận đối tượng ẩn; chủ yếu gom hàm tiện ích liên quan trong namespace của class.

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

### 2.5 Iterator và generator: sinh dữ liệu theo nhu cầu

**Iterable** trả về được iterator; **iterator** giữ trạng thái duyệt và trả phần tử lần lượt qua `__next__`. Hàm generator là cách viết iterator gọn:

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

`yield` trả một giá trị rồi tạm dừng hàm; lần duyệt tiếp theo tiếp tục từ chỗ tạm dừng. So với dựng sẵn cả list, đỉnh bộ nhớ của generator thường chỉ liên quan một batch, phù hợp pipeline dữ liệu streaming.

Cần nhớ:

- Generator thường chỉ tiêu thụ được một lần;
- Lazy evaluation hoãn ngoại lệ đến lúc duyệt;
- Đưa generator thành `list` sẽ materialize lại mọi phần tử và mất lợi thế tiết kiệm bộ nhớ.

### 2.6 Decorator: gắn hành vi ở biên lời gọi

Decorator nhận một callable và trả về một callable khác. Bộ đếm thời gian dưới đây giữ metadata của hàm gốc:

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

`@timed` tương đương `preprocess = timed(preprocess)`. `functools.wraps` giữ tên hàm, docstring và thuộc tính tương tự; điều đó quan trọng với logging, kiểm thử và reflection của framework.

Decorator phù hợp logging, cache, kiểm tra quyền và quan sát hiệu năng. Nếu chỉ một điểm gọi cần đo thời gian, một `with timer():` thường thẳng hơn; không cần decorator chỉ vì trông "cao cấp".

### 2.7 Context manager: bảo đảm tài nguyên được giải phóng

`with` đặt lấy và giải phóng tài nguyên trong cùng phạm vi từ vựng, nên dù có ngoại lệ, logic dọn dẹp vẫn chạy:

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

File, khóa, kết nối cơ sở dữ liệu, thư mục tạm và context mixed precision tự động đều hợp pattern này. Nó giải cùng bài toán với RAII trong C++: **vòng đời tài nguyên phải có biên rõ**.

### 2.8 Xử lý ngoại lệ: chỉ bắt lỗi bạn xử lý được

```python
def parse_world_size(raw: str) -> int:
    try:
        world_size = int(raw)
    except ValueError as exc:
        raise ValueError(f"WORLD_SIZE phải là số nguyên, thực tế là {raw!r}") from exc

    if world_size <= 0:
        raise ValueError("WORLD_SIZE phải lớn hơn 0")
    return world_size
```

Nên theo ba nguyên tắc:

1. Bắt ngoại lệ cụ thể; đừng dùng `except:` trống nuốt luôn `KeyboardInterrupt` và tín hiệu tương tự;
2. Chỉ bắt khi có thể phục hồi, bổ sung ngữ cảnh, hoặc chuyển lớp trừu tượng;
3. Dùng `raise ... from exc` để giữ chuỗi nguyên nhân gốc.

Đừng dùng ngoại lệ cho luồng điều khiển bình thường tần suất cao, nhất là trong vòng lặp nhạy hiệu năng.

### 2.9 Chú thích kiểu và dataclass

Chú thích kiểu chủ yếu phục vụ kiểm tra tĩnh, IDE và người đọc; chúng không tự xác thực kiểu lúc chạy:

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

Lựa chọn thường dùng:

- Khi tham số chỉ cần khả năng "iterable" hoặc "sequence", chú thích protocol trừu tượng như `Iterable[T]`, `Sequence[T]`;
- Khi trả về container cụ thể có thể chú thích `list[T]`, `dict[K, V]`;
- `T | None` nghĩa là giá trị có thể thiếu;
- `Protocol` mô tả interface theo cấu trúc, giảm kế thừa không cần thiết;
- `dataclass` phù hợp đối tượng chủ yếu mang dữ liệu, quy tắc đơn giản.

Mục tiêu của hệ thống kiểu là sớm lộ ra việc dùng sai interface, không phải ép ngôn ngữ động thành Java. Ưu tiên chú thích rõ ở biên công khai; biến cục bộ hiển nhiên không cần gắn từng cái.

---

## 3. Đồng thời Python và phân tích hiệu năng

### 3.1 Tách concurrency và parallelism trước

- **Concurrency** (đồng thời): nhiều tác vụ tiến triển trong cùng khoảng thời gian;
- **Parallelism** (song song): nhiều tác vụ thật sự thực thi cùng một lúc.

Trước khi chọn mô hình đồng thời, hãy xác định nút thắt:

| Loại tác vụ | Ví dụ điển hình | Lựa chọn thường gặp |
|----------|----------|----------|
| I/O bound | Gọi dịch vụ, đọc nhiều file nhỏ | Thread hoặc `asyncio` |
| Python CPU bound | Parse thuần Python, vòng lặp phức tạp | Multiprocessing hoặc extension native |
| Native-operator bound | Tính NumPy / PyTorch / CUDA | Thread pool / song song GPU tầng dưới; Python điều phối |
| Dữ liệu lớn xuyên process | Nạp dữ liệu, cache dùng chung | Multiprocessing + shared memory; serialize cẩn thận |

### 3.2 GIL thực sự hạn chế gì

Bản dựng CPython phổ biến dùng Global Interpreter Lock (GIL) để bảo vệ trạng thái nội bộ interpreter. Trong một process, thường chỉ một thread thực thi bytecode Python tại một thời điểm. Vì vậy hai thread chạy vòng lặp CPU thuần Python nói chung không được tốc độ gấp đôi.

Điều đó không có nghĩa "thread Python vô dụng":

- Khi I/O blocking, interpreter nhả GIL;
- Extension native như NumPy, PyTorch có thể nhả GIL khi tính toán nặng;
- Logging nền, prefetch và tác vụ điều khiển nhẹ thường hợp thread.

GIL là chi tiết triển khai của CPython; đừng khái quát thành quy tắc vĩnh viễn cho mọi triển khai Python hay mọi bản dựng. Khi đánh giá hiệu năng, hãy đo interpreter và dependency đang chạy, đừng chỉ thuộc lòng kết luận.

### 3.3 Thread: chia sẻ bộ nhớ, cũng chia sẻ rủi ro

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

Thread chia sẻ bộ nhớ process, truyền dữ liệu rẻ, nhưng read-modify-write trên cùng đối tượng khả biến có thể race. Khi cần khóa, giữ critical section càng nhỏ càng tốt:

```python
from threading import Lock

lock = Lock()
counter = 0

def increment() -> None:
    global counter
    with lock:
        counter += 1
```

Đừng thiết kế tính đúng dựa trên "thao tác builtin này trông như atomic"; phiên bản interpreter, triển khai đối tượng, hoặc thao tác ghép đều có thể phá giả định đó.

### 3.4 Multiprocessing: tránh GIL, nhưng vận chuyển dữ liệu có giá

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

`if __name__ == "__main__":` đặc biệt quan trọng trên nền tảng dùng `spawn` để tạo process con; nó tránh tạo process mới đệ quy khi process con import lại module.

Chi phí chính của multiprocessing gồm:

- Tạo process và interpreter riêng;
- Serialize / deserialize tham số và giá trị trả về;
- Sao chép hoặc map bộ nhớ;
- Giao tiếp và đồng bộ giữa các process.

Nếu mỗi tác vụ chỉ làm vài chục micro giây, chi phí lập lịch có thể cao hơn tính toán. Hãy tăng độ hạt của tác vụ và tránh đưa qua lại mảng hoặc tensor lớn.

> **Liên hệ GPU**: kết hợp sai CUDA context với cách khởi động multiprocessing có thể gây lỗi init hoặc chiếm thêm bộ nhớ thiết bị. Khi dùng khả năng nạp dữ liệu và phân tán của framework, hãy theo launcher và start method mà framework khuyến nghị; đừng `fork` tùy tiện sau khi process cha đã init CUDA.

### 3.5 `asyncio`: đồng thời I/O hợp tác

`asyncio` dùng event loop để lập lịch coroutine. Coroutine chỉ chủ động nhường quyền tại `await`:

```python
import asyncio

async def worker(name: str, delay: float) -> str:
    await asyncio.sleep(delay)  # mô phỏng I/O không blocking
    return name

async def main() -> None:
    results = await asyncio.gather(
        worker("a", 0.2),
        worker("b", 0.1),
    )
    print(results)

asyncio.run(main())
```

Chạy I/O blocking hoặc tính CPU lâu trực tiếp trên event loop sẽ kẹt mọi coroutine. Khi đó hãy dùng thư viện async, hoặc đưa việc blocking sang executor thread/process.

### 3.6 Benchmark đúng cách

Lỗi phổ biến nhất trong đo hiệu năng là chỉ đo một lần, tính luôn phần khởi tạo, hoặc quên GPU chạy bất đồng bộ.

Đoạn CPU thuần nhỏ có thể dùng `timeit`:

```bash
python -m timeit -s 'xs = list(range(1000))' 'sum(xs)'
```

Đo thủ công ít nhất nên làm thế này:

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

Với GPU, khởi chạy kernel thường bất đồng bộ. Đồng hồ CPU thường chỉ đo tới lúc "nộp việc"; cần sync ở biên, hoặc dùng GPU Event của framework. Sync làm thay đổi hành vi pipeline, nên đo throughput đầu-cuối và độ trễ từng toán tử riêng.

### 3.7 Từ "chậm" đến đúng dòng code

Nên đi từ thô đến tinh:

1. **Đo đầu-cuối**: xác nhận regression là thật;
2. **Đo theo giai đoạn**: tách dữ liệu, tính toán, giao tiếp và lưu;
3. **Profiling cấp hàm**: dùng `cProfile` tìm hàm có cumulative time cao;
4. **Profiling cấp dòng / sampling**: hẹp dần tới dòng nóng hoặc native stack;
5. **Profiling bộ nhớ**: dùng công cụ như `tracemalloc` tìm tăng cấp phát Python;
6. **Profiling GPU**: chuyển sang công cụ chuyên dụng như PyTorch Profiler, Nsight Systems/Compute.

```bash
python -m cProfile -o profile.out train.py
python -m pstats profile.out
```

Trong `pstats` có thể sắp theo cumulative time:

```text
sort cumulative
stats 30
```

Kiểm tra cấp phát bộ nhớ Python:

```python
import tracemalloc

tracemalloc.start()
# chạy code cần kiểm tra
snapshot = tracemalloc.take_snapshot()
for stat in snapshot.statistics("lineno")[:10]:
    print(stat)
```

### 3.8 Thứ tự ưu tiên tối ưu hiệu năng Python

Thứ tự gợi ý:

1. Sửa độ phức tạp thuật toán và cấu trúc dữ liệu trước;
2. Giảm I/O, sao chép và serialize không cần thiết;
3. Đổi vòng lặp Python hạt mịn thành toán tử native theo batch / vector hóa;
4. Giảm đi-về thường xuyên qua biên Python và C++/GPU;
5. Chỉ sau khi xác nhận hotspot mới viết extension C++, CUDA hoặc Triton.

Ví dụ, một triệu lời gọi toán tử nhỏ dù mỗi lần nhanh vẫn cộng dồn chi phí lập lịch lớn. Gom dữ liệu thành batch rồi gọi một toán tử lớn một lần thường hiệu quả hơn tinh chỉnh cú pháp Python.

---

## 4. C/C++: bộ nhớ, vòng đời và quản lý tài nguyên

### 4.1 Giá trị, địa chỉ, pointer và reference

```cpp
#include <iostream>

int main() {
    int value = 42;
    int* ptr = &value;  // ptr giữ địa chỉ của value
    int& ref = value;   // ref là bí danh của value

    *ptr = 7;           // dereference pointer và ghi
    ref += 1;

    std::cout << value << '\n';  // 8
}
```

Có thể dùng bảng này để dựng trực giác trước:

| Khái niệm | Có thể null? | Có đổi đích? | Công dụng thường gặp |
|------|----------|------------|----------|
| Giá trị `T` | Không | Không áp dụng | Sở hữu một đối tượng độc lập |
| Pointer `T*` | Có thể `nullptr` | Có | Đối tượng tùy chọn, mảng, API tầng thấp, quan sát không sở hữu |
| Reference `T&` | Theo ngữ nghĩa ngôn ngữ phải gắn đối tượng | Không gắn lại được | Bí danh bắt buộc có, tham số hàm |
| Const reference `const T&` | Phải gắn đối tượng | Không | Tránh sao chép và không cho sửa |

Tham số hàm nên thể hiện ý định interface càng rõ càng tốt:

```cpp
void consume(T value);          // theo giá trị: hàm nhận đối tượng của riêng nó
void update(T& value);          // mượn khả biến: hàm sẽ sửa đối tượng phía gọi
void inspect(const T& value);   // mượn chỉ-đọc: tránh sao chép
void optional(T* value);        // pointer có thể null; thường không sở hữu đối tượng
```

Với scalar nhỏ (`int`, `float`, pointer), truyền theo giá trị thường rõ nhất; đừng máy móc đổi mọi tham số thành reference.

### 4.2 Mảng, pointer và bộ nhớ liên tục

Trong hầu hết biểu thức, mảng native suy biến thành pointer tới phần tử đầu, nên một `T*` trần không mang độ dài:

```cpp
void scale(float* data, std::size_t size, float factor) {
    for (std::size_t i = 0; i < size; ++i) {
        data[i] *= factor;
    }
}
```

Trong C++ hiện đại, ưu tiên kiểu biểu đạt được độ dài:

```cpp
#include <span>

void scale(std::span<float> values, float factor) {
    for (float& value : values) {
        value *= factor;
    }
}
```

`std::span` không sở hữu dữ liệu; nó chỉ là view nhẹ "pointer + độ dài". Bộ nhớ nền phải vẫn sống trong suốt lời gọi.

Ma trận 2D thường lưu tuyến tính. Với ma trận row-major `A[M][N]`, offset của `(row, col)` là:

$$
\text{offset} = row \times N + col
$$

Công thức này sẽ xuất hiện lại liên tục trong CUDA. Thứ tự truy cập có liên tục hay không ảnh hưởng trực tiếp tới cache CPU và coalesced memory access trên GPU.

### 4.3 Thời gian lưu trữ và vòng đời đối tượng

Người ta thường nói "stack" và "heap", nhưng chính xác hơn cần quan tâm **thời gian lưu trữ và quyền sở hữu của đối tượng**:

```cpp
void example() {
    int local = 1;                  // automatic storage duration; hủy khi rời scope
    auto buffer = new float[1024];  // cấp phát động; phải delete[] tường minh
    delete[] buffer;
}
```

`new`/`delete` thủ công dễ gây:

- Memory leak: quên giải phóng;
- Double free: giải phóng cùng địa chỉ hai lần;
- Dangling pointer: đối tượng đã hủy, pointer vẫn được dùng;
- Form cấp phát/giải phóng không khớp: `new[]` đi với `delete`;
- Rò trên đường ngoại lệ: hàm thoát trước khi giải phóng.

Vì vậy nguyên tắc mặc định của C++ hiện đại là: **để đối tượng quản lý tài nguyên, để scope quản lý đối tượng.**

### 4.4 RAII: gắn tài nguyên vào vòng đời đối tượng

RAII (Resource Acquisition Is Initialization) lấy tài nguyên khi khởi tạo và giải phóng khi hủy:

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

Dù hàm trả về bình thường hay ném ngoại lệ, destructor của đối tượng cục bộ vẫn chạy. File, khóa, bộ nhớ host, bộ nhớ GPU, CUDA Stream đều có thể quản lý theo cùng ý tưởng.

Trong dự án thực tế, ưu tiên kiểu RAII sẵn có của thư viện chuẩn như `std::vector`, `std::string`, `std::fstream`, `std::lock_guard` và smart pointer; đừng tự chế lại.

### 4.5 Smart pointer và quyền sở hữu

```cpp
#include <memory>

auto unique = std::make_unique<Buffer>(1024);  // sở hữu duy nhất
auto shared = std::make_shared<Buffer>(1024);  // sở hữu chia sẻ
```

| Kiểu | Ngữ nghĩa | Chi phí và lưu ý |
|------|------|--------------|
| `std::unique_ptr<T>` | Sở hữu duy nhất; move được, không copy | Chi phí gần pointer trần; lựa chọn mặc định |
| `std::shared_ptr<T>` | Sở hữu chia sẻ theo đếm tham chiếu | Có chi phí đếm và control block; vòng tham chiếu cần `weak_ptr` |
| `std::weak_ptr<T>` | Quan sát không tăng đếm tham chiếu | Trước khi dùng phải `lock()` để kiểm tra đối tượng còn sống |
| `T*` / `T&` | Thường là truy cập không sở hữu | Đối tượng được trỏ tới phải sống đủ lâu |

Nếu không cần chia sẻ sở hữu, đừng dùng `shared_ptr`. "Tránh phải nghĩ về ownership" không phải lý do hợp lý cho shared ownership.

### 4.6 Copy, move và Rule of Zero

Container như `std::vector` sở hữu tài nguyên; copy sao chép nội dung, còn move chuyển tài nguyên nội bộ:

```cpp
#include <utility>
#include <vector>

std::vector<float> make_buffer() {
    std::vector<float> values(1'000'000, 0.0f);
    return values;  // trình biên dịch thường elide copy; nếu cần thì move
}

auto a = make_buffer();
auto b = std::move(a);  // a vẫn hợp lệ, nhưng nội dung ở trạng thái không xác định
```

Nếu class chỉ gồm thành viên RAII của thư viện chuẩn, thường không cần tự viết destructor, copy và move; đó là **Rule of Zero**. Chỉ khi trực tiếp quản lý tài nguyên đặc biệt mới cần tùy chỉnh các thao tác này và định nghĩa ownership rõ ràng.

Đừng tùy tiện viết `std::move` trên giá trị trả về; nó có thể cản return-value optimization. Sau khi move, đừng giả định đối tượng nguồn vẫn giữ nội dung cũ.

### 4.7 `const`: đưa tính bất biến vào interface

```cpp
const float* p1;        // trỏ tới const float: không đổi giá trị qua p1
float* const p2 = ptr;  // const pointer: không đổi đích của p2
const float* const p3 = ptr;
```

`const` ở cuối member function nghĩa là không sửa trạng thái quan sát được của đối tượng:

```cpp
class TensorView {
public:
    std::size_t size() const { return size_; }
private:
    std::size_t size_{};
};
```

`const` giúp trình biên dịch và người đọc kiểm tra interface; nó không tự giải quyết an toàn đồng thời hay vấn đề aliasing tầng thấp.

### 4.8 Vì sao undefined behavior nguy hiểm

Truy cập ngoài biên, dùng dangling reference, tràn số nguyên có dấu, data race và tương tự có thể kích hoạt undefined behavior (UB). Trình biên dịch có thể giả định UB không bao giờ xảy ra rồi tối ưu theo đó; kết quả không nhất thiết chỉ là "báo lỗi". Bản debug có thể chạy ổn trong khi bản tối ưu âm thầm tính sai.

```cpp
std::vector<int> values(4);
values[4] = 1;  // ngoài biên; undefined behavior
```

Khi gỡ lỗi, bật cảnh báo và sanitizer:

```bash
g++ -std=c++20 -Wall -Wextra -Wpedantic \
    -fsanitize=address,undefined -g main.cpp -o app
./app
```

AddressSanitizer thường dùng để tìm truy cập ngoài biên và use-after-free; UndefinedBehaviorSanitizer kiểm nhiều loại UB; ThreadSanitizer định vị được data race, nhưng thường không bật cùng AddressSanitizer.

---

## 5. Biên dịch C++, liên kết, template và hệ thống build

### 5.1 Từ mã nguồn đến executable

```text
File nguồn .cpp
  └─ Tiền xử lý: mở rộng #include, macro và biên dịch có điều kiện
      └─ Biên dịch: kiểm tra cú pháp/kiểu và sinh assembly
          └─ Assemble: tạo object file .o
              └─ Liên kết: giải quyết symbol và tạo executable hoặc thư viện
```

Có thể quan sát từng bước:

```bash
g++ -E main.cpp -o main.ii       # chỉ tiền xử lý
g++ -S main.ii -o main.s         # sinh assembly
g++ -c main.cpp -o main.o        # tạo object file
g++ main.o -o app                # liên kết
```

Header thường chứa khai báo; file nguồn chứa định nghĩa:

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

### 5.2 Lỗi biên dịch, lỗi liên kết và lỗi nạp lúc chạy

Đừng lẫn ba loại lỗi:

| Giai đoạn | Báo lỗi điển hình | Nguyên nhân thường gặp |
|------|----------|----------|
| Biên dịch | `not declared`, kiểu không khớp | Thiếu khai báo, instantiate template thất bại, lỗi cú pháp |
| Liên kết | `undefined reference` | Chỉ khai báo chưa định nghĩa, thiếu link thư viện, ABI/chữ ký không khớp |
| Nạp lúc chạy | `cannot open shared object file` | Thư viện động không nằm trên search path, kiến trúc không khớp |

Lệnh thường dùng để xem symbol và dependency:

```bash
nm -C build/app | less          # xem và demangle symbol C++
ldd build/app                   # Linux: dependency thư viện động
readelf -d build/app            # xem ELF dynamic section
```

Trên macOS, công cụ tương ứng thường là `otool -L`.

### 5.3 Thư viện tĩnh và thư viện động

- Thư viện tĩnh (trên Linux thường `.a`) gộp code cần thiết vào target lúc link; triển khai trực tiếp nhưng artifact có thể lớn hơn;
- Thư viện động (Linux `.so`, macOS `.dylib`) giải quyết lúc nạp/chạy; nhiều chương trình có thể dùng chung, nhưng phải xử lý search path và tương thích ABI.

Search thư viện động trên Linux chịu ảnh hưởng của RPATH/RUNPATH của binary, `LD_LIBRARY_PATH`, cache hệ thống và thư mục mặc định. Đặt tạm `LD_LIBRARY_PATH` phù hợp để chẩn đoán; không nên biến nó thành cách dài hạn để che layout cài đặt hỏng.

### 5.4 ABI: tương thích mã nguồn không có nghĩa tương thích nhị phân

ABI (Application Binary Interface) quy định calling convention, name mangling, layout đối tượng và cách triển khai kiểu thư viện chuẩn ở tầng nhị phân. Các tình huống sau đều có thể phá tương thích ABI:

- Trình biên dịch hoặc thư viện chuẩn không thống nhất;
- Tùy chọn biên dịch làm đổi layout đối tượng hoặc C++ ABI;
- Symbol xuất khẩu đổi sau khi nâng cấp thư viện;
- Trộn runtime Debug/Release;
- Kiến trúc CPU khác nhau.

Khi extension C++ cho Python "biên dịch được nhưng import thất bại", ngoài đường dẫn Python còn phải kiểm ABI, kiến trúc và symbol của extension cùng thư viện phụ thuộc.

### 5.5 Template: sinh code chuyên biệt theo kiểu lúc biên dịch

```cpp
template <typename T>
T add(T lhs, T rhs) {
    return lhs + rhs;
}

auto i = add(1, 2);        // instantiate add<int>
auto f = add(1.0f, 2.0f);  // instantiate add<float>
```

Định nghĩa template thường đặt trong header, vì trình biên dịch cần thấy định nghĩa đầy đủ tại điểm dùng để instantiate. CUDA và mã hiệu năng cao dùng template nhiều để biểu đạt:

- Kiểu dữ liệu: `float`, `half`, `bfloat16`;
- Kích thước tile và cấu hình thread block;
- Tùy chọn layout và transpose;
- Nhánh biên dịch và triển khai chuyên biệt.

Tham số template không phải kiểu có thể biến kích thước thành hằng biên dịch:

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

Lỗi template thường rất dài. Hãy bắt đầu từ "vị trí instantiate ở mã người dùng" sớm nhất và ràng buộc thất bại sâu nhất; đừng bị hàng chục lớp mở rộng ở giữa làm choáng.

### 5.6 Container STL và iterator

| Kiểu | Đặc trưng bộ nhớ / truy cập | Công dụng thường gặp |
|------|---------------|----------|
| `std::vector<T>` | Bộ nhớ liên tục, truy cập ngẫu nhiên O(1) | Buffer số; container tuần tự mặc định |
| `std::array<T, N>` | Kích thước cố định, bộ nhớ liên tục | Mảng nhỏ biết trước lúc biên dịch |
| `std::deque<T>` | Lưu theo đoạn; chèn hai đầu hiệu quả | Hàng đợi, cửa sổ trượt |
| `std::unordered_map<K,V>` | Bảng băm; tìm trung bình O(1) | Chỉ mục key-value không thứ tự |
| `std::map<K,V>` | Cây có thứ tự; O(log n) | Duyệt có thứ tự / truy vấn khoảng |

`std::vector` khi mở rộng có thể cấp phát lại và làm vô hiệu pointer, reference, iterator cũ:

```cpp
std::vector<int> values;
values.reserve(1024);  // khi biết quy mô, giảm mở rộng lặp lại
```

Trong mã nhạy hiệu năng, ngoài big-O còn phải nhìn tính liên tục, số lần cấp phát, locality cache và kích thước phần tử. Nhiều tình huống quét tuyến tính `vector` liên tục nhanh hơn cấu trúc có độ phức tạp tìm kiếm lý thuyết thấp hơn nhưng nhảy pointer thường xuyên.

### 5.7 Dự án CMake tối thiểu

Cấu trúc thư mục:

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

Trọng tâm của CMake hiện đại là **target**. Dùng `target_include_directories`, `target_link_libraries`, `target_compile_features` để mô tả dependency; hạn chế `include_directories` toàn cục và ghép lệnh biên dịch thủ công.

### 5.8 Nhận thức tối thiểu về đồng thời C++

Nhiều thread truy cập đồng thời cùng bộ nhớ, trong đó ít nhất một bên ghi và thiếu đồng bộ, có thể tạo data race, thuộc undefined behavior.

```cpp
#include <mutex>

std::mutex mutex;
int counter = 0;

void increment() {
    std::lock_guard<std::mutex> guard(mutex);
    ++counter;
}
```

Mutex bảo đảm loại trừ lẫn nhau trong critical section; kiểu atomic phù hợp trạng thái chia sẻ đơn giản; condition variable dùng để chờ thay đổi trạng thái. Đừng coi `volatile` là công cụ đồng bộ thread; nó không cung cấp atomicity hay quan hệ happens-before giữa các thread.

Trước khi tối ưu đồng thời hiệu năng cao, hãy bảo đảm tính đúng trước, và dùng công cụ như ThreadSanitizer để kiểm data race.

---

## 6. Tương tác Python và C++

### 6.1 Vì sao cần vượt biên ngôn ngữ

Mục đích thường gặp gồm:

- Tái sử dụng thư viện C/C++ sẵn có;
- Đẩy vòng lặp hotspot Python xuống mã native;
- Kết nối driver, thư viện giao tiếp hoặc interface phần cứng tùy chỉnh;
- Thêm toán tử tùy chỉnh cho framework như PyTorch.

Vượt ngôn ngữ không tự động nhanh hơn. Một lời gọi phải kiểm tham số, chuyển kiểu/layout, có thể sao chép dữ liệu và dịch ngoại lệ. Nếu mỗi phần tử vượt biên một lần, chi phí binding có thể nuốt hết lợi ích tính native. Interface lý tưởng thường truyền một batch hoặc buffer liên tục.

### 6.2 `ctypes`: gọi C ABI ổn định

Trước hết xuất một interface C từ C++:

```cpp
// scale.cpp
extern "C" void scale(float* data, int size, float factor) {
    for (int i = 0; i < size; ++i) {
        data[i] *= factor;
    }
}
```

Biên dịch thư viện động:

```bash
g++ -O3 -shared -fPIC scale.cpp -o libscale.so
```

Gọi phía Python:

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

`ctypes` đơn giản trực tiếp, nhưng kiểu, độ dài và vòng đời chủ yếu do người gọi bảo đảm. Sai chữ ký có thể crash, không chỉ là ngoại lệ Python thông thường.

### 6.3 `pybind11`: gắn interface C++ một cách tự nhiên

`pybind11` có thể ánh xạ hàm, class và ngoại lệ C++ thành đối tượng Python:

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

Khi bind mảng lớn cần đặc biệt đối chiếu:

- dtype có khớp không;
- shape và stride có đúng giả định thuật toán không;
- dữ liệu có liên tục không;
- bộ nhớ nằm trên CPU hay GPU;
- ai sở hữu storage nền, sau khi trả về còn hợp lệ không;
- trong lúc tính có nên nhả GIL không;
- ngoại lệ native chuyển thành ngoại lệ Python thế nào.

### 6.4 Zero-copy không phải "không điều kiện"

Zero-copy thường chỉ nghĩa là dùng chung cùng một khối bộ nhớ nền. Hai bên phải thống nhất hợp đồng sau:

```text
địa chỉ + kiểu phần tử + shape + stride + device + vòng đời + tính khả biến
```

Bất kỳ mục nào không khớp đều có thể gây sao chép ẩn, kết quả sai hoặc truy cập ngoài biên. Ví dụ, mảng NumPy sau transpose có thể không còn layout C-contiguous; chỉ lấy địa chỉ đầu rồi đọc như mảng liên tục sẽ ra dữ liệu sai.

> **Nguyên tắc cốt lõi**: coi layout dữ liệu và ownership là một phần của interface, không phải chi tiết triển khai.

---

## 7. Kỹ năng Linux cơ bản

### 7.1 File, thư mục và luồng văn bản

```bash
pwd                         # thư mục hiện tại
ls -lah                     # liệt kê chi tiết, gồm file ẩn
mkdir -p runs/exp01/logs     # tạo thư mục đệ quy
cp -a configs runs/exp01/   # sao chép giữ thuộc tính
mv old_name new_name
rm -i file                  # xóa tương tác; lệnh nguy hiểm hãy xác nhận đường dẫn trước

find runs -type f -name '*.log'
rg 'CUDA error|out of memory' runs
```

Sức mạnh tổ hợp của shell đến từ stdin, stdout và stderr:

```bash
python train.py >train.log 2>train.err       # chuyển hướng riêng
python train.py >train.all.log 2>&1           # gộp output
rg 'loss=' train.log | tail -n 20
```

Pipe nối stdout bên trái vào stdin bên phải. Mỗi đoạn nên làm một việc rõ; quy trình phức tạp cần bảo trì nên viết thành script và đưa vào kiểm soát phiên bản.

### 7.2 Dấu ngoặc, biến và mã thoát

```bash
name='experiment 01'
printf '%s\n' "$name"   # ngoặc kép giữ thành một tham số
printf '%s\n' '$name'   # ngoặc đơn không mở rộng biến
```

Biến không ngoặc có thể bị word splitting và glob. Trong script shell hầu như luôn nên viết `"$variable"`.

Mỗi lệnh có mã thoát: `0` là thành công, khác 0 là thất bại.

```bash
python check_data.py
status=$?
printf 'exit code: %s\n' "$status"
```

Script Bash vững thường mở đầu bằng:

```bash
#!/usr/bin/env bash
set -euo pipefail
```

- `-e`: thất bại chưa xử lý làm script thoát;
- `-u`: dùng biến chưa định nghĩa thì báo lỗi;
- `pipefail`: bất kỳ lệnh nào trong pipe thất bại thì cả pipe thất bại.

Các tùy chọn này không thay thế xử lý lỗi rõ ràng, đặc biệt cần hiểu ngữ nghĩa ngoại lệ trong câu điều kiện và pipe.

### 7.3 Mô hình quyền

```bash
ls -l script.sh
# -rwxr-x---  owner group ... script.sh

chmod u+x script.sh          # thêm quyền thực thi cho owner
chmod 640 config.yaml        # owner: rw, group: r, other: none
```

`r/w/x` với file lần lượt là đọc, ghi, thực thi; với thư mục là liệt kê tên, sửa mục thư mục, đi xuyên thư mục. Lỗi quyền hãy dùng `ls -l`, `id` và quyền thư mục cha để định vị; đừng quen tay `chmod 777`.

### 7.4 Tiến trình, job và tín hiệu

```bash
ps -ef | rg train.py
pgrep -af train.py
top                         # hoặc htop

python train.py &           # chạy nền
jobs
fg %1                       # kéo về foreground
```

Gửi tín hiệu:

```bash
kill -TERM <pid>            # yêu cầu tiến trình thoát gọn
kill -INT <pid>             # giống Ctrl-C
kill -KILL <pid>            # kernel dừng ngay, không dọn; biện pháp cuối
```

Ưu tiên `SIGTERM` để chương trình kịp lưu checkpoint, giải phóng tài nguyên dùng chung và flush log. `SIGKILL` không bắt được và không dọn được.

Tác vụ từ xa dài nên dùng `tmux` hoặc hệ thống lập lịch cluster, thay vì chỉ dựa vào terminal SSH:

```bash
tmux new -s train
# Ctrl-b d: thoát nhưng giữ session
tmux attach -t train
```

### 7.5 Xem CPU, bộ nhớ, đĩa và mạng

| Câu hỏi | Lệnh thường dùng |
|------|----------|
| CPU / tải | `top`, `uptime`, `mpstat` |
| Bộ nhớ tiến trình | `ps`, `top`, `pmap -x PID` |
| Bộ nhớ hệ thống | `free -h`, `vmstat` |
| Dung lượng đĩa | `df -h` |
| Dung lượng thư mục | `du -sh PATH`, `du -h --max-depth=1` |
| I/O file | `iostat`, `iotop` |
| Cổng đang lắng nghe | `ss -lntp` |
| Kiểm tra kết nối | `curl -v URL`, `nc -vz HOST PORT` |

"Thiếu bộ nhớ" ít nhất có thể là RAM máy chủ, bộ nhớ GPU, shared memory `/dev/shm`, giới hạn không gian địa chỉ tiến trình, hoặc đĩa đầy. Xác nhận loại tài nguyên rồi mới hành động.

### 7.6 Biến môi trường và đường dẫn thư viện động

```bash
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

env | sort
which nvcc
type -a python
```

Biến môi trường được process cha truyền cho process con. Sửa `~/.bashrc` không thần kỳ đổi terminal, IDE hay dịch vụ đang chạy; cần nạp lại cấu hình hoặc khởi động lại process tương ứng.

`env -i` có thể tái hiện vấn đề trong môi trường gần như trống, nhưng phải bổ sung tường minh các biến cần thiết:

```bash
env -i HOME="$HOME" PATH=/usr/bin:/bin bash --noprofile --norc
```

### 7.7 SSH và truyền file

Tạo khóa và cấu hình đăng nhập:

```bash
ssh-keygen -t ed25519 -C "your-name"
ssh-copy-id user@server
ssh user@server
```

`~/.ssh/config` có thể lưu alias:

```text
Host gpu-dev
    HostName 192.0.2.10
    User lxy
    IdentityFile ~/.ssh/id_ed25519
    ServerAliveInterval 60
```

Sau đó chỉ cần `ssh gpu-dev`. Đồng bộ thư mục lớn nên ưu tiên `rsync` hỗ trợ truyền tăng dần:

```bash
rsync -av --progress ./project/ gpu-dev:~/project/
```

Trước khi sync hãy kiểm kỹ đường dẫn nguồn có kết thúc bằng `/` hay không; điều đó quyết định copy "bản thân thư mục" hay "nội dung thư mục". Dữ liệu nhạy cảm và khóa riêng đừng đưa vào repo, cũng đừng tùy tiện copy lên máy dùng chung.

### 7.8 Gỡ lỗi chương trình native

Biên dịch kèm thông tin gỡ lỗi:

```bash
g++ -g -O0 main.cpp -o app
gdb ./app
```

Thao tác GDB cơ bản:

```text
break main       # đặt breakpoint
run              # khởi chạy
next / step      # bước đơn; có vào hàm hay không
print variable   # xem biến
bt               # xem call stack
continue         # tiếp tục chạy
```

Bản tối ưu khi crash cũng nên giữ symbol, ví dụ `-O2 -g`, để core dump có call stack đọc được. Debugger trả lời "crash ở đâu"; sanitizer giỏi hơn ở câu "thao tác bộ nhớ bất hợp pháp nào gây ra nó".

Vấn đề tầng system call có thể quan sát bằng `strace` (Linux):

```bash
strace -f -o trace.log ./app
rg 'ENOENT|EACCES' trace.log
```

Cách này rất hiệu quả để định vị "chương trình thực sự đang tìm file cấu hình / thư viện động nào".

---

## 8. Cô lập môi trường và phần mềm GPU

### 8.1 `venv`, conda và container giải quyết các lớp khác nhau

| Công cụ | Phạm vi cô lập | Phù hợp |
|------|----------|----------|
| `venv` | Gói Python và môi trường interpreter | Dự án thuần Python, dependency đơn giản |
| conda | Python + một phần thư viện native và toolchain | Tính toán khoa học, dependency nhị phân phức tạp |
| Docker / container | Hệ thống file user-space, dependency và môi trường tiến trình | Triển khai, CI, tái hiện xuyên máy |

Ví dụ `venv`:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Luôn dùng `python -m pip` để giảm nhầm lẫn kiểu "chạy Python của môi trường A nhưng gọi pip của môi trường B".

Ví dụ conda:

```bash
conda create -n ai-infra python=3.12
conda activate ai-infra
conda env export --from-history > environment.yml
```

Khóa phiên bản dependency trực tiếp giúp tái hiện, nhưng môi trường đầy đủ còn gồm hệ điều hành, driver, trình biên dịch, kiến trúc phần cứng và tham số khởi chạy. Đừng coi một `requirements.txt` là toàn bộ mô tả môi trường.

### 8.2 Container không phải máy ảo

Container dùng chung kernel với host, chỉ cô lập hệ thống file user-space, tiến trình, mạng và tài nguyên tương tự. Container GPU thường vẫn dùng **NVIDIA driver của host**, rồi NVIDIA Container Toolkit đưa thiết bị và thư viện driver vào container.

```bash
docker run --rm -it \
  --gpus all \
  -v "$PWD:/workspace" \
  -w /workspace \
  nvidia/cuda:12.4.1-runtime-ubuntu22.04 \
  nvidia-smi
```

Tag image nên cố định tới phiên bản đủ cụ thể. Volume mount cho phép tiến trình trong container sửa trực tiếp thư mục host; trước khi chạy lệnh lạ hãy hiểu mount và quyền người dùng.

### 8.3 Phần mềm GPU: bốn phần dễ lẫn

```text
Ứng dụng / PyTorch / extension tùy chỉnh
        │
CUDA Runtime và thư viện toán (cudart, cuBLAS, cuDNN, ...)
        │
Thư viện driver user-space (libcuda)
        │
NVIDIA Driver kernel-space
        │
GPU
```

- **NVIDIA Driver**: nền tảng giao tiếp giữa hệ điều hành và GPU; `nvidia-smi` chủ yếu phản ánh trạng thái driver;
- **CUDA Toolkit**: bộ công cụ phát triển gồm `nvcc`, header, công cụ gỡ lỗi/phân tích và thư viện;
- **CUDA Runtime / thư viện toán**: thành phần user-space được nạp khi ứng dụng chạy;
- **cuDNN**: thư viện tăng tốc độc lập hướng tới toán tử deep learning; không phải toàn bộ CUDA.

"CUDA Version" mà `nvidia-smi` hiện thường nghĩa là mức tương thích CUDA cao nhất mà driver hỗ trợ; không chứng minh máy đã cài cùng phiên bản `nvcc`. Để phán toolchain biên dịch hãy xem `nvcc --version`; để phán runtime framework thực sự dùng hãy xem thông tin của chính framework.

### 8.4 Thứ tự cố định khi chẩn đoán môi trường GPU

```bash
# 1. Driver có thấy thiết bị không?
nvidia-smi

# 2. Toolchain biên dịch đến từ đâu?
which nvcc
nvcc --version

# 3. Python và framework đến từ đâu?
which python
python -c 'import sys; print(sys.executable)'

# 4. Framework thấy gì?
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("built with CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
PY
```

Ý nghĩa của các hiện tượng thường gặp:

| Hiện tượng | Kiểm trước |
|------|----------|
| `nvidia-smi` thất bại | Driver, mount thiết bị, trạng thái host |
| `nvidia-smi` thành công nhưng không có `nvcc` | Toolkit chưa cài hoặc `PATH` sai; chạy framework prebuilt chưa chắc cần `nvcc` |
| Import framework được nhưng CUDA không khả dụng | Cài bản CPU, tương thích driver, container không passthrough GPU, thiết bị bị che |
| Extension tùy chỉnh biên dịch thất bại | Trình biên dịch, Toolkit, header, ABI, kiến trúc GPU đích |
| Runtime thiếu `.so` | Search path thư viện động, dependency wheel/conda, kiến trúc không khớp |

Mỗi lần chỉ đổi một biến, và giữ nguyên báo lỗi gốc. Hầu hết lỗi môi trường định vị được theo chuỗi này, không cần "xóa hết cài lại".

---

## 9. Cộng tác Git và cách định vị vấn đề

### 9.1 Một quy trình đóng góp sạch

```bash
git clone https://github.com/owner/project.git
cd project
git switch -c docs/improve-foundations

# sửa và kiểm chứng
git status --short
git diff --check
git diff

git add path/to/changed-file
git commit -m "docs: expand programming foundations"
```

Commit nên chỉ chứa một chủ đề rõ. Đừng trộn format cả repo, cấu hình editor và thay đổi mục tiêu vào cùng một commit.

### 9.2 Đọc trạng thái working tree

Git cần phân biệt ba lớp:

```text
commit trong HEAD  ← git commit ← vùng staging ← git add ← working tree
```

```bash
git diff                 # working tree vs staging
git diff --staged        # staging vs HEAD
git status --short
git log --oneline --decorate -10
```

Trước khi commit hãy xem đồng thời `git diff` và `git diff --staged`, để tránh sót file hoặc kéo file không liên quan vào.

### 9.3 Merge, rebase và xung đột

- `merge` giữ lịch sử hai nhánh và tạo merge commit;
- `rebase` phát lại các commit của nhánh này lên baseline mới; lịch sử tuyến tính nhưng đổi ID commit.

Nhánh tính năng cá nhân chưa chia sẻ có thể rebase; nhánh công khai đừng tùy tiện viết lại lịch sử. Giải xung đột không phải xóa marker một cách máy móc; phải hiểu ý hai phía, giữ tổ hợp đúng và chạy lại kiểm thử.

### 9.4 Một khung định vị tái sử dụng được

Khi gặp vấn đề, ghi theo thứ tự:

1. **Hành vi kỳ vọng**: lẽ ra phải xảy ra gì;
2. **Hành vi thực tế**: báo lỗi đầy đủ, mã thoát, stderr;
3. **Tái hiện tối thiểu**: lệnh ngắn nhất, input nhỏ nhất;
4. **Môi trường**: commit code, phiên bản dependency, hệ thống, phần cứng;
5. **Định vị biên**: lớp nào trong dữ liệu, Python, C++, CUDA, driver, mạng;
6. **Thí nghiệm một biến**: mỗi lần chỉ đổi một mục và ghi kết quả.

"Thỉnh thoảng OOM" không phải mô tả vấn đề khả thi; "commit abc123, batch size 8, độ dài chuỗi 4096, CUDA OOM ở step 23 khi cấp phát 1.2 GiB; batch size 4 không tái hiện" mới là.

---

## 10. Thực chiến: thêm một toán tử C++ vào Python

Dưới đây dùng `pybind11` triển khai scale theo phần tử, đi đủ đường nguồn, build, import và kiểm chứng. Ví dụ chỉ để hiện chuỗi xuyên ngôn ngữ; tính số thật nên ưu tiên thư viện vector hóa chín.

### 10.1 Cấu trúc dự án

```text
fast_scale/
├── CMakeLists.txt
├── scale.cpp
└── test_scale.py
```

### 10.2 Triển khai C++ và binding

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

Dùng `const std::vector<float>&` tránh sao chép lại input bên trong hàm, nhưng chuyển list Python thành `std::vector` vẫn tạo một lần chuyển biên; giá trị trả về cũng chuyển thành list Python. Đây không phải zero-copy.

### 10.3 Build bằng CMake

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

### 10.4 Kiểm chứng tính đúng

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

### 10.5 Từ ví dụ dạy học đến interface hiệu năng cao còn thiếu gì

Ví dụ này không hơn list comprehension Python, vì biên đã chuyển từng phần tử. Extension hiệu năng thật còn phải:

1. Nhận buffer liên tục NumPy/PyTorch thay vì list Python;
2. Kiểm dtype, shape, stride và device;
3. Tránh sao chép không cần thiết và làm rõ ownership;
4. Nhả GIL trong lúc tính nặng;
5. Cung cấp triển khai theo batch vector hóa CPU hoặc CUDA;
6. So với triển khai tham chiếu đáng tin và kiểm input biên;
7. Đo riêng chuyển biên và tính lõi.

Đó chính là các vấn đề các chương sau về extension PyTorch, CUDA Kernel và tối ưu toán tử sẽ tiếp tục giải quyết.

---

## Tóm tắt

Cốt lõi chương này không phải ba bộ cú pháp tách rời, mà một cách nghĩ xuyên lớp:

- Trong Python, nhìn rõ tham chiếu đối tượng, sao chép dữ liệu, mô hình đồng thời và biên interpreter/native;
- Trong C++, nhìn rõ layout bộ nhớ, vòng đời, ownership, biên dịch/liên kết và ABI;
- Trên Linux, nhìn rõ tiến trình, tài nguyên, đường dẫn, quyền, thư viện động và kế thừa môi trường;
- Vấn đề hiệu năng đo trước rồi mới tối ưu; vấn đề môi trường kiểm từng lớp dọc stack phần mềm;
- Interface tốt không chỉ mô tả "truyền giá trị gì", mà còn kiểu, shape, layout, device, ownership và vòng đời.

Khi học tiếp CUDA, PyTorch, huấn luyện phân tán và framework suy luận, bạn sẽ gặp lại các khái niệm này, chỉ là chúng hóa thành bộ nhớ GPU, Stream, Tensor, tiến trình giao tiếp và dạng phục vụ mô hình.

## Checklist tự kiểm

- [ ] Giải thích được khác biệt giữa `is` và `==`, giữa shallow copy và deep copy
- [ ] Giải thích được vì sao tham số mặc định khả biến bị dùng chung giữa các lần gọi
- [ ] Viết được decorator giữ metadata hàm và một context manager
- [ ] Chọn được thread, process hoặc I/O bất đồng bộ theo I/O bound hay CPU bound
- [ ] Biết vì sao thực thi GPU bất đồng bộ làm lệch đo thời gian CPU thông thường
- [ ] Phân biệt được ngữ nghĩa interface của pointer, reference, giá trị và `const` reference trong C++
- [ ] Giải thích được RAII, `unique_ptr`, `shared_ptr` mỗi cái giải quyết gì
- [ ] Giải thích được các giai đoạn tiền xử lý, biên dịch, assemble, liên kết và nạp động
- [ ] Dùng CMake build được một thư viện C++ tối thiểu
- [ ] Giải thích được vì sao shape, stride, dtype, device và lifetime thuộc interface xuyên ngôn ngữ
- [ ] Trên Linux tra được tiến trình, mức dùng tài nguyên, cổng và dependency thư viện động
- [ ] Phân biệt được NVIDIA Driver, CUDA Toolkit, CUDA Runtime và cuDNN
- [ ] Tạo được nhánh và commit Git nhỏ, rõ, và kiểm diff trước khi commit
- [ ] Định vị vấn đề bằng "tái hiện tối thiểu + snapshot môi trường + thí nghiệm một biến"

## Bài tập

1. Viết đoạn code tái hiện vấn đề shallow copy dùng chung list trong, rồi lần lượt sửa bằng tái cấu trúc dữ liệu và deep copy.
2. Triển khai decorator có tham số `@retry(max_attempts=3)`, chỉ bắt các kiểu ngoại lệ do phía gọi chỉ định.
3. Dùng thread pool và process pool lần lượt chạy tác vụ chờ I/O và tác vụ CPU thuần Python; ghi lại và giải thích chênh lệch thời gian.
4. Dùng `cProfile` phân tích một script, tìm ba hàm có cumulative time cao nhất; tối ưu một trong số đó và đo lại.
5. Viết class RAII C++ giữ file handle, cấm copy và hỗ trợ move.
6. Cố ý gây heap out-of-bounds; so sánh báo lỗi khi chạy thường và khi bật AddressSanitizer.
7. Tách một thư viện C++ thành header, file nguồn và chương trình thực thi; build bằng CMake theo target.
8. Dùng `ldd`/`otool -L` và `nm -C` phân tích dependency và symbol xuất khẩu của một extension native Python.
9. Trong môi trường ảo sạch, cài PyTorch; ghi `sys.executable`, phiên bản framework, phiên bản CUDA lúc build và khả năng thấy thiết bị.
10. Cải tiến phần thực chiến: nhận mảng NumPy, kiểm dtype và tính liên tục, rồi so thời gian giữa bản sao chép và bản dùng chung buffer.

## Tài liệu tham khảo

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
