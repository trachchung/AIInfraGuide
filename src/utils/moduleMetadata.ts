import type { CategorySlug } from './categories';
import type { Lang } from '../i18n/ui';

/** Text in every supported language. */
type T = Record<Lang, string>;

export interface ModuleResource {
  /** 资源类型标签，如「练习平台」 */
  type: T;
  /** 资源名称 */
  title: T;
  /** 外部链接 */
  href: string;
  /** 一句话说明 */
  description: T;
}

export interface ModuleMeta {
  title: T;
  description: T;
  /** 主题色（与 GuideSidebar 的 gradientMap 一致） */
  color: string;
  /** 学习建议 */
  tips: T[];
  /** 前置要求（简短描述） */
  prerequisites?: T;
  /** 延伸资源（站外链接） */
  resources?: ModuleResource[];
}

const PRACTICE: T = { zh: '练习平台', en: 'Practice', vi: 'Luyện tập' };

export const MODULE_METADATA: Partial<Record<CategorySlug, ModuleMeta>> = {
  prerequisites: {
    title: { zh: '模块一：前置知识', en: 'Module 1: Prerequisites', vi: 'Module 1: Kiến thức nền' },
    description: {
      zh: '涵盖 GPU 架构、编程语言基础、数学基础、Transformer 架构、PyTorch 框架和集合通信等核心前置知识，为后续深入 AI Infra 打好坚实基础。',
      en: 'GPU architecture, programming basics, the math you need, the Transformer architecture, PyTorch and collective communication. Everything later in the guide builds on this.',
      vi: 'Kiến trúc GPU, nền tảng lập trình, toán cần thiết, kiến trúc Transformer, PyTorch và giao tiếp tập thể (collective communication). Các phần sau của tài liệu đều dựa trên những kiến thức này.',
    },
    color: '#06B6D4',
    tips: [
      {
        zh: '建议按章节顺序学习，每章内容相对独立但有递进关系',
        en: 'Read the chapters in order. Each one stands on its own, but they build on each other.',
        vi: 'Nên học theo thứ tự chương. Mỗi chương khá độc lập nhưng chương sau dựa trên chương trước.',
      },
      {
        zh: '如果已有相关背景知识，可以跳过对应章节直接进入 CUDA 编程模块',
        en: 'If you already know a topic, skip that chapter and go straight to the CUDA module.',
        vi: 'Nếu đã nắm một chủ đề, bạn có thể bỏ qua chương đó và chuyển thẳng sang module CUDA.',
      },
      {
        zh: 'Transformer 架构和 PyTorch 框架是后续所有模块的基础，建议重点掌握',
        en: 'Every later module assumes you know the Transformer architecture and PyTorch well.',
        vi: 'Mọi module sau đều giả định bạn đã nắm chắc kiến trúc Transformer và PyTorch.',
      },
    ],
  },
  'cuda-optimization': {
    title: {
      zh: '模块二：CUDA 编程与算子优化',
      en: 'Module 2: CUDA programming and kernel optimization',
      vi: 'Module 2: Lập trình CUDA và tối ưu kernel',
    },
    description: {
      zh: '从 CUDA 编程入门到经典算子实现（Reduce、GEMM、Softmax、Attention），再到 AI 编译器和性能分析工具链，系统掌握 GPU 编程与算子优化技术。',
      en: 'Starts with CUDA basics, works through classic kernels (Reduce, GEMM, Softmax, Attention), then covers AI compilers and profiling tools.',
      vi: 'Bắt đầu từ CUDA cơ bản, đi qua các kernel kinh điển (Reduce, GEMM, Softmax, Attention), rồi đến trình biên dịch AI và công cụ profiling.',
    },
    color: '#F59E0B',
    tips: [
      {
        zh: '建议先完成前置知识模块中的 GPU 硬件概论',
        en: 'Finish the GPU hardware chapter in the prerequisites module first.',
        vi: 'Nên học xong chương phần cứng GPU trong module kiến thức nền trước.',
      },
      {
        zh: '每个算子实现章节都有配套代码，建议动手实践',
        en: 'Every kernel chapter comes with code. Run it and change it yourself.',
        vi: 'Mỗi chương về kernel đều có mã đi kèm. Hãy tự chạy và sửa thử.',
      },
      {
        zh: '性能分析工具链是贯穿整个模块的重要工具，建议尽早熟悉',
        en: 'The profiling tools come up throughout the module, so learn them early.',
        vi: 'Công cụ profiling được dùng xuyên suốt module, nên làm quen sớm.',
      },
    ],
    prerequisites: {
      zh: '前置知识模块（尤其是第5章 GPU 硬件概论）',
      en: 'The prerequisites module, especially chapter 5 on GPU hardware',
      vi: 'Module kiến thức nền, đặc biệt là chương 5 về phần cứng GPU',
    },
    resources: [
      {
        type: PRACTICE,
        title: {
          zh: 'CUDA 手撕算子 · 面试练习',
          en: 'Hand-written CUDA kernels · interview practice',
          vi: 'Tự viết kernel CUDA · luyện phỏng vấn',
        },
        href: 'https://kkcocoa.github.io/cuda-operator-interview/',
        description: {
          zh: '覆盖访存、Reduce、Softmax、Attention、GEMM 等算子的分级手写练习，附提示、自检清单与浏览器内代码编辑器',
          en: 'Graded exercises for memory access, Reduce, Softmax, Attention, GEMM and more, with hints, self-check lists and an in-browser code editor',
          vi: 'Bài tập theo cấp độ về truy cập bộ nhớ, Reduce, Softmax, Attention, GEMM và các kernel khác, có gợi ý, danh sách tự kiểm tra và trình soạn mã ngay trên trình duyệt',
        },
      },
      {
        type: PRACTICE,
        title: { zh: 'LeetGPU', en: 'LeetGPU', vi: 'LeetGPU' },
        href: 'https://leetgpu.com/',
        description: {
          zh: 'GPU 编程版的 LeetCode，按 easy/medium/hard 分级的算子题库，支持 CUDA、Triton、PyTorch 等多种写法，无需本地 GPU 即可提交运行，可查看 PTX/SASS 与性能排行',
          en: 'LeetCode for GPU programming. Kernel problems graded easy, medium and hard, solvable in CUDA, Triton, PyTorch and more. You can submit without a local GPU and inspect PTX/SASS and the performance leaderboard.',
          vi: 'LeetCode dành cho lập trình GPU. Bài tập kernel chia theo mức easy, medium, hard, giải được bằng CUDA, Triton, PyTorch và nhiều cách khác. Không cần GPU trên máy, có thể xem PTX/SASS và bảng xếp hạng hiệu năng.',
        },
      },
    ],
  },
  'distributed-training': {
    title: { zh: '模块三：分布式训练', en: 'Module 3: Distributed training', vi: 'Module 3: Huấn luyện phân tán' },
    description: {
      zh: '从分布式训练总论出发，深入数据并行、ZeRO 系列、张量并行、流水线并行、3D 并行策略，最终通过训练框架实战串联全部知识。',
      en: 'Starts with an overview, then covers data parallelism, the ZeRO family, tensor parallelism, pipeline parallelism and 3D parallelism, and ties them together with a training framework.',
      vi: 'Bắt đầu bằng phần tổng quan, sau đó đi vào song song dữ liệu, họ ZeRO, song song tensor, song song pipeline và song song 3D, cuối cùng ghép tất cả lại qua một framework huấn luyện.',
    },
    color: '#10B981',
    tips: [
      {
        zh: '分布式训练总论提供全局视角，建议首先阅读',
        en: 'Read the overview first. It gives you the big picture.',
        vi: 'Hãy đọc phần tổng quan trước để có cái nhìn toàn cảnh.',
      },
      {
        zh: 'ZeRO 系列是理解现代分布式训练的关键，建议重点学习',
        en: 'Spend extra time on ZeRO. Modern distributed training is hard to follow without it.',
        vi: 'Dành nhiều thời gian cho ZeRO. Thiếu nó sẽ khó hiểu huấn luyện phân tán hiện đại.',
      },
      {
        zh: '训练框架实战章节会将前面的理论知识串联起来',
        en: 'The training framework chapter connects the earlier theory.',
        vi: 'Chương thực hành với framework huấn luyện sẽ kết nối các phần lý thuyết trước đó.',
      },
    ],
    prerequisites: {
      zh: '前置知识模块（尤其是第6章 集合通信基础）',
      en: 'The prerequisites module, especially chapter 6 on collective communication',
      vi: 'Module kiến thức nền, đặc biệt là chương 6 về giao tiếp tập thể',
    },
  },
  'inference-optimization': {
    title: { zh: '模块四：推理优化', en: 'Module 4: Inference optimization', vi: 'Module 4: Tối ưu suy luận' },
    description: {
      zh: '覆盖 LLM 推理基础、推理引擎核心技术、主流推理框架、量化、Speculative Decoding、PD 解耦架构、端侧推理，以及性能分析与端到端实战。',
      en: 'LLM inference basics, inference engine internals, common frameworks, quantization, speculative decoding, prefill/decode disaggregation, on-device inference, profiling and end-to-end practice.',
      vi: 'Cơ bản về suy luận LLM, bên trong engine suy luận, các framework phổ biến, lượng tử hóa, speculative decoding, tách prefill/decode, suy luận trên thiết bị, profiling và thực hành từ đầu đến cuối.',
    },
    color: '#8B5CF6',
    tips: [
      {
        zh: 'LLM 推理基础章节帮助理解推理场景的独特挑战',
        en: 'The LLM inference basics chapter explains what makes inference different from training.',
        vi: 'Chương cơ bản về suy luận LLM giải thích suy luận khác huấn luyện ở điểm nào.',
      },
      {
        zh: '量化和 Speculative Decoding 是当前最热门的推理优化方向',
        en: 'Quantization and speculative decoding are where most inference work is happening now.',
        vi: 'Lượng tử hóa và speculative decoding là hai hướng tối ưu suy luận được làm nhiều nhất hiện nay.',
      },
      {
        zh: '建议结合主流推理框架（如 vLLM）进行实际部署练习',
        en: 'Practice deploying with a real inference framework such as vLLM.',
        vi: 'Hãy thực hành triển khai với một framework suy luận thật như vLLM.',
      },
      {
        zh: '端侧推理建议在真实设备上同时评估延迟、内存、功耗和精度',
        en: 'For on-device inference, measure latency, memory, power and accuracy on a real device.',
        vi: 'Với suy luận trên thiết bị, hãy đo độ trễ, bộ nhớ, điện năng và độ chính xác trên thiết bị thật.',
      },
    ],
    prerequisites: {
      zh: '前置知识模块 + CUDA 编程基础',
      en: 'The prerequisites module plus basic CUDA programming',
      vi: 'Module kiến thức nền và lập trình CUDA cơ bản',
    },
  },
};

/** category slug → URL 前缀映射 */
export const CATEGORY_URL_PREFIX: Partial<Record<CategorySlug, string>> = {
  prerequisites: '/prerequisites',
  'cuda-optimization': '/cuda',
  'distributed-training': '/distributed',
  'inference-optimization': '/inference',
};
