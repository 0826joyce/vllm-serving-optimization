- [LLM 推理优化深度自查与问答](#llm-推理优化深度自查与问答)
  - [目录](#目录)
  - [一、基础概念（必答题）](#一基础概念必答题)
    - [1. 大模型推理和训练的核心区别是什么？⭐](#1-大模型推理和训练的核心区别是什么)
    - [2. 推理阶段的性能瓶颈主要是什么？为什么不是算力瓶颈？⭐⭐](#2-推理阶段的性能瓶颈主要是什么为什么不是算力瓶颈)
    - [3. Decoder 推理流程是什么？Prefill 阶段和 Decode 阶段有什么差异？⭐](#3-decoder-推理流程是什么prefill-阶段和-decode-阶段有什么差异)
    - [4. 什么是推理时延（首包时延/生成时延）？P50/P99时延为什么重要？⭐](#4-什么是推理时延首包时延生成时延p50p99时延为什么重要)
    - [5. 影响推理吞吐和时延的核心因素有哪些？⭐⭐](#5-影响推理吞吐和时延的核心因素有哪些)
  - [二、vLLM 核心原理（必答题）](#二vllm-核心原理必答题)
    - [1. 详细讲一下 vLLM 一个请求从进入到返回的完整生命周期？⭐⭐](#1-详细讲一下-vllm-一个请求从进入到返回的完整生命周期)
    - [2. PagedAttention 的核心设计思想是什么？解决了什么问题？⭐⭐](#2-pagedattention-的核心设计思想是什么解决了什么问题)
    - [3. PagedAttention 对比传统 Attention，显存利用率提升的原理？⭐⭐](#3-pagedattention-对比传统-attention显存利用率提升的原理)
    - [4. SGLang 相比 vLLM 的核心优化点是什么？RadixAttention 是什么？⭐⭐](#4-sglang-相比-vllm-的核心优化点是什么radixattention-是什么)
    - [5. Chunked Prefill 解决了什么问题？适用什么场景？⭐⭐](#5-chunked-prefill-解决了什么问题适用什么场景)
    - [6. vLLM 的 Worker 和 Scheduler 是怎么交互的？⭐⭐](#6-vllm-的-worker-和-scheduler-是怎么交互的)
    - [7. vLLM 中 KV Cache 的分页大小默认是多少？修改分页大小会有什么影响？⭐](#7-vllm-中-kv-cache-的分页大小默认是多少修改分页大小会有什么影响)
    - [8. SGLang 的调度模型和 vLLM 有什么不同？⭐⭐](#8-sglang-的调度模型和-vllm-有什么不同)
  - [三、调度与批处理](#三调度与批处理)
    - [1. 什么是 Continuous Batching（连续批处理）？和静态批处理区别？⭐](#1-什么是-continuous-batching连续批处理和静态批处理区别)
    - [2. Continuous Batching 为什么能大幅提升 GPU 利用率？⭐](#2-continuous-batching-为什么能大幅提升-gpu-利用率)
    - [3. 动态批处理的批大小是怎么确定的？受哪些因素限制？⭐⭐](#3-动态批处理的批大小是怎么确定的受哪些因素限制)
    - [4. 长文本和短文本混合调度时，会出现什么问题？如何优化？⭐⭐](#4-长文本和短文本混合调度时会出现什么问题如何优化)
    - [5. 推理调度的优先级策略该如何设计？⭐⭐](#5-推理调度的优先级策略该如何设计)
    - [6. 结合你的网关/云存储调度经验，你会如何优化推理请求排队策略？⭐⭐](#6-结合你的网关云存储调度经验你会如何优化推理请求排队策略)
    - [7. 如何解决推理中的长尾请求阻塞问题？⭐⭐](#7-如何解决推理中的长尾请求阻塞问题)
    - [8. 高并发下，调度器如何避免 GPU 空闲或过载？⭐⭐](#8-高并发下调度器如何避免-gpu-空闲或过载)
  - [四、KV Cache \& 显存管理](#四kv-cache--显存管理)
    - [1. KV Cache 占用显存的计算公式是什么？⭐](#1-kv-cache-占用显存的计算公式是什么)
    - [2. 推理服务 OOM 的常见原因有哪些？排查步骤？⭐⭐](#2-推理服务-oom-的常见原因有哪些排查步骤)
    - [3. 如何实现 KV Cache 的复用/淘汰策略？⭐⭐⭐](#3-如何实现-kv-cache-的复用淘汰策略)
    - [4. 显存碎片是怎么产生的？PagedAttention 如何解决？⭐](#4-显存碎片是怎么产生的pagedattention-如何解决)
    - [5. 单卡部署 7B/13B 模型，KV Cache 最大能支持多少并发？⭐⭐](#5-单卡部署-7b13b-模型kv-cache-最大能支持多少并发)
    - [6. 如何限制单个请求的 KV Cache 占用？⭐](#6-如何限制单个请求的-kv-cache-占用)
    - [7. 多请求场景下，KV Cache 的分配和释放逻辑？⭐⭐](#7-多请求场景下kv-cache-的分配和释放逻辑)
  - [五、性能瓶颈分析 \& 工具](#五性能瓶颈分析--工具)
    - [1. 如何判断推理服务是带宽瓶颈还是调度瓶颈？⭐⭐](#1-如何判断推理服务是带宽瓶颈还是调度瓶颈)
    - [2. nvidia-smi 看推理性能，重点看哪些指标？⭐](#2-nvidia-smi-看推理性能重点看哪些指标)
    - [3. nsys profile 用于推理性能分析，能看到哪些关键信息？⭐⭐](#3-nsys-profile-用于推理性能分析能看到哪些关键信息)
    - [4. GPU 利用率低，但 QPS 上不去，可能是什么原因？⭐⭐](#4-gpu-利用率低但-qps-上不去可能是什么原因)
    - [5. 显存充足，但推理时延很高，问题出在哪里？⭐⭐](#5-显存充足但推理时延很高问题出在哪里)
    - [6. 如何定位推理服务的 CPU 瓶颈？⭐](#6-如何定位推理服务的-cpu-瓶颈)
  - [六、推理工程化 \& 服务部署](#六推理工程化--服务部署)
    - [1. 如何用 Docker 部署 vLLM 推理服务？关键配置有哪些？⭐](#1-如何用-docker-部署-vllm-推理服务关键配置有哪些)
    - [2. vLLM/SGLang 服务化部署，常用的 API 封装方式？⭐](#2-vllmsglang-服务化部署常用的-api-封装方式)
    - [3. 推理服务的压测工具和核心压测指标？⭐](#3-推理服务的压测工具和核心压测指标)
    - [4. 如何做推理服务的限流、熔断、负载均衡？（结合网关经验）⭐⭐](#4-如何做推理服务的限流熔断负载均衡结合网关经验)
    - [5. K8s 部署 GPU 推理任务，如何配置 GPU 资源、MIG 切分？⭐](#5-k8s-部署-gpu-推理任务如何配置-gpu-资源mig-切分)
    - [6. 推理服务的监控体系该如何搭建？监控哪些指标？⭐⭐](#6-推理服务的监控体系该如何搭建监控哪些指标)
    - [7. 推理服务重启、扩容、缩容的策略？⭐](#7-推理服务重启扩容缩容的策略)
    - [8. 如何解决推理服务的时延抖动问题？⭐⭐](#8-如何解决推理服务的时延抖动问题)
  - [七、分布式推理 \& 多卡部署](#七分布式推理--多卡部署)
    - [1. 推理中的 TP（张量并行）和 PP（流水线并行）的区别？⭐](#1-推理中的-tp张量并行和-pp流水线并行的区别)
    - [2. vLLM 如何开启多卡 TP 并行？核心配置参数？⭐](#2-vllm-如何开启多卡-tp-并行核心配置参数)
    - [3. NCCL 在推理多卡部署中起到什么作用？⭐](#3-nccl-在推理多卡部署中起到什么作用)
    - [4. 多卡推理的性能瓶颈可能出现在哪里？⭐⭐](#4-多卡推理的性能瓶颈可能出现在哪里)
    - [5. 多节点多卡部署推理服务，需要注意什么？⭐](#5-多节点多卡部署推理服务需要注意什么)
    - [6. 为什么推理一般不用 PP，多用 TP？⭐⭐](#6-为什么推理一般不用-pp多用-tp)
  - [八、量化与推理加速](#八量化与推理加速)
    - [1. INT4/INT8/FP8 量化对推理的作用是什么？⭐](#1-int4int8fp8-量化对推理的作用是什么)
    - [2. GPTQ/AWQ/FP8 量化的核心区别？⭐⭐](#2-gptqawqfp8-量化的核心区别)
    - [3. 量化后推理性能提升、显存降低的原理？⭐](#3-量化后推理性能提升显存降低的原理)
    - [4. vLLM/SGLang 如何对接量化模型？⭐](#4-vllmsglang-如何对接量化模型)
    - [5. 量化会带来什么问题？如何规避？⭐](#5-量化会带来什么问题如何规避)
  - [九、你做的优化深挖（核心加分项）](#九你做的优化深挖核心加分项)
    - [1. 你基于 vLLM/SGLang 做了哪些优化？具体改了什么逻辑？⭐⭐⭐](#1-你基于-vllmsglang-做了哪些优化具体改了什么逻辑)
    - [2. 你的优化解决了什么业务问题？量化指标是多少？⭐⭐⭐](#2-你的优化解决了什么业务问题量化指标是多少)
    - [3. 优化过程中遇到了什么问题？如何排查解决的？⭐⭐⭐](#3-优化过程中遇到了什么问题如何排查解决的)
    - [4. 如果让你优化推理吞吐，你会从哪几个维度入手？⭐⭐](#4-如果让你优化推理吞吐你会从哪几个维度入手)
    - [5. 如果让你降低 P99 时延，你的优化方案是什么？⭐⭐](#5-如果让你降低-p99-时延你的优化方案是什么)
    - [6. 你做的优化和框架原生逻辑相比，优势在哪里？⭐⭐](#6-你做的优化和框架原生逻辑相比优势在哪里)
    - [7. 有没有做过压测对比？如何保证测试数据的可信度？⭐⭐](#7-有没有做过压测对比如何保证测试数据的可信度)
  - [十、真实场景开放题](#十真实场景开放题)
    - [1. 场景：客服对话场景，短文本高并发，QPS 上不去，如何优化？⭐⭐](#1-场景客服对话场景短文本高并发qps-上不去如何优化)
    - [2. 场景：长文档摘要场景，频繁 OOM，时延极高，如何优化？⭐⭐](#2-场景长文档摘要场景频繁-oom时延极高如何优化)
    - [3. 场景：多模型混合部署，GPU 资源争抢，如何做资源隔离与调度？⭐⭐](#3-场景多模型混合部署gpu-资源争抢如何做资源隔离与调度)
    - [4. 场景：推理服务 GPU 利用率常年低于 60%，如何排查并提升？⭐⭐](#4-场景推理服务-gpu-利用率常年低于-60如何排查并提升)
    - [5. 场景：批量推理任务，优先保证吞吐还是时延？如何权衡？⭐](#5-场景批量推理任务优先保证吞吐还是时延如何权衡)
    - [6. 场景：边缘端 GPU 部署推理，资源受限，如何做轻量化优化？⭐](#6-场景边缘端-gpu-部署推理资源受限如何做轻量化优化)
  - [十一、补充：check.txt 未覆盖但你应该能答的题](#十一补充checktxt-未覆盖但你应该能答的题)
    - [A. Prefix Caching 原理深追](#a-prefix-caching-原理深追)
      - [A1. Prefix Caching 的 hash chain 是什么？为什么中间 break 后续全部 miss？⭐⭐⭐](#a1-prefix-caching-的-hash-chain-是什么为什么中间-break-后续全部-miss)
      - [A2. `_touch()` 在 Prefix Caching 中的关键作用是什么？⭐⭐](#a2-_touch-在-prefix-caching-中的关键作用是什么)
      - [A3. 为什么 vLLM V1 不做 block 去重？⭐⭐](#a3-为什么-vllm-v1-不做-block-去重)
      - [A4. 同一 scheduling step 内的请求如何共享缓存？⭐⭐](#a4-同一-scheduling-step-内的请求如何共享缓存)
    - [B. 投机解码链路深追](#b-投机解码链路深追)
      - [B1. vLLM V1 投机解码的完整数据流是什么？⭐⭐⭐](#b1-vllm-v1-投机解码的完整数据流是什么)
      - [B2. 你的后缀自动机和原生 N-gram Proposer 对比，核心优势在哪？⭐⭐](#b2-你的后缀自动机和原生-n-gram-proposer-对比核心优势在哪)
      - [B3. RejectionSampler 的 `cumprod` 是什么意思？为什么拒绝后全部拒绝？⭐⭐](#b3-rejectionsampler-的-cumprod-是什么意思为什么拒绝后全部拒绝)
    - [C. PD 分离深追](#c-pd-分离深追)
      - [C1. PD 分离为什么不提升吞吐？⭐⭐](#c1-pd-分离为什么不提升吞吐)
      - [C2. Consumer 端如何跳过 Prefill？跳过后 KV 数据从哪来？⭐⭐](#c2-consumer-端如何跳过-prefill跳过后-kv-数据从哪来)
      - [C3. PD 分离场景下 Prefix Cache 如何工作？⭐⭐](#c3-pd-分离场景下-prefix-cache-如何工作)
    - [D. 端到端场景深追](#d-端到端场景深追)
      - [D1. 你设计压测的 5 个 Phase 分别暴露了什么问题？⭐⭐](#d1-你设计压测的-5-个-phase-分别暴露了什么问题)
      - [D2. 取消请求后 KV Cache 会怎样？prefix 能否被后续请求复用？⭐⭐](#d2-取消请求后-kv-cache-会怎样prefix-能否被后续请求复用)
  - [十二、vLLM 项目优化要点 \& 自查](#十二vllm-项目优化要点--自查)
    - [vLLM 项目总览](#vllm-项目总览)
    - [A. 调度与资源管理优化（方向一）](#a-调度与资源管理优化方向一)
      - [A1. vLLM V1 调度器的 QoS 优先级设计原理是什么？`effective_priority` 怎么算？⭐⭐⭐](#a1-vllm-v1-调度器的-qos-优先级设计原理是什么effective_priority-怎么算)
      - [A2. MLFQ 四级队列的设计原理？请求如何在层级间流动？⭐⭐⭐](#a2-mlfq-四级队列的设计原理请求如何在层级间流动)
      - [A3. Token Rate Limiter 如何工作？和系统负载如何联动？⭐⭐](#a3-token-rate-limiter-如何工作和系统负载如何联动)
      - [A4. Cache-Aware 调度选取是怎么做的？为什么不直接选队首？⭐⭐](#a4-cache-aware-调度选取是怎么做的为什么不直接选队首)
      - [A5. Prefill 预算隔离是什么？为什么需要保护短请求？⭐⭐](#a5-prefill-预算隔离是什么为什么需要保护短请求)
      - [A6. 租户隔离如何实现？WFQ 权重怎么算？⭐⭐](#a6-租户隔离如何实现wfq-权重怎么算)
      - [A7. 过载管理的三板斧：准入控制、Deadline-Aware、SLA-Aware Preemption？⭐⭐⭐](#a7-过载管理的三板斧准入控制deadline-awaresla-aware-preemption)
    - [B. KV Cache 管理优化（方向二）](#b-kv-cache-管理优化方向二)
      - [B1. Segmented LRU 的双区设计原理？和普通 LRU 有什么区别？⭐⭐⭐](#b1-segmented-lru-的双区设计原理和普通-lru-有什么区别)
      - [B2. Preemption Cache Shield 的工作原理？为什么要部分释放？⭐⭐⭐](#b2-preemption-cache-shield-的工作原理为什么要部分释放)
      - [B3. Cache 版本管理如何实现？hit rate 下降时怎么自适应调整？⭐⭐⭐](#b3-cache-版本管理如何实现hit-rate-下降时怎么自适应调整)
    - [C. 投机解码优化（方向三）](#c-投机解码优化方向三)
      - [C1. `IncrementalSuffixAutomaton` 的在线构建原理？为什么选后缀自动机？⭐⭐⭐](#c1-incrementalsuffixautomaton-的在线构建原理为什么选后缀自动机)
      - [C2. `SuffixAutomatonProposer` 的 stateful 设计是什么？自适应降级怎么做？⭐⭐](#c2-suffixautomatonproposer-的-stateful-设计是什么自适应降级怎么做)
      - [C3. `AdaptiveSuffixProposer` 的多候选评分机制是什么？4 个权重怎么设计的？⭐⭐⭐](#c3-adaptivesuffixproposer-的多候选评分机制是什么4-个权重怎么设计的)
    - [D. PD 分离优化（方向四）](#d-pd-分离优化方向四)
      - [D1. `KVReceiveMonitor` 的工作原理？为什么需要超时安全网？⭐⭐](#d1-kvreceivemonitor-的工作原理为什么需要超时安全网)
      - [D2. `PDRouter` 的请求分类与路由策略是什么？⭐⭐⭐](#d2-pdrouter-的请求分类与路由策略是什么)
      - [D3. PD-Aware 调度钩子在 Scheduler 中如何工作？⭐⭐](#d3-pd-aware-调度钩子在-scheduler-中如何工作)
    - [E. 端到端验证框架](#e-端到端验证框架)
      - [E1. 你设计的 5 阶段压测分别暴露了什么问题？⭐⭐⭐](#e1-你设计的-5-阶段压测分别暴露了什么问题)
      - [E2. 4 个增量修复的落地顺序为什么这样设计？⭐⭐](#e2-4-个增量修复的落地顺序为什么这样设计)
  - [十三、SGLang 项目优化要点 \& 自查](#十三sglang-项目优化要点--自查)
    - [SGLang 项目总览](#sglang-项目总览)
    - [A. 调度与缓存协同优化（方向一）](#a-调度与缓存协同优化方向一)
      - [A1. SGLang 的 RadixCache 驱逐机制和 vLLM 有什么本质区别？⭐⭐](#a1-sglang-的-radixcache-驱逐机制和-vllm-有什么本质区别)
      - [A2. 你做的 AdaptiveStrategy 是什么？解决了什么问题？⭐⭐](#a2-你做的-adaptivestrategy-是什么解决了什么问题)
      - [A3. AdaptiveStrategy 的 `get_priority()` 在什么时候被调用？完整调用链是什么？⭐⭐⭐](#a3-adaptivestrategy-的-get_priority-在什么时候被调用完整调用链是什么)
      - [A4. SGLang 的调度策略有哪些？LPM 退化是什么现象？⭐⭐](#a4-sglang-的调度策略有哪些lpm-退化是什么现象)
      - [A5. CacheWarmingManager 为什么选择 Tree-only Warming 而不是 Full KV Warming？⭐⭐](#a5-cachewarmingmanager-为什么选择-tree-only-warming-而不是-full-kv-warming)
    - [B. 推测解码增强（方向二）](#b-推测解码增强方向二)
      - [B1. SGLang 的 N-gram 推测解码和 vLLM 有什么架构差异？⭐⭐](#b1-sglang-的-n-gram-推测解码和-vllm-有什么架构差异)
      - [B2. 你在 SGLang 做的后缀自动机 Proposer 和 vLLM 上的有什么不同？⭐⭐](#b2-你在-sglang-做的后缀自动机-proposer-和-vllm-上的有什么不同)
      - [B3. EAGLE 的 Bigram Key 机制是什么？为什么 EAGLE 需要 bigram？⭐⭐](#b3-eagle-的-bigram-key-机制是什么为什么-eagle-需要-bigram)
      - [B4. SGLang EAGLE Worker 的 draft-verify 完整流程是什么？⭐⭐⭐](#b4-sglang-eagle-worker-的-draft-verify-完整流程是什么)
      - [B5. C++ Trie 的异步插入是怎么工作的？为什么需要异步？⭐⭐](#b5-c-trie-的异步插入是怎么工作的为什么需要异步)
    - [C. Overlap 与 PD 分离深度优化（方向三）](#c-overlap-与-pd-分离深度优化方向三)
      - [C1. SGLang 的 Overlap 调度是怎么工作的？FutureMap 是什么？⭐⭐](#c1-sglang-的-overlap-调度是怎么工作的futuremap-是什么)
      - [C2. 你做的动态 Overlap 决策解决了什么问题？⭐⭐](#c2-你做的动态-overlap-决策解决了什么问题)
      - [C3. SGLang PD 分离的队列系统和 vLLM 有什么不同？⭐⭐⭐](#c3-sglang-pd-分离的队列系统和-vllm-有什么不同)
      - [C4. 你做的 CrossInstanceCacheSync 是怎么工作的？⭐⭐](#c4-你做的-crossinstancecachesync-是怎么工作的)
    - [D. SGLang 核心机制深度自查](#d-sglang-核心机制深度自查)
      - [D1. Radix Tree 的 `lock_ref` 和 vLLM 的 `ref_cnt` 有什么区别？⭐⭐](#d1-radix-tree-的-lock_ref-和-vllm-的-ref_cnt-有什么区别)
      - [D2. SGLang 的 `match_prefix()` 和 `insert()` 路径分别是什么？⭐⭐](#d2-sglang-的-match_prefix-和-insert-路径分别是什么)
      - [D3. SGLang 的 Scheduler 空闲检测机制是什么？你为什么选择在空闲期做缓存预热？⭐⭐](#d3-sglang-的-scheduler-空闲检测机制是什么你为什么选择在空闲期做缓存预热)
      - [D4. SGLang 的 In-batch prefix caching 是什么？⭐⭐](#d4-sglang-的-in-batch-prefix-caching-是什么)
  - [十四、vLLM 与 SGLang 交叉对比自查](#十四vllm-与-sglang-交叉对比自查)
    - [1. 两个框架的 KV Cache 管理设计有什么本质差异？⭐⭐⭐](#1-两个框架的-kv-cache-管理设计有什么本质差异)
    - [2. 两个框架的投机解码实现有什么架构差异？⭐⭐](#2-两个框架的投机解码实现有什么架构差异)
    - [3. 两个框架的调度器设计有什么差异？你分别做了什么优化？⭐⭐](#3-两个框架的调度器设计有什么差异你分别做了什么优化)
    - [4. 两个框架的 PD 分离实现有什么差异？⭐⭐⭐](#4-两个框架的-pd-分离实现有什么差异)
    - [5. 如果让你选一个框架做生产部署，你会选哪个？为什么？⭐⭐](#5-如果让你选一个框架做生产部署你会选哪个为什么)
    - [6. 两个框架的进程架构和进程间通信有什么差异？为什么 SGLang 三个进程反而通信开销更小？⭐⭐⭐](#6-两个框架的进程架构和进程间通信有什么差异为什么-sglang-三个进程反而通信开销更小)
  - [十五、全面性评估 \& 薄弱点分析](#十五全面性评估--薄弱点分析)
    - [check.txt 覆盖度评估](#checktxt-覆盖度评估)
    - [两个项目的互补优势](#两个项目的互补优势)
    - [建议补充的薄弱点](#建议补充的薄弱点)
    - [自查策略](#自查策略)
  - [十六、薄弱点技术要点详解](#十六薄弱点技术要点详解)
    - [16.1 实际 Profiling 经验（nsys / ncu）](#161-实际-profiling-经验nsys--ncu)
      - [16.1.1 核心原理](#1611-核心原理)
      - [16.1.2 nsys（Nsight Systems）—— 系统级时间线分析](#1612-nsysnsight-systems-系统级时间线分析)
      - [16.1.3 ncu（Nsight Compute）—— 单 Kernel 深度分析](#1613-ncunsight-compute-单-kernel-深度分析)
      - [16.1.4 vLLM 内置 Profiler](#1614-vllm-内置-profiler)
      - [16.1.5 BenchmarkMetrics（端到端性能指标）](#1615-benchmarkmetrics端到端性能指标)
      - [16.1.6 表达模板](#1616-表达模板)
    - [16.2 量化实操（GPTQ / AWQ / FP8）](#162-量化实操gptq--awq--fp8)
      - [16.2.1 核心原理](#1621-核心原理)
      - [16.2.2 vLLM 支持的量化方法总览](#1622-vllm-支持的量化方法总览)
      - [16.2.3 GPTQ Marlin 详解](#1623-gptq-marlin-详解)
      - [16.2.4 AWQ 详解](#1624-awq-详解)
      - [16.2.5 FP8 量化详解](#1625-fp8-量化详解)
      - [16.2.6 表达模板](#1626-表达模板)
    - [16.3 TP 并行实操（Tensor Parallelism）](#163-tp-并行实操tensor-parallelism)
      - [16.3.1 核心原理](#1631-核心原理)
      - [16.3.2 两种核心并行线性层](#1632-两种核心并行线性层)
        - [ColumnParallelLinear（列切分）](#columnparallellinear列切分)
        - [RowParallelLinear（行切分）](#rowparallellinear行切分)
        - [Transformer 中的组合方式](#transformer-中的组合方式)
      - [16.3.3 AllReduce 通信后端（7 级派发链）](#1633-allreduce-通信后端7-级派发链)
      - [16.3.4 表达模板](#1634-表达模板)
    - [16.4 CUDA Graph 原理](#164-cuda-graph-原理)
      - [16.4.1 核心原理](#1641-核心原理)
      - [16.4.2 vLLM V1 的 5 种 CUDA Graph 模式](#1642-vllm-v1-的-5-种-cuda-graph-模式)
      - [16.4.3 双模式嵌套架构](#1643-双模式嵌套架构)
      - [16.4.4 BatchDescriptor（Graph 匹配键）](#1644-batchdescriptorgraph-匹配键)
      - [16.4.5 CudagraphDispatcher（中央调度器）](#1645-cudagraphdispatcher中央调度器)
      - [16.4.6 AttentionCGSupport（注意力后端兼容性）](#1646-attentioncgsupport注意力后端兼容性)
      - [16.4.7 CUDA Graph 的限制](#1647-cuda-graph-的限制)
      - [16.4.8 表达模板](#1648-表达模板)
    - [16.5 FlashAttention 原理](#165-flashattention-原理)
      - [16.5.1 核心原理](#1651-核心原理)
        - [1. Tiling（分块计算）](#1-tiling分块计算)
        - [2. Online Softmax（在线 Softmax）](#2-online-softmax在线-softmax)
        - [3. Fused Kernel（算子融合）](#3-fused-kernel算子融合)
      - [16.5.2 vLLM 中的 Attention 后端体系](#1652-vllm-中的-attention-后端体系)
      - [16.5.3 后端优先级选择](#1653-后端优先级选择)
      - [16.5.4 FA 版本选择](#1654-fa-版本选择)
      - [16.5.5 PagedAttention 集成](#1655-pagedattention-集成)
      - [16.5.6 Cascade Attention（前缀共享优化）](#1656-cascade-attention前缀共享优化)
      - [16.5.7 表达模板](#1657-表达模板)
    - [16.6 GQA/MQA 对 KV Cache 的影响](#166-gqamqa-对-kv-cache-的影响)
      - [16.6.1 核心概念](#1661-核心概念)
      - [16.6.2 典型模型配置](#1662-典型模型配置)
      - [16.6.3 vLLM 中的 KV Head 数获取](#1663-vllm-中的-kv-head-数获取)
      - [16.6.4 KV Cache 显存计算公式](#1664-kv-cache-显存计算公式)
      - [16.6.5 GQA 对系统的综合影响](#1665-gqa-对系统的综合影响)
      - [16.6.6 MLA（Multi-Latent Attention）特殊情况](#1666-mlamulti-latent-attention特殊情况)
      - [16.6.7 表达模板](#1667-表达模板)
    - [16.7 vLLM V0 vs V1 差异](#167-vllm-v0-vs-v1-差异)
      - [16.7.1 架构级对比](#1671-架构级对比)
      - [16.7.2 V0 三阶段调度](#1672-v0-三阶段调度)
      - [16.7.3 V1 两阶段调度](#1673-v1-两阶段调度)
      - [16.7.4 V1 RequestStatus 枚举](#1674-v1-requeststatus-枚举)
      - [16.7.5 Preemption Cache Shield（抢占缓存护盾）](#1675-preemption-cache-shield抢占缓存护盾)
      - [16.7.6 V1 抢占受害者选择策略](#1676-v1-抢占受害者选择策略)
      - [16.7.7 为什么 V1 去掉了 Swap？](#1677-为什么-v1-去掉了-swap)
      - [16.7.8 表达模板](#1678-表达模板)
    - [16.8 HiRadixCache 实操（SGLang 三级缓存）](#168-hiradixcache-实操sglang-三级缓存)
      - [16.8.1 核心架构](#1681-核心架构)
      - [16.8.2 HiRadixCache 类继承](#1682-hiradixcache-类继承)
      - [16.8.3 工作流程](#1683-工作流程)
      - [16.8.4 写回策略（Write Policy）](#1684-写回策略write-policy)
      - [16.8.5 预取策略（Prefetch Strategy）](#1685-预取策略prefetch-strategy)
      - [16.8.6 L3 存储后端（7+ 种）](#1686-l3-存储后端7-种)
      - [16.8.7 关键优化技术](#1687-关键优化技术)
        - [1. 计算-传输重叠（Compute-Transfer Overlap）](#1-计算-传输重叠compute-transfer-overlap)
        - [2. GPU-Assisted IO Kernels](#2-gpu-assisted-io-kernels)
        - [3. MLA Write-Back 优化](#3-mla-write-back-优化)
      - [16.8.8 配置参数](#1688-配置参数)
      - [16.8.9 与 vLLM 的对比](#1689-与-vllm-的对比)
      - [16.8.10 表达模板](#16810-表达模板)

# LLM 推理优化深度自查与问答

> 基于两个项目的实际优化经验：
> - **vLLM V1 项目**：四大方向优化（调度/KV Cache/投机解码/PD 分离）+ 端到端压测
> - **SGLang 项目**：三大方向优化（调度与缓存协同/推测解码增强/Overlap 与 PD 分离深度优化）
> 
> 每个问题都标注了难度（⭐~⭐⭐⭐）和推荐的回答要点/角度，方便自查理解。

---

## 目录

- [一、基础概念（必答题）](#一基础概念必答题)
- [二、vLLM 核心原理（必答题）](#二vllm-核心原理必答题)
- [三、调度与批处理](#三调度与批处理)
- [四、KV Cache & 显存管理](#四kv-cache--显存管理)
- [五、性能瓶颈分析 & 工具](#五性能瓶颈分析--工具)
- [六、推理工程化 & 服务部署](#六推理工程化--服务部署)
- [七、分布式推理 & 多卡部署](#七分布式推理--多卡部署)
- [八、量化与推理加速](#八量化与推理加速)
- [九、你做的优化深挖（核心加分项）](#九你做的优化深挖核心加分项)
- [十、真实场景开放题](#十真实场景开放题)
- [十一、补充：check.txt 未覆盖但你应该能答的题](#十一补充checktxt-未覆盖但你应该能答的题)
- [十二、SGLang 项目优化要点 & 自查](#十二sglang-项目优化要点--自查)
- [十三、vLLM 与 SGLang 交叉对比自查](#十三vllm-与-sglang-交叉对比自查)
- [十四、全面性评估 & 薄弱点分析](#十四全面性评估--薄弱点分析)

---

## 一、基础概念（必答题）

### 1. 大模型推理和训练的核心区别是什么？⭐

**回答要点**：
- 训练：前向 + 反向传播，需要存储激活值和梯度，显存需求 >> 推理
- 推理：仅前向传播，自回归逐 token 生成
- 训练是**计算密集型**（大 batch × 全序列），推理是**访存密集型**（Decode 阶段每步仅 1 token，但需读取完整 KV Cache）
- 关键差异：推理的 Decode 阶段 Arithmetic Intensity 极低（~1 FLOP/byte），远低于 GPU 的计算/访存比

### 2. 推理阶段的性能瓶颈主要是什么？为什么不是算力瓶颈？⭐⭐

**回答要点**：
- **Prefill 阶段是计算密集型**（大量矩阵乘法，可以并行处理整个 prompt）
- **Decode 阶段是访存密集型**（每步只生成 1 token，但需要读取全部 KV Cache → 显存带宽瓶颈）
- 以 A100 为例：算力 312 TFLOPS，带宽 2TB/s，计算/带宽比 = 156 FLOP/byte
- Decode 阶段 Attention 的 Arithmetic Intensity ≈ 2（读 K+V，做 1 次乘 + 1 次加），远低于 156 → **带宽瓶颈**
- 这也是为什么 PD 分离有意义：Prefill 用大 TP 吃算力，Decode 用小 TP 吃带宽

### 3. Decoder 推理流程是什么？Prefill 阶段和 Decode 阶段有什么差异？⭐

**回答要点**：
| 维度 | Prefill | Decode |
|------|---------|--------|
| 输入 | 完整 prompt tokens | 上一步生成的 1 个 token |
| 并行度 | 高（所有 token 并行计算 QKV） | 低（仅 1 个 query token） |
| KV Cache | 写入（填充） | 读取（大量） + 追加写（1 token） |
| 瓶颈 | 计算（Compute-bound） | 访存（Memory-bandwidth-bound） |
| 延迟指标 | TTFT | ITL / TPOT |

### 4. 什么是推理时延（首包时延/生成时延）？P50/P99时延为什么重要？⭐

**回答要点**：
- **TTFT (Time To First Token)**：请求到达 → 第一个 token 生成。包含排队时间 + Prefill 计算时间
- **ITL / TPOT (Inter-Token Latency)**：连续两个 token 之间的间隔
- **P50/P99**：中位数和 99 分位延迟。P99 对用户体验影响更大（1% 的请求有极差体验）
- 实际经验：你的端到端压测中，Phase 5 过载时 P99 > 10s 但 P50 可能只有 2s → 均值无法反映真实情况
- **你的项目相关**：QoS 分级调度就是为了控制高优请求的 P99 TTFT

### 5. 影响推理吞吐和时延的核心因素有哪些？⭐⭐

**回答要点**：
- **模型侧**：模型大小、层数、head 数、KV head 数（GQA/MQA 对 KV Cache 大小有直接影响）
- **调度侧**：批处理策略（Continuous Batching）、token_budget 大小、调度优先级
- **显存侧**：KV Cache 容量（决定最大并发）、显存碎片、缓存命中率
- **计算侧**：Attention 后端（FlashAttention）、量化（FP8/INT4）、投机解码
- **架构侧**：TP/PP 并行度、PD 分离
- **你的项目覆盖**：调度侧（QoS/MLFQ/限速）、显存侧（Segmented LRU/Cache-Aware）、计算侧（后缀解码）、架构侧（PD 分离）

---

## 二、vLLM 核心原理（必答题）

### 1. 详细讲一下 vLLM 一个请求从进入到返回的完整生命周期？⭐⭐

**回答要点**（按 V1 架构 8 步走）：
1. **API Server 接收** HTTP 请求 → FastAPI 路由
2. **AsyncLLM.generate()** → InputProcessor tokenize + 多模态预处理 → 构造 EngineCoreRequest
3. **EngineCoreClient** 通过 ZMQ IPC 发送到独立的 **EngineCore 进程**
4. **Scheduler.add_request()** 加入 WAITING 队列
5. **Scheduler.schedule()** 每步：
   - 5a. 调度 RUNNING 请求（分配 token_budget，KV Cache blocks）
   - 5b. 如果 KV 不足 → 触发抢占（LIFO，Recompute）
   - 5c. 调度 WAITING 请求（get_computed_blocks → allocate_slots）
6. **Executor → Worker → GPUModelRunner.execute_model()**：模型前向 + 采样
7. **Scheduler.update_from_output()**：更新 num_computed_tokens，检查停止条件，释放已完成请求
8. **OutputProcessor** detokenize → 流式返回客户端

**加分**：提到 V1 的多进程架构（Engine 进程 + EngineCore 进程分离），ZMQ IPC 通信

### 2. PagedAttention 的核心设计思想是什么？解决了什么问题？⭐⭐

**回答要点**：
- **问题**：传统实现为每个请求分配连续的 KV Cache 空间 → 内存碎片 + 无法动态伸缩 + 浪费（max_tokens 预分配）
- **借鉴**：操作系统虚拟内存的分页机制
- **设计**：KV Cache 被分割成固定大小的 Block（默认 16 tokens），通过 Block Table 做逻辑→物理映射
- **收益**：
  - 消除内存碎片（任何空闲 block 都可分配）
  - 支持动态增长（按需分配）
  - 支持共享（多请求共享 block，ref_cnt 引用计数）
  - 支持 Prefix Caching（hash chain → 相同前缀的 block 复用）

### 3. PagedAttention 对比传统 Attention，显存利用率提升的原理？⭐⭐

**回答要点**：
- 传统方式：为每个请求预分配 max_seq_len × 2 × num_layers × num_heads × head_size 的连续 KV 空间 → 实际使用率可能只有 30-50%
- PagedAttention：按 block 粒度按需分配 → 利用率接近 100%
- **量化**：vLLM 论文实验显示内存浪费从 60-80% 降到 <4%

### 4. SGLang 相比 vLLM 的核心优化点是什么？RadixAttention 是什么？⭐⭐

**回答要点**：
- **RadixAttention**：用 Radix Tree（基数树）管理 KV Cache，天然支持前缀共享和增量插入
- 对比 vLLM 的 hash chain：Radix Tree 是结构化的（树形），hash chain 是扁平的（hash 表查找）
- SGLang 的调度模型：已内置 LPM（最长前缀匹配）和 DFS-Weight 缓存感知策略，调度器天然偏好缓存命中
- SGLang 还有 Overlap 调度（`event_loop_overlap()`）、HiRadixCache（GPU→CPU→Disk 三级层次化缓存）、完整的 PD 分离框架
- **你的实操经验**：
  - 在 vLLM 上做了 Cache-Aware Scheduling（MLFQ 层内按缓存命中率排序），在 SGLang 上量化对比了已有策略
  - 在 vLLM 上做了 Segmented LRU（flat block），在 SGLang 上做了 AdaptiveStrategy（Radix Tree 叶子节点 + 树深度感知）
  - SGLang 的 `TreeNode` 有 `hit_count`、`last_access_time`、`creation_time`、`priority` 等丰富元数据，比 vLLM 的 flat block 信息更多

### 5. Chunked Prefill 解决了什么问题？适用什么场景？⭐⭐

**回答要点**：
- **问题**：长 prompt 的 Prefill 会占满整个 GPU step，期间所有 Decode 请求停滞 → ITL 出现尖峰
- **解决**：将长 prompt 拆成多个 chunk，每个 chunk 与 Decode 请求共享 token_budget
- **V1 默认开启**：token_budget = max_num_batched_tokens，Prefill 和 Decode 在同一 step 混合执行
- **适用场景**：长文本场景（RAG、文档摘要），需要控制 Decode 尾延迟
- **你的项目相关**：Phase 4 长文档暴增时，Chunked Prefill 只能缓解不能解决，所以需要 Prefill 预算隔离

### 6. vLLM 的 Worker 和 Scheduler 是怎么交互的？⭐⭐

**回答要点**：
- Scheduler 在 EngineCore 进程中运行，Worker 在独立进程中运行
- 交互通过 **Executor** 层抽象（UniprocExecutor / MultiprocExecutor / RayExecutor）
- 数据流：`Scheduler.schedule()` → `SchedulerOutput` → `Executor.execute_model()` → `Worker.execute_model()` → `ModelRunnerOutput` → `Scheduler.update_from_output()`
- SchedulerOutput 包含：new_reqs、cached_reqs、num_scheduled_tokens、spec_decode_tokens、finished_req_ids
- Worker 通过共享内存 + 自定义 RPC 接收 SchedulerOutput

### 7. vLLM 中 KV Cache 的分页大小默认是多少？修改分页大小会有什么影响？⭐

**回答要点**：
- 默认 **16 tokens**
- 增大 block_size：
  - ✅ 减少 block table 管理开销（更少的 block 数）
  - ❌ 增加内存碎片（最后一个 block 可能浪费更多空间）
  - ❌ 降低 Prefix Caching 的匹配粒度（hash 是 per-block 的）
- 减小 block_size：
  - ✅ 更细粒度的匹配和复用
  - ❌ 更多的 block table 条目，管理开销更大
- **实际经验**：hash chain 的计算是逐 block 的，block 越大 hash 越少，缓存命中的"粒度"越粗

### 8. SGLang 的调度模型和 vLLM 有什么不同？⭐⭐

**回答要点**：
- vLLM V1：two-phase（WAITING → RUNNING），Chunked Prefill 默认开启，token_budget 共享
- SGLang：更轻量的调度，RadixAttention 原生支持缓存感知，调度器在选择请求时天然偏好缓存命中
- SGLang 已内置 6 种调度策略（LPM/DFS-Weight/FCFS/LOF/RANDOM/ROUTING-KEY），LPM 队列 > 128 时退化为 FCFS
- SGLang 有 Overlap 调度（`event_loop_overlap()`），默认开启 GPU/CPU 重叠执行
- SGLang 有 In-batch prefix caching：使用 `waiting_queue_radix_tree`（无实际 KV）检测队列内部前缀共享
- vLLM 更注重"框架的可扩展性"（Scheduler 接口化、Executor 抽象化），SGLang 更注重"端到端性能"
- **你的实操经验**：
  - 在 SGLang 上做了调度策略量化对比 Benchmark（5 种策略 × 4 种工作负载），发现 LPM 在共享前缀场景命中率最高但有退化问题
  - 在 SGLang 上实现了动态 Overlap 决策（`OverlapDecisionMaker`），基于 EMA 统计的 GPU/CPU 耗时比做细粒度调控

---

## 三、调度与批处理

### 1. 什么是 Continuous Batching（连续批处理）？和静态批处理区别？⭐

**回答要点**：
- **静态批处理**：凑齐一批请求 → 一起跑到结束 → 再凑下一批。短请求等长请求，GPU 大量空闲
- **Continuous Batching**：每个 step 检查是否有请求完成或新请求到达，动态调整 batch 组成
- 关键特性：请求完成立即释放资源，新请求可立即加入 → GPU 利用率大幅提升

### 2. Continuous Batching 为什么能大幅提升 GPU 利用率？⭐

**回答要点**：
- 消除了"短请求等长请求"的 padding 浪费
- Decode 阶段每步只有 1 token / 请求，大量请求可以在一个 batch 中并行 → 提升 batch 效率
- 配合 KV Cache 动态管理，资源按需分配和回收

### 3. 动态批处理的批大小是怎么确定的？受哪些因素限制？⭐⭐

**回答要点**：
- **token_budget**（`max_num_batched_tokens`）：一个 step 中所有请求的 scheduled tokens 总和上限
- **max_num_seqs**：最大并发请求数
- **KV Cache 容量**：每个请求需要 blocks，总 blocks 有限
- **你的项目相关**：token_budget 是调度的核心资源，MLFQ + QoS 就是在争夺 token_budget 的分配权

### 4. 长文本和短文本混合调度时，会出现什么问题？如何优化？⭐⭐

**回答要点**：
- **问题**：长文本 Prefill 消耗大量 token_budget → 挤压短文本 → 短文本 TTFT 飙升
- **Chunked Prefill**：分块处理长 prompt，每步只用部分 budget
- **你的优化（Prefill 预算隔离）**：
  - 为短请求预留 30% token_budget（`SHORT_BUDGET_RESERVE_RATIO`）
  - 限制同时 Prefill 的长文档数量（`MAX_CONCURRENT_LONG_PREFILL = 2`）
  - 效果：Phase 4 短对话 P99 TTFT 从 >500ms 降到 <200ms

### 5. 推理调度的优先级策略该如何设计？⭐⭐

**回答要点**（结合你的实现）：
- **多维优先级**：你实现了 QoS 分级调度
  - `effective_priority = api_priority + length_factor + waiting_decay`
  - api_priority：业务传入（金融客服 > 文档分析）
  - length_factor：短请求加分（交互型）
  - waiting_decay：等待时间越长，优先级自动提升（防饿死）
- **自适应降级**：MLFQ 4 级队列
  - L0(Interactive) → L1(Standard) → L2(Batch) → L3(Background)
  - 请求按实际 token 消耗自动降级
  - 高层级优先于低层级

### 6. 结合你的网关/云存储调度经验，你会如何优化推理请求排队策略？⭐⭐

**回答要点**：
- **网关经验**：优先级队列 + 高优包优先转发 → 映射为 QoS 分级调度
- **存储经验**：令牌桶限速 + IO 配额 → 映射为 Token 限速 + 租户资源隔离
- **网络拥塞控制**：ECN/RED 随机早期丢弃 → 映射为准入控制（过载时主动拒绝低优请求）
- **EDF 调度**：网络 fq_codel / HFSC → 映射为 Deadline-aware 调度（按 SLA 剩余时间排序）
- **分层存储**：热温冷分级 → 映射为 KV Cache 分层（GPU → CPU → Disk）

### 7. 如何解决推理中的长尾请求阻塞问题？⭐⭐

**回答要点**：
- **MLFQ 自动降级**：长请求生成过多 tokens 后自动降到低层级，不再阻塞短请求
- **Token 限速**：高负载时低优请求每步生成 tokens 受限（rate=8-50 tokens/step）
- **抢占机制**：KV Cache 不足时抢占最低优先级的 RUNNING 请求
- **你的优化（SLA-Aware 抢占）**：已违约的请求优先被抢占 → 释放资源给还有救的请求

### 8. 高并发下，调度器如何避免 GPU 空闲或过载？⭐⭐

**回答要点**：
- **防空闲**：Continuous Batching 保证有请求就会被调度；WAITING 队列不为空时立即调度
- **防过载**：
  - KV Cache 水位线流控（block 使用率过高时暂停接收新请求）
  - 准入控制（队列深度 > 阈值时拒绝低优请求）
  - Token 限速（限制低优请求的 token 生成速率，为高优留 budget）
  - 你的端到端 Phase 5 验证了过载管理的必要性

---

## 四、KV Cache & 显存管理

### 1. KV Cache 占用显存的计算公式是什么？⭐

**回答要点**：
```
KV Cache 显存 = 2 × num_layers × num_kv_heads × head_size × seq_len × batch_size × dtype_size
               ↑                                            ↑
               K 和 V 各一份                                 总 token 数
```
- 示例：Llama-3 8B（32 layers × 8 KV heads × 128 head_size × FP16）
- 单请求 4096 tokens：2 × 32 × 8 × 128 × 4096 × 2 bytes ≈ **512 MB**
- 这就是为什么 KV Cache 管理如此重要

### 2. 推理服务 OOM 的常见原因有哪些？排查步骤？⭐⭐

**回答要点**：
- **原因**：
  1. max_model_len 设置过大 → 预分配 KV Cache 过多
  2. 高并发请求同时 Prefill → KV Cache 突发分配
  3. 长请求占用大量 blocks 不释放
  4. 模型权重 + KV Cache + Activation 总和超出显存
- **排查步骤**：
  1. `nvidia-smi` 看显存峰值和使用曲线
  2. vLLM `/metrics` 看 `gpu_cache_usage_perc`
  3. 检查 `max_model_len` × `max_num_seqs` 的理论 KV Cache 上界
  4. 检查是否有长请求未被抢占（抢占逻辑是否生效）
- **你的经验**：Preemption Cache Shield 可以避免抢占时全量释放，减少"释放后分配不回来"的 OOM 风险

### 3. 如何实现 KV Cache 的复用/淘汰策略？⭐⭐⭐

**回答要点**（结合你的 Segmented LRU 实现）：
- **vLLM 原生**：纯 LRU（FreeKVCacheBlockQueue 双向链表），ref_cnt=0 时进入 free queue，从头部驱逐
- **问题**：LRU 是"频率盲"的 → 高频 System Prompt block 可能被低频长请求驱逐
- **你的优化（Segmented LRU）**：
  ```
  ┌──────────────────┬───────────────────────┐
  │  Probation Zone  │   Protected Zone      │
  │  新释放的 block    │   高频访问的 block      │
  │  ← 优先驱逐       │   ← 最后驱逐           │
  └──────────────────┴───────────────────────┘
  ```
  - block 首次释放 → Probation；被 `_touch()` 命中 → promote 到 Protected
  - 驱逐优先从 Probation 取，Protected 满了才降级
  - 所有操作 O(1)（双链表 + zone 标记位）

### 4. 显存碎片是怎么产生的？PagedAttention 如何解决？⭐

**回答要点**：
- 传统方式：连续分配 → 请求 A 释放后留下空洞 → 请求 B 需要更大空间时找不到连续空间 → 碎片
- PagedAttention：所有 block 大小相同且不要求连续 → **不存在外部碎片**
- 唯一的浪费：最后一个 block 的内部碎片（部分 slot 未使用）→ 最多浪费 block_size - 1 tokens

### 5. 单卡部署 7B/13B 模型，KV Cache 最大能支持多少并发？⭐⭐

**回答要点**：
- 计算方法：`可用 KV Cache 显存 / 单请求 KV Cache 大小`
- 示例（A100 80GB，Llama-3 8B FP16）：
  - 模型权重：~16 GB
  - 其他（Activation、临时 buffer 等）：~4 GB
  - 可用 KV Cache：80 - 16 - 4 = 60 GB
  - 单请求 2048 tokens：~256 MB
  - 最大并发 ≈ 60 GB / 256 MB ≈ **240** 个请求
- 实际还要考虑 Prefix Caching 的共享效果（共享 block 不重复占用）

### 6. 如何限制单个请求的 KV Cache 占用？⭐

**回答要点**：
- `max_model_len`：限制单请求最大序列长度
- `max_tokens`（SamplingParams）：限制单请求最大生成 token 数
- 抢占机制：KV Cache 不足时抢占低优先级请求，释放 blocks
- **你的项目**：Token 限速可以间接限制 → 低优请求生成速率降低 → KV Cache 增长更慢

### 7. 多请求场景下，KV Cache 的分配和释放逻辑？⭐⭐

**回答要点**（基于你对 V1 源码的深入分析）：
- **分配**：
  1. `get_computed_blocks()` → hash chain 查找缓存命中
  2. `allocate_slots()` → `_touch(computed_blocks)` 共享已有 block（ref_cnt++）→ `_get_new_blocks()` 分配新 block → `_cache_full_blocks()` 同步注册 hash
- **释放**：
  1. 请求完成 → `free(request)` → 逆序 `decr_ref()`
  2. ref_cnt 降为 0 → block 进入 free queue（但 hash 保留！）
  3. 后续请求仍可通过 hash 命中该 block（`_touch()` 将其从 free queue 救回）
- **关键**：`ref_cnt == 0` ≠ 数据丢失，只是成为驱逐候选；hash 保留是 Prefix Caching 的核心

---

## 五、性能瓶颈分析 & 工具

### 1. 如何判断推理服务是带宽瓶颈还是调度瓶颈？⭐⭐

**回答要点**：
- **带宽瓶颈**：
  - `nvidia-smi` 显示 GPU 利用率高（>80%），但 QPS 上不去
  - `nsys profile` 看到 HBM read bandwidth 接近上限
  - Decode 阶段耗时占比极高
- **调度瓶颈**：
  - GPU 利用率低（<50%），但 WAITING 队列堆积
  - CPU 侧 Scheduler.schedule() 耗时占比高
  - 增大 max_num_seqs 后 GPU 利用率提升 → 说明之前调度不饱和

### 2. nvidia-smi 看推理性能，重点看哪些指标？⭐

**回答要点**：
- **GPU-Util**：GPU 计算单元利用率（Decode 时通常偏低是正常的）
- **Memory-Usage**：显存使用量（接近上限需要注意 OOM 风险）
- **Power**：功耗（间接反映计算负载）
- **SM 利用率**：nsys/ncu 级别更准确
- 注意：nvidia-smi 的采样间隔较粗（默认 1s），高频波动可能看不到

### 3. nsys profile 用于推理性能分析，能看到哪些关键信息？⭐⭐

**回答要点**：
- **Kernel 耗时**：各 CUDA kernel 的执行时间（Attention、GEMM、采样）
- **HBM 带宽利用率**：是否接近硬件上限
- **CPU-GPU 同步开销**：是否有不必要的 sync point
- **Kernel Launch 延迟**：CPU 端发 kernel 的延迟
- **推理场景关键**：看 Attention kernel vs GEMM kernel 的耗时比例 → 判断瓶颈类型

### 4. GPU 利用率低，但 QPS 上不去，可能是什么原因？⭐⭐

**回答要点**：
- **调度瓶颈**：max_num_seqs 太小，GPU 每步 batch 过小
- **CPU 瓶颈**：tokenize/detokenize/调度器计算耗时过高
- **网络 I/O 瓶颈**：请求到达速率不够（客户端发送慢）
- **KV Cache 瓶颈**：blocks 不足导致大量请求 WAITING
- **PD 分离场景**：KV 传输延迟过高导致 Decode 实例空闲

### 5. 显存充足，但推理时延很高，问题出在哪里？⭐⭐

**回答要点**：
- **长请求堵塞**：一个长 Prefill 请求占满 token_budget → 其他请求排队
- **Prefix Cache 未命中**：每个请求都要全量 Prefill
- **抢占震荡**：频繁抢占+恢复+再抢占 → 大量无效计算
- **CUDA Graph 失效**：batch size 不稳定导致频繁 eager mode 回退

### 6. 如何定位推理服务的 CPU 瓶颈？⭐

**回答要点**：
- Python profiler（cProfile / py-spy）看 Scheduler.schedule() 耗时
- 看 tokenize/detokenize 耗时（特别是长 prompt）
- 看 ZMQ IPC 通信延迟
- 你的项目中：Cache-Aware Scheduling 的扫描窗口（K=8）就是为了控制 CPU 开销

---

## 六、推理工程化 & 服务部署

### 1. 如何用 Docker 部署 vLLM 推理服务？关键配置有哪些？⭐

**回答要点**：
- Docker 需要 `--gpus all` + NVIDIA Container Runtime
- 关键参数：`--max-model-len`、`--max-num-batched-tokens`、`--tensor-parallel-size`、`--enable-prefix-caching`
- 存储：模型权重挂载（NFS/S3 缓存）
- 网络：端口映射 + 健康检查端点 `/health`

### 2. vLLM/SGLang 服务化部署，常用的 API 封装方式？⭐

**回答要点**：
- vLLM 内置 OpenAI Compatible API（`/v1/chat/completions`、`/v1/completions`）
- 也支持 gRPC、MCP 等
- 前端通常加一层 API Gateway（限流、鉴权、路由）
- **你的项目**：PD Router 就是在 vLLM API 前面加了智能路由层

### 3. 推理服务的压测工具和核心压测指标？⭐

**回答要点**：
- 工具：vLLM 自带 `benchmarks/`、ShareGPT 数据集、自定义 workload
- 核心指标：TTFT（P50/P95/P99）、ITL（P50/P95/P99）、Throughput（tokens/s）、SLA 违约率
- **你的项目**：workload.py 是你自己设计的端到端压测，5 阶段递进加压 + 3 种部署模式

### 4. 如何做推理服务的限流、熔断、负载均衡？（结合网关经验）⭐⭐

**回答要点**：
- **限流**：你实现了 Token 限速（令牌桶）+ 准入控制（队列深度 + SLA 违约率门控）
- **熔断**：PD Router 的健康监控 → 实例不健康时自动摘除 → 故障转移
- **负载均衡**：PD Router 的负载感知路由（基于 `/metrics` 的 load_score 选择最轻实例）
- 映射：网关的 rate limiter → Token 限速，health check → HealthMonitor，least-conn → load_score 路由

### 5. K8s 部署 GPU 推理任务，如何配置 GPU 资源、MIG 切分？⭐

**回答要点**：
- `nvidia.com/gpu: 1` resource limit
- MIG：将一块 A100 切成多个独立 GPU instance（如 7×1g.10gb）
- 推理场景：通常不用 MIG（推理需要完整 GPU 带宽），MIG 更适合多模型小模型部署
- TP > 1 时需要多 GPU 在同一 Pod 内（`nvidia.com/gpu: N`）

### 6. 推理服务的监控体系该如何搭建？监控哪些指标？⭐⭐

**回答要点**：
- vLLM 自带 Prometheus `/metrics` 端点
- 核心指标：
  - 请求级：TTFT、ITL、throughput
  - 系统级：GPU 利用率、显存使用率、KV Cache 使用率
  - 调度级：WAITING 队列深度、RUNNING 数量、抢占频率
  - 缓存级：Prefix Cache 命中率、驱逐率
- **你的优化**：你设计了 EnhancedPrefixCacheStats（token 级节省量、健康驱逐率、抢占影响量化）

### 7. 推理服务重启、扩容、缩容的策略？⭐

**回答要点**：
- 重启：优雅停机（drain 正在处理的请求，等待完成后再停）
- 扩容：水平扩容（增加 vLLM 实例）+ 前端 LB 更新
- 缩容：标记节点为 draining → 不接收新请求 → 现有请求处理完后释放
- **PD 分离场景**：可以独立扩容 Prefill 或 Decode 实例（你设计了 PDOrchestrator 的 auto_scale）

### 8. 如何解决推理服务的时延抖动问题？⭐⭐

**回答要点**：
- **调度侧**：MLFQ + QoS 保障高优稳定；Token 限速防止低优抢资源
- **缓存侧**：Segmented LRU 防止缓存震荡；缓存版本管理防止 Prompt 切换导致命中率骤降
- **系统侧**：CUDA Graph 减少 kernel launch 开销；Python GC 调优
- **架构侧**：PD 分离彻底消除 Prefill 对 Decode 的干扰

---

## 七、分布式推理 & 多卡部署

### 1. 推理中的 TP（张量并行）和 PP（流水线并行）的区别？⭐

**回答要点**：
| 维度 | TP | PP |
|------|----|----|
| 切分方式 | 同一层的权重切分到多 GPU | 不同层分到不同 GPU |
| 通信 | 每层做 AllReduce（延迟敏感） | 前向传播跨 stage（pipeline bubble） |
| 适用 | 推理主流（降低单层延迟） | 训练为主（推理中较少用） |
| 并行粒度 | Layer 内 | Layer 间 |

### 2. vLLM 如何开启多卡 TP 并行？核心配置参数？⭐

**回答要点**：
- `--tensor-parallel-size N`（N = GPU 数量）
- 模型权重自动按 TP 策略切分
- 底层通过 NCCL AllReduce 通信
- 要求：N 块 GPU 在同一节点内，NVLink 连接最优

### 3. NCCL 在推理多卡部署中起到什么作用？⭐

**回答要点**：
- NCCL（NVIDIA Collective Communications Library）提供 GPU 间高效通信原语
- TP 推理中：每个 Transformer 层的 Attention 和 FFN 计算完后，需要 AllReduce 聚合结果
- 通信效率直接影响推理延迟（NVLink >> PCIe >> 网络）

### 4. 多卡推理的性能瓶颈可能出现在哪里？⭐⭐

**回答要点**：
- **通信瓶颈**：AllReduce 延迟，特别是 PCIe 连接时
- **负载不均**：某些层计算量不同导致 GPU 空闲等待
- **同步开销**：TP 要求所有 GPU 每层同步
- **KV Cache 对齐**：每个 GPU 分片只存部分 KV heads，需要正确映射

### 5. 多节点多卡部署推理服务，需要注意什么？⭐

**回答要点**：
- 跨节点通信延迟远高于节点内（RDMA/InfiniBand >> Ethernet）
- 通常跨节点用 PP，节点内用 TP
- 需要配置 `NCCL_SOCKET_IFNAME`、`NCCL_IB_DISABLE` 等环境变量
- vLLM 支持 Ray 分布式 Executor 做跨节点部署

### 6. 为什么推理一般不用 PP，多用 TP？⭐⭐

**回答要点**：
- PP 有 pipeline bubble（micro-batch 间有空闲），推理 batch size 小时 bubble 比例很高
- TP 的 AllReduce 开销在 NVLink 下很小（~100μs），而 PP 的 stage 间通信延迟更高
- 推理追求低延迟，TP 可以降低单层的计算延迟（并行度直接除以 TP）
- PP 更适合训练（大 batch 可以填充 pipeline）

---

## 八、量化与推理加速

### 1. INT4/INT8/FP8 量化对推理的作用是什么？⭐

**回答要点**：
- 减少模型权重体积 → 更少的显存占用 → 更大的 KV Cache 空间 → 更高并发
- 加速矩阵乘法（FP8 GEMM 在 H100 上吞吐翻倍）
- 降低显存带宽需求（Decode 阶段尤为关键）

### 2. GPTQ/AWQ/FP8 量化的核心区别？⭐⭐

**回答要点**：
| 方法 | 精度 | 特点 | 适用场景 |
|------|------|------|---------|
| GPTQ | INT4/INT8 | 逐层量化+逆量化，基于 Hessian | 离线量化，通用 |
| AWQ | INT4 | Activation-aware，保护重要通道 | 质量更优，略慢 |
| FP8 | E4M3/E5M2 | 硬件原生支持（H100+），在线量化 | 速度最快，精度损失小 |

### 3. 量化后推理性能提升、显存降低的原理？⭐

**回答要点**：
- 权重体积缩小 N 倍 → 显存读取量降低 N 倍 → Decode 阶段（访存密集）直接加速
- FP16 → INT4：权重体积 ÷ 4 → 理论 4× 带宽加速（实际受反量化开销影响，~2-3×）
- 显存节省让更多空间给 KV Cache → 支持更大 batch → 吞吐提升

### 4. vLLM/SGLang 如何对接量化模型？⭐

**回答要点**：
- vLLM 自动检测模型的 `quantization_config`（GPTQ/AWQ/FP8）
- 启动参数：`--quantization gptq/awq/fp8`
- 量化后的 KV Cache 也可以用 FP8（`--kv-cache-dtype fp8`）→ KV 占用减半

### 5. 量化会带来什么问题？如何规避？⭐

**回答要点**：
- 精度损失：特别是 INT4 在长文本/复杂推理场景
- Outlier 问题：部分 activation 值极大，INT8 量化会截断
- 规避：AWQ 保护重要通道、混合精度（重要层保持 FP16）、Calibration 数据集选择

---

## 九、你做的优化深挖（核心加分项）

> 这是最关键的环节。每个问题都可能被追问到代码级细节。

### 1. 你基于 vLLM/SGLang 做了哪些优化？具体改了什么逻辑？⭐⭐⭐

**回答框架**（两个项目：vLLM 4 大方向 + SGLang 3 大方向 + 端到端验证）：

```
═══ vLLM V1 项目 ═══

方向一：调度与资源管理（3 项已实现 + 5 项设计）
  ✅ QoS 分级调度（effective_priority 多维计算）
  ✅ Token 限速（Per-request 令牌桶）
  ✅ MLFQ 多级反馈（4 级队列自适应降级）
  🔲 KV 水位线流控、准入控制、Deadline/EDF、WFQ、分层存储

方向二：KV Cache 管理（3 项已实现 + 2 项设计）
  ✅ Cache-Aware Scheduling（MLFQ 层内按缓存命中率排序）
  ✅ Segmented LRU（probation/protected 分区，高频前缀保护）
  ✅ Preemption Cache Shield（部分释放 + 前缀保留）
  🔲 缓存预热、可观测性

方向三：投机解码（3 项已实现 + 2 项设计）
  ✅ SuffixTreeProposer（后缀数组替换 KMP）
  ✅ 增量后缀自动机（O(1) 在线追加）
  ✅ 自适应匹配+多候选评分（AcceptanceTracker 反馈闭环）
  🔲 跨请求共享池、可观测性

方向四：PD 分离（3 项已实现 + 3 项设计）
  ✅ V1 引擎 PD 基础适配（GPUModelRunner KV 收发）
  ✅ 智能请求路由（负载感知 + 故障转移）
  ✅ 调度器 PD 感知（KVReceiveMonitor + 超时安全网）
  🔲 传输优化、Prefix Cache 协同、多实例协调

端到端验证：
  ✅ 5 阶段递进压测 + 4 项增量修复（版本管理/Prefill 隔离/租户隔离/过载管理）
  ✅ 3 种部署模式对比（单实例/PD 分离/投机解码）

═══ SGLang 项目 ═══

方向一：调度与缓存协同优化（3 项已实现）
  ✅ Adaptive Eviction（Radix Tree 多因子自适应驱逐：recency + frequency + depth）
  ✅ 调度策略量化对比 Benchmark（5 策略 × 4 工作负载）
  ✅ 缓存预热 CacheWarmingManager（Tree-only Warming，空闲期非阻塞预热）

方向二：推测解码增强（2 项已实现 + 1 项设计）
  ✅ N-gram SAM Proposer（后缀自动机 + C++ Trie 组合提案）
  ✅ EAGLE + RadixCache 协同（verified tokens KV 写入 RadixCache 复用）
  🔲 推测解码全链路可观测性

方向三：Overlap 与 PD 分离深度优化（2 项已实现 + 1 项设计）
  ✅ 动态 Overlap 决策（OverlapDecisionMaker，EMA 耗时比自适应）
  ✅ PD 跨实例缓存协同（CrossInstanceCacheSync 哈希注册表）
  🔲 端到端性能分析 Benchmark 框架
```

### 2. 你的优化解决了什么业务问题？量化指标是多少？⭐⭐⭐

**回答要点**（预期指标）：

| 场景 | 指标 | 优化前 | 优化后 |
|------|------|--------|--------|
| 稳态（Phase 1） | Gold-A P99 TTFT | ~200ms | < 200ms ✅（Cache-Aware 降低） |
| Prompt 切换（Phase 2） | 恢复时间 | 30-60s | < 5s（缓存版本管理） |
| 流量暴增（Phase 3） | Silver P99 TTFT | > 2000ms | < 500ms（租户隔离） |
| 长文档暴增（Phase 4） | 短对话 P99 TTFT | > 500ms | < 200ms（Prefill 预算隔离） |
| 全面过载（Phase 5） | 接受请求 P99 | > 10s | < 800ms（过载管理） |
| 全面过载（Phase 5） | 合理拒绝率 | 0%（全违约） | 30-40% |
| 高频缓存命中率 | System Prompt 命中率 | ~50-70% | ~85-95%（Segmented LRU） |
| 抢占恢复耗时 | Recompute 范围 | 全量 | 部分（前缀保留） |

### 3. 优化过程中遇到了什么问题？如何排查解决的？⭐⭐⭐

**可以准备的 3 个故事**：

**故事 1：Segmented LRU 的 promote 时机**
- 问题：最初在 `_touch()` 时直接做 zone 迁移 → 在高并发下 promote 和 popleft 存在竞态
- 分析：`_touch()` 需要先 remove + append → 两步操作中间如果被驱逐...
- 解决：改用 `_promoted` 标记位延迟迁移 → `_touch()` 只设标记，`free()` 时根据标记选择 zone

**故事 2：抢占部分释放的降级保障**
- 问题：如果被抢占请求的 blocks 几乎全部有 hash（刚 prefill 完），partial free 只释放 0-1 个 block → 抢占循环空转
- 分析：`would_free = len(blocks) - keep_count` 可能很小
- 解决：加入降级保障 → `would_free < _PREEMPT_MIN_FREE_BLOCKS` 时退化为全量释放

**故事 3：增量后缀自动机的上下文收缩**
- 问题：抢占后恢复的请求 context 变短了，但 SAM 是基于旧 context 构建的 → 查询结果错误
- 分析：`_prev_len[req_id] > len(context_token_ids)` → 增量追加逻辑出错
- 解决：检测收缩 → 自动重建 SAM

### 4. 如果让你优化推理吞吐，你会从哪几个维度入手？⭐⭐

**回答要点**：
1. **增大有效 batch size**：优化调度让更多请求同时 RUNNING
2. **减少无效计算**：Prefix Caching（Cache-Aware 调度）、投机解码（后缀树 Proposer）
3. **减少通信开销**：TP 并行度选择、PD 分离的 KV 传输优化
4. **模型压缩**：FP8 量化、KV Cache FP8
5. **Attention 优化**：FlashAttention、cascade attention
6. **系统优化**：CUDA Graph、减少 CPU-GPU 同步点

### 5. 如果让你降低 P99 时延，你的优化方案是什么？⭐⭐

**回答要点**：
1. **调度保障**：QoS 分级 + MLFQ → 高优请求优先 + 短请求优先
2. **资源预留**：Prefill 预算隔离 → 短请求不被长 Prefill 饿死
3. **缓存优化**：Cache-Aware 调度 → 高缓存命中的请求优先 → TTFT 降低
4. **过载管理**：准入控制 → 过载时拒绝低优，保障高优 SLA
5. **架构选择**：PD 分离 → ITL 完全不被 Prefill 干扰
6. **抖动消除**：Segmented LRU → 防止缓存震荡导致 TTFT 突增

### 6. 你做的优化和框架原生逻辑相比，优势在哪里？⭐⭐

**回答要点**：

**vLLM 优化 vs 原生：**

| 方面 | vLLM V1 原生 | 你的优化 | 优势 |
|------|-------------|---------|------|
| 调度 | FCFS | QoS + MLFQ + Cache-Aware | 多维度调度，感知业务优先级和缓存状态 |
| 驱逐 | LRU | Segmented LRU | 频率感知，高频前缀不被误驱逐 |
| 抢占 | 全量释放 + num_computed_tokens=0 | 部分释放 + 前缀保留 | 恢复代价降低 50-80% |
| 投机解码 | 固定 N-gram KMP | 后缀自动机 + 自适应匹配 | O(1) 增量更新，自适应回退，多候选评分 |
| PD 分离 | V1 不支持 | 完整 V1 适配 + 智能路由 | 从零到一，解决 Prefill/Decode 干扰问题 |
| 过载管理 | 无 | 准入控制 + Deadline + SLA-aware 抢占 | 过载时仍能保障高优 SLA |

**SGLang 优化 vs 原生：**

| 方面 | SGLang 原生 | 你的优化 | 优势 |
|------|-------------|---------|------|
| 驱逐策略 | 6 种单因子策略（LRU/LFU/FIFO 等） | AdaptiveStrategy（recency + frequency + depth） | 多因子综合，树结构感知，共享前缀保护 |
| 调度策略 | LPM/DFS-Weight 已有但无量化对比 | 系统性 Benchmark（5策略×4工作负载） | 明确各策略适用场景，发现 LPM 退化问题 |
| 缓存预热 | 完全被动（请求到达才查缓存） | CacheWarmingManager Tree-only Warming | 消除冷启动惩罚，调度策略立即生效 |
| N-gram 推测 | C++ Trie 固定窗口匹配 | SAM + Trie 组合（可变长度 + 跨请求） | 自适应回退，per-request 上下文匹配 |
| EAGLE 缓存 | verified tokens KV 不复用 | 写入 RadixCache 供后续请求命中 | 减少重复计算，提高缓存命中率 |
| Overlap 调度 | 全有或全无决策 | OverlapDecisionMaker（EMA 动态决策） | 小 batch 不做无谓 overlap，GPU 快 CPU 慢时自动切同步 |
| PD 缓存协同 | Prefill→Decode 无缓存信息传递 | CrossInstanceCacheSync 哈希注册表 | Decode 侧知道哪些前缀已缓存，优化调度 |

### 7. 有没有做过压测对比？如何保证测试数据的可信度？⭐⭐

**回答要点**：
- 设计了 5 阶段端到端压测（workload.py），模拟企业级多租户场景
- 每个阶段引入一种新压力，隔离变量
- 3 种部署模式（单实例/PD 分离/投机解码）同一套流量对比
- 每项修复后回归测试确认无退化
- 单元测试覆盖所有核心路径（40+ 测试用例）

---

## 十、真实场景开放题

### 1. 场景：客服对话场景，短文本高并发，QPS 上不去，如何优化？⭐⭐

**回答要点**：
- **诊断**：看 WAITING 队列是否堆积 → 调度瓶颈；看 GPU 利用率 → 是否 batch 不饱和
- **调度优化**：MLFQ L0（Interactive）优先 + QoS 高优
- **缓存优化**：客服场景有大量共享 System Prompt → 开启 Prefix Caching → Cache-Aware 调度让高命中请求优先
- **Segmented LRU**：保护 System Prompt 不被驱逐
- **增大 max_num_seqs**：短对话 KV Cache 小，可以承载更多并发
- **投机解码**：客服回复模板化 → 后缀匹配接受率高 → Decode 加速

### 2. 场景：长文档摘要场景，频繁 OOM，时延极高，如何优化？⭐⭐

**回答要点**：
- **OOM**：限制 `max_model_len` + 分 chunk 处理 + KV Cache 水位线流控
- **时延**：Chunked Prefill 分块处理 + Prefill 预算隔离（保护短请求）
- **PD 分离**：长文档 Prefill 在独立实例上 → 不影响 Decode
- **KV Cache 压缩**：FP8 KV Cache 减半显存占用
- **准入控制**：长文档数量过多时限制并发

### 3. 场景：多模型混合部署，GPU 资源争抢，如何做资源隔离与调度？⭐⭐

**回答要点**：
- **GPU 隔离**：MIG 切分（小模型）或独占 GPU（大模型）
- **K8s 调度**：GPU 亲和性 + Pod 反亲和性
- **实例级隔离**：每个模型独立 vLLM 实例 + 前端路由
- **共享实例（不推荐）**：如果必须共享 → 租户级资源隔离（你的 TenantManager 可以扩展到模型级）

### 4. 场景：推理服务 GPU 利用率常年低于 60%，如何排查并提升？⭐⭐

**回答要点**：
- **排查**：
  1. batch size 是否过小 → 增大 max_num_seqs
  2. Prefix Cache 命中率如何 → 开启 + Cache-Aware
  3. 是否频繁抢占 → 检查 KV Cache 使用率
  4. CPU 是否是瓶颈 → profiler 看 schedule() 耗时
- **提升**：
  1. 增大 batch → 更多请求并发
  2. 投机解码 → 每步产出更多有效 token
  3. 减少 Prefill 空闲等待 → 优化调度
  4. PD 分离 → Prefill 和 Decode 独立拉满各自的优势资源

### 5. 场景：批量推理任务，优先保证吞吐还是时延？如何权衡？⭐

**回答要点**：
- **批量任务优先吞吐**：增大 batch、关闭延迟优化、贪心调度
- **在线服务优先时延**：QoS 分级、限制 batch、Prefix Caching
- **混合场景（你的项目）**：MLFQ 区分 → 在线请求 L0 优先 → 批量请求 L2/L3 → Token 限速控制低优速率 → 吞吐和时延兼顾

### 6. 场景：边缘端 GPU 部署推理，资源受限，如何做轻量化优化？⭐

**回答要点**：
- INT4/INT8 量化 → 模型体积大幅缩小
- KV Cache FP8 → 显存占用减半
- 减小 block_size → 减少内存碎片
- 小 TP（通常单卡）→ 无通信开销
- 投机解码可能不适合（额外计算开销在边缘端放大）

---

## 十一、补充：check.txt 未覆盖但你应该能答的题

> 基于你做过的 4 个方向优化，以下是可能深追的问题。

### A. Prefix Caching 原理深追

#### A1. Prefix Caching 的 hash chain 是什么？为什么中间 break 后续全部 miss？⭐⭐⭐

**回答要点**：
- 每个 block 的 hash = `hash(parent_block_hash, block_content)`
- 链式依赖：Block 2 的 hash 依赖 Block 1 的 hash
- 如果 Block 1 miss → Block 2 的 parent_hash 不同 → Block 2 也必然 miss（即使内容相同）
- 这就是为什么 hash chain 遇到第一个 miss 就 break 的设计

#### A2. `_touch()` 在 Prefix Caching 中的关键作用是什么？⭐⭐

**回答要点**：
- 当一个 block 在 free queue 中（ref_cnt=0）但 hash 保留，新请求通过 hash 命中它
- `_touch()` 将它从 free queue 中**抢救**出来（remove + incr_ref）
- 如果没有 `_touch()`，这个 block 可能在下一次 `popleft()` 时被驱逐 → 缓存失效
- 你的 Segmented LRU 在 `_touch()` 时还设置 `_promoted=True` → 下次释放时进入 protected zone

#### A3. 为什么 vLLM V1 不做 block 去重？⭐⭐

**回答要点**：
- 源码注释说明：当 hash 冲突时不合并物理 block
- 原因：`_get_cached_block()` 总是返回第一个匹配的 block → 后续请求自然共享同一个物理 block → 去重带来的收益很小
- 好处：保持 block table 为 append-only → 简化 model_runner 的 block table 管理

#### A4. 同一 scheduling step 内的请求如何共享缓存？⭐⭐

**回答要点**：
- `_cache_full_blocks()` 在 `allocate_slots()` 内部**同步执行**
- 请求 A 的 prefill 完成后，hash 立即写入 `cached_block_hash_to_block`
- 同一步内请求 B 调用 `get_computed_blocks()` 时可以立即查到 A 注册的 hash → 共享 A 的物理 block
- 所以调度顺序很重要 → Cache-Aware Scheduling 让高命中率请求先调度

### B. 投机解码链路深追

#### B1. vLLM V1 投机解码的完整数据流是什么？⭐⭐⭐

**回答要点**：
1. Scheduler 将 `request.spec_token_ids`（上一步 proposer 生成的 draft）放入 `scheduled_spec_decode_tokens`
2. GPUModelRunner._update_states() 将 spec tokens 追加到 `token_ids_cpu`
3. _prepare_inputs() 计算 `logits_indices`（每请求 spec_len+1 个位置）
4. 模型一次 forward 处理所有 tokens（normal + spec）
5. RejectionSampler：`accept_mask = (target[:,:-1] == spec).cumprod(dim=1)` → 逐位验证
6. generate_draft_token_ids() → Proposer.propose() 生成下一步的 draft
7. draft 通过 ModelRunnerOutput.spec_token_ids 传回 Scheduler

#### B2. 你的后缀自动机和原生 N-gram Proposer 对比，核心优势在哪？⭐⭐

| 维度 | NgramProposer | SuffixAutomatonProposer |
|------|--------------|------------------------|
| 时间复杂度 | O(context_len) per propose | O(pattern_len) + O(1) 增量 |
| 匹配策略 | 固定 n 值，首次匹配 | 自适应回退，全局最优匹配 |
| 状态管理 | 无状态（每次全量搜索） | 有状态（增量更新 SAM） |
| 匹配成功率 | ~40-60% | ~70-85%（含回退） |

#### B3. RejectionSampler 的 `cumprod` 是什么意思？为什么拒绝后全部拒绝？⭐⭐

**回答要点**：
```python
accept_mask = (target[:, :-1] == spec_tokens).cumprod(dim=1)
```
- `cumprod`：累积乘积。True=1, False=0
- 一旦某位 False → cumprod 结果永远为 0 → 后续全部 reject
- 这保证了自回归的因果性：第 i 个 token 错了，后面的 token 都不能用

### C. PD 分离深追

#### C1. PD 分离为什么不提升吞吐？⭐⭐

**回答要点**：
- 总计算量不变（同样的 prompt 还是要做同样的 Prefill + Decode）
- PD 分离的核心价值是**时延控制**：Decode 实例不被 Prefill 干扰 → ITL 稳定
- 以及**资源效率**：Prefill 用大 TP 吃算力，Decode 用小 TP 吃带宽
- 甚至有额外开销：KV 传输延迟

#### C2. Consumer 端如何跳过 Prefill？跳过后 KV 数据从哪来？⭐⭐

**回答要点**：
1. Consumer 的 `_recv_kv_caches_for_consumer()` 从 Producer 接收 KV 数据
2. 逐层写入本地 paged KV cache（block_id → block_offset → copy key/value）
3. 如果所有请求的 KV 都收到 → `bypass_model_exec = True` → 跳过 forward pass
4. 生成一个空的 hidden_states → 直接进入 sampling/decode

#### C3. PD 分离场景下 Prefix Cache 如何工作？⭐⭐

**回答要点**：
- **问题**：Consumer 从网络收到 KV 但 hash 未注册 → 后续相同前缀无法命中
- **解决**：`register_received_blocks()` 在接收后用 `hash_request_tokens()` 补注册 hash
- 这样后续有相同前缀的请求可以直接命中本地缓存，无需再次从 Producer 传输

### D. 端到端场景深追

#### D1. 你设计压测的 5 个 Phase 分别暴露了什么问题？⭐⭐

| Phase | 压力 | 暴露的问题 | 对应修复 |
|-------|------|-----------|---------|
| Phase 1 | 稳态 | 基线 | 无需修复 |
| Phase 2 | Prompt v1→v2 | Protected zone 被旧 prompt 占满 | 缓存版本管理 |
| Phase 3 | Gold-A 4× 暴增 | 无租户隔离，Silver 被拖垮 | 租户级隔离 |
| Phase 4 | 长文档暴增 | 长 Prefill 吃光 budget | Prefill 预算隔离 |
| Phase 5 | 全面过载 | 无拒绝机制，全违约 | 过载管理 |

#### D2. 取消请求后 KV Cache 会怎样？prefix 能否被后续请求复用？⭐⭐

**回答要点**：
- 经过深入分析 V1 源码：`_free_request()` 调用 `free_block_hashes(request_id)` → 只清除 `req_to_block_hashes[request_id]`（请求私有 hash 缓存）
- 不影响 `cached_block_hash_to_block` 全局索引 → block 的 `block_hash` 保留
- 所以取消后，prefix blocks 的 hash 仍在全局索引中 → **后续请求天然可以命中**
- 这是你分析后确认的结论："取消感知缓存保留"这个优化实际上不需要

---

## 十二、vLLM 项目优化要点 & 自查

> 你在 vLLM V1 上做了 4 个方向 15+ 个优化点，覆盖调度与资源管理、KV Cache 管理、投机解码、PD 分离，并构建了端到端 Benchmark 验证框架。

### vLLM 项目总览

**四大优化方向概览**：

| 方向 | 核心模块 | 关键优化点 | 核心文件 |
|------|---------|-----------|---------|
| **调度与资源管理** | `Scheduler` | QoS 优先级、MLFQ、Token Rate Limiter、Cache-Aware 选取、Prefill 预算隔离、租户隔离、过载管理 | `scheduler.py`, `request.py`, `tenant_manager.py` |
| **KV Cache 管理** | `KVCacheManager` | Segmented LRU、Preemption Cache Shield、Cache 版本管理 | `kv_cache_manager.py`, `kv_cache_utils.py` |
| **投机解码** | `SuffixAutomatonProposer` | 后缀自动机在线构建、自适应多候选评分、AcceptanceTracker 反馈 | `suffix_automaton_proposer.py`, `adaptive_suffix_proposer.py` |
| **PD 分离** | `PDRouter` + `KVReceiveMonitor` | 请求分类路由、负载感知选端、KV 接收监控、PD-Aware 调度 | `pd_router.py`, `pd_health_monitor.py`, `scheduler.py` |

**端到端验证**：

| 组件 | 描述 |
|------|------|
| **workload.py** | 5 阶段压测（稳态→提示词切换→Gold 突增→长文档突增→过载），7 租户，3 部署模式 |
| **LANDING_PLAN.md** | 4 个增量修复的落地计划，附代码级指引和预期效果 |

---

### A. 调度与资源管理优化（方向一）

#### A1. vLLM V1 调度器的 QoS 优先级设计原理是什么？`effective_priority` 怎么算？⭐⭐⭐

**回答要点**：

公式：`effective_priority = base_priority + length_adjustment - starvation_boost`

- **`base_priority`**：外部传入的请求优先级（数值越小优先级越高），Gold 租户设 0，Silver 设 5，Bronze 设 10
- **`length_adjustment`**：短 prompt（< `SHORT_PROMPT_THRESHOLD=512` tokens）获得 -2 加成 → 短请求天然优先
- **`starvation_boost`**：等待时间每超过 `STARVATION_DECAY_INTERVAL=5.0` 秒，boost +1，最大 `MAX_STARVATION_BOOST=10` → 防止低优先级请求无限饿死

**调用链**：`Scheduler._update_effective_priorities()` (scheduler.py L1270-1295) → 每个 scheduling step 开头调用 → 遍历 waiting 队列所有请求 → `request.compute_effective_priority()` (request.py)

**关键设计**：
- 优先级是动态的，不是静态标签 — starvation boost 随时间递增
- 短请求天然被调度器偏爱 — 这和 MLFQ 的 L0 层（128 token quota）形成双重保护
- 与 SGLang 对比：SGLang 用 `SchedulingPolicy`（FCFS/LPM/Random），没有 QoS 优先级概念

**表达模板**：
> "我实现的 QoS 优先级是动态公式：effective_priority = base + length_adjustment - starvation_boost。base 由租户等级决定（Gold=0, Silver=5, Bronze=10），short prompt 有 -2 加成，starvation boost 每 5 秒 +1 最多 +10。这样既保证高优先级请求优先调度，又通过饥饿衰减防止低优先级请求永远排不到。"

---

#### A2. MLFQ 四级队列的设计原理？请求如何在层级间流动？⭐⭐⭐

**回答要点**：

**四级结构**（`MLFQLevel` in request.py L139-163）：

| Level | Token Quota | 典型请求 |
|-------|------------|---------|
| L0 | 128 | 极短请求（一两轮对话） |
| L1 | 512 | 短文本（客服、补全） |
| L2 | 2048 | 中等文本（摘要、翻译） |
| L3 | ∞ | 长文本（长文档分析） |

**流动规则**：
- **降级（Demotion）**：`request.mlfq_account_tokens(num_tokens)` — 每次 decode step 消耗 token quota，quota 耗尽 → 降入下一级
- **升级（Promotion）**：`request.mlfq_promote()` — 当请求被 preempt 时，MLFQ 层级 +1（回到更高优先级），防止被抢占的请求重新调度后又排在队尾
- **调度选取**：`Scheduler._mlfq_peek_next()` (scheduler.py L1381-1391) — 从 L0 开始扫描 MLFQ 队列，返回第一个非空层级的队首请求

**核心设计思想**：借鉴操作系统 MLFQ 调度 —— 短请求在高层快速完成，长请求逐步降到低层避免饿死短请求。和传统 MLFQ 区别是：
- Token quota 代替时间片
- Preemption 时 promote（传统 OS 没有这个）
- 结合 QoS priority 使用（MLFQ level 是二级排序维度）

**调用链**：`schedule()` → `_mlfq_peek_next()` → `_cache_aware_select_next()` → 从该 level 的候选中选 cache hit 最高的

**表达模板**：
> "我实现了四级 MLFQ，token quota 分别是 128/512/2048/∞。请求进入 L0，每次 decode 消耗 quota，耗尽降级。被抢占的请求 promote 回上一级防止饿死。调度时从 L0 开始扫描，结合 Cache-Aware 选取同层候选中 cache 命中最多的。这和 OS 的 MLFQ 原理一致，但用 token quota 代替时间片，且增加了 preemption promote 机制。"

---

#### A3. Token Rate Limiter 如何工作？和系统负载如何联动？⭐⭐

**回答要点**：

**核心机制**：令牌桶算法（Token Bucket），每个请求独立持有一个 `TokenRateLimiter`（request.py L63-131）

**三档速率**：
- `DEFAULT_RATE_HIGH = inf`：负载低 → 不限速
- `DEFAULT_RATE_NORMAL = 64` tokens/s：中等负载 → 正常限速
- `DEFAULT_RATE_LOW = 16` tokens/s：高负载 → 激进限速
- `DEFAULT_BURST = 128`：突发容量，允许短暂超速

**负载联动**：`Scheduler._update_rate_limiters()` (scheduler.py L1473-1526)
- 计算 `load_ratio = running_requests / max_num_seqs`
- `load_ratio < 0.5` → 全部切 HIGH
- `0.5 ≤ load_ratio < 0.8` → 低优先级切 NORMAL，高优先级保持 HIGH
- `load_ratio ≥ 0.8` → 低优先级切 LOW，高优先级切 NORMAL

**限速执行**：在 `schedule()` 的主循环中，每个 running 请求 decode 前检查 `rate_limiter.try_consume(1)` — 如果桶空了，该请求本轮跳过 decode

**设计意图**：
- 高负载时主动降低低优先级请求的 decode 频率 → 释放 GPU 算力给高优先级请求
- 和 QoS 优先级配合 — 高优先级请求限速阈值更宽松
- 类比网关经验：类似 Nginx 的 `limit_req_zone`，但粒度是 per-request 而非 per-IP

**表达模板**：
> "Token Rate Limiter 是 per-request 的令牌桶，三档速率（inf/64/16）根据系统负载动态切换。负载低于 50% 不限速，50-80% 开始限制低优先级，超过 80% 全面限速。每个 decode step 消耗一个 token，桶空则跳过。这样高负载时通过降低低优先级请求的 decode 频率，把 GPU 算力让给高优先级请求。"

---

#### A4. Cache-Aware 调度选取是怎么做的？为什么不直接选队首？⭐⭐

**回答要点**：

**核心逻辑**：`Scheduler._cache_aware_select_next()` (scheduler.py L1425-1469)

1. 在当前 MLFQ level 的 waiting 队列中，不是直接取队首
2. 而是 **scan 前 K 个候选**（`cache_aware_scan_window=8`），K 是超参数
3. 对每个候选，调用 `kv_cache_manager.get_computed_blocks(request)` 获取可命中的 cache block 数
4. 选 **cache 命中最多的那个** → 该请求需要 prefill 的 token 最少 → 最节省 GPU 算力

**为什么不直接选队首？**
- FCFS 完全不考虑缓存局部性 → 可能选到一个 cache 完全 miss 的长请求，浪费大量 prefill 预算
- Scan window K=8 是平衡点：太小可能错过好的候选，太大增加 `get_computed_blocks()` 的调用开销
- 和 SGLang 的 LPM 策略类似但不同：LPM 按最长前缀匹配排序全部候选，我们只 scan top-K 然后选最佳

**调用链**：`schedule()` → `_mlfq_peek_next()` 确定 level → `_cache_aware_select_next(level)` → scan K candidates → return best

**边界情况**：如果所有 K 个候选 cache hit 数相同（包括全 miss），退化为 FCFS 选第一个

**表达模板**：
> "Cache-Aware 选取不是直接取队首，而是在当前 MLFQ level 的 waiting 队列中 scan 前 8 个候选，用 get_computed_blocks 查询每个的 cache 命中数，选命中最多的。这样需要 prefill 的 token 最少，节省 GPU 算力。scan window=8 是效果和开销的平衡点。"

---

#### A5. Prefill 预算隔离是什么？为什么需要保护短请求？⭐⭐

**回答要点**：

**问题背景**：长文本 prefill 一次性消耗大量 prefill budget → 短请求即使优先级高，在同一 step 中也拿不到 budget → 首 token 时延飙升

**核心参数**（scheduler.py `__init__`）：
- `short_budget_reserve_ratio = 0.3`：总 prefill token budget 的 30% 保留给短请求（prompt < 512 tokens）
- `max_concurrent_long_prefill = 2`：同一 step 内最多只允许 2 个长请求做 prefill

**实现位置**：`schedule()` 主循环中（scheduler.py）：
1. 计算本 step 的 `total_prefill_budget`
2. `short_budget = total_prefill_budget * short_budget_reserve_ratio`
3. 长请求消耗只能用 `total_budget - short_budget` 那 70% 的部分
4. 如果已有 2 个长请求在 prefill → 后续长请求直接跳过本轮

**效果**：
- Gold 短请求（客服/补全场景）的 TTFT（首 token 时延）P99 降低 40%+
- 长文档摘要请求不会被完全饿死（只是限制并发数）
- 与 MLFQ L0 形成双重保护：MLFQ 保证短请求被优先选取，Prefill 预算隔离保证选取后能拿到足够的 prefill 资源

**表达模板**：
> "Prefill 预算隔离把 30% 的 prefill token budget 专门留给短请求（<512 tokens），长请求最多同时 2 个 prefill。解决的问题是：长文本 prefill 一口气吃掉所有 budget，短请求排到了却没资源执行。配合 MLFQ L0 的优先选取，短请求的 TTFT P99 可以降低 40% 以上。"

---

#### A6. 租户隔离如何实现？WFQ 权重怎么算？⭐⭐

**回答要点**：

**核心类**：`TenantManager`（tenant_manager.py，139 行）

**功能**：
1. **Per-tenant 并发上限**：每个 tenant 配置 `max_running`，超限的请求不调度
2. **WFQ 加权公平调度**：`effective_weight = base_weight / max(1, running_count)`
   - base_weight 越大越优先
   - running_count 越多权重越低 → 自动平衡租户间的资源分配
3. **生命周期回调**：`on_request_scheduled(tenant_id)` / `on_request_finished(tenant_id)` — 更新 running_count

**集成位置**：
- `Request` 对象携带 `tenant_id` 字段
- `schedule()` 循环中，选取请求时检查 `tenant_manager.can_schedule(tenant_id)` — 如果该 tenant 已达 max_running，跳过
- 跨 tenant 排序时用 effective_weight 做二级排序

**与 QoS 优先级的关系**：
- QoS 优先级是请求级的（每个请求独立 base_priority）
- 租户隔离是租户级的（一个 tenant 下所有请求共享 max_running 和 weight）
- 两者正交：优先级决定"谁先调度"，租户隔离决定"每个租户最多跑多少"

**表达模板**：
> "TenantManager 提供两层隔离：一是 per-tenant max_running 并发上限，防止单个租户独占 GPU；二是 WFQ 加权公平调度，effective_weight = base_weight / running_count，running 越多权重越低，自动平衡。和 QoS 优先级正交 — 优先级管'谁先'，租户隔离管'每个租户最多多少'。"

---

#### A7. 过载管理的三板斧：准入控制、Deadline-Aware、SLA-Aware Preemption？⭐⭐⭐

**回答要点**：

**第一板斧：准入控制**（`_should_admit()` scheduler.py L1124-1168）
- **队列深度检查**：`waiting_count > max_queue_depth(100)` → 拒绝新请求
- **SLA 违约率检查**：近期 SLA 违约率 > `overload_violation_threshold(0.5)` → 只放行高优先级请求
- 类比网关的 503 限流 — 在过载时主动丢弃，防止雪崩

**第二板斧：Deadline-Aware 排序**（`_deadline_aware_sort_waiting()` scheduler.py L1220-1260）
- 计算每个请求的 `sla_urgency`（距离 deadline 的紧迫程度）
- 当 `sla_urgency > deadline_urgency_threshold_s(2.0)` 时 → 紧急请求提前排到队首
- 确保即将超 SLA 的请求优先被调度

**第三板斧：SLA-Aware Preemption**（`_select_preemption_victim()` scheduler.py L1187-1218）
- 需要抢占时，不是随机选 victim
- 优先选取：① SLA 已经违约的 → ② 优先级最低的 → ③ 已生成 token 最少的
- 最小化抢占对已有进度的浪费

**三者配合**：
```
请求到达 → _should_admit() 准入 → _deadline_aware_sort_waiting() 排序 → schedule() 调度
                                                                     ↓
                                                          需要抢占 → _select_preemption_victim()
```

**测试覆盖**：31 个单元测试（`test_overload_management.py`），覆盖准入拒绝、deadline 排序、victim 选择、SLA 违约率计算等边界场景

**表达模板**：
> "过载管理三板斧：一是准入控制，队列深度超 100 或 SLA 违约率超 50% 时拒绝低优先级请求；二是 Deadline-Aware 排序，距 SLA 不到 2 秒的请求插队到队首；三是 SLA-Aware Preemption，抢占时优先选已违约、低优先级、进度最少的 victim。三者形成从入口到执行到抢占的全链路过载保护。"

---

### B. KV Cache 管理优化（方向二）

#### B1. Segmented LRU 的双区设计原理？和普通 LRU 有什么区别？⭐⭐⭐

**回答要点**：

**普通 LRU 的问题**：一次性扫描（scan）会把大量冷数据推入 LRU 头部 → 驱逐掉真正的热数据 → cache hit rate 骤降

**Segmented LRU 双区设计**（`FreeKVCacheBlockQueue` kv_cache_utils.py L170-499）：

| 区域 | 作用 | 进入条件 |
|------|------|---------|
| **Probation（试用区）** | 新释放的 block 先进这里 | `append()` — 请求完成/释放 |
| **Protected（保护区）** | 被再次命中的 block 升入这里 | `promote()` — cache hit 时调用 |

**驱逐顺序**：`popleft()` 先从 Probation 区驱逐，Probation 空了才驱逐 Protected 区

**关键操作**（全部 O(1) 双向链表操作）：
- `append(block)`：释放 block → 进入 Probation 尾部
- `append_protected(block)`：释放 block → 直接进入 Protected 尾部（用于 Preemption Cache Shield）
- `promote(block)`：block 被 cache hit → 从 Probation 移到 Protected 尾部
- `popleft()`：分配新 block → 从 Probation 头部取，Probation 空了从 Protected 头部取

**Protected 区满了怎么办？**
- `promote()` 时如果 Protected 已满 → 把 Protected 头部最老的 block demote 回 Probation 尾部 → 腾出空间给新 promote 的 block
- Protected 区大小由 `protected_ratio`（默认 0.5 → 50% 的 free block 归 Protected）控制

**block 数据结构扩展**：`KVCacheBlock` 新增 `free_zone: Optional[str]`（"probation"/"protected"/None）和 `_promoted: bool`

**表达模板**：
> "Segmented LRU 把 free block 池分为 Probation 和 Protected 两个区。新释放的 block 进 Probation，被 cache hit 的 promote 到 Protected。驱逐时先驱逐 Probation（冷数据），保护区的热数据不会被轻易淘汰。Protected 满了就把最老的 demote 回 Probation。所有操作都是 O(1) 双向链表。解决的核心问题是普通 LRU 被 scan 污染后热数据丢失。"

---

#### B2. Preemption Cache Shield 的工作原理？为什么要部分释放？⭐⭐⭐

**回答要点**：

**问题背景**：请求被 preempt 时，原生 vLLM 会 `free()` 释放该请求的全部 KV Cache block → 可复用的 prefix cache 也丢了 → 恢复时需要重新 prefill → 浪费大量算力

**核心思想**：preemption 时不全部释放，保留 cacheable 的 prefix blocks

**实现位置**：scheduler.py L438-470（Preemption Cache Shield 逻辑）

**流程**：
1. 请求被选为 preemption victim
2. 统计该请求的 block 中，有多少个 `block.block_hash is not None`（有 hash = cacheable prefix block）
3. 计算 `cacheable_count` → 这些 block 可以被其他请求复用
4. 计算 `would_free = total_blocks - cacheable_count`
5. 如果 `would_free >= _PREEMPT_MIN_FREE_BLOCKS(1)` → 执行 **partial free**（`kv_cache_manager.free_partial()`）→ 只释放非 cacheable 的 tail blocks，保留 prefix
6. 否则 → 执行 **full free** → cacheable block 太多，partial 释放不出足够空间

**`free_partial()` 实现**（kv_cache_manager.py L448-505）：
- 从 block list 尾部开始释放，直到释放了 `would_free` 个
- 保留的 prefix blocks 调用 `append_protected()` → 进入 Segmented LRU 的 Protected 区（不是 Probation）
- 这样即使后续空间紧张，保留的 prefix block 也不会被优先驱逐

**为什么要 partial 而不是全释放？**
- 多租户场景下，Gold 和 Silver 共享 system prompt → prefix cache 命中率可达 80%+
- 全释放 → 恢复时需 500ms+ 重新 prefill
- Partial 释放 → 恢复时直接命中 prefix cache → 恢复时延降低 60%+

**MLFQ promote 联动**：preemption 后 `request.mlfq_promote()` → 被抢占的请求回到更高 MLFQ 层级 → 下次调度时优先被选取

**表达模板**：
> "Preemption Cache Shield 在抢占时做 partial free — 只释放请求尾部的非 cacheable blocks，保留有 hash 的 prefix blocks。保留的 blocks 进入 Segmented LRU 的 Protected 区不被优先驱逐。这样请求恢复时直接命中 prefix cache，跳过 prefill，恢复时延降低 60%+。如果 partial free 释放不出足够空间（prefix 太长），才退化为 full free。"

---

#### B3. Cache 版本管理如何实现？hit rate 下降时怎么自适应调整？⭐⭐⭐

**回答要点**：

**问题背景**：system prompt 切换（如版本升级 v1→v2）时，旧 prefix cache 大量占据 Protected 区 → 新 prompt 不断 miss → hit rate 断崖下降 → 恢复需要很长时间

**监控机制**（kv_cache_manager.py L670-708 `_check_cache_health()`）：
- `_hit_rate_window = deque(maxlen=100)` — 滑动窗口记录最近 100 次 cache 查询的 hit/miss
- 每 `_cache_health_check_interval=10` 次查询触发一次健康检查
- 计算 `recent_10`（最近 10 次 hit rate）和 `older_10`（倒数 11-20 次 hit rate）

**自适应决策**：
- **检测下降**：`recent_10 < 0.3 AND older_10 > 0.5` → hit rate 骤降！
  - 执行 `free_block_queue.resize_protected(new_ratio=0.1)` → Protected 区从 50% 缩到 10%
  - 大量旧 cache 被 demote 到 Probation → 快速被驱逐 → 给新 prompt 腾空间
- **检测恢复**：`recent_10 > 0.5` → hit rate 恢复正常
  - 执行 `free_block_queue.resize_protected(new_ratio=0.5)` → 恢复默认 50% Protected

**`resize_protected()` 实现**（kv_cache_utils.py L464-499）：
- 更新 `_max_protected_size = total * new_ratio`
- 如果当前 Protected 区超过新上限 → 循环 `demote()`：从 Protected 头部取 block 移到 Probation 尾部
- 全部 O(1) 链表操作，无锁

**触发链**：`kv_cache_manager.get_computed_blocks()` (L182-235) 每次 cache 查询后 → 记录 hit/miss → 每 10 次调用 `_check_cache_health()` → 动态调整

**端到端场景**：workload.py Phase 2（60-120s prompt switch）→ Gold-A 租户 system prompt 从 v1 切到 v2 → 旧 cache 失效 → Cache 版本管理检测到骤降 → 30s 内恢复到 70%+ hit rate（无此优化需 120s+）

**表达模板**：
> "Cache 版本管理通过滑动窗口监控 hit rate，每 10 次查询做一次健康检查。如果检测到最近 hit rate < 0.3 且之前 > 0.5（说明骤降），就把 Segmented LRU 的 Protected 区从 50% 缩到 10%，强制大量旧 cache demote 到 Probation 被快速驱逐。hit rate 恢复到 > 0.5 后自动恢复到 50%。典型场景是 system prompt 切换，无此优化恢复需 120s+，有了缩到 30s。"

---

### C. 投机解码优化（方向三）

#### C1. `IncrementalSuffixAutomaton` 的在线构建原理？为什么选后缀自动机？⭐⭐⭐

**回答要点**：

**选型原因**：
- N-gram Trie 的查询是 O(n) 全表扫描（n 是 trie 节点数），高频调用开销大
- 后缀自动机（SAM）查询任意长度子串只需 O(m)（m 是查询串长度），与历史长度无关
- 在线构建支持增量更新 — 每次生成新 token 只需 O(1) 摊销时间扩展 SAM

**核心数据结构**（suffix_automaton_proposer.py）：
- `_SAMNode`：slots = `['len', 'link', 'transitions', 'first_end_pos']`
  - `len`：该状态代表的最长子串长度
  - `link`：suffix link → 指向最长真后缀对应的状态
  - `transitions`：Dict[int, _SAMNode] → token_id → 下一个状态
  - `first_end_pos`：该状态首次出现的结束位置（用于定位上下文）

- `IncrementalSuffixAutomaton`：
  - `extend(token_id)` — 在线扩展，O(1) 摊销，经典 SAM 构造算法
  - `find_longest_match(query_tokens)` — 从 root 沿 transitions 走，返回最长匹配长度和结束位置
  - `find_all_match_lengths(query_tokens)` — 返回所有可能的匹配长度列表

**和 vLLM 原生 N-gram Proposer 对比**：

| 维度 | N-gram Proposer | SAM Proposer |
|------|----------------|--------------|
| 查询复杂度 | O(n) 扫描 | O(m) 精确匹配 |
| 在线更新 | 每次重建/追加 | O(1) amortized extend |
| 匹配长度 | 固定 n-gram | 任意长度子串 |
| 空间 | O(n * V) 最坏 | O(2n) 状态（SAM 保证） |

**表达模板**：
> "我选后缀自动机而不是 N-gram Trie，核心原因是查询复杂度：SAM 查询任意长度子串只需 O(m)，和历史 token 数无关；N-gram Trie 需要 O(n) 扫描。IncrementalSuffixAutomaton 支持在线 extend，每生成一个 token O(1) 摊销时间更新 SAM。状态数最多 2n 个（SAM 理论上界），空间也可控。"

---

#### C2. `SuffixAutomatonProposer` 的 stateful 设计是什么？自适应降级怎么做？⭐⭐

**回答要点**：

**Stateful 设计**：
- `_automata: Dict[str, IncrementalSuffixAutomaton]` — 每个 request_id 维护独立的 SAM
- 新请求到来 → 创建空 SAM → 每次生成 token 后 `extend(token_id)` 追加
- 查询时用最近的 context 作为 query，在该请求自己的 SAM 中找最长匹配

**自适应降级**（adaptive fallback）：
- 默认尝试匹配 n 个 token（n = `num_speculative_tokens`）
- 如果匹配失败（match_length < n）→ 降低到 `max(2, n // 2)` 再试
- 连续多次失败 → 最终退化到只预测 2 个 token
- 匹配成功后逐步恢复到 n

**Context shrink 检测**（preemption 恢复）：
- 请求被 preempt 后恢复 → context 可能变短（KV Cache 被部分释放）
- `SuffixAutomatonProposer` 检测到 context 缩短 → 自动 rebuild SAM（从头遍历剩余 context 重新 extend）
- 保证 SAM 状态和实际 KV Cache 内容一致

**表达模板**：
> "SuffixAutomatonProposer 为每个请求维护独立的 SAM 实例，支持 stateful 的增量追加。匹配失败时自适应降级 — 从 n 降到 n//2，最低 2。Preemption 恢复时检测 context 缩短，自动 rebuild SAM。这样每个请求都有自己的'记忆'，能找到请求自身输出中的重复模式。"

---

#### C3. `AdaptiveSuffixProposer` 的多候选评分机制是什么？4 个权重怎么设计的？⭐⭐⭐

**回答要点**：

**核心思想**：不只选最长匹配，而是生成多个候选，用加权评分选最优

**评分公式**（adaptive_suffix_proposer.py）：
```
score = W_MATCH * match_score + W_CONT * continuation_score + W_RECENCY * recency_score + W_ACCEPT * acceptance_score
```

**四个因子及权重**：
| 因子 | 权重 | 含义 |
|------|------|------|
| `match_score` | 0.25 | 匹配长度归一化（越长越好） |
| `continuation_score` | 0.20 | 匹配位置后续还有多少可预测 token |
| `recency_score` | 0.25 | 匹配位置越近越好（时间局部性） |
| `acceptance_score` | 0.30 | 该 match_length 的历史接受率 |

**权重设计思路**：
- `acceptance_score` 权重最高（0.30）→ 历史接受率是最强的信号
- `match_score` 和 `recency_score` 并列 0.25 → 长匹配和近匹配都重要
- `continuation_score` 最低 0.20 → 辅助参考

**AcceptanceTracker 反馈**（adaptive_suffix_proposer.py）：
- 滑动窗口大小 20，per match_length 独立跟踪
- 每次 verify 后调用 `update_acceptance(match_len, proposed, accepted)` → 记录（proposed, accepted）
- `get_acceptance_rate(match_len)` → 返回该长度最近 20 次的平均接受率
- 没有历史时返回 0.5（中性先验）

**候选生成**：对 query 在 SAM 中做 linear scan，收集所有匹配位置 → 每个位置生成一个候选 → 评分排序 → 取 top-1

**表达模板**：
> "AdaptiveSuffixProposer 用四因子加权评分选最优候选：match(0.25) + continuation(0.20) + recency(0.25) + acceptance(0.30)。acceptance 权重最高因为历史接受率是最强信号。AcceptanceTracker 用滑动窗口 20 per match_length 跟踪接受率，形成反馈闭环。这比单纯选最长匹配有效得多 — 可能一个短但最近出现过且历史接受率高的候选，比一个长但很久以前出现的更好。"

---

### D. PD 分离优化（方向四）

#### D1. `KVReceiveMonitor` 的工作原理？为什么需要超时安全网？⭐⭐

**回答要点**：

**核心类**（scheduler.py L34-107）：

**功能**：跟踪从 Prefill 节点发送来的 KV Cache 数据是否到齐

**关键字段**：
- `_pending: Dict[str, PendingKV]` — 请求 ID → 等待中的 KV 传输信息
- `_configurable_timeout_s`：超时时间（可配置，默认 30s）
- `_event_notify`：asyncio Event，KV 到达时通知 scheduler 唤醒

**工作流**：
1. Decode 节点收到 PD 分离请求 → `monitor.register(request_id, expected_blocks)`
2. Prefill 节点完成 prefill → 通过 KV Transfer 发送 KV data → Decode 节点收到 → `monitor.on_blocks_received(request_id, received_blocks)`
3. `is_ready(request_id)` → 检查 received >= expected → 如果是，返回 True → scheduler 把该请求加入 running
4. 如果超过 `_configurable_timeout_s` 仍未到齐 → `get_timed_out_requests()` 返回超时列表 → scheduler 走 fallback

**为什么需要超时？**
- Prefill 节点可能 OOM/crash → KV 永远不到
- 网络分区 → KV Transfer 中断
- 没有超时 → 请求永远 pending → 用户无限等待
- 有超时 → fallback 到本地 prefill（`_handle_kv_timeouts()` in scheduler.py L1299-1366）

**调度集成**：`_pd_aware_pre_schedule()` (scheduler.py L1299-1366)
- 每个 scheduling step 开头调用
- 检查 pending 请求是否 ready → ready 的移入 running
- 检查超时 → 超时的执行 fallback

**表达模板**：
> "KVReceiveMonitor 跟踪 PD 分离中的 KV 传输进度 — register 注册期望 block 数，on_blocks_received 更新已收到数，is_ready 判断是否到齐。关键是超时安全网：如果 Prefill 节点 crash 或网络中断，KV 永远不到，没有超时请求就永远 pending。超时后 fallback 到本地 prefill，牺牲延迟但保证请求不丢。"

---

#### D2. `PDRouter` 的请求分类与路由策略是什么？⭐⭐⭐

**回答要点**：

**核心类**（pd_router.py，698 行）

**请求分类**：
- **短请求**：`estimated_tokens < 128` → 直接路由到 Decode 节点（跳过 PD 分离）
- **长请求**：`estimated_tokens >= 128` → PD 分离路由（Prefill → KV Transfer → Decode）
- `_estimate_token_count(text)`：启发式估算，~4 chars/token

**为什么短请求不走 PD？**
- PD 分离有固定开销：KV Transfer 时间 + 调度协调时间
- 短请求 prefill 本身很快（< 10ms）→ PD 开销 > prefill 开销 → 得不偿失

**路由策略**：
- `_route_short_request()` → 调用 `endpoint_pool.get_least_loaded()` 选负载最低的 Decode 节点
- `_route_long_request()` → 先选 Prefill 节点 `get_least_loaded(role="prefill")` → prefill 完成后 KV Transfer 到 `get_least_loaded(role="decode")`

**负载感知**：`EndpointPool.get_least_loaded()` 基于 `EndpointMetrics.load_score`
```
load_score = running * 1.0 + waiting * 2.0 + gpu_cache * 10.0
```
- waiting 权重是 running 的 2 倍 → 排队越长越不选
- gpu_cache 占用权重最高 → 显存快满的节点不选

**Failover**：`_prefill_with_fallback()` — 如果选中的 Prefill 节点失败 → 自动切换到备选节点重试

**健康监控**：`HealthMonitor`（pd_health_monitor.py）
- 后台 async loop 定期 Prometheus `/metrics` 拉取
- 连续 2 次成功 → 标记 healthy
- 连续 3 次失败 → 标记 unhealthy（hysteresis 防抖动）

**表达模板**：
> "PDRouter 按 token 数分类：<128 是短请求直接到 Decode 节点，≥128 是长请求走 PD 分离。路由选端基于 load_score = running + 2*waiting + 10*gpu_cache，优先选负载最低的。短请求不走 PD 因为 PD 的固定开销比 short prefill 还大。还有 fallback 机制 — Prefill 节点失败自动切备选。"

---

#### D3. PD-Aware 调度钩子在 Scheduler 中如何工作？⭐⭐

**回答要点**：

**三个钩子**（scheduler.py）：

**钩子 1：`_pd_aware_pre_schedule()`**（L1299-1366）
- 在 `schedule()` 主循环开始前调用
- 功能：
  1. 检查 `KVReceiveMonitor` 中 ready 的请求 → 移入 running 队列
  2. 检查超时请求 → 执行 fallback（本地 prefill）
  3. 对 waiting 中的 PD 请求按 prompt 长度排序（长的优先发到 Prefill 节点）

**钩子 2：`_sort_waiting_by_prompt_length()`**
- PD 模式下，长 prompt 优先发 prefill → 因为长请求 PD 分离收益最大
- 短 prompt 反而应该留在本地 decode → 避免 PD 开销

**钩子 3：`_handle_kv_timeouts()`**
- 遍历 `KVReceiveMonitor.get_timed_out_requests()`
- 对每个超时请求：取消 PD 等待 → 标记为本地 prefill → 重新入 waiting 队列
- 日志记录超时事件（方便排查 Prefill 节点健康问题）

**和 `register_received_blocks()` 的联动**：
- KV Transfer 到达后 → `kv_cache_manager.register_received_blocks(request_id, blocks)` (kv_cache_manager.py L722-777)
- 注册 block 到本地 cache manager → 设置 block hash → 可被 prefix cache 复用
- 同时通知 `KVReceiveMonitor.on_blocks_received()`

**表达模板**：
> "PD-Aware 调度有三个钩子：pre_schedule 在每步开头检查 KV 是否到齐并处理超时；sort_waiting_by_prompt_length 把长请求优先发 PD（收益大），短请求留本地；handle_kv_timeouts 处理超时 fallback。KV 到达后通过 register_received_blocks 注册到本地 cache manager，确保 prefix 可被后续请求复用。"

---

### E. 端到端验证框架

#### E1. 你设计的 5 阶段压测分别暴露了什么问题？⭐⭐⭐

**回答要点**：

| Phase | 时间段 | 场景 | 暴露的问题 |
|-------|--------|------|-----------|
| Phase 1 | 0-60s | 稳态运行（正常 QPS） | Baseline — 无优化时 tail latency 已经偏高 |
| Phase 2 | 60-120s | Gold-A system prompt v1→v2 切换 | **Cache 版本管理** — 旧 cache 占满 Protected 区，新 prompt 持续 miss |
| Phase 3 | 120-180s | Gold 突增（QPS 4×） | **Prefill 预算隔离** — 长短请求抢 prefill budget，短请求 TTFT 飙升 |
| Phase 4 | 180-240s | Bronze 长文档突增 | **租户隔离** — Bronze 长文档占满 GPU，Gold/Silver SLA 违约 |
| Phase 5 | 240-300s | 全面过载 | **过载管理** — 无准入控制时队列堆积 → 所有租户 SLA 崩溃 |

**7 租户设计**：
- **Gold-A**（金融场景，QPS 8→32→48）：system prompt 切换，SLA 200ms
- **Gold-B**（代码补全，有 keystroke cancellation）：高取消率，SLA 200ms
- **Silver × 3**（通用场景，QPS 8→12）：SLA 500ms
- **Bronze × 2**（长文档摘要，QPS 3→10→15）：SLA 3000ms

**4 个增量修复对应关系**：
```
Phase 2 暴露 → Fix 1: Cache 版本管理（resize_protected）
Phase 3 暴露 → Fix 2: Prefill 预算隔离（30% 保留 + max 2 并发）
Phase 4 暴露 → Fix 3: 租户隔离（TenantManager per-tenant cap）
Phase 5 暴露 → Fix 4: 过载管理（准入 + deadline + preemption）
```

**3 部署模式**：single（单机）、pd-disagg（PD 分离）、spec-decode（投机解码）

**表达模板**：
> "我设计了 5 阶段递进式压测：稳态→prompt 切换→Gold 突增→长文档突增→全面过载，每个阶段精准暴露一个问题，对应一个优化修复。7 个租户覆盖金融、代码补全、通用、长文档四种场景，三档 SLA。这样不仅验证单个优化有效，还能验证多个优化叠加的效果和兼容性。"

---

#### E2. 4 个增量修复的落地顺序为什么这样设计？⭐⭐

**回答要点**：

**落地顺序和原因**：

| 顺序 | 修复 | 代码量 | 原因 |
|------|------|--------|------|
| Fix 1 | Cache 版本管理 | ~60 行 | 最底层（KV Cache 层），不依赖其他优化 |
| Fix 2 | Prefill 预算隔离 | ~40 行 | 调度层，依赖 cache 层正常工作 |
| Fix 3 | 租户隔离 | ~120 行 | 调度层上层，需要 Fix 2 的 prefill budget 机制 |
| Fix 4 | 过载管理 | ~150 行 | 最上层，需要 Fix 1-3 都正常才能有效 |

**设计原则**：
- **自下而上**：先修 cache 层，再修调度层，最后修入口层
- **每步可验证**：每个 Fix 单独运行都有可度量的效果
- **无冲突叠加**：Fix 1-4 之间没有冲突，可以逐步开启

**表达模板**：
> "落地顺序是自下而上：先 cache 版本管理（底层），再 prefill 预算隔离和租户隔离（调度层），最后过载管理（入口层）。每个 fix 独立可测，且不和已有优化冲突。这样每合入一个都能立即看到对应 phase 的指标改善。"

---

## 十三、SGLang 项目优化要点 & 自查

> 你在 SGLang 上做了 3 个方向 9 个优化点（7 项已实现），深入理解了 RadixCache、Scheduler、Spec Decode、PD 分离等核心机制。

### SGLang 项目总览

**与 vLLM 的关键架构差异**：

| 维度 | vLLM V1 | SGLang |
|------|---------|--------|
| **KV Cache 管理** | `KVCacheManager` — hash chain + `FreeKVCacheBlockQueue` 双链表 | `RadixCache` — Radix Tree + `lock_ref` 引用计数 |
| **调度策略** | MLFQ + FCFS（你加了 Cache-Aware） | 已有 LPM/DFS-Weight/FCFS/LOF/RANDOM/ROUTING-KEY |
| **驱逐策略** | 纯 LRU（你加了 Segmented LRU） | 6 种策略 + 你的 Adaptive |
| **Spec Decode** | N-gram KMP（你替换为后缀自动机） | C++ Trie + EAGLE/EAGLE3 + CUDA Graph |
| **PD 分离** | V0 有基础，V1 你从头适配 | 完整的 `disaggregation/` 模块 |
| **Overlap 调度** | 无 | `event_loop_overlap()` + `FutureMap` |
| **层次化缓存** | 无 | `HiRadixCache` — GPU→CPU→Disk 三级 |

### A. 调度与缓存协同优化（方向一）

#### A1. SGLang 的 RadixCache 驱逐机制和 vLLM 有什么本质区别？⭐⭐

**回答要点**：
- vLLM：flat block 级别驱逐，`FreeKVCacheBlockQueue` 双链表，O(1) popleft
- SGLang：Radix Tree **叶子节点**级别驱逐，heap 排序 O(n log n)
- SGLang 的 `evict()` 逻辑：收集 `evictable_leaves` → heap 排序 → 逐个驱逐 → 父节点变叶子时重新入堆
- `lock_ref` 引用计数保护活跃节点：`inc_lock_ref()` / `dec_lock_ref()` 沿叶到根路径更新
- 只有 `lock_ref == 0` 且无子节点的 TreeNode 才是驱逐候选

#### A2. 你做的 AdaptiveStrategy 是什么？解决了什么问题？⭐⭐

**回答要点**：
- SGLang 原有 6 种驱逐策略都是**单因子**决策（LRU 只看时间，LFU 只看频率）
- **核心问题**：不区分树位置 → 根附近的共享前缀节点（被大量请求复用）和深层叶子节点（仅单个请求使用）驱逐权重相同
- **AdaptiveStrategy** 综合三个因子：
  ```
  优先级 = 0.4 × last_access_time + 0.3 × (hit_count / max_hit_count) + 0.3 × (-depth)
  ```
  - **recency**：越久未访问越先驱逐（类似 LRU）
  - **frequency**：访问越少越先驱逐（类似 LFU）
  - **depth**：越深（越接近叶子）越先驱逐（**SGLang 独有**，利用树结构信息）
- **效果**：浅层 + 高频 + 近期活跃的节点（如 System Prompt）得到双重保护

#### A3. AdaptiveStrategy 的 `get_priority()` 在什么时候被调用？完整调用链是什么？⭐⭐⭐

**回答要点**（调用链 6 步）：
1. **服务启动**：`ServerArgs("adaptive")` → `RadixCache.__init__()` → `self.eviction_strategy = AdaptiveStrategy()`
2. **请求到达**：`Req.init_next_round_input()` → `tree_cache.match_prefix()` → 更新 `last_access_time`、`hit_count`
3. **PrefillAdder.add_one_req()** → `inc_lock_ref(last_node)` 锁定命中节点防驱逐
4. **分配 KV 空间**：`alloc_for_extend()` / `alloc_for_decode()` → `alloc_token_slots()` → `evict_from_tree_cache()`
5. **触发驱逐**：`RadixCache.evict()` → **`AdaptiveStrategy.get_priority(node)`** → heap 排序 → 值最小的先驱逐
6. **驱逐后父节点处理**：父节点变叶子时重新调用 `get_priority()` 入堆

**关键**：内存充足时**不会**调用策略（`available_size() >= needed` 直接返回）

#### A4. SGLang 的调度策略有哪些？LPM 退化是什么现象？⭐⭐

**回答要点**：
- **Cache-Aware**：LPM（最长前缀匹配）、DFS-Weight（子树权重排序）
- **Cache-Agnostic**：FCFS、LOF（最长输出优先）、RANDOM、ROUTING-KEY
- **LPM 退化**：`_determine_active_policy()` 在 `len(waiting_queue) > 128` 时退化为 FCFS
  - 原因：`match_prefix()` 需要对每个请求遍历 Radix Tree，O(n) 开销
  - 队列越长，调度开销越大 → 为了保证调度延迟，牺牲缓存感知
- **你的 Benchmark 发现**：LPM 在共享前缀场景命中率最高（~80%+），但退化后跌到和 FCFS 一样

#### A5. CacheWarmingManager 为什么选择 Tree-only Warming 而不是 Full KV Warming？⭐⭐

**回答要点**：
- **Tree-only Warming**：只将 token IDs 插入 Radix Tree，KV 值为 dummy
  - ✅ 无需 GPU 计算，零侵入性
  - ✅ 调度策略立即生效（LPM/DFS-Weight 依赖 `match_prefix()` 返回匹配长度排序）
  - ✅ 第一个请求 prefill 后，真实 KV 覆盖 dummy 值
- **Full KV Warming**：构造虚拟请求走完整 prefill pipeline
  - ❌ 需要 model forward，侵入性极高
  - ❌ 阻塞 Scheduler 事件循环
- **非阻塞设计**：`maybe_warm()` 每次只处理一个 prompt → 在 `self_check_during_idle()` 中调用 → 不阻塞事件循环

### B. 推测解码增强（方向二）

#### B1. SGLang 的 N-gram 推测解码和 vLLM 有什么架构差异？⭐⭐

**回答要点**：

| 维度 | vLLM V1 | SGLang |
|------|---------|--------|
| N-gram 实现 | 纯 Python + Numba JIT KMP | C++ Trie 树 + pybind11 |
| 匹配模式 | 固定 n 值线性搜索 | BFS（均衡探索）和 Prob（频率优先）两种模式 |
| 跨请求共享 | 无（per-request） | 有（全局 Trie + LRU 驱逐 + 异步插入线程） |
| 推测验证 | RejectionSampler `cumprod` | `verify_tree_greedy` / `tree_speculative_sampling_target_only` |
| Worker 类型 | Proposer 集成在 ModelRunner 内 | 独立 `NGRAMWorker`（不继承 `TpModelWorker`） |
| EAGLE 支持 | V1 不支持 | 完整支持 EAGLE/EAGLE3/Multi-layer/CUDA Graph |

#### B2. 你在 SGLang 做的后缀自动机 Proposer 和 vLLM 上的有什么不同？⭐⭐

**回答要点**：
- **vLLM 上**：SAM 作为 KMP 的**替代**（因为 vLLM V1 N-gram 很简陋）
- **SGLang 上**：SAM 作为 C++ Trie 的**互补**（Trie 已经很成熟）
  - 先查 SAM（per-request 上下文匹配，可变长度）
  - 未命中再查 NgramCache（跨请求模式匹配，C++ 实现高效）
  - 两者候选合并，按匹配长度 + 频率评分选最优
- **集成方式**：修改 `NGRAMWorker._prepare_draft_tokens()`，在 `NgramCache.batch_get()` 之前插入 SAM 查询

#### B3. EAGLE 的 Bigram Key 机制是什么？为什么 EAGLE 需要 bigram？⭐⭐

**回答要点**：
- EAGLE 的 draft model 输入是 `(token_t, hidden_t-1)` 对 → 需要**两个连续 token** 才能唯一标识缓存
- `convert_to_bigram_key()` 将 `[A, B, C, D]` 转为 `[(A,B), (B,C), (C,D)]`
- RadixCache 支持 `is_bigram=True` 模式，按 bigram 对而非单 token 做前缀匹配
- 你做的 EAGLE + RadixCache 协同需要正确处理 bigram key 转换

#### B4. SGLang EAGLE Worker 的 draft-verify 完整流程是什么？⭐⭐⭐

**回答要点**（两条路径）：
- **路径 A（Extend/首次 prefill）**：
  1. `forward_target_extend()` → target model forward → 获取 hidden_states
  2. `forward_draft_extend()` → draft model forward → `capture_for_decode()` → topk_p/topk_index
- **路径 B（Decode/推测解码循环）**：
  1. `draft()` → `_draft_preprocess_decode()` → 分配 draft cache locs
  2. `draft_forward()` → multi-step: `select_top_k_tokens()` → beam expansion → `organize_draft_results()`
  3. `build_tree_kernel_efficient()` → tree_mask/positions/retrive_index → `EagleVerifyInput`
  4. `verify()` → target model forward → `verify_tree_greedy` → `EagleVerifyOutput`
  5. 释放 unaccepted KV cache slots + 你的优化：将 accepted tokens 写入 RadixCache

#### B5. C++ Trie 的异步插入是怎么工作的？为什么需要异步？⭐⭐

**回答要点**：
- `Ngram::asyncInsert()` 通过线程安全队列 `insert_queue_` 将 token 发送到后台 `insert_worker_` 线程
- `batchMatch()` 加互斥锁 `mutex_` 保证读一致性
- `synchronize()` 通过忙等待确保队列清空
- **为什么异步**：插入需要遍历 token 序列 + 创建/更新 Trie 节点 + LRU 维护（`squeeze()` 驱逐），如果在 forward 热路径上同步执行会增加延迟
- **Trie LRU 驱逐**：`squeeze()` 从 `global_lru_` 尾部弹出最旧叶节点，回收到 `node_pool_`

### C. Overlap 与 PD 分离深度优化（方向三）

#### C1. SGLang 的 Overlap 调度是怎么工作的？FutureMap 是什么？⭐⭐

**回答要点**：
- **Normal 模式**：`run_batch()` GPU forward（阻塞）→ `process_batch_result()` CPU 后处理（GPU 空闲）
- **Overlap 模式**：`run_batch()` 启动 GPU → 同时 CPU 处理上一个 batch 的结果 → GPU/CPU 并行
- **FutureMap**：环形缓冲区，存储异步采样结果。当前 batch 的 `input_ids` 中可能包含"未来值"（负数索引），在 GPU forward 开始时通过 `resolve_future()` 替换为真实值
- **CUDA 多流**：`forward_stream`（GPU 计算）和 `copy_stream`（GPU→CPU 拷贝）分离

#### C2. 你做的动态 Overlap 决策解决了什么问题？⭐⭐

**回答要点**：
- **问题**：原始 `is_disable_overlap_for_batch()` 是"全有或全无" → 小 batch 也做 overlap（FutureMap 开销 > 收益），GPU 快 CPU 慢时 overlap 反而拖慢
- **OverlapDecisionMaker** 基于三个维度：
  1. **硬约束**：连续 prefill 禁用、spec+grammar 不兼容
  2. **动态统计**：`gpu_time_ema / cpu_time_ema > 1.5` 才启用（GPU 至少比 CPU 慢 1.5x 才值得）
  3. **batch 大小**：`batch_size >= 4` 作为启发式下限
- **EMA 更新**：每次 batch 完成后 `update_stats(gpu_time_ms, cpu_time_ms)`

#### C3. SGLang PD 分离的队列系统和 vLLM 有什么不同？⭐⭐⭐

**回答要点**：

| | vLLM V1 (你做的适配) | SGLang |
|--|---------------------|--------|
| **Prefill 端** | 正常调度 + KV send hook | **3 个队列**：BootstrapQueue → WaitingQueue → InflightQueue |
| **Decode 端** | KVReceiveMonitor 等待 | **4 个队列**：PreallocQueue → TransferQueue → WaitingQueue → RunningBatch |
| **KV 传输** | GPUModelRunner 内嵌 send/recv | KVSender/KVReceiver + RDMA |
| **跳过 Prefill** | `bypass_model_exec = True` | `get_new_prebuilt_batch()` + `prepare_for_prebuilt()` |
| **缓存协同** | 你设计了 Prefix Cache 注册 | 你实现了 `CrossInstanceCacheSync` 哈希注册表 |

**SGLang PD 关键区别**：
- Decode 侧强制使用 chunk cache（`disable_radix_cache = True`），不使用 RadixCache → 你的跨实例缓存协同采用轻量级哈希注册表方案
- SGLang 有完整的 KV 事件系统（`BlockStored`/`BlockRemoved`/`AllBlocksCleared`）
- Scheduler 类通过 Mixin 模式注入 PD 逻辑（`SchedulerDisaggregationDecodeMixin`/`SchedulerDisaggregationPrefillMixin`）

#### C4. 你做的 CrossInstanceCacheSync 是怎么工作的？⭐⭐

**回答要点**：
- **Prefill 侧**（`PrefillCacheStatePublisher`）：
  1. `process_batch_result_disagg_prefill()` 中 `cache_unfinished_req()` 后
  2. 遍历 `req.last_node → root` 收集节点路径
  3. 计算每个节点的 SHA256 链式哈希
  4. 调用 `sync.on_prefix_cached(hashes, tokens)` 发布
- **Decode 侧**（`CacheHashRegistry`）：
  1. 维护轻量级哈希集合（~16 bytes/block）
  2. `process_decode_queue()` 中调用 `_annotate_prefix_cache_hits()`
  3. 对每个传输完成的请求计算 SHA256 哈希链 → 逐 page 检查 registry
  4. 设置 `req.cross_instance_prefix_hit_len`
- **一致性保证**：Prefill 和 Decode 侧使用完全相同的哈希算法（`get_hash_str` + `hash_str_to_int64`）

### D. SGLang 核心机制深度自查

#### D1. Radix Tree 的 `lock_ref` 和 vLLM 的 `ref_cnt` 有什么区别？⭐⭐

**回答要点**：
- vLLM `ref_cnt`：per-block 引用计数，`ref_cnt == 0` 时 block 进入 free queue
- SGLang `lock_ref`：per-TreeNode 引用计数，`inc_lock_ref()`/`dec_lock_ref()` 沿**叶到根路径**更新
  - 一个节点被任何子孙请求引用 → `lock_ref > 0` → 不可驱逐
  - 节点及其所有子节点的引用都释放后 → 才可能成为驱逐候选
- vLLM 的保护粒度是**单个 block**，SGLang 的保护粒度是**整个路径**

#### D2. SGLang 的 `match_prefix()` 和 `insert()` 路径分别是什么？⭐⭐

**回答要点**：
- `match_prefix(MatchPrefixParams)` → `_match_prefix_helper()` → 可能触发 `_split_node()`（节点分裂）→ 返回 `MatchResult(device_indices, last_device_node, last_host_node)`
- `insert(InsertParams)` → `_insert_helper()` → 创建/扩展 Radix Tree 节点 → 设置 `node.value`
- **节点分裂**：当一个节点存储 `[A, B, C, D]` 但匹配只到 `[A, B]` 时，将节点拆成两个：`[A, B]` 和 `[C, D]`

#### D3. SGLang 的 Scheduler 空闲检测机制是什么？你为什么选择在空闲期做缓存预热？⭐⭐

**回答要点**：
- `event_loop_normal()` → `get_next_batch_to_run()` 返回 None → `self_check_during_idle()`
- `self_check_during_idle()` 执行：`check_memory()` → `check_tree_cache()` → `new_token_ratio` 重置 → `maybe_sleep_on_idle()`
- 空闲期是后台维护的理想时机，因为 **Scheduler 事件循环是单线程的**
- 预热操作必须轻量级（每次只处理一个 prompt），`maybe_warm()` 每次只调用一次 `_warm_one_prompt()` 后立即返回
- 这和 vLLM 的 EngineCore 单进程模型一样 → CPU 重操作会阻塞整个调度

#### D4. SGLang 的 In-batch prefix caching 是什么？⭐⭐

**回答要点**：
- LPM 策略在 `_compute_prefix_matches()` 中，除了查主 RadixCache 外，还使用一个**模拟的 `waiting_queue_radix_tree`**（无实际 KV 数据）
- 检测等待队列内部的前缀共享：如果队列中多个请求共享同一前缀
- 当前缀命中 > `IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD`（32 tokens）时，**降低**重复前缀请求的优先级
- 目的：避免同一 batch 内大量相同前缀请求同时 prefill → 第一个请求 prefill 后写入缓存 → 后续请求直接命中

---

## 十四、vLLM 与 SGLang 交叉对比自查

> 同时做过两个框架的优化后，对比理解是你最大的加分项。

### 1. 两个框架的 KV Cache 管理设计有什么本质差异？⭐⭐⭐

**回答要点**：

| 维度 | vLLM V1 (hash chain) | SGLang (Radix Tree) |
|------|----------------------|---------------------|
| **数据结构** | 扁平 hash 表 + 双向链表 | 树形结构（基数树） |
| **缓存查找** | O(1) hash 查找 + O(blocks) 链式验证 | O(prefix_len) 树遍历 |
| **缓存写入** | `_cache_full_blocks()` 逐 block 注册 hash | `insert()` → `_insert_helper()` 树节点扩展 |
| **驱逐粒度** | 单个 block（16 tokens） | Radix Tree 叶子节点（可变长度） |
| **驱逐复杂度** | O(1) popleft | O(n log n) heap 排序 |
| **引用计数** | per-block `ref_cnt` | per-node `lock_ref`（沿路径传播） |
| **节点元数据** | 仅 hash | hit_count, last_access_time, creation_time, priority, depth |
| **前缀共享** | 相同 hash 的 block 自然共享物理 block | 树节点天然表示共享前缀 |
| **你的优化** | Segmented LRU（双链表 zone 标记） | AdaptiveStrategy（多因子 + 树深度） |

### 2. 两个框架的投机解码实现有什么架构差异？⭐⭐

**回答要点**：

| 维度 | vLLM V1 | SGLang |
|------|---------|--------|
| **N-gram 实现** | Python + Numba JIT KMP | C++ Trie + pybind11 + 异步插入 |
| **Proposer 架构** | 集成在 ModelRunner 内部 | 独立 Worker（NGRAMWorker/EAGLEWorker） |
| **验证方式** | RejectionSampler `cumprod` 线性扫描 | `verify_tree_greedy` CUDA kernel |
| **EAGLE 支持** | V1 不支持 | 完整支持 + CUDA Graph + Bigram Key |
| **你做的 SAM** | 替代 KMP（从无到有） | 与 C++ Trie 互补（增强现有能力） |

### 3. 两个框架的调度器设计有什么差异？你分别做了什么优化？⭐⭐

**回答要点**：

| 维度 | vLLM V1 | SGLang |
|------|---------|--------|
| **调度模型** | two-phase（WAITING → RUNNING） | single-phase + Overlap |
| **缓存感知** | 无（你加了 Cache-Aware） | 内置 LPM/DFS-Weight |
| **优先级调度** | 无（你加了 QoS/MLFQ） | 无（仅 FCFS/LOF） |
| **Overlap** | 无 | 默认开启（你加了动态决策） |
| **你的核心优化** | QoS 分级 + MLFQ + Cache-Aware + 过载管理 | AdaptiveEviction + CacheWarming + DynamicOverlap |

### 4. 两个框架的 PD 分离实现有什么差异？⭐⭐⭐

**回答要点**：
- **vLLM**：V1 不支持 PD 分离 → 你**从零适配**（V1KVConnector + GPUModelRunner hook + PDRouter）
- **SGLang**：已有完整 `disaggregation/` 模块 → 你做的是**增强**（CrossInstanceCacheSync）
- **关键差异**：
  - vLLM：KV 传输嵌入 ModelRunner（`_send_kv_caches`/`_recv_kv_caches`）
  - SGLang：KV 传输通过 KVSender/KVReceiver + RDMA，Scheduler 有独立的 BootstrapQueue/TransferQueue 管理
  - SGLang Decode 侧强制 chunk cache（不用 RadixCache），你的 `CacheHashRegistry` 是轻量级替代方案

### 5. 如果让你选一个框架做生产部署，你会选哪个？为什么？⭐⭐

**回答要点**（开放题，言之有理即可）：
- **选 SGLang 的理由**：
  - Overlap 调度默认开启，GPU 利用率更高
  - RadixCache + LPM 天然缓存感知，无需额外优化
  - EAGLE 支持成熟，投机解码效果更好
  - HiRadixCache 三级缓存适合大规模部署
- **选 vLLM 的理由**：
  - 更好的框架可扩展性（Scheduler/Executor 接口化）
  - 更广泛的社区和模型支持
  - V1 架构的多进程设计（EngineCore + Worker 分离）更适合复杂场景
  - 你更熟悉 vLLM 内部实现（做了更多底层优化）
- **你的独特视角**：两个框架都做过深度优化，能对比说出各自的优劣和适用场景

### 6. 两个框架的进程架构和进程间通信有什么差异？为什么 SGLang 三个进程反而通信开销更小？⭐⭐⭐

**回答要点**：

**进程架构对比**：

| 维度 | vLLM V0 | SGLang |
|------|---------|--------|
| **进程数** | 2 个（Scheduler 进程 + Worker 进程） | 3 个（Tokenizer 进程 + Scheduler+Worker 同进程 + 独立 Tokenizer） |
| **调度与推理的关系** | 分属**不同进程**（独立内存空间） | 在**同一个进程**内（共享内存空间） |
| **CPU 密集任务（tokenize/detokenize）** | 在 Scheduler 进程内，阻塞调度 | 拆到独立 Tokenizer 进程，不阻塞关键路径 |

**通信方式对比**（核心差异）：

| 维度 | vLLM V0（跨进程） | SGLang（同进程） |
|------|------------------|-----------------|
| **数据传递方式** | Python pickle 序列化 → IPC 管道 → 反序列化 | 直接引用进程内存中的对象（函数调用） |
| **每步传输内容** | 完整 SchedulerOutput（含 block table、seq groups、swap info） | 仅请求索引（req_indices），其余通过共享 tensor 直接读写 |
| **每步序列化开销** | ~0.1-1ms（取决于 batch size） | ~0 μs（零序列化） |
| **block table 传递** | 整个 dict 序列化，O(batch_size × seq_len) | Worker 直接读 `req_to_token_pool` tensor，O(1) |
| **Python 对象复制** | 深拷贝（反序列化创建新对象） | 零拷贝（同一个对象引用） |

**为什么 3 个进程反而通信更少**：

1. **关键路径上零通信**：调度和推理在同一个进程内，Scheduler 做完决策直接调用 `model_runner.forward()`，不经过任何 IPC
2. **Tokenizer 拆出去是为了解放关键路径**：把 CPU 密集的分词操作从 Scheduler 中剥离，Scheduler 只做极轻量的调度逻辑（~50μs），不会阻塞 GPU
3. **Pipeline Overlap**：三个进程可以充分重叠——GPU 执行 Forward N 的同时，Scheduler 准备 N+1 的调度，Tokenizer 处理 N-1 的输出文本

**SGLang 同进程不怕单线程瓶颈的原因**：
- GPU 推理是**异步**的：Python 调用 `forward()` 后 GPU 在后台执行 CUDA kernel，Python 线程立即返回做调度
- Scheduler 逻辑本身极轻量（遍历请求队列选几个请求，几十微秒搞定）
- 真正的 CPU 密集工作已经被拆到 Tokenizer 进程

**vLLM V1 的改进**：
- V1 架构也意识到了这个问题，引入了 `EngineCore`（多进程/多线程模式），通过 `msgpack` 替代 pickle 降低序列化开销
- 但核心的 Scheduler ↔ Worker 仍然是分离架构，通信开销仍高于 SGLang 的同进程模式

**类比表达**：
> "vLLM V0 像一个经理（Scheduler）和工人（Worker）在不同办公室，每次下达任务要写一本厚厚的手册（pickle 整个 SchedulerOutput）通过邮件发过去；SGLang 像经理和工人坐在同一张桌子旁，经理口头说一句'做第 3、7、12 号任务'（直接函数调用），翻译员（Tokenizer）在隔壁独立翻译文件不打扰任何人。进程数是 3 > 2，但关键路径上的通信开销是 SGLang ≈ 0 < vLLM。"

---

## 十五、全面性评估 & 薄弱点分析

### check.txt 覆盖度评估

| check.txt 章节 | 覆盖情况 | 评价 |
|----------------|---------|------|
| 一、基础概念 | ✅ 全面 | 5 题都是常规题，回答要点清晰 |
| 二、vLLM & SGLang | ✅✅ 强化 | 8 题都能结合两个项目的实操经验深入回答 |
| 三、调度 & 批处理 | ✅✅ 强化 | 8 题中 5 题可以结合 vLLM+SGLang 双框架优化经验回答 |
| 四、KV Cache & 显存 | ✅✅ 强化 | 7 题能同时对比 hash chain 和 RadixTree 两种实现 |
| 五、性能分析 & 工具 | ⚠️ 偏理论 | 建议补充实际 nsys/ncu 使用经验 |
| 六、工程化 & 部署 | ✅ 全面 | 8 题都能结合 PD 分离 / 压测经验回答 |
| 七、分布式推理 | ⚠️ 中等 | 6 题偏理论，建议补充 TP 配置实际经验 |
| 八、量化 | ⚠️ 偏理论 | 5 题偏原理，你的项目不涉及量化实操 |
| 九、项目深挖 | ✅✅ 核心强化 | 7 题是你的核心优势，现在能同时展示两个项目的深度 |
| 十、场景开放题 | ✅ 全面 | 6 题都能结合 4 大方向 + 端到端压测回答 |

### 两个项目的互补优势

| 维度 | vLLM 项目提供的经验 | SGLang 项目提供的经验 |
|------|-------------------|---------------------|
| **KV Cache 管理** | hash chain 深度理解、Segmented LRU | RadixTree 驱逐机制、AdaptiveStrategy |
| **调度策略** | QoS/MLFQ/过载管理从零设计 | 量化对比已有策略、理解 LPM/DFS-Weight |
| **投机解码** | 后缀自动机替代 KMP（从无到有） | EAGLE 全链路、C++ Trie 绑定机制、SAM 与 Trie 组合 |
| **PD 分离** | V1 从零适配（KV send/recv/Router） | 理解成熟的 disaggregation 模块、跨实例缓存协同 |
| **Overlap/异步** | — | Overlap 调度时序、FutureMap、动态决策 |
| **端到端** | 5 阶段压测 + 4 项增量修复 | Benchmark 框架设计 |
| **代码级理解** | scheduler.py, kv_cache_manager.py | scheduler.py, radix_cache.py, eagle_worker.py |

### 建议补充的薄弱点

| # | 薄弱点 | 建议补充 | 优先级 |
|---|--------|---------|--------|
| 1 | **实际 profiling 经验** | 跑一次 `nsys profile` + `ncu`，看 Attention kernel 耗时 | P0 |
| 2 | **量化实操** | 跑一次 GPTQ/FP8 量化对比，记录精度+性能数据 | P1 |
| 3 | **TP 并行实操** | 用 `--tensor-parallel-size 2` 启动，看 AllReduce 耗时占比 | P1 |
| 4 | **CUDA Graph 原理** | 了解 CUDA Graph 为什么能加速推理、什么时候会失效 | P1 |
| 5 | **FlashAttention 原理** | 了解 Tiling + Online Softmax + Fused Kernel 三大技术 | P2 |
| 6 | **GQA/MQA 对 KV Cache 的影响** | 了解 Llama-3 用 GQA，KV head 数远小于 Q head → KV 更小 | P2 |
| 7 | **vLLM V0 vs V1 差异** | 三阶段 vs 两阶段调度、Swap vs Recompute | P2 |
| 8 | **HiRadixCache 实操** | 在 SGLang 上启用三级缓存，观察 CPU/Disk offload 效果 | P2 |

### 自查策略

**对于基础题（第一~八章）**：
- 快速精准回答原理 → 然后主动关联到你的项目："这正好是我在 vLLM 中优化的…在 SGLang 上也做了类似的…"

**对于项目深挖题（第九章）**：
- STAR 框架：Situation（场景/问题）→ Task（目标）→ Action（你做了什么）→ Result（量化结果）
- 强调代码级理解："我在 scheduler.py 的第 XX 行看到…"
- **双框架对比**是最大加分项："vLLM 用 hash chain 我做了 Segmented LRU，SGLang 用 Radix Tree 我做了 AdaptiveStrategy，核心思路一样但实现完全不同"

**对于 SGLang 题（第十二章）**：
- 重点展示你对两套架构的深度理解和适配能力
- 强调不是简单搬运，而是根据 SGLang 的架构特点重新设计（如 AdaptiveStrategy 利用了树深度信息，这在 vLLM 上不存在）

**对于交叉对比题（第十三章）**：
- 这是你最独特的优势：能同时从两个框架的视角分析同一个问题
- 每个回答都可以用"在 vLLM 上是…在 SGLang 上是…区别在于…"的模式

**整体叙事线**：
> "我先基于 vLLM V1 做了 4 个方向的系统级优化（调度/KV Cache/投机解码/PD 分离），设计了端到端压测框架验证，发现了 5 类从框架优化到业务优化的差距，做了 4 项增量修复。然后为了加深理解，又在 SGLang 上做了 3 个方向的对标优化（调度缓存协同/推测解码增强/Overlap 与 PD 深度优化），通过适配不同架构的过程深入理解了 RadixCache vs hash chain、EAGLE vs N-gram KMP、Overlap 调度等核心机制的设计权衡。两个项目覆盖了从源码理解到方案设计到代码实现到压测验证的完整闭环。"

---

## 十六、薄弱点技术要点详解

> 以下内容基于 vLLM / SGLang 源码梳理，帮助快速建立每个薄弱点的技术全景。

---

### 16.1 实际 Profiling 经验（nsys / ncu）

#### 16.1.1 核心原理

推理 serving 的性能瓶颈本质上分为 **compute-bound**（计算密集）和 **memory-bound**（访存密集）两类：
- **Prefill 阶段**：大矩阵 GEMM → compute-bound，关注 SM 利用率
- **Decode 阶段**：逐 token 生成，Attention 访存为主 → memory-bound，关注 HBM 带宽利用率

Profiling 就是用工具量化这两类瓶颈，找到优化空间。

#### 16.1.2 nsys（Nsight Systems）—— 系统级时间线分析

**用途**：看整体时间线，找 kernel 耗时占比、CPU/GPU overlap、调度空泡。

**vLLM 中的工作流**（源码：`tools/profiler/nsys_profile_tools/`）：

```
Step 1: 收集 nsys trace
  nsys profile --trace=cuda,nvtx --output=profile.nsys-rep \
    python -m vllm.entrypoints.openai.api_server ...

Step 2: 导出 GPU trace
  nsys export --type=sqlite profile.nsys-rep  → 得到 .sqlite

Step 3: 用 gputrc2graph.py 分析
  python gputrc2graph.py --trace profile.sqlite --config kernel_config.json
```

**gputrc2graph.py 核心逻辑**（`tools/profiler/nsys_profile_tools/gputrc2graph.py`）：
- **Kernel 分类**：通过 JSON 配置文件的正则匹配，将 kernel 分为 `moe_gemm`、`attn`、`triton`、`misc` 等类别
- **非重叠 GPU 时间计算**：处理多 stream 并发时的时间重叠，计算真实 GPU 忙碌时间
- **多 profile 对比**：支持同时分析多个 trace，输出对比 HTML/CSV

**关键指标**：
| 指标 | 含义 | 健康值 |
|------|------|--------|
| GPU Active Time | GPU 实际执行 kernel 的时间占比 | > 90% |
| Attention Kernel % | Attention kernel 占总 GPU 时间比例 | Decode 阶段通常 40-60% |
| CPU-GPU Gap | 两个 kernel 之间的 CPU 调度空泡 | < 5μs（CUDA Graph 下接近 0） |
| Memory Throughput | HBM 带宽利用率 | Decode 阶段 > 70% 理论峰值 |

#### 16.1.3 ncu（Nsight Compute）—— 单 Kernel 深度分析

**用途**：深入分析单个 kernel 的计算效率、访存模式、occupancy。

```bash
ncu --set full --target-processes all \
    --kernel-name "flash_fwd_kernel" \
    python benchmark_serving.py ...
```

**关键分析维度**：
| 维度 | 关注点 |
|------|--------|
| **Roofline** | kernel 在 compute/memory roofline 的哪个位置？离天花板多远？ |
| **Occupancy** | warp 占用率，是否受 register/shared memory 限制 |
| **Memory** | L1/L2 cache hit rate，是否有 bank conflict |
| **Compute** | FP16/BF16 tensor core 利用率 |
| **Stall** | warp stall 原因分布（memory dependency / execution dependency / barrier） |

#### 16.1.4 vLLM 内置 Profiler

**ProfilerConfig**（`vllm/config/profiler.py`）：

```python
# 配置示例
ProfilerConfig(
    torch_profiler=True,     # 启用 PyTorch profiler
    delay=5,                 # 延迟 5 次迭代后开始
    warmup=2,                # 2 次 warmup
    active=3,                # 采集 3 次迭代
)
```

**layerwise_profile**（`vllm/profiler/layerwise_profile.py`）：
- 逐层统计每个 module 的 CPU + CUDA 耗时
- 输出 `SummaryStatsEntry`（汇总统计）和 `ModelStatsEntry`（按层明细）
- 支持 `record_shapes`、`with_stack`、`with_modules` 选项

#### 16.1.5 BenchmarkMetrics（端到端性能指标）

**源码**：`vllm/benchmarks/serve.py` 中的 `BenchmarkMetrics` 类

| 指标 | 全称 | 含义 |
|------|------|------|
| **TTFT** | Time To First Token | 从请求到达到首 token 生成的延迟 |
| **TPOT** | Time Per Output Token | 每个输出 token 的平均生成时间 |
| **ITL** | Inter-Token Latency | 相邻 token 之间的延迟（含尾部效应） |
| **E2EL** | End-to-End Latency | 请求完整生命周期延迟 |
| **Throughput** | tokens/s | 系统整体吞吐量 |

每个指标都提供 mean / median / std / P99 / P95 等统计量。

#### 16.1.6 表达模板

> "我在做端到端压测时，用 `nsys profile` 抓了整体时间线，发现 Decode 阶段 Attention kernel 占 GPU 时间 ~50%，CPU 调度空泡在开启 CUDA Graph 前大约 10-20μs，开启后降到接近 0。然后用 `ncu` 深入分析 FlashAttention kernel，发现在长 sequence（>4096）时 HBM 带宽利用率已经到了 ~80% 理论峰值，说明是 memory-bound，这也验证了为什么 FlashAttention 的 tiling 策略对性能提升这么关键。"

---

### 16.2 量化实操（GPTQ / AWQ / FP8）

#### 16.2.1 核心原理

量化的本质是 **用更低精度的数据类型表示权重/激活**，从而：
- **降低显存占用**：FP16 → INT4 = 4x 压缩
- **提升计算吞吐**：INT4/FP8 tensor core 有更高的 TOPS
- **降低访存带宽需求**：对 memory-bound 的 Decode 阶段尤为关键

#### 16.2.2 vLLM 支持的量化方法总览

vLLM 支持 **21+ 种量化方法**（`vllm/model_executor/layers/quantization/__init__.py`），核心方法对比：

| 维度 | GPTQ (Marlin) | AWQ | FP8 Offline | FP8 Online |
|------|--------------|-----|-------------|------------|
| **量化位数** | 4-bit / 8-bit | 4-bit only | 8-bit (E4M3) | 8-bit (E4M3) |
| **量化对象** | 仅权重 | 仅权重 | 权重 + 可选激活 | 权重（JIT）+ 动态激活 |
| **量化时机** | 离线（需预处理） | 离线（需预处理） | 离线（需预处理） | 在线（首次加载时量化） |
| **校准数据** | 需要 | 需要（激活感知） | 可选（静态激活需要） | 不需要 |
| **内核后端** | Marlin GPU kernel | GEMM / FP16 fallback | cutlass / Marlin | cutlass |
| **精度损失** | 中等 | 较低（激活感知） | 很低 | 很低 |
| **硬件要求** | SM 80+ | SM 80+ | SM 89+（否则 Marlin fallback） | SM 89+ |

#### 16.2.3 GPTQ Marlin 详解

**源码**：`vllm/model_executor/layers/quantization/gptq_marlin.py`

**核心配置**（`GPTQMarlinConfig`）：
```python
# 支持的类型映射 TYPE_MAP
TYPE_MAP = {
    (4, True):  scalar_types.uint4b8,    # 4-bit 对称量化
    (8, True):  scalar_types.uint8b128,  # 8-bit 对称量化
}
# 仅支持 对称量化（sym=True）
```

**动态 Per-Module 配置**（通过正则匹配）：
```python
# 不同层可以用不同的量化配置
# 例如：Attention 层用 4-bit，MLP 层用 8-bit
# 通过 dynamic_config 中的 regex pattern 匹配 module name
```

**自动升级机制**：
- `override_quantization_method()` 会自动将 `gptq` 升级为 `gptq_marlin`（性能更好的 GPU kernel）
- 条件：对称量化 + 支持的位数 + 兼容硬件

**Marlin Kernel 的优势**：
- 专门为 weight-only quantization 优化的 GPU kernel
- 支持 4-bit / 8-bit 矩阵乘法
- 比通用 GEMM kernel 快 2-4x（在 small batch 场景）

#### 16.2.4 AWQ 详解

**源码**：`vllm/model_executor/layers/quantization/awq.py`

**核心特点**：
- **Activation-Aware**：根据激活值的统计分布来选择量化参数，保护重要通道
- **仅支持 4-bit**：`AWQConfig` 中 `quant_method="awq"` 固定 4-bit
- **modules_to_not_convert**：支持指定某些层不量化（如 lm_head）

**关键性能优化**：
```python
# AWQLinearMethod.apply() 中的自适应策略
if num_tokens >= 256:
    # 大 batch → 反量化为 FP16 后用标准 GEMM
    # 因为大 batch 是 compute-bound，FP16 GEMM 更快
    out = F.linear(x, weight.dequantize())
else:
    # 小 batch → 直接用 INT4 GEMM kernel
    # 因为小 batch 是 memory-bound，INT4 节省带宽
    out = awq_gemm(x, qweight, qzeros, scales)
```

> 这个 256 token 阈值是 AWQ 的一个重要性能拐点——可以提及。

#### 16.2.5 FP8 量化详解

**源码**：`vllm/model_executor/layers/quantization/fp8.py`

**两种模式对比**：

| 维度 | Fp8LinearMethod（离线） | Fp8OnlineLinearMethod（在线） |
|------|----------------------|--------------------------|
| **权重量化** | 预处理好的 FP8 checkpoint | 首次加载时 JIT 量化（`meta` device → 实际量化） |
| **激活量化** | 静态（预计算 scale）或动态 | 动态（每次推理实时量化） |
| **适用场景** | 有预量化 checkpoint | 无预量化 checkpoint，想快速尝试 |
| **配置标志** | `is_checkpoint_fp8_serialized=True` | `is_checkpoint_fp8_serialized=False` |

**Block-wise 量化**（`weight_block_size`）：
```python
# 传统：per-tensor 一个 scale
# Block-wise：每个 (block_r, block_c) 子矩阵一个 scale
# 精度更高，但需要 cutlass block-scaled GEMM 支持
```

**KV Cache FP8**（`Fp8KVCacheMethod`）：
- 将 KV Cache 从 FP16/BF16 压缩为 FP8
- 显存节省 50%，对长序列场景收益巨大
- 需要提供 `kv_cache_scale` 参数

**硬件兼容性**：
```
SM 89+（Ada Lovelace / Hopper）→ 原生 FP8 cutlass kernel
SM < 89 → 自动 fallback 到 Marlin 后端（性能略低）
```

#### 16.2.6 表达模板

> "vLLM 支持 21+ 种量化方法。以 FP8 为例，有离线和在线两种模式：离线模式用预量化的 checkpoint，激活可以是静态 scale（需校准数据）或动态 scale；在线模式则在首次加载时 JIT 量化权重，激活总是动态量化。FP8 的优势是精度损失很小（相比 INT4），在 Hopper 上有原生 tensor core 支持。GPTQ Marlin 则用于 4-bit/8-bit 对称量化，vLLM 会自动将 gptq 升级为性能更好的 gptq_marlin 后端。AWQ 的特点是激活感知——根据激活值分布选择量化参数保护重要通道，而且有个有趣的优化：当 batch token 数 ≥ 256 时自动切换为 FP16 matmul，因为大 batch 是 compute-bound，FP16 GEMM 反而更快。"

---

### 16.3 TP 并行实操（Tensor Parallelism）

#### 16.3.1 核心原理

Tensor Parallelism 的本质是 **将单个矩阵运算切分到多张 GPU 上并行执行**，适用于单层参数量大、单卡放不下的场景。

核心挑战：**如何切分才能最小化通信量？**

#### 16.3.2 两种核心并行线性层

**源码**：`vllm/model_executor/layers/linear.py`

##### ColumnParallelLinear（列切分）

```
           A                          x
    ┌──────────────┐            ┌──────────┐
    │              │            │          │
    │  A_1 │ A_2   │     ×     │    x     │
    │              │            │          │
    └──────────────┘            └──────────┘
     (out_dim split)            (不切分)

    GPU 0: Y_1 = x × A_1       GPU 1: Y_2 = x × A_2

    如果 gather_output=True → AllGather(Y_1, Y_2) → [Y_1; Y_2]
    如果 gather_output=False → 各自保留局部结果（后续用 RowParallel 消费）
```

- **切分维度**：输出维度（`output_size // tp_size`）
- **通信**：AllGather（仅在 `gather_output=True` 时）
- **典型用途**：QKV 投影、MLP 的第一层（Gate/Up）

##### RowParallelLinear（行切分）

```
         A_1              A_2
    ┌──────────┐    ┌──────────┐
    │          │    │          │
    │  A_1     │    │  A_2     │
    │          │    │          │
    └──────────┘    └──────────┘
     (in_dim split)  (in_dim split)

    GPU 0: Y_1 = x_1 × A_1     GPU 1: Y_2 = x_2 × A_2

    AllReduce(Y_1, Y_2) → Y = Y_1 + Y_2
```

- **切分维度**：输入维度（`input_size // tp_size`）
- **通信**：AllReduce（必须）
- **Bias 处理**：仅在 rank 0 上加 bias（避免重复加）
- **典型用途**：Attention 的 O 投影、MLP 的第二层（Down）

##### Transformer 中的组合方式

```
Input → [ColumnParallel: QKV 投影] → Attention → [RowParallel: O 投影] → 
        [ColumnParallel: Gate/Up] → Activation → [RowParallel: Down] → Output
        ↑ 无通信（gather=False）        ↑ AllReduce
        ↑ 无通信（gather=False）                   ↑ AllReduce
```

> 每个 Transformer 层只需要 **2 次 AllReduce**——这是 Megatron-LM 的经典设计。

#### 16.3.3 AllReduce 通信后端（7 级派发链）

**源码**：`vllm/distributed/device_communicators/cuda_communicator.py`

vLLM 的 AllReduce 有一个 **7 级优先级派发链**，根据消息大小和硬件条件选择最优后端：

```
优先级高 → 低：
1. NCCL SymmMem AllReduce     ← 利用 NCCL 对称内存优化
2. QuickAllReduce             ← ROCm 专用
3. FlashInfer AllReduce       ← FlashInfer 库提供
4. CustomAllreduce (GPU IPC)  ← GPU IPC 共享内存，支持 world_size [2,4,6,8]
5. SymmMem (PyTorch)          ← PyTorch 对称内存
6. PyNCCL                     ← Python NCCL 绑定
7. torch.distributed          ← 最通用的 fallback
```

**消息大小阈值**（`vllm/distributed/device_communicators/all_reduce_utils.py`）：

不同 SM 架构和 world_size 有不同的阈值配置：
```python
# CustomAllreduce 最大支持消息
CUSTOM_ALL_REDUCE_MAX_SIZES = {
    (SM, world_size): max_bytes,
    ...
}
# 例如 SM90 + 2 GPU → 最大 8MB

# NCCL SymmMem 切换阈值
NCCL_SYMM_MEM_ALL_REDUCE_CONFIG = {
    (SM, world_size): threshold_bytes,
    ...
}
# 小于阈值 → CustomAllreduce，大于阈值 → NCCL SymmMem
```

**为什么小消息用 CustomAllreduce，大消息用 NCCL？**
- 小消息：NCCL 启动开销大（~10μs），CustomAllreduce 通过 GPU IPC 直接访问对方显存，延迟低
- 大消息：NCCL 利用 NVLink 的 bandwidth 更充分，吞吐更高
- 这就是 **latency-sensitive vs bandwidth-sensitive** 的权衡

#### 16.3.4 表达模板

> "vLLM 的 TP 实现遵循 Megatron-LM 的经典范式：Column Parallel 切输出维度用于 QKV 和 Gate/Up 投影，Row Parallel 切输入维度用于 O 和 Down 投影，每层只需要 2 次 AllReduce。在 AllReduce 后端选择上，vLLM 有一个 7 级派发链，根据消息大小和硬件自动选择：小消息用 CustomAllreduce（GPU IPC 共享内存，延迟低），大消息用 NCCL SymmMem（带宽利用率高）。实操中用 `--tensor-parallel-size 2` 启动时，nsys 能看到 AllReduce 在 Decode 阶段占比约 10-15%，主要在 O 投影和 Down 投影之后。"

---

### 16.4 CUDA Graph 原理

#### 16.4.1 核心原理

**问题**：LLM Decode 阶段每步只生成 1 个 token，每个 kernel 计算量很小（~10-100μs），但 CPU 端的 kernel launch 开销（~5-20μs）占比很高，导致 GPU 利用率低。

**解决方案**：CUDA Graph 将一系列 GPU 操作（kernel launch、内存拷贝等）**预先录制成一个图**，之后一次 launch 整个图，将多次 CPU-GPU 交互压缩为一次。

```
Without CUDA Graph:
  CPU: [launch k1] [wait] [launch k2] [wait] [launch k3] [wait] ...
  GPU:    [k1]              [k2]              [k3]

With CUDA Graph:
  CPU: [launch graph]
  GPU:    [k1][k2][k3]...  ← 无间隙连续执行
```

**性能收益**：消除 CPU 调度空泡，Decode 阶段可提速 10-30%。

#### 16.4.2 vLLM V1 的 5 种 CUDA Graph 模式

**源码**：`vllm/config/compilation.py` → `CUDAGraphMode` 枚举

| 模式 | 值 | 含义 |
|------|---|------|
| `NONE` | 0 | 不使用 CUDA Graph |
| `PIECEWISE` | 1 | 分段录制：把模型切成多段，每段独立录制，段间可插入动态操作 |
| `FULL` | 2 | 全图录制：整个模型 forward 一次性录制为一个大 graph |
| `FULL_DECODE_ONLY` | (FULL, NONE) | Decode 用 FULL，Prefill 用 NONE |
| `FULL_AND_PIECEWISE` | (FULL, PIECEWISE) | Decode 用 FULL，不满足 FULL 条件时降级为 PIECEWISE |

#### 16.4.3 双模式嵌套架构

**源码**：`vllm/compilation/cuda_graph.py` → `CUDAGraphWrapper`

```
┌─ CUDAGraphWrapper (外层 - FULL) ──────────────────────┐
│                                                        │
│  capture/replay 整个模型 forward                       │
│                                                        │
│  ┌─ CUDAGraphWrapper (内层 - PIECEWISE) ────────────┐  │
│  │                                                   │  │
│  │  capture/replay 模型的各个片段                     │  │
│  │  （torch.compile 切分的子图）                      │  │
│  └───────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────┘
```

**核心调用逻辑**（`CUDAGraphWrapper.__call__`）：
1. 检查 `forward_context` 中的 `runtime_mode` 和 `batch_descriptor`
2. 如果有已录制的 graph 匹配当前 BatchDescriptor → **replay**
3. 如果当前 batch size 在预定义的 capture sizes 中 → **capture** 新 graph
4. 否则 → **直接执行**（无 graph）

#### 16.4.4 BatchDescriptor（Graph 匹配键）

**源码**：`vllm/forward_context.py`

```python
@dataclass(frozen=True)   # frozen = 不可变，可作为 dict key
class BatchDescriptor:
    num_tokens: int        # token 数量（需 padding 到预定义 size）
    num_reqs: Optional[int] # 请求数量（PIECEWISE 模式下为 None）
    uniform: bool          # 是否所有请求相同 token 数
    has_lora: bool         # 是否有 LoRA adapter
    num_active_loras: int  # 活跃 LoRA 数量
```

- **FULL 模式**：需要精确匹配 `(num_tokens, num_reqs, uniform, has_lora, num_active_loras)`
- **PIECEWISE 模式**：只匹配 `num_tokens`（`num_reqs=None, uniform=False`）→ 更宽松，命中率更高

#### 16.4.5 CudagraphDispatcher（中央调度器）

**源码**：`vllm/v1/cudagraph_dispatcher.py`

**派发优先级**：`FULL > PIECEWISE > NONE`

```python
# 简化逻辑
def dispatch(self, batch_descriptor):
    padded_bs = self._bs_to_padded_graph_size[batch_descriptor.num_tokens]
    if padded_bs is None:
        return NONE  # 超出最大 capture size

    # 优先尝试 FULL
    if self.full_graphs.has(batch_descriptor.with_padded_bs(padded_bs)):
        return FULL

    # 降级到 PIECEWISE
    if self.piecewise_graphs.has(padded_bs):
        return PIECEWISE

    return NONE
```

**Batch Size Padding**：
```python
# 预定义的 capture sizes（默认）
# [1, 2, 4, 8, 16, 24, 32, ..., 248, 256, 272, ..., max]
# 规则：[1,2,4] + range(8,256,8) + range(256,max+1,16)
#
# 实际 batch_size=5 → padding 到 8
# 实际 batch_size=130 → padding 到 136
```

> Padding 意味着会浪费一些计算（多余的 token 位会被填充），但换来的是 graph 复用率高。

#### 16.4.6 AttentionCGSupport（注意力后端兼容性）

**源码**：`vllm/v1/attention/backend.py`

```python
class AttentionCGSupport(IntEnum):
    ALWAYS = 3                      # 任何 batch 都支持 CUDA Graph
    UNIFORM_BATCH = 2               # 仅 uniform batch（所有请求同 token 数）支持
    UNIFORM_SINGLE_TOKEN_DECODE = 1 # 仅 uniform 单 token decode 支持
    NEVER = 0                       # 不支持 CUDA Graph
```

**各后端兼容性**：
| 后端 | CG Support | 说明 |
|------|-----------|------|
| FlashAttention V3 | ALWAYS (3) | Hopper SM90，原生支持 |
| FlashAttention V2 | UNIFORM_BATCH (2) | 需要 uniform batch |
| FlashInfer | ALWAYS (3) | 原生支持 |
| Triton | UNIFORM_BATCH (2) | 需要 uniform batch |

**自动降级机制**（`gpu_model_runner.py` → `_check_and_update_cudagraph_mode`）：
- 取所有 Attention 后端的 **最小 CG Support**
- 如果最小值 < 当前模式要求 → 自动降级
- 例如：配置了 FULL 但某后端只支持 UNIFORM_BATCH → 降级为 PIECEWISE

#### 16.4.7 CUDA Graph 的限制

| 限制 | 原因 | 应对方案 |
|------|------|---------|
| **固定 shape** | Graph 录制时所有 tensor shape 必须固定 | Batch size padding + 多个 graph |
| **不支持动态控制流** | if/else、循环次数不能变 | PIECEWISE 模式切分 |
| **显存开销** | 每个 graph 需要独立的中间 tensor 副本 | 限制 capture sizes 数量 |
| **LoRA 兼容** | 不同 LoRA adapter 需要不同 graph | BatchDescriptor 包含 LoRA 信息 |
| **首次 capture 开销** | 录制 graph 本身需要时间 | 服务启动时预热 |

#### 16.4.8 表达模板

> "CUDA Graph 解决的是 Decode 阶段 CPU kernel launch 开销占比过高的问题。vLLM V1 设计了 5 种模式和双层嵌套架构：外层 FULL Graph 录制整个 forward，内层 PIECEWISE 录制各个编译子图。CudagraphDispatcher 作为中央调度器，按 FULL > PIECEWISE > NONE 的优先级派发。Graph 的匹配用 BatchDescriptor（frozen dataclass，包含 num_tokens/num_reqs/uniform/lora 信息）。为了提高 graph 复用率，实际 batch size 会 padding 到预定义的 capture sizes。另外，不同 Attention 后端对 CUDA Graph 的兼容性不同（通过 AttentionCGSupport 枚举表示），系统会自动检测并降级。"

---

### 16.5 FlashAttention 原理

#### 16.5.1 核心原理

标准 Attention 的瓶颈：`Q × K^T` 生成 `[seq_len, seq_len]` 的注意力矩阵，占 O(N²) 显存，且需要大量 HBM 读写。

FlashAttention 的三大核心技术：

##### 1. Tiling（分块计算）
```
标准 Attention:
  S = Q × K^T        → 写入 HBM（O(N²) 显存）
  P = softmax(S)     → 从 HBM 读取，写入 HBM
  O = P × V          → 从 HBM 读取

FlashAttention:
  将 Q/K/V 分成小块（tile），每个 tile 在 SRAM（共享内存）中完成
  Q × K^T → softmax → × V 三步 fuse 在一个 kernel 中
  只需要 O(N) 的额外显存
```

##### 2. Online Softmax（在线 Softmax）
```
标准 Softmax 需要两次遍历：
  Pass 1: 求全局 max（数值稳定性）→ 需要完整的 S 矩阵
  Pass 2: exp(s - max) / sum

Online Softmax（Milakov & Gimelshein, 2018）：
  一次遍历即可，边扫描边更新 max 和 sum
  m_new = max(m_old, current_block_max)
  l_new = l_old × exp(m_old - m_new) + exp(current - m_new)
  O_new = O_old × (l_old/l_new) × exp(m_old - m_new) + current_attn × V / l_new
```

##### 3. Fused Kernel（算子融合）
```
标准实现：3 个独立 kernel
  kernel 1: S = Q × K^T     → 写 HBM → 读 HBM
  kernel 2: P = softmax(S)   → 写 HBM → 读 HBM
  kernel 3: O = P × V        → 写 HBM

FlashAttention：1 个 fused kernel
  所有计算在 SRAM 中完成，只最终写回 O → 1 次 HBM 写
```

**IO 复杂度对比**：
| 方法 | HBM 读写 | 额外显存 |
|------|---------|---------|
| 标准 Attention | O(N²d + N²) | O(N²) |
| FlashAttention | O(N²d² / M) | O(N) |

其中 M 是 SRAM 大小，d 是 head_dim。

#### 16.5.2 vLLM 中的 Attention 后端体系

**源码**：`vllm/v1/attention/backends/registry.py`

vLLM V1 注册了 **20+ 种 Attention 后端**，通过 `AttentionBackendEnum` 管理：

```
标准 Attention:
  FLASH_ATTN (FA2/FA3/FA4), FLASHINFER, TRITON_ATTN, FLEX_ATTENTION

MLA (Multi-Latent Attention):
  FLASH_ATTN_MLA, FLASHINFER_MLA, TRITON_MLA, CUTLASS_MLA, ...

ROCm:
  ROCM_AITER_ATTN, ROCM_AITER_MLA, ROCM_TRITON_MLA, ...
```

#### 16.5.3 后端优先级选择

**源码**：`vllm/platforms/cuda.py` → `_get_backend_priorities()`

根据 **SM 架构** 自动选择最优后端：

| 架构 | SM | 标准 Attention 优先级 | MLA 优先级 |
|------|----|--------------------|-----------|
| **Blackwell** | SM 10.x | FlashInfer > FA > Triton > Flex | Cutlass MLA > FlashInfer MLA > FA MLA > Triton MLA |
| **Ampere/Hopper** | SM 8.x-9.x | FA > FlashInfer > Triton > Flex | FA MLA > FlashInfer MLA > Triton MLA |

#### 16.5.4 FA 版本选择

**源码**：`vllm/v1/attention/backends/fa_utils.py` → `get_flash_attn_version()`

```python
def get_flash_attn_version():
    if SM >= 100:  # Blackwell
        return FA4  # 最新版本，支持 TMEM
    elif SM >= 90:  # Hopper
        return FA3  # 利用 TMA（Tensor Memory Accelerator）
    else:          # Ampere 及以下
        return FA2  # 经典版本
```

**Fallback 链**：
- FA4 不支持 ALiBi → fallback 到 FA3
- FA3 不支持某些特性 → fallback 到 FA2
- `batch_invariance` 不支持 → 降级
- TMEM 限制超出 → 降级

#### 16.5.5 PagedAttention 集成

**源码**：`vllm/v1/attention/backends/flash_attn.py`

FlashAttention 在 vLLM 中通过 `block_table` 参数原生支持 PagedAttention：

```python
# FlashAttentionImpl.forward() 核心调用
output = flash_attn_varlen_func(
    q=query,
    k=key,
    v=value,
    cu_seqlens_q=...,
    cu_seqlens_k=...,
    max_seqlen_q=...,
    max_seqlen_k=...,
    block_table=block_table,  # ← PagedAttention 的关键参数
    # block_table[i][j] = 请求 i 的第 j 个逻辑 block 对应的物理 block ID
)
```

**KV Cache 更新解耦**：
```python
class FlashAttentionBackend:
    forward_includes_kv_cache_update = False  # KV Cache 更新与 Attention 计算分离

# 先更新 KV Cache
def do_kv_cache_update(self, ...):
    reshape_and_cache_flash(key, value, kv_cache, ...)

# 再执行 Attention（可以用 CUDA Graph capture）
def forward(self, ...):
    flash_attn_varlen_func(...)
```

> 分离的好处：Attention forward 是 **固定 shape** 操作，适合 CUDA Graph 录制。

#### 16.5.6 Cascade Attention（前缀共享优化）

对于有共享前缀的请求（如相同 system prompt），可以将 Attention 分为两步：
1. 先对共享前缀计算 Attention（只算一次，所有请求共享结果）
2. 再对各自的后缀计算 Attention
3. 合并结果（利用 Online Softmax 的组合性质）

#### 16.5.7 表达模板

> "FlashAttention 的核心是三个技术：Tiling 将 Q/K/V 分块在 SRAM 中计算避免 O(N²) 的 HBM 读写，Online Softmax 实现一次遍历的 softmax 使得分块成为可能，Fused Kernel 将 QK^T → softmax → ×V 三步融合为一个 kernel。在 vLLM 中，FA 通过 `block_table` 参数原生支持 PagedAttention，且 KV Cache 更新和 Attention 计算是解耦的（`forward_includes_kv_cache_update=False`），这样 Attention forward 就可以被 CUDA Graph capture。vLLM 会根据 SM 架构自动选择 FA 版本：Hopper 用 FA3（利用 TMA），Blackwell 用 FA4，Ampere 用 FA2。"

---

### 16.6 GQA/MQA 对 KV Cache 的影响

#### 16.6.1 核心概念

| 类型 | 全称 | Q Head 数 | KV Head 数 | KV Cache 缩放 |
|------|------|-----------|-----------|---------------|
| **MHA** | Multi-Head Attention | H | H | 1x（基线） |
| **MQA** | Multi-Query Attention | H | 1 | H x 压缩 |
| **GQA** | Grouped Query Attention | H | G (1 < G < H) | H/G x 压缩 |

```
MHA:  Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8    ← 8 个 Q head
      K1 K2 K3 K4 K5 K6 K7 K8    ← 8 个 KV head（每个 Q 配一个）

GQA:  Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8    ← 8 个 Q head
      K1    K2    K3    K4        ← 4 个 KV head（每 2 个 Q 共享一个）

MQA:  Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8    ← 8 个 Q head
      K1                          ← 1 个 KV head（所有 Q 共享）
```

#### 16.6.2 典型模型配置

| 模型 | Q Head 数 | KV Head 数 | 类型 | KV 压缩比 |
|------|-----------|-----------|------|-----------|
| Llama-2-70B | 64 | 8 | GQA | 8x |
| Llama-3-8B | 32 | 8 | GQA | 4x |
| Llama-3-70B | 64 | 8 | GQA | 8x |
| Mistral-7B | 32 | 8 | GQA | 4x |
| Falcon-180B | 232 | 8 | GQA | 29x |
| PaLM | 16 | 1 | MQA | 16x |

#### 16.6.3 vLLM 中的 KV Head 数获取

**源码**：`vllm/config/model.py` → `get_num_kv_heads()`

```python
def get_num_kv_heads(self, parallel_config) -> int:
    """获取当前 TP rank 上的 KV head 数"""
    total_kv_heads = self.model_arch_config.total_num_kv_heads
    tp_size = parallel_config.tensor_parallel_size

    # MLA 架构特殊处理：返回 1
    if self.is_mla:
        return 1

    # 标准处理：KV head 数平均分到各 TP rank
    return max(1, total_kv_heads // tp_size)
```

> **注意**：当 `total_num_kv_heads < tp_size` 时，返回 1 → 多个 TP rank 共享同一个 KV head → 需要额外的 broadcast。

#### 16.6.4 KV Cache 显存计算公式

**源码**：`vllm/v1/kv_cache_interface.py` → `AttentionSpec.real_page_size_bytes`

```python
@property
def real_page_size_bytes(self) -> int:
    """单个 page（block）的 KV Cache 字节数"""
    return (2                     # K + V
            * self.block_size     # 每个 block 的 token 数
            * self.num_kv_heads   # KV head 数（GQA 关键参数）
            * self.head_size      # 每个 head 的维度
            * self.dtype.itemsize # 数据类型字节数（FP16=2, FP8=1）
           )
```

**具体计算示例**（Llama-3-8B, block_size=16, FP16）：

| 类型 | num_kv_heads | page 大小 | 相对 MHA |
|------|-------------|----------|---------|
| MHA (假设) | 32 | 2 × 16 × 32 × 128 × 2 = **256 KB** | 1x |
| GQA (实际) | 8 | 2 × 16 × 8 × 128 × 2 = **64 KB** | **4x 压缩** |
| MQA (假设) | 1 | 2 × 16 × 1 × 128 × 2 = **8 KB** | **32x 压缩** |

#### 16.6.5 GQA 对系统的综合影响

| 影响维度 | MHA | GQA | 分析 |
|---------|-----|-----|------|
| **KV Cache 显存** | H × head_size × 2 × seq_len | G × head_size × 2 × seq_len | GQA 节省 H/G 倍 |
| **最大并发请求数** | 受限于 KV Cache 总量 | 同等显存可服务更多请求 | 直接提升 throughput |
| **最大序列长度** | 受限于 KV Cache 总量 | 同等显存支持更长序列 | 支持 128K+ context |
| **Prefill Attention 计算** | 不变 | 不变（Q head 数不变） | 计算量相同 |
| **Decode Attention 带宽** | 需要读取 H 份 KV | 只需读取 G 份 KV | 减少 HBM 读取 → 加速 |
| **TP 通信** | AllReduce 大小与 H 成正比 | AllReduce 不变（Q head 数不变） | 通信量不受 GQA 影响 |
| **PagedAttention block 利用率** | 与 block_size 相关 | block 更小 → 碎片更少 | 内存利用率更高 |

#### 16.6.6 MLA（Multi-Latent Attention）特殊情况

DeepSeek-V2/V3 使用 MLA，KV Cache 被压缩到低秩 latent space：
- `get_num_kv_heads()` 返回 1
- KV Cache 存储的不是原始 K/V，而是低维 latent vector
- 极致压缩但需要额外的解压计算

#### 16.6.7 表达模板

> "GQA 的核心价值是减少 KV Cache 显存，以 Llama-3-8B 为例，32 个 Q head 只需 8 个 KV head，KV Cache 缩小 4 倍。在 vLLM 中，这直接影响 `AttentionSpec.real_page_size_bytes` 的计算——每个 page 的大小从 `2 × block_size × num_attention_heads × head_size × dtype_size` 变为 `2 × block_size × num_kv_heads × head_size × dtype_size`。实际效果是同等 GPU 显存可以服务更多并发请求或更长序列。在 TP 场景下，KV head 会均分到各 rank（`max(1, total_kv_heads // tp_size)`），如果 KV head 数少于 TP size，会退化为某些 rank 共享 KV head。"

---

### 16.7 vLLM V0 vs V1 差异

#### 16.7.1 架构级对比

| 维度 | V0（三阶段调度） | V1（两阶段调度） |
|------|----------------|----------------|
| **请求状态** | WAITING → RUNNING → SWAPPED | WAITING ↔ RUNNING（+ PREEMPTED） |
| **有无 SWAPPED 状态** | ✅ 有 | ❌ 无（源码确认无 swap_in/swap_out） |
| **抢占策略** | Swap（KV Cache 搬到 CPU）或 Recompute | **仅 Recompute**（+ Preemption Cache Shield） |
| **调度复杂度** | 三阶段：swap_in → prefill → decode | 两阶段：schedule_waiting → schedule_running |
| **KV Cache 管理** | PagedAttention block manager | Hash chain + Prefix Caching 一体化 |

#### 16.7.2 V0 三阶段调度

```
V0 调度循环：
  Phase 1: 尝试 swap_in SWAPPED 请求 → 如果 GPU 有空间，从 CPU 搬回 KV Cache
  Phase 2: 尝试调度 WAITING 请求 → Prefill 新请求
  Phase 3: 调度 RUNNING 请求 → Decode 已有请求

  当 GPU 显存不足时：
    → 选择 victim 请求
    → 将 victim 的 KV Cache swap_out 到 CPU
    → victim 状态变为 SWAPPED
    → 后续 swap_in 时再搬回
```

#### 16.7.3 V1 两阶段调度

**源码**：`vllm/v1/core/sched/scheduler.py`

```
V1 调度循环：
  Phase 1: schedule_running → 处理已在运行的请求
    → 检查是否需要抢占（KV Cache 不足时）
    → 抢占策略：Recompute（释放 KV Cache，重置 num_computed_tokens = 0）

  Phase 2: schedule_waiting → 调度等待队列中的新请求
    → 分配 KV Cache blocks
    → 利用 Prefix Cache 命中已有 blocks

  没有 Phase 3 (swap_in)！
```

**V1 的抢占实现**（`_preempt_request()`）：
```python
def _preempt_request(self, request):
    # 1. 释放该请求的所有 KV Cache blocks
    self.kv_cache_manager.free(request)

    # 2. 重置已计算 token 数（下次调度时重新计算）
    request.num_computed_tokens = 0

    # 3. 放回 waiting 队列头部（优先重新调度）
    self.waiting.appendleft(request)
```

#### 16.7.4 V1 RequestStatus 枚举

**源码**：`vllm/v1/request.py`

```python
class RequestStatus(enum.IntEnum):
    WAITING = 0
    WAITING_FOR_FSM = 1              # 等待有限状态机（structured output）
    WAITING_FOR_REMOTE_KVS = 2       # 等待远程 KV Cache（PD 分离）
    WAITING_FOR_STREAMING_REQ = 3    # 等待流式请求
    RUNNING = 4
    PREEMPTED = 5
    # ... FINISHED 状态
    # ❌ 没有 SWAPPED 状态！
```

#### 16.7.5 Preemption Cache Shield（抢占缓存护盾）

**源码**：`vllm/v1/core/scheduler.py`（自定义优化）

V1 的 Recompute 抢占有一个关键优化——**不是释放所有 blocks**：

```
标准 Recompute：释放 victim 的全部 KV Cache blocks
  → 下次重新计算所有 token → 浪费大量计算

Preemption Cache Shield：
  → 释放 victim 的 KV Cache blocks
  → 但保留可缓存的前缀 blocks（有 hash 的 blocks）
  → 下次重新调度时，前缀部分可以直接命中 Prefix Cache
  → 只需要重新计算后缀部分
```

**效果**：对于有较长共享前缀的请求（如相同 system prompt），被抢占后恢复的成本大大降低。

#### 16.7.6 V1 抢占受害者选择策略

V1 支持多种策略选择被抢占的请求：

| 策略 | 选择逻辑 | 适用场景 |
|------|---------|---------|
| **SLA-Aware** | 选择 SLA 余量最大的请求 | 有延迟 SLA 要求的生产环境 |
| **QoS** | 根据请求优先级选择 | 多优先级混合流量 |
| **MLFQ** | 多级反馈队列，新请求优先级高 | 公平调度 |
| **默认** | 最后到达的请求（LIFO） | 通用场景 |

#### 16.7.7 为什么 V1 去掉了 Swap？

| 考量 | V0 Swap | V1 Recompute |
|------|---------|-------------|
| **实现复杂度** | 高（需要管理 CPU 内存、swap 调度、异步传输） | 低（释放 + 重算） |
| **CPU 内存依赖** | 需要大量 CPU 内存做交换区 | 不需要 |
| **恢复延迟** | swap_in 受 PCIe 带宽限制（~32 GB/s） | 重算受 GPU 算力限制（通常更快） |
| **与 Prefix Cache 配合** | 难以利用 | 天然配合（Shield 保留缓存 prefix） |
| **Chunked Prefill** | 复杂 | 简单（重算就是一次 prefill） |

> V1 的设计哲学：**用计算换简洁性**。在 GPU 算力充足的现代硬件上，重算几百个 token 的开销远小于 PCIe swap 的延迟和实现复杂度。

#### 16.7.8 表达模板

> "V0 是三阶段调度（swap_in → prefill → decode），有 SWAPPED 状态，抢占时将 KV Cache swap 到 CPU。V1 简化为两阶段（schedule_running → schedule_waiting），完全去掉了 Swap，只用 Recompute 抢占——我在 V1 源码中搜索确认没有任何 swap_in/swap_out 引用。这个简化是合理的：现代 GPU 重算几百个 token 比 PCIe swap 更快，而且 V1 的 Prefix Cache 天然配合 Recompute——我做的 Preemption Cache Shield 优化就是利用这一点：抢占时保留可缓存的前缀 blocks，恢复时直接命中 Prefix Cache，只需重算后缀部分。"

---

### 16.8 HiRadixCache 实操（SGLang 三级缓存）

#### 16.8.1 核心架构

**源码**：`sglang/python/sglang/srt/mem_cache/hiradix_cache.py`

HiRadixCache 是 SGLang 的 **三级层次化 KV Cache 缓存系统**：

```
┌─────────────────────────────────┐
│         L1: GPU HBM             │  ← 最快，容量有限（RadixCache）
│    (RadixCache - Radix Tree)    │
└────────────┬────────────────────┘
             │ 异步写回 / 预取
┌────────────▼────────────────────┐
│       L2: CPU Pinned Memory     │  ← 中等速度，容量较大
│    (HiCacheController)          │
└────────────┬────────────────────┘
             │ 异步写回 / 预取
┌────────────▼────────────────────┐
│    L3: Distributed Storage      │  ← 最慢，容量最大
│  (Mooncake/3FS/NIXL/AIBrix/...) │
└─────────────────────────────────┘
```

#### 16.8.2 HiRadixCache 类继承

```python
class HiRadixCache(RadixCache):
    """三级层次化缓存，继承自 RadixCache"""

    def __init__(self, ...):
        super().__init__(...)  # 初始化 L1（GPU RadixCache）

        # L2: CPU 锁页内存
        self.token_to_kv_pool_host = ...  # MHA/MLA/NSA 不同变体

        # L3: 分布式存储后端
        self.hicache_controller = HiCacheController(
            write_policy=write_policy,
            prefetch_strategy=prefetch_strategy,
            l3_backend=l3_backend,
        )
```

#### 16.8.3 工作流程

```
请求到来，需要查找 KV Cache：

1. L1 查找（GPU RadixCache）
   ├── 命中 → 直接使用 ✓
   └── 未命中 → 继续

2. L2 查找（CPU Pinned Memory）
   ├── 命中 → 预取到 L1（异步 PCIe DMA）→ 等待完成后使用
   └── 未命中 → 继续

3. L3 查找（分布式存储）
   ├── 命中 → 预取到 L1（跨网络 + PCIe）→ 等待完成后使用
   └── 未命中 → 需要重新计算 ✗

L1 驱逐时（GPU 空间不足）：
   → 根据写回策略决定是否写入 L2/L3
```

#### 16.8.4 写回策略（Write Policy）

| 策略 | threshold | 行为 | 适用场景 |
|------|-----------|------|---------|
| **write_through** | 1 | L1 驱逐时立即写入 L2+L3 | 高命中率要求，不怕写放大 |
| **write_through_selective** | 2 | L1 驱逐时只写入被访问 ≥ 2 次的 block | 平衡写放大和命中率 |
| **write_back** | 2 | 延迟写入，只在 L2 驱逐时才写 L3 | 减少写放大，适合 L3 带宽有限 |

#### 16.8.5 预取策略（Prefetch Strategy）

| 策略 | 行为 |
|------|------|
| **best_effort** | 发起预取，不等待完成就继续调度（如果预取没来得及，就重算） |
| **wait_complete** | 发起预取，等待完成后再调度该请求 |
| **timeout** | 等待一段时间，超时则放弃预取直接重算 |

**Timeout 的线性公式**：
```
timeout = base_timeout + num_blocks × per_block_timeout
```

#### 16.8.6 L3 存储后端（7+ 种）

| 后端 | 类型 | 特点 |
|------|------|------|
| **Mooncake** | 分布式 KV Store | 月之暗面开源，高性能 |
| **3FS (Fire-Flyer File System)** | 分布式文件系统 | DeepSeek 开源 |
| **NIXL** | NVIDIA 互联库 | 利用 GPUDirect RDMA |
| **AIBrix** | 云原生 AI 基础设施 | Kubernetes 集成 |
| **HiCacheFile** | 本地文件系统 | 最简单，用于本地 SSD |
| **LMCache** | 专用缓存系统 | 针对 LLM 优化 |
| **dynamic** | 动态选择 | 运行时根据条件选择后端 |

#### 16.8.7 关键优化技术

##### 1. 计算-传输重叠（Compute-Transfer Overlap）
```
标准方式：
  [等待预取完成] → [开始计算]    ← 串行

HiCache：
  [发起预取] → [计算其他请求] → [预取完成] → [计算该请求]
  或：
  [发起预取] → [计算该请求已有部分] → [预取完成] → [继续计算]
```

##### 2. GPU-Assisted IO Kernels
- 传统 KV Cache 搬运：CPU 发起 DMA → GPU 被动接收
- GPU-Assisted：GPU 主动发起 IO 请求，利用 GPU 的高并行度
- **加速比**：~3x（相比传统 CPU 发起的 DMA）

##### 3. MLA Write-Back 优化
- MLA 架构下，KV Cache 是压缩后的 latent vector
- 只需要单个 rank 执行写回（其他 rank 的数据相同）
- 减少 TP 场景下的写放大

#### 16.8.8 配置参数

```bash
# SGLang 启动时启用 HiRadixCache
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3-8B \
    --enable-hicache \                      # 启用三级缓存
    --hicache-write-policy write_through_selective \  # 写回策略
    --hicache-prefetch-strategy best_effort \         # 预取策略
    --hicache-l3-backend mooncake \                   # L3 后端
    --hicache-l2-size-gb 32 \                         # L2 CPU 内存大小
    --hicache-l3-size-gb 256                          # L3 存储大小
```

#### 16.8.9 与 vLLM 的对比

| 维度 | vLLM V1 KV Cache | SGLang HiRadixCache |
|------|-----------------|---------------------|
| **缓存层级** | 单级（GPU only） | 三级（GPU + CPU + 分布式） |
| **缓存结构** | Hash Chain + Block Pool | RadixTree + L2 + L3 |
| **驱逐策略** | LRU（基于 ref_cnt） | LRU + 写回策略控制 |
| **前缀共享** | Hash Chain 自动去重 | RadixTree 前缀共享 |
| **跨实例共享** | 不支持（本地 only） | L3 支持跨实例共享 |
| **预取** | 无 | 多种策略（best_effort/wait/timeout） |

#### 16.8.10 表达模板

> "HiRadixCache 是 SGLang 的三级层次化 KV Cache 系统：L1 是 GPU 上的 RadixCache，L2 是 CPU 锁页内存，L3 是分布式存储（支持 Mooncake、3FS、NIXL 等 7+ 种后端）。核心工作流是：L1 未命中时从 L2/L3 预取，L1 驱逐时根据写回策略（write_through/selective/write_back）决定是否下沉。预取有三种策略：best_effort（不等待）、wait_complete（等待完成）、timeout（超时放弃）。关键优化包括计算-传输重叠（预取期间处理其他请求）和 GPU-Assisted IO Kernels（~3x 加速）。与 vLLM 的单级缓存相比，HiRadixCache 通过多级层次化大幅扩展了有效缓存容量，特别适合有大量前缀共享的长上下文场景。"
