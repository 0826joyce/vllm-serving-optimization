#!/usr/bin/env python3
"""企业级 AI 平台综合端到端压测

将原 5 个独立 Case 融合为一次运行，通过 5 个阶段递进引入压力：
  Phase 1 (0-60s)   : 稳态预热 — 建立性能基线
  Phase 2 (60-120s)  : System Prompt 灰度切换 — 暴露缓存版本管理不足
  Phase 3 (120-180s) : Gold-A 流量暴增 4× — 暴露租户隔离不足
  Phase 4 (180-240s) : Bronze 长文档暴增 — 暴露 prefill 预算隔离不足
  Phase 5 (240-300s) : 全面过载 — 暴露 SLA 保障 / 过载管理不足

全程持续: Gold-B 代码补全高频取消 — 暴露取消后缓存未保留

支持三种部署模式（通过 --mode 切换），同一套流量在不同后端配置下跑出对比数据：

  Mode A (single)     : 单实例 + Prefix Cache 优化（默认）
  Mode B (pd-disagg)  : PD 分离 + 智能路由（Prefill/Decode 独立实例）
  Mode C (spec-decode) : 单实例 + Suffix Decoding（投机解码加速）
  Mode D (all)         : 依次跑三种模式，自动生成交叉对比报告

用法:
    # Mode A: 单实例
    python workload.py --model <model> --host 127.0.0.1 --port 8000

    # Mode B: PD 分离（需要先启动 Router + Prefill + Decode 实例）
    python workload.py --model <model> --mode pd-disagg \
        --host 127.0.0.1 --port 8000 \
        --prefill-host 127.0.0.1:8100 --decode-host 127.0.0.1:8200

    # Mode C: Suffix Decoding（需要启动带 --speculative-config 的实例）
    python workload.py --model <model> --mode spec-decode

    # Mode D: 全部模式依次运行
    python workload.py --model <model> --mode all \
        --prefill-host 127.0.0.1:8100 --decode-host 127.0.0.1:8200
"""

import argparse
import asyncio
import json
import os
import random
import time
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple

import aiohttp
import numpy as np


# ============================================================
# 1. System Prompt / 消息模板定义
# ============================================================

# --- Gold-A: 金融客服（会做 Prompt 灰度切换） ---
GOLD_A_SYSTEM_PROMPT_V1 = (
    "你是金融助手 Alpha，为高端客户提供专业投资咨询。规则：\n"
    "1. 所有建议附风险提示\n2. 不推荐具体股票\n"
    "3. 合规声明：非投资建议\n4. 回答简洁专业\n\n"
    "产品知识库(2024版)：\n"
    "- 稳健理财A：年化3.5%，低风险，T+1赎回\n"
    "- 增长基金B：年化8-15%，中风险，锁定期90天\n"
    "- 股票组合C：高风险高回报，需50万起投\n"
    "当前活动：新客首购理财A送30天体验金\n"
)

GOLD_A_SYSTEM_PROMPT_V2 = (
    "你是金融助手 Alpha，为高端客户提供专业投资咨询。规则：\n"
    "1. 所有建议附风险提示\n2. 不推荐具体股票\n"
    "3. 合规声明：非投资建议\n4. 回答简洁专业\n\n"
    "产品知识库(2025版-已更新)：\n"
    "- 稳健理财A Plus：年化4.0%，低风险，T+0快速赎回\n"
    "- AI量化基金D：年化10-20%，中高风险，AI策略\n"
    "- ESG绿色基金E：年化6-12%，中风险，社会责任投资\n"
    "春季活动：购买D基金免首年管理费\n"
)

GOLD_A_USER_MESSAGES = [
    "最近有什么好的理财产品推荐吗？",
    "我想了解一下你们的基金产品",
    "低风险的产品有哪些？",
    "现在是入场的好时机吗？",
    "怎么做资产配置比较合理？",
    "你们的赎回到账时间是多久？",
    "新客户有什么优惠？",
    "我想转入50万，有什么建议？",
]

# --- Gold-B: 代码补全（高频取消） ---
CODE_CONTEXT = '''"""Data processing utilities."""
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class Config:
    input_path: str
    output_path: str
    batch_size: int = 32

class DataProcessor:
    def __init__(self, config: Config):
        self.config = config
        self._cache = {}
        self._stats = {"processed": 0, "errors": 0}

    def process_batch(self, batch: List[Dict]) -> List[Dict]:
        results = []
        for record in batch:
            try:
                processed = self._transform(record)
                results.append(processed)
                self._stats["processed"] += 1
            except Exception as e:
                self._stats["errors"] += 1
        return results

    def _transform(self, record: Dict) -> Dict:
        return {"id": record.get("id"), "value": np.log1p(abs(record.get("value", 0)))}

    def '''

TYPING_SEQUENCES = [
    "export_to_csv",
    "validate_input",
    "merge_datasets",
    "filter_outliers",
    "compute_stats",
    "load_from_db",
    "save_results",
    "run_pipeline",
]

# --- Silver: 通用客服 ---
SILVER_SYSTEM_PROMPTS = [
    "你是电商客服助手，帮助用户解决购物问题。回答热情、耐心。",
    "你是旅游规划助手，帮助用户制定旅行计划。提供实用建议。",
    "你是教育辅导助手，帮助学生解答学习问题。讲解清晰。",
]

SILVER_USER_MESSAGES = [
    "你好，我想查一下订单状态",
    "怎么修改收货地址？",
    "退货流程是怎样的？",
    "有什么优惠活动吗？",
    "这个产品支持七天无理由退换吗？",
    "推荐一下适合学生的产品",
    "帮我规划一个三天的旅行",
    "这道数学题怎么解？",
]

# --- Bronze: 长文档 RAG ---
BRONZE_SYSTEM_PROMPT = (
    "你是文档分析助手。请仔细阅读提供的文档，给出准确全面的分析。"
)

LONG_DOCUMENTS = [
    (
        "以下是云原生架构迁移调研报告，请分析关键风险：\n\n"
        "# 云原生迁移报告\n\n## 背景\n"
        "当前单体架构面临：部署周期2周，扩缩容困难，故障影响全站。\n\n"
        "## 目标架构\nKubernetes + Istio + ArgoCD + Prometheus\n\n"
        "## 拆分策略\n按DDD拆分为：用户服务、订单服务、商品服务、搜索服务、通知服务\n\n"
        "## 迁移方案\n采用Strangler Fig Pattern：\n"
        "- 第一阶段：用户服务（低风险）\n- 第二阶段：订单+商品（分布式事务Saga）\n"
        "- 第三阶段：搜索+通知（Kafka消息队列）\n- 第四阶段：下线旧系统\n\n"
        "## 风险\n- 数据一致性复杂\n- 网络延迟增加\n- 运维复杂度提升\n- 团队技能缺口\n\n"
        "请分析最大技术风险及缓解措施。"
    ),
    (
        "以下是AI编程产品PRD，请分析对推理服务的性能挑战：\n\n"
        "# AI Code Assistant PRD\n\n## 核心功能\n"
        "F1：智能补全（TTFT<200ms，多行）\nF2：代码解释（选中代码插入注释）\n"
        "F3：重构建议（代码异味检测+一键重构）\nF4：测试生成（pytest/jest/JUnit）\n"
        "F5：文档生成（docstring/JSDoc/README）\n\n"
        "## 技术架构\n前端IDE插件 → 请求路由 → vLLM集群(A100) → Redis缓存\n"
        "模型：CodeLlama 34B\n\n"
        "## 性能指标\n补全TTFT<200ms QPS>100 | 解释TTFT<1s QPS>20 | 测试<3s QPS>10\n\n"
        "请分析核心性能挑战。"
    ),
    (
        "以下是SaaS服务协议关键条款，请审查对我方不利的条款：\n\n"
        "# SaaS服务协议\n\n## 费用\n按token计费：输入$0.01/1K，输出$0.03/1K\n"
        "微调按GPU小时：$2.5/h(A100)\n逾期每日0.05%滞纳金\n\n"
        "## SLA\n月可用性≥99.9%\n未达标赔偿：99-99.9%赔10%，95-99%赔30%，<95%赔100%\n"
        "排除：客户原因/不可抗力/计划维护\n\n"
        "## 数据\n不用于训练，AES-256加密，合同终止30天删除\n\n"
        "## 知识产权\n输入归客户，输出归客户，平台归服务商\n\n"
        "## 责任限制\n间接损失不赔，最大赔偿=12个月费用\n\n"
        "## 期限\n1年自动续约，需提前90天通知不续约\n\n"
        "请指出对乙方不利的条款并给出修改建议。"
    ),
    (
        "以下是Q1项目评审会议纪要，请提取关键决策和Action Items：\n\n"
        "# 项目评审会议纪要\n日期：2025-03-15 时长2h\n\n"
        "## 进度回顾\n项目A(用户增长)完成80% | 项目B(支付重构)延期2周 | 项目C(推荐v2)提前完成CTR+15%\n"
        "决策：从C抽调2人支援B\n\n## 稳定性\nQ1发生3次P1故障\n"
        "目标Q2零P1。措施：上线前压测+混沌工程\n\n"
        "## Q2规划\nP0：AI助手集成+安全合规 | P1：国际化+性能优化\n"
        "资源：研发40人分配15AI/10国际化/5性能/10安全\n\n"
        "## 技术债务\n旧API三版本并行，TS迁移30%，测试覆盖率后端60%前端25%\n"
        "决策：Q2投入20%时间处理技术债务\n\n请按优先级排序所有决策和Action Items。"
    ),
]


# ============================================================
# 2. 数据结构
# ============================================================

@dataclass
class RequestRecord:
    """单个请求记录"""
    request_id: str
    tenant_id: str          # gold_a, gold_b, silver_0, bronze_0 ...
    sla_tier: str            # gold, silver, bronze
    request_type: str        # short, long, code_completion
    phase: str               # phase_1 .. phase_5
    send_time: float
    deploy_mode: str = "single"  # single, pd-disagg, spec-decode
    first_token_time: Optional[float] = None
    complete_time: Optional[float] = None
    ttft_ms: Optional[float] = None
    e2e_ms: Optional[float] = None
    output_tokens: int = 0
    sla_ttft_target_ms: float = 500.0
    sla_violated: bool = False
    prompt_version: Optional[str] = None   # v1/v2 (gold_a only)
    was_cancelled: bool = False
    error: Optional[str] = None

    # --- PD Disaggregation 指标（mode=pd-disagg 时采集） ---
    pd_route_type: Optional[str] = None       # "pd_split" | "decode_only" | None
    pd_prefill_instance: Optional[str] = None  # Prefill 实例地址
    pd_decode_instance: Optional[str] = None   # Decode 实例地址
    pd_kv_transfer_ms: Optional[float] = None  # KV 传输耗时（从 Router 响应头解析）

    # --- Spec Decode 指标（mode=spec-decode 时采集） ---
    spec_tokens_per_step: Optional[float] = None     # 每步有效 token 数
    spec_acceptance_rate: Optional[float] = None      # draft 接受率
    spec_draft_count: Optional[int] = None            # 本请求生成的 draft 数


# ============================================================
# 3. 辅助函数
# ============================================================

def get_phase(elapsed: float) -> str:
    """根据经过时间判断所属阶段"""
    if elapsed < 60:
        return "phase_1"
    elif elapsed < 120:
        return "phase_2"
    elif elapsed < 180:
        return "phase_3"
    elif elapsed < 240:
        return "phase_4"
    else:
        return "phase_5"


async def collect_pd_metrics(
    session: aiohttp.ClientSession,
    router_url: str,
) -> Dict:
    """从 PD Router 采集路由统计指标（/router/status 端点）"""
    try:
        async with session.get(
            f"{router_url}/router/status",
            timeout=aiohttp.ClientTimeout(total=5),
        ) as resp:
            if resp.status == 200:
                return await resp.json()
    except Exception:
        pass
    return {}


async def collect_spec_decode_metrics(
    session: aiohttp.ClientSession,
    url: str,
) -> Dict:
    """从 vLLM /metrics 端点采集投机解码指标"""
    try:
        async with session.get(
            f"{url}/metrics",
            timeout=aiohttp.ClientTimeout(total=5),
        ) as resp:
            if resp.status == 200:
                text = await resp.text()
                metrics = {}
                for line in text.split("\n"):
                    if line.startswith("#"):
                        continue
                    # 解析 Prometheus 指标
                    if "spec_decode_num_accepted_tokens" in line:
                        parts = line.strip().split()
                        if len(parts) == 2:
                            metrics["accepted_tokens"] = float(parts[1])
                    elif "spec_decode_num_draft_tokens" in line:
                        parts = line.strip().split()
                        if len(parts) == 2:
                            metrics["draft_tokens"] = float(parts[1])
                    elif "spec_decode_num_emitted_tokens" in line:
                        parts = line.strip().split()
                        if len(parts) == 2:
                            metrics["emitted_tokens"] = float(parts[1])
                return metrics
    except Exception:
        pass
    return {}


async def send_chat_request(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    system_prompt: str,
    user_message: str,
    max_tokens: int,
    record: RequestRecord,
) -> RequestRecord:
    """发送 chat completion 请求并记录指标"""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": True,
    }
    try:
        async with session.post(
            f"{url}/v1/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            # PD 模式：从响应头提取路由信息
            if record.deploy_mode == "pd-disagg":
                record.pd_route_type = resp.headers.get("X-PD-Route-Type")
                record.pd_prefill_instance = resp.headers.get("X-PD-Prefill-Instance")
                record.pd_decode_instance = resp.headers.get("X-PD-Decode-Instance")
                kv_ms = resp.headers.get("X-PD-KV-Transfer-Ms")
                if kv_ms:
                    try:
                        record.pd_kv_transfer_ms = float(kv_ms)
                    except ValueError:
                        pass

            first_token_received = False
            output_tokens = 0
            async for line in resp.content:
                line = line.decode("utf-8").strip()
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                    choices = data.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        if delta.get("content"):
                            output_tokens += 1
                            if not first_token_received:
                                record.first_token_time = time.monotonic()
                                record.ttft_ms = (
                                    record.first_token_time - record.send_time
                                ) * 1000
                                first_token_received = True
                except json.JSONDecodeError:
                    continue
            record.complete_time = time.monotonic()
            record.e2e_ms = (record.complete_time - record.send_time) * 1000
            record.output_tokens = output_tokens
            if record.ttft_ms is not None:
                record.sla_violated = record.ttft_ms > record.sla_ttft_target_ms
    except Exception as e:
        record.error = str(e)
        record.complete_time = time.monotonic()
        record.e2e_ms = (record.complete_time - record.send_time) * 1000
        record.sla_violated = True
    return record


async def send_completion_request(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    record: RequestRecord,
    cancel_event: asyncio.Event,
) -> RequestRecord:
    """发送 completion 请求（代码补全），支持取消"""
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }
    try:
        async with session.post(
            f"{url}/v1/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            first_token_received = False
            output_tokens = 0
            async for line in resp.content:
                if cancel_event.is_set():
                    record.was_cancelled = True
                    break
                line = line.decode("utf-8").strip()
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                    choices = data.get("choices", [])
                    if choices and choices[0].get("text"):
                        output_tokens += 1
                        if not first_token_received:
                            record.first_token_time = time.monotonic()
                            record.ttft_ms = (
                                record.first_token_time - record.send_time
                            ) * 1000
                            first_token_received = True
                except json.JSONDecodeError:
                    continue
            record.complete_time = time.monotonic()
            record.e2e_ms = (record.complete_time - record.send_time) * 1000
            record.output_tokens = output_tokens
            if record.ttft_ms is not None and not record.was_cancelled:
                record.sla_violated = record.ttft_ms > record.sla_ttft_target_ms
    except asyncio.CancelledError:
        record.was_cancelled = True
        record.complete_time = time.monotonic()
        record.e2e_ms = (record.complete_time - record.send_time) * 1000
    except Exception as e:
        record.error = str(e)
        record.complete_time = time.monotonic()
        record.e2e_ms = (record.complete_time - record.send_time) * 1000
    return record


async def append_record(
    record: RequestRecord,
    all_records: List[RequestRecord],
    lock: asyncio.Lock,
):
    async with lock:
        all_records.append(record)


# ============================================================
# 4. 各租户的流量生成协程
# ============================================================

async def gold_a_generator(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    start_time: float,
    duration: float,
    all_records: List[RequestRecord],
    lock: asyncio.Lock,
    deploy_mode: str = "single",
):
    """Gold-A: 金融客服，Phase 2 做 Prompt 切换，Phase 3/5 暴增"""
    count = 0
    end = start_time + duration

    while time.monotonic() < end:
        now = time.monotonic()
        elapsed = now - start_time
        phase = get_phase(elapsed)

        # 确定 QPS
        if phase in ("phase_3", "phase_4"):
            qps = 32.0   # 暴增 4x
        elif phase == "phase_5":
            qps = 48.0   # 过载
        else:
            qps = 8.0

        # 确定 System Prompt 版本
        if phase == "phase_1":
            sp = GOLD_A_SYSTEM_PROMPT_V1
            ver = "v1"
        elif phase == "phase_2":
            # 灰度切换：前 30s 10% v2，后 30s 100% v2
            phase_elapsed = elapsed - 60
            if phase_elapsed < 30 and random.random() > 0.1:
                sp = GOLD_A_SYSTEM_PROMPT_V1
                ver = "v1"
            else:
                sp = GOLD_A_SYSTEM_PROMPT_V2
                ver = "v2"
        else:
            sp = GOLD_A_SYSTEM_PROMPT_V2
            ver = "v2"

        record = RequestRecord(
            request_id=f"gold_a_{count}",
            tenant_id="gold_a",
            sla_tier="gold",
            request_type="short",
            phase=phase,
            send_time=time.monotonic(),
            deploy_mode=deploy_mode,
            sla_ttft_target_ms=200.0,
            prompt_version=ver,
        )

        asyncio.create_task(_chat_and_record(
            session, url, model, sp,
            random.choice(GOLD_A_USER_MESSAGES),
            100, record, all_records, lock,
        ))

        count += 1
        await asyncio.sleep(np.random.exponential(1.0 / qps))


async def gold_b_generator(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    start_time: float,
    duration: float,
    all_records: List[RequestRecord],
    lock: asyncio.Lock,
    deploy_mode: str = "single",
):
    """Gold-B: 代码补全，全程高频取消"""
    session_count = 0
    end = start_time + duration

    while time.monotonic() < end:
        elapsed = time.monotonic() - start_time
        phase = get_phase(elapsed)

        typing_seq = TYPING_SEQUENCES[session_count % len(TYPING_SEQUENCES)]
        prev_cancel_event: Optional[asyncio.Event] = None
        prev_task: Optional[asyncio.Task] = None

        for char_idx, char in enumerate(typing_seq):
            if time.monotonic() >= end:
                break

            typed_so_far = typing_seq[:char_idx + 1]
            prompt = CODE_CONTEXT + typed_so_far
            is_last = (char_idx == len(typing_seq) - 1)

            record = RequestRecord(
                request_id=f"gold_b_s{session_count}_c{char_idx}",
                tenant_id="gold_b",
                sla_tier="gold",
                request_type="code_completion",
                phase=get_phase(time.monotonic() - start_time),
                send_time=time.monotonic(),
                deploy_mode=deploy_mode,
                sla_ttft_target_ms=200.0,
            )

            # 取消上一个请求
            if prev_cancel_event is not None:
                prev_cancel_event.set()

            cancel_event = asyncio.Event()

            if is_last:
                # 最后一个字符：等待完成
                result = await send_completion_request(
                    session, url, model, prompt, 50,
                    record, cancel_event,
                )
                await append_record(result, all_records, lock)
            else:
                prev_cancel_event = cancel_event

                async def _fire(rec, ce, p):
                    r = await send_completion_request(
                        session, url, model, p, 50, rec, ce,
                    )
                    await append_record(r, all_records, lock)

                prev_task = asyncio.create_task(
                    _fire(record, cancel_event, prompt)
                )
                # 模拟击键间隔 200-300ms
                await asyncio.sleep(random.uniform(0.2, 0.3))

        # 等待最后一个后台任务
        if prev_task is not None:
            try:
                await asyncio.wait_for(prev_task, timeout=5.0)
            except asyncio.TimeoutError:
                pass

        session_count += 1
        # session 间隔 1-2s
        await asyncio.sleep(random.uniform(1.0, 2.0))


async def silver_generator(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    tenant_idx: int,
    start_time: float,
    duration: float,
    all_records: List[RequestRecord],
    lock: asyncio.Lock,
    deploy_mode: str = "single",
):
    """Silver 租户：稳定短对话"""
    count = 0
    end = start_time + duration
    sp = SILVER_SYSTEM_PROMPTS[tenant_idx % len(SILVER_SYSTEM_PROMPTS)]
    tid = f"silver_{tenant_idx}"

    while time.monotonic() < end:
        elapsed = time.monotonic() - start_time
        phase = get_phase(elapsed)
        qps = 12.0 if phase == "phase_5" else 8.0  # Phase 5 略增

        record = RequestRecord(
            request_id=f"{tid}_{count}",
            tenant_id=tid,
            sla_tier="silver",
            request_type="short",
            phase=phase,
            send_time=time.monotonic(),
            deploy_mode=deploy_mode,
            sla_ttft_target_ms=500.0,
        )

        asyncio.create_task(_chat_and_record(
            session, url, model, sp,
            random.choice(SILVER_USER_MESSAGES),
            120, record, all_records, lock,
        ))

        count += 1
        await asyncio.sleep(np.random.exponential(1.0 / qps))


async def bronze_generator(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    tenant_idx: int,
    start_time: float,
    duration: float,
    all_records: List[RequestRecord],
    lock: asyncio.Lock,
    deploy_mode: str = "single",
):
    """Bronze 租户：长文档 RAG，Phase 4/5 暴增"""
    count = 0
    end = start_time + duration
    tid = f"bronze_{tenant_idx}"

    while time.monotonic() < end:
        elapsed = time.monotonic() - start_time
        phase = get_phase(elapsed)

        if phase == "phase_4":
            qps = 10.0    # 暴增
        elif phase == "phase_5":
            qps = 15.0    # 过载
        else:
            qps = 3.0

        record = RequestRecord(
            request_id=f"{tid}_{count}",
            tenant_id=tid,
            sla_tier="bronze",
            request_type="long",
            phase=phase,
            send_time=time.monotonic(),
            deploy_mode=deploy_mode,
            sla_ttft_target_ms=3000.0,
        )

        asyncio.create_task(_chat_and_record(
            session, url, model,
            BRONZE_SYSTEM_PROMPT,
            random.choice(LONG_DOCUMENTS),
            300, record, all_records, lock,
        ))

        count += 1
        if qps > 0:
            await asyncio.sleep(np.random.exponential(1.0 / qps))
        else:
            await asyncio.sleep(5.0)


async def _chat_and_record(
    session, url, model, system_prompt, user_message,
    max_tokens, record, all_records, lock,
):
    """发送 chat 请求并记录"""
    result = await send_chat_request(
        session, url, model,
        system_prompt, user_message,
        max_tokens, record,
    )
    await append_record(result, all_records, lock)


# ============================================================
# 5. 实时监控
# ============================================================

async def realtime_monitor(
    all_records: List[RequestRecord],
    lock: asyncio.Lock,
    start_time: float,
    duration: float,
):
    """每 10 秒打印一次各阶段、各租户等级的实时指标"""
    try:
        while True:
            await asyncio.sleep(10)
            elapsed = time.monotonic() - start_time
            if elapsed > duration + 30:
                break

            phase = get_phase(min(elapsed, duration - 1))

            async with lock:
                # 最近 10 秒的已完成请求
                now = time.monotonic()
                recent = [
                    r for r in all_records
                    if r.ttft_ms is not None
                    and (now - r.send_time) < 15
                ]
                total = len(all_records)

            if not recent:
                print(f"  [t={elapsed:.0f}s {phase}] waiting for responses...")
                continue

            # 按 SLA 等级统计
            parts = [f"[t={elapsed:.0f}s {phase}] total={total}"]
            for tier in ["gold", "silver", "bronze"]:
                tier_recs = [r for r in recent if r.sla_tier == tier]
                if not tier_recs:
                    continue
                ttfts = [r.ttft_ms for r in tier_recs]
                violated = sum(1 for r in tier_recs if r.sla_violated)
                cancelled = sum(1 for r in tier_recs if r.was_cancelled)
                p99 = np.percentile(ttfts, 99) if len(ttfts) >= 5 else max(ttfts)
                status = "✅" if violated == 0 else "❌"
                extra = f" cancel={cancelled}" if cancelled > 0 else ""
                parts.append(
                    f"{status}{tier}:n={len(tier_recs)}"
                    f" p99={p99:.0f}ms viol={violated}{extra}"
                )
            print(f"  {' | '.join(parts)}")

    except asyncio.CancelledError:
        pass


# ============================================================
# 6. 统计报告
# ============================================================

def print_report(records: List[RequestRecord]):
    """打印分阶段、分租户的详细统计"""
    if not records:
        print("No records collected!")
        return

    valid = [r for r in records if r.ttft_ms is not None]
    cancelled = [r for r in records if r.was_cancelled]
    errors = [r for r in records if r.error is not None]

    print(f"\n{'='*70}")
    print(f"  综合压测报告 — {len(records)} 请求")
    print(f"{'='*70}")
    print(f"  有效: {len(valid)}  取消: {len(cancelled)}  错误: {len(errors)}")

    # ---- 1. 分阶段总览 ----
    print(f"\n{'─'*70}")
    print(f"  1. 分阶段总览")
    print(f"{'─'*70}")

    phase_names = {
        "phase_1": "Phase 1: 稳态预热",
        "phase_2": "Phase 2: Prompt切换",
        "phase_3": "Phase 3: Gold暴增",
        "phase_4": "Phase 4: 长文档暴增",
        "phase_5": "Phase 5: 全面过载",
    }

    for phase_key in sorted(phase_names.keys()):
        phase_recs = [r for r in valid if r.phase == phase_key]
        if not phase_recs:
            continue
        ttfts = [r.ttft_ms for r in phase_recs]
        violated = sum(1 for r in phase_recs if r.sla_violated)
        print(f"\n  {phase_names[phase_key]}")
        print(f"    请求数: {len(phase_recs)}  "
              f"SLA违约: {violated} ({violated/len(phase_recs)*100:.1f}%)")
        print(f"    TTFT: P50={np.percentile(ttfts,50):.0f}ms "
              f"P95={np.percentile(ttfts,95):.0f}ms "
              f"P99={np.percentile(ttfts,99):.0f}ms "
              f"max={max(ttfts):.0f}ms")

        # 按租户等级
        for tier in ["gold", "silver", "bronze"]:
            tier_recs = [r for r in phase_recs if r.sla_tier == tier]
            if not tier_recs:
                continue
            t_ttfts = [r.ttft_ms for r in tier_recs]
            t_viol = sum(1 for r in tier_recs if r.sla_violated)
            target = tier_recs[0].sla_ttft_target_ms
            status = "✅" if t_viol == 0 else "❌"
            p99 = np.percentile(t_ttfts, 99) if len(t_ttfts) >= 2 else max(t_ttfts)
            print(f"      {status} {tier.upper()} (SLA {target:.0f}ms): "
                  f"n={len(tier_recs)} "
                  f"P99={p99:.0f}ms "
                  f"violated={t_viol}/{len(tier_recs)}")

    # ---- 2. 关键场景分析 ----
    print(f"\n{'─'*70}")
    print(f"  2. 关键场景分析")
    print(f"{'─'*70}")

    # 2a. Prompt 切换影响 (Phase 2 Gold-A)
    p2_gold_a = [r for r in valid
                 if r.phase == "phase_2" and r.tenant_id == "gold_a"]
    if p2_gold_a:
        v1_recs = [r for r in p2_gold_a if r.prompt_version == "v1"]
        v2_recs = [r for r in p2_gold_a if r.prompt_version == "v2"]
        print(f"\n  2a. Prompt 切换 (Phase 2, Gold-A)")
        for label, subset in [("v1(旧)", v1_recs), ("v2(新)", v2_recs)]:
            if subset:
                ttfts = [r.ttft_ms for r in subset]
                print(f"      {label}: n={len(subset)} "
                      f"mean={np.mean(ttfts):.0f}ms "
                      f"P99={np.percentile(ttfts,99):.0f}ms")
        # 对比 Phase 1 的 Gold-A
        p1_gold_a = [r for r in valid
                     if r.phase == "phase_1" and r.tenant_id == "gold_a"]
        if p1_gold_a:
            p1_ttfts = [r.ttft_ms for r in p1_gold_a]
            print(f"      对比 Phase 1 基线: mean={np.mean(p1_ttfts):.0f}ms "
                  f"P99={np.percentile(p1_ttfts,99):.0f}ms")

    # 2b. 租户隔离 (Phase 3 Silver vs Gold-A)
    p3_silver = [r for r in valid
                 if r.phase == "phase_3" and r.sla_tier == "silver"]
    p1_silver = [r for r in valid
                 if r.phase == "phase_1" and r.sla_tier == "silver"]
    if p3_silver:
        print(f"\n  2b. 租户隔离 (Phase 3, Silver 受 Gold-A 暴增影响)")
        s_ttfts = [r.ttft_ms for r in p3_silver]
        s_viol = sum(1 for r in p3_silver if r.sla_violated)
        print(f"      Phase 3 Silver: n={len(p3_silver)} "
              f"P99={np.percentile(s_ttfts,99):.0f}ms "
              f"violated={s_viol}/{len(p3_silver)}")
        if p1_silver:
            p1_s_ttfts = [r.ttft_ms for r in p1_silver]
            print(f"      Phase 1 Silver基线: "
                  f"P99={np.percentile(p1_s_ttfts,99):.0f}ms")

    # 2c. 长短混合 (Phase 4 短对话受影响)
    p4_short = [r for r in valid
                if r.phase == "phase_4" and r.request_type == "short"]
    p1_short = [r for r in valid
                if r.phase == "phase_1" and r.request_type == "short"]
    if p4_short:
        print(f"\n  2c. 长短混合 (Phase 4, 短对话受长文档暴增影响)")
        sh_ttfts = [r.ttft_ms for r in p4_short]
        sh_viol = sum(1 for r in p4_short if r.sla_violated)
        print(f"      Phase 4 短对话: n={len(p4_short)} "
              f"P99={np.percentile(sh_ttfts,99):.0f}ms "
              f"violated={sh_viol}/{len(p4_short)}")
        if p1_short:
            p1_sh_ttfts = [r.ttft_ms for r in p1_short]
            print(f"      Phase 1 短对话基线: "
                  f"P99={np.percentile(p1_sh_ttfts,99):.0f}ms")

    # 2d. 过载 (Phase 5 全局)
    p5 = [r for r in valid if r.phase == "phase_5"]
    if p5:
        print(f"\n  2d. 过载 (Phase 5, 全局)")
        p5_ttfts = [r.ttft_ms for r in p5]
        p5_viol = sum(1 for r in p5 if r.sla_violated)
        print(f"      n={len(p5)} "
              f"P99={np.percentile(p5_ttfts,99):.0f}ms "
              f"SLA违约率={p5_viol/len(p5)*100:.1f}%")

    # ---- 3. 代码补全专项 (Gold-B 全程) ----
    print(f"\n{'─'*70}")
    print(f"  3. 代码补全专项 (Gold-B)")
    print(f"{'─'*70}")

    goldb_all = [r for r in records if r.tenant_id == "gold_b"]
    goldb_cancelled = [r for r in goldb_all if r.was_cancelled]
    goldb_completed = [r for r in goldb_all
                       if not r.was_cancelled and r.ttft_ms is not None]
    if goldb_all:
        print(f"  总请求: {len(goldb_all)}")
        print(f"  取消: {len(goldb_cancelled)} "
              f"({len(goldb_cancelled)/max(1,len(goldb_all))*100:.1f}%)")
        print(f"  完成: {len(goldb_completed)}")
        if goldb_completed:
            c_ttfts = [r.ttft_ms for r in goldb_completed]
            print(f"  完成请求 TTFT: mean={np.mean(c_ttfts):.0f}ms "
                  f"P99={np.percentile(c_ttfts,99):.0f}ms")
            # 如果 prefix cache 生效，连续 session 的后续请求 TTFT 应该很低
            # 如果没有生效，每次都是全量 prefill

    # ---- 4. PD 分离专项（mode=pd-disagg 时有数据） ----
    pd_recs = [r for r in valid if r.deploy_mode == "pd-disagg"]
    if pd_recs:
        print(f"\n{'─'*70}")
        print(f"  4. PD 分离专项")
        print(f"{'─'*70}")

        pd_split_recs = [r for r in pd_recs if r.pd_route_type == "pd_split"]
        decode_only_recs = [r for r in pd_recs
                           if r.pd_route_type == "decode_only"]
        print(f"  路由分布: PD分离={len(pd_split_recs)} "
              f"直发Decode={len(decode_only_recs)} "
              f"未分类={len(pd_recs)-len(pd_split_recs)-len(decode_only_recs)}")

        # 按 Phase 对比 PD 分离模式的 TTFT
        for phase_key in sorted(phase_names.keys()):
            pd_phase = [r for r in pd_recs if r.phase == phase_key]
            if not pd_phase:
                continue
            pd_ttfts = [r.ttft_ms for r in pd_phase]
            pd_viol = sum(1 for r in pd_phase if r.sla_violated)
            print(f"  {phase_names[phase_key]}:")
            print(f"    n={len(pd_phase)} "
                  f"P50={np.percentile(pd_ttfts,50):.0f}ms "
                  f"P99={np.percentile(pd_ttfts,99):.0f}ms "
                  f"violated={pd_viol}")

        # KV 传输延迟分析
        kv_recs = [r for r in pd_recs if r.pd_kv_transfer_ms is not None]
        if kv_recs:
            kv_ms = [r.pd_kv_transfer_ms for r in kv_recs]
            print(f"\n  KV 传输延迟 (n={len(kv_recs)}):")
            print(f"    P50={np.percentile(kv_ms,50):.1f}ms "
                  f"P95={np.percentile(kv_ms,95):.1f}ms "
                  f"P99={np.percentile(kv_ms,99):.1f}ms")

        # PD 分离对长文档暴增 (Phase 4) 的效果
        pd_p4_short = [r for r in pd_recs
                       if r.phase == "phase_4" and r.request_type == "short"]
        if pd_p4_short:
            ttfts = [r.ttft_ms for r in pd_p4_short]
            print(f"\n  PD 分离下 Phase 4 短对话 TTFT:")
            print(f"    n={len(pd_p4_short)} "
                  f"P99={np.percentile(ttfts,99):.0f}ms "
                  f"(长 Prefill 不干扰 Decode ITL)")

    # ---- 5. Spec Decode 专项（mode=spec-decode 时有数据） ----
    sd_recs = [r for r in valid if r.deploy_mode == "spec-decode"]
    if sd_recs:
        print(f"\n{'─'*70}")
        print(f"  5. Spec Decode 专项")
        print(f"{'─'*70}")

        # 各阶段的生成速度对比
        for phase_key in sorted(phase_names.keys()):
            sd_phase = [r for r in sd_recs if r.phase == phase_key]
            if not sd_phase:
                continue
            sd_ttfts = [r.ttft_ms for r in sd_phase]
            sd_e2e = [r.e2e_ms for r in sd_phase if r.e2e_ms is not None]
            sd_out = [r.output_tokens for r in sd_phase if r.output_tokens > 0]
            print(f"  {phase_names[phase_key]}:")
            parts = [f"n={len(sd_phase)}",
                     f"P99_TTFT={np.percentile(sd_ttfts,99):.0f}ms"]
            if sd_e2e:
                parts.append(f"P50_E2E={np.percentile(sd_e2e,50):.0f}ms")
            if sd_out:
                parts.append(f"avg_out={np.mean(sd_out):.0f}tok")
            print(f"    {' '.join(parts)}")

        # 代码补全场景的 Spec Decode 效果（Gold-B 的后缀匹配很适合）
        sd_goldb = [r for r in sd_recs
                    if r.tenant_id == "gold_b" and not r.was_cancelled
                    and r.ttft_ms is not None]
        if sd_goldb:
            gb_ttfts = [r.ttft_ms for r in sd_goldb]
            gb_e2e = [r.e2e_ms for r in sd_goldb if r.e2e_ms is not None]
            print(f"\n  Gold-B 代码补全 + Spec Decode:")
            print(f"    n={len(sd_goldb)} "
                  f"TTFT P99={np.percentile(gb_ttfts,99):.0f}ms")
            if gb_e2e:
                print(f"    E2E P50={np.percentile(gb_e2e,50):.0f}ms "
                      f"(后缀匹配加速代码补全生成)")

    # ---- 6. 多模式交叉对比（mode=all 时有数据） ----
    mode_set = set(r.deploy_mode for r in valid)
    if len(mode_set) > 1:
        print(f"\n{'─'*70}")
        print(f"  6. 多模式交叉对比")
        print(f"{'─'*70}")

        mode_labels = {
            "single": "单实例",
            "pd-disagg": "PD分离",
            "spec-decode": "SpecDec",
        }

        print(f"\n  {'Phase':<18}", end="")
        for m in sorted(mode_set):
            print(f"{'['+mode_labels.get(m,m)+'] P99':>16}", end="")
        print()
        print(f"  {'─'*18}", end="")
        for _ in mode_set:
            print(f"{'─'*16}", end="")
        print()

        for phase_key in sorted(phase_names.keys()):
            print(f"  {phase_names[phase_key][:16]:<18}", end="")
            for m in sorted(mode_set):
                m_recs = [r for r in valid
                          if r.deploy_mode == m and r.phase == phase_key]
                if m_recs:
                    ttfts = [r.ttft_ms for r in m_recs]
                    p99 = np.percentile(ttfts, 99)
                    print(f"{p99:>14.0f}ms", end="")
                else:
                    print(f"{'—':>16}", end="")
            print()

        # Gold-B 模式对比
        print(f"\n  Gold-B 补全对比:")
        for m in sorted(mode_set):
            m_gb = [r for r in valid
                    if r.deploy_mode == m and r.tenant_id == "gold_b"
                    and not r.was_cancelled and r.ttft_ms is not None]
            if m_gb:
                gb_t = [r.ttft_ms for r in m_gb]
                print(f"    [{mode_labels.get(m,m)}] n={len(m_gb)} "
                      f"TTFT P99={np.percentile(gb_t,99):.0f}ms")

    # ---- SLA 违约汇总 ----
    section_num = 4
    if pd_recs:
        section_num += 1
    if sd_recs:
        section_num += 1
    if len(mode_set) > 1:
        section_num += 1

    print(f"\n{'─'*70}")
    print(f"  {section_num}. SLA 违约汇总")
    print(f"{'─'*70}")

    print(f"\n  {'Tier':<8} {'Phase':<12} {'Total':<8} {'Violated':<10} {'Rate':<8}")
    print(f"  {'─'*46}")
    for tier in ["gold", "silver", "bronze"]:
        for pk in sorted(phase_names.keys()):
            subset = [r for r in valid if r.sla_tier == tier and r.phase == pk]
            if not subset:
                continue
            viol = sum(1 for r in subset if r.sla_violated)
            rate = viol / len(subset) * 100
            marker = " ❌" if rate > 5 else " ⚠️" if rate > 1 else ""
            print(f"  {tier:<8} {pk:<12} {len(subset):<8} "
                  f"{viol:<10} {rate:<7.1f}%{marker}")

    print()


# ============================================================
# 7. 主函数
# ============================================================

async def run_workload(args):
    """运行综合压测（支持多模式）"""
    modes_to_run = []
    if args.mode == "all":
        modes_to_run = ["single", "pd-disagg", "spec-decode"]
    else:
        modes_to_run = [args.mode]

    all_mode_records: List[RequestRecord] = []

    for mode in modes_to_run:
        # 根据模式确定目标 URL
        if mode == "pd-disagg":
            if args.router_host:
                url = f"http://{args.router_host}"
            else:
                url = f"http://{args.host}:{args.port}"
        else:
            url = f"http://{args.host}:{args.port}"

        os.makedirs(args.output_dir, exist_ok=True)

        mode_label = {
            "single": "单实例 + Prefix Cache",
            "pd-disagg": "PD 分离 + 智能路由",
            "spec-decode": "单实例 + Suffix Decoding",
        }.get(mode, mode)

        print(f"\n{'='*70}")
        print(f"  企业级 AI 平台 — 综合端到端压测")
        print(f"  部署模式: {mode_label}")
        print(f"{'='*70}")
        print(f"  Model: {args.model}")
        print(f"  Target: {url}")
        print(f"  Duration: {args.duration}s (5 phases × 60s)")
        print(f"  Tenants: Gold-A, Gold-B, Silver×3, Bronze×2")
        if mode == "pd-disagg":
            print(f"  Prefill: {args.prefill_host or '(via router)'}")
            print(f"  Decode:  {args.decode_host or '(via router)'}")
        print(f"  Output: {args.output_dir}")
        print()
        print(f"  Phase 1 (0-60s)   : 稳态预热")
        print(f"  Phase 2 (60-120s)  : Prompt 切换")
        print(f"  Phase 3 (120-180s) : Gold-A 暴增 4×")
        print(f"  Phase 4 (180-240s) : 长文档暴增")
        print(f"  Phase 5 (240-300s) : 全面过载")
        print(f"  全程: Gold-B 代码补全高频取消")
        print()

        all_records: List[RequestRecord] = []
        lock = asyncio.Lock()

        connector = aiohttp.TCPConnector(limit=500)
        async with aiohttp.ClientSession(connector=connector) as session:
            start_time = time.monotonic()

            tasks = []

            # Gold-A: 金融客服
            tasks.append(asyncio.create_task(
                gold_a_generator(session, url, args.model,
                                 start_time, args.duration,
                                 all_records, lock, mode)
            ))

            # Gold-B: 代码补全
            tasks.append(asyncio.create_task(
                gold_b_generator(session, url, args.model,
                                 start_time, args.duration,
                                 all_records, lock, mode)
            ))

            # Silver × 3
            for i in range(3):
                tasks.append(asyncio.create_task(
                    silver_generator(session, url, args.model, i,
                                     start_time, args.duration,
                                     all_records, lock, mode)
                ))

            # Bronze × 2
            for i in range(2):
                tasks.append(asyncio.create_task(
                    bronze_generator(session, url, args.model, i,
                                     start_time, args.duration,
                                     all_records, lock, mode)
                ))

            # 实时监控
            monitor = asyncio.create_task(
                realtime_monitor(all_records, lock,
                                 start_time, args.duration)
            )

            # PD 模式：定期采集路由指标
            pd_metrics_task = None
            if mode == "pd-disagg":
                async def _collect_pd_loop():
                    try:
                        while True:
                            await asyncio.sleep(30)
                            metrics = await collect_pd_metrics(session, url)
                            if metrics:
                                print(f"  [PD Router] {json.dumps(metrics, indent=None)}")
                    except asyncio.CancelledError:
                        pass
                pd_metrics_task = asyncio.create_task(_collect_pd_loop())

            # Spec Decode 模式：定期采集指标
            sd_metrics_task = None
            if mode == "spec-decode":
                async def _collect_sd_loop():
                    try:
                        while True:
                            await asyncio.sleep(30)
                            metrics = await collect_spec_decode_metrics(
                                session, url)
                            if metrics:
                                acc = metrics.get("accepted_tokens", 0)
                                draft = metrics.get("draft_tokens", 0)
                                rate = acc / max(1, draft)
                                print(f"  [SpecDecode] accept_rate={rate:.1%} "
                                      f"accepted={acc:.0f} draft={draft:.0f}")
                    except asyncio.CancelledError:
                        pass
                sd_metrics_task = asyncio.create_task(_collect_sd_loop())

            await asyncio.gather(*tasks)
            monitor.cancel()
            if pd_metrics_task:
                pd_metrics_task.cancel()
            if sd_metrics_task:
                sd_metrics_task.cancel()

        # 等待剩余请求
        print("\n等待剩余请求完成...")
        await asyncio.sleep(10)

        all_mode_records.extend(all_records)

        # 保存单模式结果
        mode_output = os.path.join(args.output_dir, f"results_{mode}.json")
        output_data = {
            "args": vars(args),
            "deploy_mode": mode,
            "total_requests": len(all_records),
            "records": [asdict(r) for r in all_records],
        }
        with open(mode_output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"结果已保存: {mode_output}")

    # 保存合并结果（all 模式时）
    if len(modes_to_run) > 1:
        combined_file = os.path.join(args.output_dir, "results_combined.json")
        combined_data = {
            "args": vars(args),
            "modes": modes_to_run,
            "total_requests": len(all_mode_records),
            "records": [asdict(r) for r in all_mode_records],
        }
        with open(combined_file, "w") as f:
            json.dump(combined_data, f, indent=2)
        print(f"\n合并结果已保存: {combined_file}")

    # 打印报告（包含所有模式的数据）
    print_report(all_mode_records)


def main():
    parser = argparse.ArgumentParser(
        description="企业级 AI 平台综合端到端压测（支持多部署模式对比）"
    )
    parser.add_argument("--model", type=str, required=True,
                        help="模型名称")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--duration", type=int, default=300,
                        help="总运行时间(秒)，默认 300 (5 phases × 60s)")
    parser.add_argument("--output-dir", type=str,
                        default="results/combined/",
                        help="结果输出目录")
    parser.add_argument("--seed", type=int, default=42)

    # ---- 部署模式 ----
    parser.add_argument(
        "--mode",
        choices=["single", "pd-disagg", "spec-decode", "all"],
        default="single",
        help=("部署模式: "
              "single=单实例+PrefixCache(默认), "
              "pd-disagg=PD分离+智能路由, "
              "spec-decode=单实例+SuffixDecoding, "
              "all=依次跑三种模式并交叉对比")
    )

    # ---- PD 分离参数 ----
    parser.add_argument("--router-host", type=str, default=None,
                        help="PD 模式: Router 代理地址 (host:port)，"
                             "默认使用 --host:--port")
    parser.add_argument("--prefill-host", type=str, default=None,
                        help="PD 模式: Prefill 实例地址 (host:port)，"
                             "仅用于报告显示")
    parser.add_argument("--decode-host", type=str, default=None,
                        help="PD 模式: Decode 实例地址 (host:port)，"
                             "仅用于报告显示")

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    asyncio.run(run_workload(args))


if __name__ == "__main__":
    main()
