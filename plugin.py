"""表情包智能审核插件（自主实现）。

问题背景：
  - 现有表情包选择只靠「文字标签 + 低阈值语义匹配」，不看图也不看语境
  - 结果经常乱发：气氛严肃时发搞笑表情、文字安慰人配大笑表情
  - 群友反馈「表情包用的不到位」

本插件做法（发送前二次把关）：
  1. 从待发消息里取出表情段（含 base64 图片数据）和文字
  2. 拉最近聊天上下文
  3. 先把表情图交给视觉模型「看图」得出情绪判断（可选）
  4. 再结合上下文让模型判断这个表情配这句话合不合适
  5. 不合适就撤掉表情、保留文字；合适就原样放行

设计要点：
  - 全程零成本（用 OFM 网关的免费渠道模型）
  - 任何异常/超时都放行，绝不因为审核失败导致消息发不出去
  - 严格程度、审核模型、并发、概率都可配置
"""

from __future__ import annotations

import asyncio
import base64
import json
import random
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import Field
from maibot_sdk import HookHandler, MaiBotPlugin
from maibot_sdk.config import PluginConfigBase
from maibot_sdk.types import HookMode, HookOrder

# ── 动态生成「看图模型」的下拉选项 ──
# 在导入阶段读取宿主 model_config.toml 里所有 visual=true 的模型，
# 这样下拉框列出的就是本机真实可用的视觉模型，换机器自动适配。
try:
    from .vision_models import build_vision_choices as _bmc

    _VISION_CHOICES = tuple(_bmc())
except Exception:  # noqa: BLE001
    _VISION_CHOICES = ("auto", "off")

VisionModelChoice = Literal[_VISION_CHOICES]  # type: ignore[valid-type]

VISION_PROMPT = "这张表情包表达的什么情绪或什么意思？用不超过 8 个中文词概括，只输出词组，不要解释。"

AUDIT_PROMPT = """你在帮一个 QQ 群聊机器人把关表情包用得对不对。

【最近聊天】
{context}

【机器人准备发的内容】
文字：{text}
表情包（{emoji_count} 个）：{emoji_desc}

请判断：这些表情包配这句话、在这个语境下合不合适？

判「不合适」：
- 文字在安慰/认真回应，表情却是搞笑、嘲讽、无所谓
- 群里有人的事比较严肃（吵架、难过、有人被骂），这时候发表情很突兀
- 表情和文字/话题完全无关，明显硬凑
- 文字已经把情绪说清楚，表情纯属多余
- 同一段对话里刚连发过表情，再发像刷屏

判「合适」：
- 表情自然强化了文字情绪
- 语境轻松，发表情显得有人味、不机械
- 接梗玩梗时表情正好对上

只输出 JSON，不要其他内容：
{{"suitable": true, "reason": "一句话理由"}}"""


class PluginMeta(PluginConfigBase):
    """插件元信息（WebUI 上显示为「通用设置」）"""

    __ui_label__ = "通用设置"

    enabled: bool = Field(
        default=True,
        description="是否启用本插件",
        json_schema_extra={"label": "启用插件", "order": 1},
    )
    config_version: str = Field(
        default="1.0.0",
        description="配置版本号，请勿手动修改",
        json_schema_extra={"label": "配置版本", "order": 2},
    )


class AuditSection(PluginConfigBase):
    """审核行为配置"""

    __ui_label__ = "审核行为"

    enabled: bool = Field(
        default=True,
        description="是否启用表情包审核",
        json_schema_extra={
            "label": "启用审核",
            "hint": "关闭后表情包不再经过审核，按原样发出",
            "order": 1,
        },
    )

    strip_emoji: bool = Field(
        default=True,
        description="是否自动清除回复里的 emoji 表情符号",
        json_schema_extra={
            "label": "清除 emoji",
            "hint": "模型爱发😂这类表情，QQ 会转成系统表情显示，一条消息多个时会连发。开启后自动从文本里删掉",
            "order": 11,
        },
    )
    vision_model: VisionModelChoice = Field(  # type: ignore[valid-type]
        default="auto",
        description="看图用的模型",
        json_schema_extra={
            "label": "看图模型",
            "hint": "auto = 自动从你配置里支持视觉（visual=true）的模型挑一个，换机器不用改。下拉里列出的是你本机所有支持视觉的模型，可以直接选一个固定使用。off = 跳过看图，只靠文字标签判断",
            "order": 2,
        },
    )
    auto_try_limit: int = Field(
        default=3,
        description="auto 模式下最多尝试几个视觉模型",
        json_schema_extra={
            "label": "自动尝试数量",
            "hint": "看图模型选 auto 时，最多依次尝试几个模型。3 表示前 3 个都不行就回退到文字标签判断",
            "order": 3,
        },
    )
    judge_model: Literal[
        "fast_model",
        "utils",
        "planner",
        "replyer",
        "emoji",
    ] = Field(
        default="fast_model",
        description="最终判断用哪个模型任务",
        json_schema_extra={
            "label": "判断模型",
            "hint": "用哪个模型任务做最终判断。fast_model 最便宜快速，planner 更聪明但更贵",
            "order": 3,
        },
    )
    strictness: Literal["loose", "normal", "strict"] = Field(
        default="normal",
        description="严格程度",
        json_schema_extra={
            "label": "严格程度",
            "hint": "loose=只在文字和表情明显冲突时才拦；normal=默认；strict=稍不搭就拦，宁可少发也不乱发",
            "order": 4,
        },
    )
    context_limit: int = Field(
        default=12,
        description="审核时参考最近多少条聊天记录",
        json_schema_extra={
            "label": "参考消息条数",
            "hint": "条数越多判断越准，但更慢。建议 8~20",
            "order": 5,
        },
    )
    timeout_seconds: float = Field(
        default=12.0,
        description="单次审核超时（秒）",
        json_schema_extra={
            "label": "审核超时（秒）",
            "hint": "超过这个时间就按「超时处理方式」执行，避免卡住消息发送",
            "order": 6,
        },
    )
    timeout_action: Literal["keep", "drop"] = Field(
        default="keep",
        description="超时或出错时怎么办",
        json_schema_extra={
            "label": "超时处理方式",
            "hint": "keep=保留表情正常发送（推荐，审核失败不影响说话）；drop=撤掉表情只发文字",
            "order": 7,
        },
    )
    audit_rate: float = Field(
        default=1.0,
        description="审核概率 0~1，1.0 表示每条都审",
        json_schema_extra={
            "label": "审核概率",
            "hint": "1.0=每条都审；0.5=一半概率审。降低可以省调用",
            "order": 8,
        },
    )
    skip_if_text_too_short: int = Field(
        default=2,
        description="文字少于这个字数就跳过审核",
        json_schema_extra={
            "label": "跳过审核的字数门槛",
            "hint": "纯表情或极短文字没有语境可判断，直接放行",
            "order": 9,
        },
    )
    log_decisions: bool = Field(
        default=True,
        description="是否记录每次审核结果",
        json_schema_extra={
            "label": "记录审核日志",
            "hint": "在日志里能看到每次审核是「通过」还是「拦下」以及理由",
            "order": 10,
        },
    )


class EmojiSmartAuditConfig(PluginConfigBase):
    """配置根"""

    __ui_label__ = "表情包智能审核"

    plugin: PluginMeta = Field(
        default_factory=PluginMeta,
        json_schema_extra={"label": "通用设置", "order": 0},
    )
    audit: AuditSection = Field(
        default_factory=AuditSection,
        json_schema_extra={"label": "审核行为", "order": 1},
    )


def _find_mai_root() -> Path | None:
    """从插件自身路径推断 MaiBot 根目录（plugins/<id>/plugin.py → 上两级）。"""
    try:
        me = Path(__file__).resolve()
        for parent in me.parents:
            if (parent / "config" / "model_config.toml").exists():
                return parent
    except Exception:  # noqa: BLE001
        pass
    return None


def _list_vision_models() -> list[str]:
    """读取宿主 model_config.toml，返回所有 visual=true 的模型名（按优先级排序）。"""
    root = _find_mai_root()
    if root is None:
        return []
    cfg = root / "config" / "model_config.toml"
    try:
        import tomllib

        data = tomllib.loads(cfg.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return []

    models = data.get("models") or []
    names: list[str] = []
    for m in models:
        if not isinstance(m, dict):
            continue
        if not m.get("visual"):
            continue
        name = str(m.get("name") or "").strip()
        if name:
            names.append(name)
    # 优先挑名字里带常见视觉系关键字的，其余保持原顺序
    prefer = ("vl", "vision", "mini", "qwen", "glm")
    names.sort(key=lambda n: (0 if any(k in n.lower() for k in prefer) else 1))
    return names


import unicodedata


# 要清理的 emoji 范围：图形类 emoji（笑哭/emoji 表情），但保留中文标点和普通符号
def _is_emoji_char(ch: str) -> bool:
    """判断单字符是否为 emoji 类字符。"""
    cp = ord(ch)
    # 常见 emoji 区段
    if 0x1F300 <= cp <= 0x1F9FF:   # 各类 emoji（含😂🤣😅😊👍）
        return True
    if 0x1FA00 <= cp <= 0x1FAFF:
        return True
    if 0x2600 <= cp <= 0x27BF:     # 杂项符号（☀☂✂✈等）
        return True
    if 0xFE00 <= cp <= 0xFE0F:     # 变体选择符
        return True
    if 0x1F000 <= cp <= 0x1F02F:   # 麻将牌
        return True
    if 0x1F1E6 <= cp <= 0x1F1FF:   # 区域指示符（国旗 emoji）
        return True
    if cp in (0x200D, 0x20E3):      # ZWJ / 组合键帽
        return True
    return False


def _strip_emoji_from_text(text: str) -> tuple[str, int]:
    """从文本中移除 emoji，返回 (新文本, 移除数量)。"""
    if not text:
        return text, 0
    kept = []
    removed = 0
    for ch in text:
        if _is_emoji_char(ch):
            removed += 1
            continue
        kept.append(ch)
    if removed == 0:
        return text, 0
    new_text = "".join(kept)
    # 清理多余空格
    new_text = re.sub(r"\s{2,}", " ", new_text).strip()
    return new_text, removed


def _component_list(message: dict[str, Any]) -> list[Any] | None:
    """取消息组件列表，兼容 dict.components / 裸列表两种序列化。"""
    raw = message.get("raw_message")
    if isinstance(raw, dict):
        comps = raw.get("components")
        return comps if isinstance(comps, list) else None
    if isinstance(raw, list):
        return raw
    return None


def _get_text(message: dict[str, Any]) -> str:
    parts = _component_list(message)
    if parts is not None:
        chunks = [
            str(p.get("data") or "")
            for p in parts
            if isinstance(p, dict) and p.get("type") == "text"
        ]
        text = "".join(chunks).strip()
        if text:
            return text
    plain = message.get("processed_plain_text")
    return plain.strip() if isinstance(plain, str) else ""


def _get_emoji_parts(message: dict[str, Any]) -> list[dict[str, Any]]:
    parts = _component_list(message)
    if parts is None:
        return []
    out = []
    for idx, p in enumerate(parts):
        if isinstance(p, dict) and p.get("type") == "emoji":
            out.append({"index": idx, "part": p})
    return out


class EmojiSmartAuditPlugin(MaiBotPlugin):
    """发送前用视觉+语境双重判断表情包该不该发"""

    config_model = EmojiSmartAuditConfig

    def __init__(self) -> None:
        super().__init__()

    async def on_load(self) -> None:
        cfg = self.config.audit
        self.ctx.logger.info(
            f"表情包智能审核已加载：严格度={cfg.strictness}，"
            f"看图模型={('关(只用标签)' if cfg.vision_model == 'off' else ('auto(自动选择)' if cfg.vision_model == 'auto' else cfg.vision_model))}，判断模型={cfg.judge_model}，"
            f"超时={cfg.timeout_seconds}s/{cfg.timeout_action}"
        )

    async def on_unload(self) -> None:
        self.ctx.logger.info("表情包智能审核已卸载")

    async def on_config_update(self, scope: str, config_data: dict, version: str) -> None:
        del scope, config_data, version
        cfg = self.config.audit
        self.ctx.logger.info(
            f"表情包智能审核配置更新：启用={cfg.enabled} 严格度={cfg.strictness}"
        )

    @HookHandler(
        "send_service.before_send",
        name="emoji_smart_audit_before_send",
        description="发送前结合图片内容和聊天语境审核表情包",
        mode=HookMode.BLOCKING,
        order=HookOrder.LATE,
        timeout_ms=30000,
    )
    async def handle_before_send(
        self, message: dict[str, Any], **kwargs: Any
    ) -> dict[str, Any] | None:
        del kwargs
        cfg = self.config.audit

        # 无论审核开没开，先清掉文本里的 emoji（模型爱发笑哭，适配器会转成 QQ 表情连发）
        changed = False
        if cfg.strip_emoji:
            message, changed = self._strip_emoji(message)

        # Hook 契约：修改必须包在 modified_kwargs 里，否则 Host 会丢弃并照原样发送
        def with_message() -> dict[str, Any]:
            return {"action": "continue", "modified_kwargs": {"message": message}}

        if not cfg.enabled:
            return with_message() if changed else None

        emojis = _get_emoji_parts(message)
        if not emojis:
            return with_message() if changed else None

        text = _get_text(message)
        if len(text) < cfg.skip_if_text_too_short:
            return with_message() if changed else None

        if cfg.audit_rate < 1.0 and random.random() > cfg.audit_rate:
            return with_message() if changed else None

        try:
            verdict = await asyncio.wait_for(
                self._judge(cfg, message, text, emojis),
                timeout=max(3.0, cfg.timeout_seconds),
            )
        except asyncio.TimeoutError:
            self.ctx.logger.warning(
                f"表情包审核超时（{cfg.timeout_seconds}s），按 {cfg.timeout_action} 处理"
            )
            if cfg.timeout_action == "drop":
                return self._strip_emojis(message, emojis, "审核超时")
            return with_message() if changed else None
        except Exception as exc:  # noqa: BLE001
            self.ctx.logger.warning(f"表情包审核异常，放行：{exc}")
            if cfg.timeout_action == "drop":
                return self._strip_emojis(message, emojis, "审核异常")
            return with_message() if changed else None

        if verdict is None:
            return with_message() if changed else None

        suitable, reason = verdict
        if cfg.log_decisions:
            self.ctx.logger.info(
                f"表情包审核[{'通过' if suitable else '拦下'}] "
                f"文字={text[:18]!r} 表情数={len(emojis)} | {reason[:70]}"
            )
        if suitable:
            return with_message() if changed else None
        return self._strip_emojis(message, emojis, reason)

    def _strip_emoji(self, message: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        """清除文本组件和 processed_plain_text 里的 emoji。

        返回 (消息, 是否有改动)。整条全是 emoji 时不清理，避免发出空消息。
        Host 反序列化会原样带回 processed_plain_text，且非空时不再重建，
        所以两个字段必须同步清理，不能只清组件。
        """
        parts = message.get("raw_message")
        if not isinstance(parts, list):
            parts = []
        edits: list[tuple[dict[str, Any], str, str]] = []
        for part in parts:
            if not isinstance(part, dict) or part.get("type") != "text":
                continue
            text = part.get("data")
            if not isinstance(text, str) or not text:
                continue
            new_text, removed = _strip_emoji_from_text(text)
            if removed > 0:
                edits.append((part, text, new_text))
        if edits:
            # 清理后是否还有内容：非文本组件，或任一文本组件清完非空
            edited_ids = {id(part) for part, _, _ in edits}
            has_content = any(
                isinstance(p, dict) and p.get("type") != "text" for p in parts
            ) or any(new_text.strip() for _, _, new_text in edits) or any(
                isinstance(p, dict)
                and p.get("type") == "text"
                and id(p) not in edited_ids
                and str(p.get("data") or "").strip()
                for p in parts
            )
            if not has_content:
                return message, False
            for part, old_text, new_text in edits:
                part["data"] = new_text
                self.ctx.logger.info(
                    f"清除 emoji: {old_text[:30]!r} -> {new_text[:30]!r}"
                )
        plain = message.get("processed_plain_text")
        plain_changed = False
        if isinstance(plain, str) and plain:
            new_plain, removed = _strip_emoji_from_text(plain)
            if removed > 0:
                # 清成空串时 Host 会从已清理的组件重建纯文本
                message["processed_plain_text"] = new_plain
                plain_changed = True
        if not edits and not plain_changed:
            return message, False
        return message, True

    def _strip_emojis(
        self, message: dict[str, Any], emojis: list[dict[str, Any]], reason: str
    ) -> dict[str, Any] | None:
        parts = _component_list(message)
        if parts is None:
            return None
        drop = {e["index"] for e in emojis}
        kept = [p for i, p in enumerate(parts) if i not in drop]
        if len(kept) == len(parts) or not kept:
            return None
        message["raw_message"] = kept
        self.ctx.logger.info(f"已撤掉表情：{reason[:70]}")
        return {"action": "continue", "modified_kwargs": {"message": message}}

    async def _judge(
        self,
        cfg: AuditSection,
        message: dict[str, Any],
        text: str,
        emojis: list[dict[str, Any]],
    ) -> tuple[bool, str] | None:
        """返回 (是否合适, 理由)；无法判断时返回 None 表示放行。"""

        # 1) 看图（可选）
        emoji_desc = await self._describe_emoji(cfg, emojis)

        # 2) 拉上下文
        context = await self._context(str(message.get("session_id") or ""), cfg.context_limit)

        # 3) 综合判断
        hint = {
            "loose": "只有在文字和表情情绪明显冲突时才判不合适。",
            "normal": "",
            "strict": "只要表情和语境不是高度契合就判不合适，宁可少发也不要乱发。",
        }.get(cfg.strictness, "")

        prompt = AUDIT_PROMPT.format(
            context=context, text=text, emoji_count=len(emojis), emoji_desc=emoji_desc
        )
        if hint:
            prompt += f"\n\n【本次标准】{hint}"

        raw = await self._llm(cfg.judge_model, prompt, max_tokens=200, temperature=0.1)
        if not raw:
            return None

        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            return None
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
        return bool(data.get("suitable")), str(data.get("reason") or "")

    async def _describe_emoji(
        self, cfg: AuditSection, emojis: list[dict[str, Any]]
    ) -> str:
        """用视觉模型看图；失败则回退到文字描述。"""
        raw_model = (cfg.vision_model or "").strip()
        if not raw_model or raw_model == "off":
            return self._fallback_desc(emojis)

        # 只取第一张（多条表情时省 token）
        first = emojis[0]["part"]
        b64 = first.get("binary_data_base64")
        if not b64:
            return self._fallback_desc(emojis)

        content = [
            {"type": "text", "text": VISION_PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ]

        # auto：从宿主 visual=true 的模型里依次尝试，第一个成功就用
        if raw_model == "auto":
            candidates = _list_vision_models()
            if not candidates:
                self.ctx.logger.debug("auto 看图：宿主未找到 visual=true 的模型，回退文字描述")
                return self._fallback_desc(emojis)
            candidates = candidates[: self.config.audit.auto_try_limit]
        else:
            candidates = [raw_model]

        for model in candidates:
            try:
                res = await self.ctx.llm.generate(
                    [{"role": "user", "content": content}],
                    model=model,
                    temperature=0.2,
                    max_tokens=80,
                )
                txt = self._result_text(res)
                if txt:
                    if raw_model == "auto":
                        self.ctx.logger.debug(f"auto 看图命中模型：{model}")
                    extra = f"（共 {len(emojis)} 个，其余类似）" if len(emojis) > 1 else ""
                    return txt[:100] + extra
            except Exception as exc:  # noqa: BLE001
                self.ctx.logger.debug(f"看图模型 {model} 失败：{exc}")
                continue

        return self._fallback_desc(emojis)

    @staticmethod
    def _fallback_desc(emojis: list[dict[str, Any]]) -> str:
        descs = []
        for e in emojis:
            d = e["part"].get("data")
            if isinstance(d, str) and d.strip():
                descs.append(d.strip()[:60])
        return "、".join(descs) if descs else "（未知表情）"

    async def _context(self, session_id: str, limit: int) -> str:
        if not session_id:
            return "（无上下文）"
        try:
            msgs = await self.ctx.message.get_recent(session_id, limit=limit)
        except Exception:  # noqa: BLE001
            return "（无上下文）"
        if not msgs:
            return "（无上下文）"
        lines = []
        for m in msgs[-limit:]:
            if not isinstance(m, dict):
                continue
            name = m.get("user_nickname") or m.get("user_id") or "?"
            txt = str(m.get("processed_plain_text") or "").strip()
            if txt:
                lines.append(f"{name}: {txt[:70]}")
        return "\n".join(lines) if lines else "（无上下文）"

    async def _llm(
        self, model: str, prompt: str, *, max_tokens: int, temperature: float
    ) -> str:
        try:
            res = await self.ctx.llm.generate(
                prompt, model=model, max_tokens=max_tokens, temperature=temperature
            )
        except Exception as exc:  # noqa: BLE001
            self.ctx.logger.debug(f"审核 LLM 失败：{exc}")
            return ""
        return self._result_text(res)

    @staticmethod
    def _result_text(res: Any) -> str:
        if isinstance(res, dict):
            return str(res.get("response") or res.get("text") or "").strip()
        return str(res or "").strip()


def create_plugin() -> EmojiSmartAuditPlugin:
    """SDK 工厂入口"""
    return EmojiSmartAuditPlugin()
