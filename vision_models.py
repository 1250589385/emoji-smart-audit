"""动态生成视觉模型选项的辅助模块。

放在单独文件里，避免和 plugin.py 里的类定义顺序纠缠：
plugin.py 在导入阶段调用 build_vision_choices()，用返回值构造 Literal。
"""

from __future__ import annotations

from pathlib import Path

# 没有任何视觉模型时的兜底选项
FALLBACK_CHOICES = ["auto", "off"]


def find_mai_root() -> Path | None:
    """从当前文件位置往上找含有 config/model_config.toml 的目录。"""
    try:
        me = Path(__file__).resolve()
        for parent in me.parents:
            if (parent / "config" / "model_config.toml").exists():
                return parent
    except Exception:  # noqa: BLE001
        pass
    return None


def list_vision_models() -> list[str]:
    """读取宿主 model_config.toml，返回所有 visual=true 的模型名。

    按「像视觉模型」的程度排序，便于下拉框里好找。
    """
    root = find_mai_root()
    if root is None:
        return []
    cfg = root / "config" / "model_config.toml"
    if not cfg.exists():
        return []
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
        if name and name not in names:
            names.append(name)

    prefer = ("vl", "vision", "omni", "gpt-4", "gpt-5", "gemini", "qwen", "glm")
    names.sort(key=lambda n: (0 if any(k in n.lower() for k in prefer) else 1, len(n)))
    return names


def build_vision_choices() -> list[str]:
    """构造下拉选项：auto + off + 本机所有视觉模型。"""
    models = list_vision_models()
    if not models:
        return list(FALLBACK_CHOICES)
    # auto 放最前（默认），off 放最后，中间是本机视觉模型
    return ["auto"] + models + ["off"]
