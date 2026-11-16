# 更新日志

## 1.1.0

- 修复修改结果被宿主丢弃的问题：Hook 返回值改用 `modified_kwargs` 契约，之前返回 `{"message": ...}` 会被忽略，导致清理其实从未生效
- emoji 清理同时处理消息组件与 `processed_plain_text`，避免纯文本字段把 emoji 带回去
- 兼容 `raw_message` 的两种序列化形态（裸列表 / dict.components）
- 清理状态改为局部变量，避免并发发送时互相覆盖导致漏清
- 新增 emoji 清理日志，便于排查

## 1.0.0

- 首次发布：视觉模型看图 + 语境判断，不合适就撤掉表情