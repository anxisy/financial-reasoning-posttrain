# GitHub 发布前检查清单

- [ ] 运行 `python tools/release_audit.py --root .`；确保无权重、数据集、预测 JSONL、日志、虚拟环境或秘密凭据进入仓库。
- [ ] 检查 `provenance/source_files.json` 与 `results/frozen_source_manifest.json`，确认公开的文件名、原路径不包含私人项目或账号信息。
- [ ] `grep -R '/root/autodl-tmp' finqa convfinqa` 审核遗留绝对路径。修复后需要在原环境重跑自检，不得伪称“开箱即用”。
- [ ] 与原始 FinQA / ConvFinQA 数据使用条款、Qwen3 基座许可核对；不要复制完整官方数据或模型文件。
- [ ] 由项目所有者明确选择是否添加 LICENSE（仓库默认未自动选择许可证）。
- [ ] 检查 README 所有数字对应 `results/` 中的冻结证据，区分 Dev/Test 和 Oracle/Self。
- [ ] GitHub 仓库先建立为 **Private**，检查 `git status` 和拟提交文件，再根据需要设为 Public。
- [ ] 冻结归档、四个 Adapter、环境快照单独备份到可靠位置；GitHub 源码仓库不等于模型资产备份。

推荐检查命令：

```bash
python tools/release_audit.py --root .
python tools/path_audit.py --root . --limit 100
python -m compileall -q finqa convfinqa tools
find . -type l -print
find . -type f -size +5M -print
git status --short
```

**默认没有推送到 GitHub 或创建云端仓库**，不会未经授权上传私人文件。
