# Agentic-Embodied 文档索引

## 当前真机

- [真机导航](real_robot/README.md)：先读模块职责和命令分类。
- [当前部署](real_robot/current-deployment.md)：dodo 环境、当前冻结配置及复现入口。
- [相比 Zetta 的改动](real_robot/zetta-changes.md)：按固定 Git 提交比较，区分本项目新增与上游后续更新。
- [2026-10-07 分段成功](experiments/arx-no-lift-timing-20261007/README.md)：实际抬升、五帧保持及归位/失能记录。
- [2026-10-05 真机进化试点](experiments/arx-real-evolution-20261005/README.md)：旧父代/候选配对实验，候选未获晋升。

## 设计与历史记录

`integration/arx-real-*` 记录输入契约、后端、bundle 执行和抓取接口的设计过程；其中日期、旧 SHA 和“尚未完成”描述是当时的状态。**当前部署配置以 `real_robot/current-deployment.md` 为索引，历史协议保持原样。**

- [真机输入契约](integration/arx-real-input-contract.md)
- [真机后端](integration/arx-real-backend.md)
- [bundle 执行](integration/arx-real-bundle-running.md)
- [抓取恢复接口](integration/arx-real-grasp-recovery.md)
- [学习流程审查](evolution-learning-pipeline-review.md)
- [通用 MuJoCo](integration/mujoco.md)

`experiments/` 是不可覆盖的按日期实验记录；`../integrations/` 是历史仿真交付快照，不是当前真机入口。大文件的实际存储位置由实验协议记录。

## 上游 Sphinx 文档构建

Zetta's documentation is built with Sphinx. English and Simplified Chinese
sources live in parallel trees so local builds and Read the Docs use the same
layout.

## Set up the environment

From the repository root:

```bash
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
uv venv --python 3.11 .docs-venv
source .docs-venv/bin/activate
uv pip install -r docs/requirements.txt
```

The locale variables should be set in each new terminal used to build the
documentation. The Python version matches the Read the Docs build environment.

## Build and preview

Run the live-reloading English preview:

```bash
cd docs
bash autobuild.sh
```

For the Chinese documentation:

```bash
cd docs
bash autobuild.sh zh
```

Both commands serve the generated site at <http://localhost:8000> and rebuild
it when a source file changes.

If port 8000 is already in use, pass another port or use `0` to select a free
port automatically:

```bash
bash autobuild.sh zh 8001
bash autobuild.sh zh 0
```

To build without starting the preview server:

```bash
cd docs
sphinx-build -W --keep-going source-en build/html-en
sphinx-build -W --keep-going source-zh build/html-zh
```

You can also use Make:

```bash
cd docs
make html LANG=en
make html LANG=zh
```

## Clean generated files

```bash
cd docs
make clean
```

The generated `docs/build/` directory and `.docs-venv/` environment are not
tracked by Git.

## Write documentation

Documentation pages use reStructuredText (RST). Keep English and Chinese pages
at matching relative paths so the language switcher can open the same page in
the other language.

- English: `source-en/rst_source/`
- Chinese: `source-zh/rst_source/`
- RST syntax reference: <https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>
