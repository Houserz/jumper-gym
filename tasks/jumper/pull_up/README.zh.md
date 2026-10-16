<!-- tracks: README.md @ sha256:b31545d54643c566 -->

# jumper.pull_up — 在平台边缘外的单杠上做引体向上

机器人用两只钳子吊在一根 25 mm 的单杠上。单杠位于 0.22 m 高平台的边缘外 0.11 m，
高出台面 0.27 m（`scenes/bar.py`）。机器人按要求的高度拉上去、再放下来，一次接一次。
每个回合开始时已经吊在杠上；上杠由 crab gym 负责（`docs/CRABGYM.md`）。目前握杆用的是
约束（`mdp/grip.py`）。任务概览见 `__init__.py`，本文件说明各部分在哪里。

```bash
python scripts/train.py --task jumper.pull_up --backend warp --headless --max-iterations 2000
python scripts/play.py  --task jumper.pull_up --checkpoint <run>/model_N.pt
python tasks/jumper/pull_up/tools/pull_check.py --checkpoint <run>/model_N.pt --json <run>/pull_check.json
python tasks/jumper/pull_up/tools/hang_pose.py --family
```

回放时如果没人操作，它每 2 s 拉一次到范围的最高点；右扳机或键盘 ↑ 可以接管高度。

| 文件 | 是什么 |
|---|---|
| `hang.py` | 每个回合起始的悬挂姿态（实测得出），以及 `check`：模型对不上时拒绝构建 |
| `collision.py` | 让机身对手臂成为实体；以及 `pull_check` 测量的碰撞外壳间距 |
| `tools/hang_pose.py` | 推导 `hang.py` 的数值；`--family` 列出每个高度的代价 |
| `tools/pull_check.py` | 验收工具：回放一个检查点，逐条给出 PASS 或 FAIL |
| `mdp/commands.py` | 上拉命令，按一组一组的上拉生成（`rep_profile`） |
| `mdp/state.py` | 上升高度、摆动、受力角（杆子压在哪一片钳口上），以及脚相对机身的位置 |
| `env_cfg.py` | 全部参数，每个参数旁边都写着决定它的测量结果 |

**crab gym 的 bundle 里是 `model_9998`**，来自 2026-10-09 在这个场景上训练的一次运行
（`out/bundle_gym/gym.app`）。同一次运行最后通过验收的 `model_11997` 也能通过
`tools/pull_check.py`，但由 gym 的摆荡交给它时，它会把腿甩到背上，拉不起来。仓库里没有
这两个检查点：用上面的命令训练一个，再用 `tools/pull_check.py` 判断它是否通过。

## 尚未解决

- **没有荡着拉。** 它靠手臂和前倾拉上去；12 cm 用不着更多。
- **能不能抓住。** 握杆是约束，受力角也只是近似：它取的是从重心穿过杆子的方向，即
  身体静止悬挂或整体摆动时的受力方向，手臂自身加速度带来的那部分没有计入。真实的
  钳子能不能抓住——包括摆动时——是下一阶段的问题。
- **重量。** 模型重 2.54 kg；`docs/HARDWARE.md` 写的是 1.8 kg（工程样机 2.8 kg）。
  本文所有力矩都是按模型算的。
- **部署。** `pull_command` 还没有进入导出的部署契约。
