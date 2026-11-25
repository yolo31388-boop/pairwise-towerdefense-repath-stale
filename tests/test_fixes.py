"""六个修复点的行为测试（契约测试之外的正确性验证）。

覆盖：
1. invalidate_inflight —— 重算时作废在途任务
2. grid_version / validation —— 结果带版本号，过期结果不得提交
3. target_walkable —— 到达前校验目标格可通行
4. claim_order —— 同格认领按确定性顺序仲裁
5. release_cell —— 单位死亡立即释放占用
6. batch_converge / commit —— 批次收敛后统一提交
另含线程压力下的确定性重放与一致性审计。
"""

import threading

import pytest

from gameplay.towerdefense_repath_stale import (
    Cell,
    Grid,
    Occupancy,
    Pathfinder,
    RepathBatch,
    Scenario,
    TaskStatus,
    Unit,
    World,
    audit,
    batch_converge,
    claim_order,
    commit,
    determinism,
    fallback,
    grid_version,
    invalidate_inflight,
    metrics,
    occupancy,
    path_cache,
    release_cell,
    repath,
    target_walkable,
    validation,
)


# ---------------------------------------------------------------------------
# 修复点 1：重算时作废在途任务
# ---------------------------------------------------------------------------

def test_invalidate_inflight_cancels_queued_and_running():
    grid = Grid(5, 5)
    pf = Pathfinder(grid, workers=2)
    t1 = pf.submit("u1", (0, 0), (4, 4))
    t2 = pf.submit("u2", (0, 1), (4, 3))
    t3 = pf.submit("u3", (1, 0), (3, 4))
    assert {t.status for t in (t1, t2, t3)} == {TaskStatus.QUEUED}

    cancelled_ids = invalidate_inflight(pf, reason="wall dropped")
    assert set(cancelled_ids) == {t1.task_id, t2.task_id, t3.task_id}
    for t in (t1, t2, t3):
        assert t.status is TaskStatus.CANCELLED
        assert t.done_event.is_set()
    assert pf.inflight() == []
    assert pf.queue == deque_like_empty()


def deque_like_empty():
    import collections
    return collections.deque()


def test_drop_obstacle_inflights_and_keeps_fresh_results():
    world = World(6, 6)
    world.add_unit("u1", (0, 0), (5, 5))
    world.tick()  # 生成并提交一条 v0 新鲜路径
    unit = world.units["u1"]
    assert unit.path_version == 0

    # 落墙：版本升到 1，在途任务（若有）全部作废，单位标脏
    n = world.drop_obstacle((3, 3))
    assert isinstance(n, int)
    assert grid_version(world) == 1
    assert unit.needs_repath is True
    assert world.pathfinder.inflight() == []

    # 再 tick：新路径必须基于 v1
    world.tick()
    assert unit.path_version == 1
    assert all(c != (3, 3) for c in unit.path)
    assert world.audit()["ok"]


# ---------------------------------------------------------------------------
# 修复点 2：结果带网格版本号；过期结果不得覆盖新路径
# ---------------------------------------------------------------------------

def test_stale_result_is_rejected_by_version_validation():
    grid = Grid(5, 5)
    pf = Pathfinder(grid)
    task = pf.submit("u1", (0, 0), (4, 4))  # requested v0
    pf.pump()
    assert task.status is TaskStatus.DONE
    assert task.is_fresh
    assert validation(task, grid.version)["valid"] is True

    # 网格升级后，同一结果对新版本失效
    grid.add_block((2, 2))
    verdict = validation(task, grid.version)
    assert verdict["valid"] is False
    assert "version mismatch" in verdict["reason"]


def test_result_computed_on_old_grid_marked_stale():
    grid = Grid(6, 6)
    pf = Pathfinder(grid)
    task = pf.submit("u1", (0, 0), (5, 5))  # requested v0
    # 任务排队期间网格升级；pump 时在 v1 上算出结果
    grid.add_block((3, 3))
    pf.pump()
    assert task.status is TaskStatus.STALE
    assert task.result_version == 1
    assert task.requested_version == 0
    assert validation(task, 1)["valid"] is False


def test_world_never_commits_stale_path():
    world = World(6, 6)
    u = world.add_unit("u1", (0, 0), (5, 5))
    # 手工制造一个 STALE 任务
    task = world.pathfinder.submit("u1", (0, 0), (5, 5))
    world.grid.add_block((3, 3))
    world.pathfinder.pump()
    assert task.status is TaskStatus.STALE
    old_path = u.path
    world._commit_ready_paths()
    # 过期结果没有覆盖单位路径，单位被标记需要重算
    assert u.needs_repath is True
    assert u.path == old_path
    assert world.metrics.stale_rejected >= 1


# ---------------------------------------------------------------------------
# 修复点 3：到达前校验目标格可通行
# ---------------------------------------------------------------------------

def test_target_walkable_detects_new_wall_and_occupant():
    world = World(5, 5)
    u1 = world.add_unit("u1", (0, 0), (4, 4))
    u2 = world.add_unit("u2", (2, 2), (4, 0), priority=1)
    world.tick()
    own_cell = u1.cell

    # 静态阻挡（选一个不属于任何单位当前格的格子）
    far_cell = (0, 4)
    world.grid.add_block(far_cell)
    assert target_walkable(world, u1, far_cell) is False
    assert target_walkable(world, u1, own_cell) is True

    # 他人占用
    assert target_walkable(world, u1, u2.cell) is False
    # 自己当前格永远对自己可通行
    assert target_walkable(world, u1, own_cell) is True


def test_unit_does_not_walk_into_freshly_dropped_wall():
    # 墙落在单位当前路径的下一步上：移动必须被拒绝并触发重算
    world = World(5, 5)
    u = world.add_unit("u1", (0, 0), (4, 4))
    world.tick()
    idx = u.path.index(u.cell)
    next_cell = u.path[idx + 1]
    # 走正式入口：推进版本、作废在途、清旧缓存、单位标脏
    world.drop_obstacle(next_cell)
    before = u.cell
    moved = world._move_unit(u)
    assert moved is False
    assert u.cell == before
    assert u.needs_repath is True
    assert world.metrics.revalidations_blocked >= 1
    # 过期路径要么在入口被整条丢弃，要么移动被二次校验拒绝
    kinds = {e[0] for e in world.event_log}
    assert "stale_path_discarded" in kinds or "move_rejected" in kinds
    # 无论哪种路径，审计不得出现“路径穿过阻挡格”
    assert not any("crosses blocked" in p for p in world.audit()["problems"])


# ---------------------------------------------------------------------------
# 修复点 4：同格认领按确定性顺序仲裁
# ---------------------------------------------------------------------------

def test_claim_order_is_deterministic_ignoring_insertion_order():
    occ = Occupancy()
    # 故意以乱序插入
    occ.claim("z", (2, 2), priority=5)
    occ.claim("a", (2, 2), priority=0)
    occ.claim("m", (2, 2), priority=0)
    occ.claim("b", (2, 2), priority=2)

    order = claim_order(occ, (2, 2))
    unit_order = [row[0] for row in order]
    # priority 升序；同 priority 按全局 seq（即插入顺序）
    assert unit_order == ["a", "m", "b", "z"]
    winner = occ.arbitrate((2, 2))
    assert winner.unit_id == "a"

    # 获胜者消费后，下一顺位确定性接班
    occ.consume(winner)
    assert occ.arbitrate((2, 2)).unit_id == "m"


def test_claim_arbitration_stable_across_thread_timing():
    # 多线程并发认领同一格，仲裁结果只取决于 (priority, seq)
    occ = Occupancy()
    barrier = threading.Barrier(8)

    def worker(unit_id: str, priority: int):
        barrier.wait()
        occ.claim(unit_id, (4, 4), priority)

    threads = []
    spec = [("u7", 3), ("u3", 1), ("u1", 1), ("u9", 0),
            ("u2", 2), ("u5", 1), ("u0", 0), ("u4", 2)]
    for uid, prio in spec:
        threads.append(threading.Thread(target=worker, args=(uid, prio)))
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    order = [row[0] for row in claim_order(occ, (4, 4))]
    # 按 priority 分组；组内顺序与提交 spec 的先后一致（seq 单调）
    prios = {uid: p for uid, p in spec}
    keys = [(prios[uid], i) for i, uid in enumerate(order)]
    assert keys == sorted(keys)
    assert occ.arbitrate((4, 4)).unit_id in {"u9", "u0"}


# ---------------------------------------------------------------------------
# 修复点 5：死亡即释放占用
# ---------------------------------------------------------------------------

def test_death_releases_cell_and_claims():
    world = World(6, 6)
    u1 = world.add_unit("u1", (1, 1), (5, 5))
    u2 = world.add_unit("u2", (2, 1), (5, 0))
    u3 = world.add_unit("u3", (4, 4), (0, 5))
    world.tick()
    u1_cell = u1.cell
    # u1 对前方格子排队认领
    target = (3, 1)
    world.occupancy.claim("u1", target)
    assert occupancy(world, "u1") == u1_cell

    assert world.kill_unit("u1") is True
    # 当前格释放
    assert release_cell(world.occupancy, "u1") is None  # 已无占用
    assert occupancy(world, "u1") is None
    assert world.occupancy.owner_of(u1_cell) is None
    # 排队认领全部清空
    assert all("u1" not in [c.unit_id for c in q]
               for q in world.occupancy.claims.values())
    # 释放出的格子可被其他存活单位立即登记
    assert world.occupancy.occupy("u3", u1_cell) is True
    assert world.occupancy.owner_of(u1_cell) == "u3"
    world.units["u3"].cell = u1_cell  # 同步单位视图，保持审计一致
    assert world.metrics.deaths_released == 1
    assert world.audit()["ok"]
    # 重复击杀是幂防的
    assert world.kill_unit("u1") is False


def test_no_long_detour_after_death_releases_chokepoint():
    # 狭窄通道被死亡单位让出后，后续单位路径恢复直线
    world = World(5, 3, blocked={(2, 0), (2, 2)})
    world.add_unit("blocker", (2, 1), (4, 1))
    world.add_unit("mover", (0, 1), (4, 1), priority=1)
    world.tick()
    assert world.kill_unit("blocker") is True
    batch = repath(world, ["mover"])
    assert batch.committed and (2, 1) in batch.results["mover"]


# ---------------------------------------------------------------------------
# 修复点 6：批次收敛后再统一提交
# ---------------------------------------------------------------------------

def test_batch_converges_before_commit():
    world = World(8, 8)
    units = [world.add_unit(f"u{i}", (i % 3, i // 3), (7, 7), priority=i)
             for i in range(6)]
    batch = batch_converge(world)
    assert isinstance(batch, RepathBatch)
    assert batch.converged is True
    assert batch.committed is True
    # 收敛前没有任何单位能拿到半成品：提交后所有单位路径版本一致
    assert len(batch.results) == 6
    for u in units:
        assert u.path_version == grid_version(world)
        assert u.needs_repath is False


def test_commit_refuses_unconverged_batch():
    world = World(4, 4)
    world.add_unit("u1", (0, 0), (3, 3))
    batch = RepathBatch(batch_id=1, grid_version_at_start=0, task_ids=[])
    batch.converged = False
    with pytest.raises(RuntimeError):
        commit(world, batch)


def test_batch_recomputes_after_obstacle_and_never_uses_stale_paths():
    world = World(6, 6)
    world.add_unit("u1", (0, 0), (5, 5))
    world.add_unit("u2", (0, 5), (5, 0))
    world.tick()
    # 落墙打断两条路径
    world.drop_obstacle((3, 3))
    batch = repath(world)
    assert batch.committed
    assert batch.stale_units == []  # 收敛后没有过期结果被采纳
    for uid, path in batch.results.items():
        assert (3, 3) not in path
        assert world.units[uid].path_version == grid_version(world)
    assert world.audit()["ok"]


def test_fallback_never_returns_stale_route():
    world = World(5, 5)
    u = world.add_unit("u1", (0, 0), (4, 4))
    world.tick()
    safe = fallback(world, u)
    assert safe[0] == u.cell  # 从当前格出发
    # 路径过期后，兜底只允许原地等待，不许沿旧路径前进
    world.grid.add_block((2, 2))
    u.path_version = -1
    assert fallback(world, u) == [u.cell]


# ---------------------------------------------------------------------------
# 路径缓存按版本隔离
# ---------------------------------------------------------------------------

def test_path_cache_is_version_scoped():
    world = World(5, 5)
    pf = world.pathfinder
    task = pf.submit("u1", (0, 0), (4, 4))
    pf.pump()
    v0 = grid_version(world)
    assert path_cache(world, (0, 0), (4, 4), v0) is not None
    # 走正式入口落墙：旧版本缓存随版本升级被清理
    world.drop_obstacle((2, 2))
    assert path_cache(world, (0, 0), (4, 4), v0) is None
    assert path_cache(world) == 0


# ---------------------------------------------------------------------------
# 端到端：线程模式下的确定性重放与审计
# ---------------------------------------------------------------------------

def _busy_scenario() -> Scenario:
    return Scenario(
        width=7,
        height=7,
        blocked={(3, 0), (3, 6)},
        units=[
            ("u1", (0, 0), (6, 6), 0),
            ("u2", (0, 6), (6, 0), 0),
            ("u3", (6, 3), (0, 3), 1),
            ("u4", (2, 2), (5, 5), 2),
            ("u5", (1, 4), (6, 1), 1),
        ],
        obstacles=[(3, (3, 3)), (7, (4, 2))],
        deaths=[(5, "u4")],
        ticks=14,
    )


def test_determinism_inline_vs_threaded():
    report = determinism(_busy_scenario(), runs=3)
    assert report["deterministic"] is True, report["diffs"]
    assert report["runs"] == ["inline", "threaded", "threaded"]


def test_threaded_stress_audit_clean():
    scenario = _busy_scenario()
    from gameplay.towerdefense_repath_stale import _run_scenario
    result = _run_scenario(scenario, threaded=True)
    assert result["audit"]["ok"] is True, result["audit"]["problems"]
    # 死亡单位确实释放了占用
    cells = [cell for _uid, (cell, _alive, _v) in result["units"].items()]
    assert len(cells) == len(set(cells))
    m = result["metrics"]
    assert m["deaths_released"] == 1


def test_metrics_accounting():
    world = World(5, 5)
    world.add_unit("u1", (0, 0), (4, 4))
    world.tick()
    # 先制造一个在途任务，再落墙，确保作废计数器非零
    world.pathfinder.submit("u1", (0, 0), (4, 4))
    assert world.pathfinder.inflight()
    world.drop_obstacle((2, 2))
    world.tick()
    m = metrics(world)
    assert m["submitted"] >= 2
    assert m["committed"] >= 2
    assert m["cancelled"] >= 1
    assert m["deaths_released"] == 0
    assert audit(world)["ok"]


def test_threaded_inflight_invalidation_during_obstacle():
    # 线程模式下：任务在途时落墙，运行中的任务必须被作废而非迟到覆盖
    world = World(8, 8, threaded=True, workers=4)
    try:
        world.add_unit("u1", (0, 0), (7, 7))
        world.add_unit("u2", (0, 7), (7, 0))
        # 直接提交在途任务（不经过 tick 的立即收敛）
        t1 = world.pathfinder.submit("u1", (0, 0), (7, 7))
        t2 = world.pathfinder.submit("u2", (0, 7), (7, 0))
        cancelled = world.drop_obstacle((4, 4))
        assert cancelled >= 1
        world.pathfinder.wait_idle()
        for t in (t1, t2):
            assert t.status in (TaskStatus.CANCELLED, TaskStatus.STALE,
                                TaskStatus.DONE)
            if t.status is TaskStatus.DONE:
                # 若已完成则必须是作废前的新鲜结果，且单位仍被标脏重算
                assert t.is_fresh
        assert world.units["u1"].needs_repath is True
        assert audit(world)["ok"]
    finally:
        world.shutdown()
