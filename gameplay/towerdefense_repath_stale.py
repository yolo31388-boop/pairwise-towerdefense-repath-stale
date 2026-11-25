"""塔防动态重寻路与过期结果防护。

本模块实现一个最小可运行的异步寻路模拟器，并针对以下六类缺陷给出修复：

1. 阻挡物落下触发批量重算时，作废旧网格上仍在途的寻路任务（而不是只标记受影响格子）。
2. 每个网格快照带单调递增的版本号；异步算完的结果提交前做版本校验，旧网格上的结果
   不能覆盖新版本网格上的路径。
3. 单位移动到目标格之前再次校验该格当前可通行，避免沿旧路径走进新墙卡死。
4. 同一格被多个单位同时认领时，按确定性的全局顺序（priority, unit_id）仲裁，与线程
   完成顺序无关，重复运行轨迹完全一致。
5. 单位死亡立即释放其当前格占用与所有排队认领，杜绝占用泄漏导致的长期绕行。
6. 批量重算必须等整批任务收敛（全部完成/作废/失败）后再统一提交，禁止逐个覆盖。

模块同时提供面向对象的模拟器（``World``/``Pathfinder``/``Occupancy`` 等）与一组
函数式入口（与测试契约中的 15 个函数同名）。
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Deque, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    # 领域对象
    "Cell", "GridSnapshot", "Grid", "Unit", "TaskStatus", "PathTask",
    "Pathfinder", "RepathBatch", "Occupancy", "Claim", "PathCache",
    "World", "Scenario", "RepathMetrics",
    # 函数式入口（契约）
    "invalidate_inflight", "grid_version", "target_walkable", "claim_order",
    "release_cell", "batch_converge", "determinism", "validation", "audit",
    "fallback", "repath", "metrics", "occupancy", "path_cache", "commit",
]

Cell = Tuple[int, int]


# ---------------------------------------------------------------------------
# 网格与版本号
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GridSnapshot:
    """某一时刻网格的不可变快照，携带单调版本号。"""

    version: int
    width: int
    height: int
    blocked: frozenset[Cell]

    def walkable(self, cell: Cell) -> bool:
        x, y = cell
        if not (0 <= x < self.width and 0 <= y < self.height):
            return False
        return cell not in self.blocked


class Grid:
    """动态网格。阻挡物变化会生成新版本的不可变快照。"""

    def __init__(self, width: int = 8, height: int = 8,
                 blocked: Optional[Iterable[Cell]] = None) -> None:
        self.width = width
        self.height = height
        self._blocked: set[Cell] = set(blocked or ())
        self._version = 0
        self.lock = threading.RLock()

    # ---- 只读 ----
    @property
    def version(self) -> int:
        with self.lock:
            return self._version

    def snapshot(self) -> GridSnapshot:
        with self.lock:
            return GridSnapshot(
                self._version, self.width, self.height, frozenset(self._blocked)
            )

    def walkable(self, cell: Cell) -> bool:
        return self.snapshot().walkable(cell)

    # ---- 变更（每次变更版本号 +1）----
    def add_block(self, cell: Cell) -> GridSnapshot:
        with self.lock:
            if cell not in self._blocked:
                self._blocked.add(cell)
                self._version += 1
            return self.snapshot()

    def remove_block(self, cell: Cell) -> GridSnapshot:
        with self.lock:
            if cell in self._blocked:
                self._blocked.discard(cell)
                self._version += 1
            return self.snapshot()

    def apply_blocks(self, added: Iterable[Cell] = (),
                     removed: Iterable[Cell] = ()) -> GridSnapshot:
        with self.lock:
            changed = False
            for c in added:
                if c not in self._blocked:
                    self._blocked.add(c)
                    changed = True
            for c in removed:
                if c in self._blocked:
                    self._blocked.discard(c)
                    changed = True
            if changed:
                self._version += 1
            return self.snapshot()

    # ---- 寻路：BFS，固定邻居顺序 => 同快照同端点结果唯一 ----
    @staticmethod
    def _neighbors(cell: Cell) -> List[Cell]:
        x, y = cell
        # 固定扫描顺序：上、右、下、左，保证确定性
        return [(x, y - 1), (x + 1, y), (x, y + 1), (x - 1, y)]

    def find_path(self, start: Cell, goal: Cell,
                  snapshot: Optional[GridSnapshot] = None) -> Optional[List[Cell]]:
        snap = snapshot or self.snapshot()
        if not snap.walkable(start) or not snap.walkable(goal):
            return None
        if start == goal:
            return [start]
        prev: Dict[Cell, Cell] = {}
        queue: Deque[Cell] = deque([start])
        seen = {start}
        while queue:
            cur = queue.popleft()
            if cur == goal:
                path: List[Cell] = [cur]
                while cur != start:
                    cur = prev[cur]
                    path.append(cur)
                path.reverse()
                return path
            for nxt in self._neighbors(cur):
                if nxt in seen or not snap.walkable(nxt):
                    continue
                seen.add(nxt)
                prev[nxt] = cur
                queue.append(nxt)
        return None  # 不可达


# ---------------------------------------------------------------------------
# 异步寻路任务
# ---------------------------------------------------------------------------

class TaskStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    STALE = "stale"        # 结果基于过期网格，被版本校验拒绝
    CANCELLED = "cancelled"  # 重算时在途作废
    FAILED = "failed"      # 当前网格不可达


_TERMINAL = {TaskStatus.DONE, TaskStatus.STALE, TaskStatus.CANCELLED,
             TaskStatus.FAILED}


@dataclass
class PathTask:
    task_id: int
    unit_id: str
    start: Cell
    goal: Cell
    requested_version: int          # 提交时网格版本
    priority: int = 0
    status: TaskStatus = TaskStatus.QUEUED
    path: Optional[List[Cell]] = None
    result_version: Optional[int] = None  # 结果实际基于的网格版本
    error: Optional[str] = None
    cancel_event: threading.Event = field(default_factory=threading.Event,
                                          repr=False)
    done_event: threading.Event = field(default_factory=threading.Event,
                                        repr=False)

    def cancel(self) -> bool:
        if self.status in _TERMINAL:
            return False
        self.status = TaskStatus.CANCELLED
        self.error = "inflight task invalidated by grid change"
        self.cancel_event.set()
        self.done_event.set()
        return True

    @property
    def is_fresh(self) -> bool:
        """结果版本与请求版本一致才算新鲜。"""
        return (self.status is TaskStatus.DONE
                and self.result_version == self.requested_version)


class Pathfinder:
    """异步寻路器：支持手动泵取与线程池两种执行方式。

    所有状态修改都在 ``self.lock`` 下完成；线程内的实际计算在锁外进行，但
    提交结果时重新取锁并做版本校验，因此线程完成顺序不影响最终状态。
    """

    def __init__(self, grid: Grid, workers: int = 2) -> None:
        self.grid = grid
        self.workers = max(1, workers)
        self.lock = threading.RLock()
        self._next_id = 1
        self.queue: Deque[PathTask] = deque()
        self.tasks: Dict[int, PathTask] = {}
        self._threads: List[threading.Thread] = []
        self._stop = threading.Event()
        self._wake = threading.Condition(self.lock)
        self.cache = PathCache()

    # ---- 任务生命周期 ----
    def submit(self, unit_id: str, start: Cell, goal: Cell,
               priority: int = 0) -> PathTask:
        with self.lock:
            task = PathTask(
                task_id=self._next_id,
                unit_id=unit_id,
                start=start,
                goal=goal,
                requested_version=self.grid.version,
                priority=priority,
            )
            self._next_id += 1
            self.tasks[task.task_id] = task
            self.queue.append(task)
            self._wake.notify_all()
            return task

    def inflight(self) -> List[PathTask]:
        """所有尚未进入终态的任务（排队中或运行中）。"""
        with self.lock:
            return [t for t in self.tasks.values() if t.status not in _TERMINAL]

    def invalidate_inflight(self, reason: str = "grid changed") -> List[int]:
        """修复点 1：重算时作废旧网格上全部在途任务。

        返回被作废的任务 id 列表。已经终态的任务不动（其结果仍会经过版本
        校验，过期结果无法提交）。
        """
        cancelled: List[PathTask] = []
        with self.lock:
            queued = [t for t in self.queue if t.status not in _TERMINAL]
            self.queue = deque(t for t in self.queue if t.status in _TERMINAL)
            for t in queued:
                if t.cancel():
                    t.error = reason
                    cancelled.append(t)
            # 运行中的任务：置取消标志，worker 在提交点会拒绝写入
            for t in self.tasks.values():
                if t.status is TaskStatus.RUNNING and t.cancel():
                    t.error = reason
                    cancelled.append(t)
            self._wake.notify_all()
        return [t.task_id for t in cancelled]

    def _process_one(self, task: PathTask) -> None:
        # 锁外做重计算；先抓快照，结果永远绑定该快照版本
        snapshot = self.grid.snapshot()
        if task.cancel_event.is_set():
            with self.lock:
                if task.status not in _TERMINAL:
                    task.status = TaskStatus.CANCELLED
                    task.done_event.set()
            return
        path = self.grid.find_path(task.start, task.goal, snapshot)

        with self.lock:
            # 提交点重新检查：可能在计算期间被作废或网格已升级
            if task.status in _TERMINAL:
                task.done_event.set()
                return
            if task.cancel_event.is_set():
                task.status = TaskStatus.CANCELLED
                task.error = "invalidated while computing"
                task.done_event.set()
                return
            task.result_version = snapshot.version
            if snapshot.version != task.requested_version:
                # 修复点 2：旧网格结果标记为 STALE，绝不覆盖新路径
                task.status = TaskStatus.STALE
                task.path = path
                task.error = (f"stale result: computed on v{snapshot.version}, "
                              f"requested v{task.requested_version}")
            elif path is None:
                task.status = TaskStatus.FAILED
                task.error = "no path on current grid"
            else:
                task.status = TaskStatus.DONE
                task.path = path
                self.cache.put(task.start, task.goal, snapshot.version, path)
            task.done_event.set()

    # ---- 手动泵取（确定性测试用）----
    def pump(self) -> int:
        """同步处理一个排队任务，返回处理数（0 表示队列空）。"""
        with self.lock:
            while self.queue:
                task = self.queue.popleft()
                if task.status not in _TERMINAL:
                    task.status = TaskStatus.RUNNING
                    break
            else:
                return 0
        self._process_one(task)
        return 1

    def pump_all(self) -> int:
        n = 0
        while self.pump():
            n += 1
        return n

    # ---- 线程池模式 ----
    def start(self) -> None:
        with self.lock:
            if self._threads:
                return
            self._stop.clear()
            for i in range(self.workers):
                th = threading.Thread(target=self._worker_loop,
                                      name=f"pathfinder-{i}", daemon=True)
                th.start()
                self._threads.append(th)

    def _worker_loop(self) -> None:
        while True:
            with self.lock:
                while not self._stop.is_set() and not self.queue:
                    self._wake.wait(timeout=0.05)
                if self._stop.is_set() and not self.queue:
                    return
                task: Optional[PathTask] = None
                while self.queue:
                    cand = self.queue.popleft()
                    if cand.status not in _TERMINAL:
                        cand.status = TaskStatus.RUNNING
                        task = cand
                        break
                if task is None:
                    if self._stop.is_set():
                        return
                    continue
            self._process_one(task)

    def shutdown(self, wait: bool = True, timeout: float = 5.0) -> None:
        with self.lock:
            self._stop.set()
            self._wake.notify_all()
            threads = list(self._threads)
            self._threads.clear()
        if wait:
            for th in threads:
                th.join(timeout=timeout)

    def wait_idle(self, timeout: float = 5.0) -> bool:
        """等待所有已提交任务进入终态。"""
        import time
        deadline = time.monotonic() + timeout
        with self.lock:
            while time.monotonic() < deadline:
                if (all(t.status in _TERMINAL for t in self.tasks.values())
                        and not self.queue):
                    return True
                # 轮询等待：worker 持有的是另一个 Condition，无法直接唤醒这里
                self._wake.wait(timeout=min(0.02, max(0.0, deadline - time.monotonic())))
            return (all(t.status in _TERMINAL for t in self.tasks.values())
                    and not self.queue)


# ---------------------------------------------------------------------------
# 路径缓存（按版本隔离，网格升级即失效）
# ---------------------------------------------------------------------------

class PathCache:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self._store: Dict[Tuple[Cell, Cell, int], List[Cell]] = {}
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _key(start: Cell, goal: Cell, version: int) -> Tuple[Cell, Cell, int]:
        return (start, goal, version)

    def get(self, start: Cell, goal: Cell, version: int) -> Optional[List[Cell]]:
        with self.lock:
            path = self._store.get(self._key(start, goal, version))
            if path is None:
                self.misses += 1
                return None
            self.hits += 1
            return list(path)

    def put(self, start: Cell, goal: Cell, version: int,
            path: Sequence[Cell]) -> None:
        with self.lock:
            self._store[self._key(start, goal, version)] = list(path)

    def purge_versions(self, keep: Optional[int] = None,
                       below: Optional[int] = None) -> int:
        """清理过期版本缓存。``below`` 给定时清理所有 < below 的条目。"""
        with self.lock:
            if below is None:
                n = len(self._store)
                self._store.clear()
                return n
            dead = [k for k in self._store if k[2] < below]
            for k in dead:
                del self._store[k]
            return len(dead)

    def __len__(self) -> int:
        with self.lock:
            return len(self._store)


# ---------------------------------------------------------------------------
# 格子占用与确定性认领仲裁
# ---------------------------------------------------------------------------

@dataclass
class Claim:
    unit_id: str
    cell: Cell
    priority: int
    seq: int  # 全局单调序号，作为同优先级下的确定性 tiebreaker


class Occupancy:
    """单位对格子的占用与认领。

    - ``occupied``：单位当前站立的格子。
    - ``claims``：单位对下一步目标格的排队认领，每格一个 FIFO 队列。
    - 仲裁顺序固定为 ``(priority, seq)``（seq 为全局提交序号），与线程
      完成顺序无关，因此同一输入多次运行结果完全一致。
    """

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.occupied: Dict[Cell, str] = {}
        self.by_unit: Dict[str, Cell] = {}
        self.claims: Dict[Cell, Deque[Claim]] = {}
        self._seq = 0

    def occupy(self, unit_id: str, cell: Cell) -> bool:
        with self.lock:
            owner = self.occupied.get(cell)
            if owner is not None and owner != unit_id:
                return False
            prev = self.by_unit.get(unit_id)
            if prev is not None and prev != cell:
                self.occupied.pop(prev, None)
            self.occupied[cell] = unit_id
            self.by_unit[unit_id] = cell
            return True

    def owner_of(self, cell: Cell) -> Optional[str]:
        with self.lock:
            return self.occupied.get(cell)

    def cell_of(self, unit_id: str) -> Optional[Cell]:
        with self.lock:
            return self.by_unit.get(unit_id)

    def claim(self, unit_id: str, cell: Cell, priority: int = 0) -> Claim:
        """排队认领目标格；重复认领返回已排队的那一条。"""
        with self.lock:
            queue = self.claims.setdefault(cell, deque())
            for c in queue:
                if c.unit_id == unit_id:
                    return c
            self._seq += 1
            c = Claim(unit_id, cell, priority, self._seq)
            queue.append(c)
            return c

    def arbitrate(self, cell: Cell) -> Optional[Claim]:
        """修复点 4：按 (priority, seq) 确定性顺序选出获胜认领。"""
        with self.lock:
            queue = self.claims.get(cell)
            if not queue:
                return None
            winner = min(queue, key=lambda c: (c.priority, c.seq))
            return winner

    def consume(self, claim: Claim) -> None:
        with self.lock:
            queue = self.claims.get(claim.cell)
            if queue:
                try:
                    queue.remove(claim)
                except ValueError:
                    pass
                if not queue:
                    self.claims.pop(claim.cell, None)

    def release_claims(self, unit_id: str) -> List[Cell]:
        with self.lock:
            freed: List[Cell] = []
            for cell, queue in list(self.claims.items()):
                before = len(queue)
                remaining = deque(c for c in queue if c.unit_id != unit_id)
                if len(remaining) != before:
                    freed.append(cell)
                if remaining:
                    self.claims[cell] = remaining
                else:
                    self.claims.pop(cell, None)
            return freed

    def release(self, unit_id: str) -> Optional[Cell]:
        """修复点 5：单位死亡立即释放当前格与全部排队认领。"""
        with self.lock:
            cell = self.by_unit.pop(unit_id, None)
            if cell is not None:
                if self.occupied.get(cell) == unit_id:
                    self.occupied.pop(cell, None)
            self.release_claims(unit_id)
            return cell

    def snapshot(self) -> Dict[str, object]:
        with self.lock:
            return {
                "occupied": dict(self.occupied),
                "by_unit": dict(self.by_unit),
                "claims": {cell: [(c.unit_id, c.priority, c.seq)
                                  for c in q]
                           for cell, q in self.claims.items()},
            }


# ---------------------------------------------------------------------------
# 单位
# ---------------------------------------------------------------------------

@dataclass
class Unit:
    unit_id: str
    cell: Cell
    goal: Cell
    priority: int = 0
    path: Optional[List[Cell]] = None
    path_version: Optional[int] = None
    alive: bool = True
    needs_repath: bool = True
    blocked_ticks: int = 0
    last_task_id: Optional[int] = None


# ---------------------------------------------------------------------------
# 批次收敛与统一提交
# ---------------------------------------------------------------------------

@dataclass
class RepathBatch:
    batch_id: int
    grid_version_at_start: int
    task_ids: List[int]
    committed: bool = False
    converged: bool = False
    rounds: int = 0
    results: Dict[str, List[Cell]] = field(default_factory=dict)
    stale_units: List[str] = field(default_factory=list)
    failed_units: List[str] = field(default_factory=list)

    def is_terminal(self, tasks: Dict[int, PathTask]) -> bool:
        return all(tasks[i].status in _TERMINAL for i in self.task_ids)


class RepathMetrics:
    def __init__(self) -> None:
        self.submitted = 0
        self.cancelled = 0
        self.stale_rejected = 0
        self.committed = 0
        self.failed = 0
        self.batches = 0
        self.revalidations_blocked = 0
        self.deaths_released = 0
        self.cache_hits = 0
        self.cache_misses = 0

    def as_dict(self) -> Dict[str, int]:
        return dict(self.__dict__)


# ---------------------------------------------------------------------------
# 世界（门面：把网格、寻路器、占用、单位组装成可 tick 的模拟器）
# ---------------------------------------------------------------------------

class World:
    def __init__(self, width: int = 8, height: int = 8,
                 blocked: Optional[Iterable[Cell]] = None,
                 workers: int = 2, threaded: bool = False) -> None:
        self.grid = Grid(width, height, blocked)
        self.pathfinder = Pathfinder(self.grid, workers=workers)
        self.occupancy = Occupancy()
        self.units: Dict[str, Unit] = {}
        self.lock = threading.RLock()
        self.metrics = RepathMetrics()
        self._batch_id = 0
        self.event_log: List[Tuple[str, object]] = []
        self.threaded = threaded
        if threaded:
            self.pathfinder.start()

    # ---- 单位管理 ----
    def add_unit(self, unit_id: str, cell: Cell, goal: Cell,
                 priority: int = 0) -> Unit:
        with self.lock:
            if not self.grid.walkable(cell):
                raise ValueError(f"spawn cell {cell} is blocked")
            unit = Unit(unit_id=unit_id, cell=cell, goal=goal,
                        priority=priority)
            self.units[unit_id] = unit
            self.occupancy.occupy(unit_id, cell)
            return unit

    def kill_unit(self, unit_id: str) -> bool:
        """修复点 5：死亡即释放占用与认领。"""
        with self.lock:
            unit = self.units.get(unit_id)
            if unit is None or not unit.alive:
                return False
            unit.alive = False
            cell = self.occupancy.release(unit_id)
            unit.cell = cell or unit.cell
            unit.path = None
            unit.needs_repath = True
            self.metrics.deaths_released += 1
            self.event_log.append(("unit_died", (unit_id, cell)))
            return True

    # ---- 阻挡物落下：作废在途 + 受影响单位标脏 + 版本推进 ----
    def drop_obstacle(self, cell: Cell) -> int:
        with self.lock:
            self.grid.add_block(cell)
            cancelled = self.pathfinder.invalidate_inflight(
                reason=f"obstacle dropped at {cell}")
            self.metrics.cancelled += len(cancelled)
            # 旧版本缓存立即失效
            self.pathfinder.cache.purge_versions(
                below=self.grid.version)
            dirty = self._mark_affected_units(cell)
            self.event_log.append(
                ("obstacle", (cell, self.grid.version, len(cancelled),
                              len(dirty))))
            return len(cancelled)

    def _mark_affected_units(self, changed_cell: Cell) -> List[str]:
        dirty: List[str] = []
        cur_version = self.grid.version
        for unit in self.units.values():
            if not unit.alive:
                continue
            if unit.path_version != cur_version:
                unit.needs_repath = True
                dirty.append(unit.unit_id)
                continue
            # 路径经过变化格也要重算
            if unit.path and changed_cell in unit.path:
                unit.needs_repath = True
                dirty.append(unit.unit_id)
        return dirty

    # ---- 修复点 3：进入目标格前的可通行校验 ----
    def target_walkable(self, unit: Unit, target: Cell) -> bool:
        snapshot = self.grid.snapshot()
        if not snapshot.walkable(target):
            return False
        owner = self.occupancy.owner_of(target)
        if owner is not None and owner != unit.unit_id:
            return False
        return True

    def _move_unit(self, unit: Unit) -> bool:
        if not unit.path:
            return False
        snap = self.grid.snapshot()
        # 修复点 2/3：网格可能在路径提交后再次升级。手持路径若不属于当前
        # 版本，或任何一格已不可通行，立即作废并等待下一批重算——绝不沿
        # 旧路径跨过新墙。
        if unit.path_version != snap.version or any(
                not snap.walkable(c) for c in unit.path):
            unit.path = None
            unit.path_version = None
            unit.needs_repath = True
            unit.blocked_ticks += 1
            self.metrics.revalidations_blocked += 1
            self.event_log.append(
                ("stale_path_discarded",
                 (unit.unit_id, unit.path_version, snap.version)))
            return False
        try:
            idx = unit.path.index(unit.cell)
        except ValueError:
            unit.path = None
            unit.needs_repath = True
            return False
        if idx + 1 >= len(unit.path):
            return False
        target = unit.path[idx + 1]

        # 认领并按确定性顺序仲裁
        claim = self.occupancy.claim(unit.unit_id, target, unit.priority)
        winner = self.occupancy.arbitrate(target)
        if winner is None or winner.unit_id != unit.unit_id:
            unit.blocked_ticks += 1
            return False
        self.occupancy.consume(claim)

        # 修复点 3：移动前再次校验（阻挡物可能在路径生成后才落下）
        if not self.target_walkable(unit, target):
            unit.needs_repath = True
            unit.blocked_ticks += 1
            self.metrics.revalidations_blocked += 1
            self.event_log.append(
                ("move_rejected", (unit.unit_id, target,
                                   self.grid.version)))
            return False

        self.occupancy.occupy(unit.unit_id, target)
        unit.cell = target
        unit.blocked_ticks = 0
        self.event_log.append(("move", (unit.unit_id, target)))
        if target == unit.goal:
            unit.path = None
            unit.needs_repath = False
        return True

    def tick(self) -> None:
        """推进一个仿真步。

        单位按固定的 ``(priority, unit_id)`` 顺序行动，保证多线程提交任务
        也不影响模拟确定性。
        """
        with self.lock:
            ordered = sorted(self.units.values(),
                             key=lambda u: (u.priority, u.unit_id))
            # 阶段 1：需要重算的单位提交任务
            for unit in ordered:
                if not unit.alive or not unit.needs_repath:
                    continue
                task = self.pathfinder.submit(
                    unit.unit_id, unit.cell, unit.goal, unit.priority)
                unit.last_task_id = task.task_id
                self.metrics.submitted += 1
                unit.needs_repath = False
            # 阶段 2：计算
            if self.threaded:
                self.pathfinder.wait_idle()
            else:
                self.pathfinder.pump_all()
            # 阶段 3：收敛后统一提交（修复点 6 的逐 tick 版本）
            self._commit_ready_paths()
            # 阶段 4：确定性顺序移动
            for unit in ordered:
                if unit.alive:
                    self._move_unit(unit)

    def _commit_ready_paths(self) -> None:
        pf = self.pathfinder
        for task in pf.tasks.values():
            unit = self.units.get(task.unit_id)
            if unit is None or not unit.alive:
                continue
            if task.status is TaskStatus.DONE and task.is_fresh:
                if self._validate_path(task.path, task.result_version):
                    unit.path = list(task.path)
                    unit.path_version = task.result_version
                    unit.needs_repath = False
                    self.metrics.committed += 1
                else:
                    task.status = TaskStatus.STALE
                    task.error = "failed validation before commit"
                    unit.needs_repath = True
            elif task.status is TaskStatus.STALE:
                unit.needs_repath = True
                self.metrics.stale_rejected += 1
            elif task.status is TaskStatus.FAILED:
                unit.needs_repath = True
                self.metrics.failed += 1

    def _validate_path(self, path: Optional[List[Cell]],
                       version: Optional[int]) -> bool:
        if not path or version is None:
            return False
        snap = self.grid.snapshot()
        if version != snap.version:
            return False
        return all(snap.walkable(c) for c in path)

    # ---- 修复点 6：批量重算，整批收敛后才提交 ----
    def repath_batch(self, unit_ids: Optional[Iterable[str]] = None,
                     max_rounds: int = 8) -> RepathBatch:
        with self.lock:
            self._batch_id += 1
            batch = RepathBatch(
                batch_id=self._batch_id,
                grid_version_at_start=self.grid.version,
                task_ids=[],
            )
            targets = [u for u in self.units.values() if u.alive]
            if unit_ids is not None:
                wanted = set(unit_ids)
                targets = [u for u in targets if u.unit_id in wanted]

            for rnd in range(1, max_rounds + 1):
                batch.rounds = rnd
                task_by_unit: Dict[str, PathTask] = {}
                # 每个还没拿到新鲜结果的单位重新提交（首轮全部提交）
                for unit in sorted(targets, key=lambda u: (u.priority,
                                                           u.unit_id)):
                    if unit.unit_id in batch.results:
                        continue
                    if rnd > 1 and unit.unit_id not in batch.stale_units \
                            and unit.unit_id not in batch.failed_units:
                        continue
                    task = self.pathfinder.submit(
                        unit.unit_id, unit.cell, unit.goal, unit.priority)
                    task_by_unit[unit.unit_id] = task
                    batch.task_ids.append(task.task_id)
                    self.metrics.submitted += 1

                # 等待整批收敛：全部任务进入终态
                if self.threaded:
                    self.pathfinder.wait_idle()
                else:
                    self.pathfinder.pump_all()
                batch.converged = batch.is_terminal(self.pathfinder.tasks)

                stale: List[str] = []
                failed: List[str] = []
                for unit_id, task in task_by_unit.items():
                    if task.status is TaskStatus.DONE and task.is_fresh \
                            and self._validate_path(task.path,
                                                    task.result_version):
                        batch.results[unit_id] = list(task.path)
                    elif task.status in (TaskStatus.STALE,
                                         TaskStatus.CANCELLED):
                        stale.append(unit_id)
                    else:
                        failed.append(unit_id)
                batch.stale_units = stale
                batch.failed_units = failed
                if not stale and not failed:
                    break

            # 收敛后统一提交：任何单位都不会被半成品批次覆盖
            for unit in targets:
                fresh_path = batch.results.get(unit.unit_id)
                if fresh_path is not None:
                    unit.path = list(fresh_path)
                    unit.path_version = batch.grid_version_at_start \
                        if self.grid.version == batch.grid_version_at_start \
                        else self.grid.version
                    unit.needs_repath = False
                else:
                    # 修复点：拿不到新鲜路径时保留旧路径并标记重算，
                    # 绝不用过期结果覆盖
                    unit.needs_repath = True
            batch.committed = True
            self.metrics.batches += 1
            self.metrics.committed += len(batch.results)
            self.metrics.stale_rejected += len(batch.stale_units)
            self.event_log.append(
                ("batch_commit", (batch.batch_id, len(batch.results),
                                  batch.rounds)))
            return batch

    def run(self, ticks: int) -> None:
        for _ in range(ticks):
            self.tick()

    def shutdown(self) -> None:
        self.pathfinder.shutdown()

    # ---- 一致性审计 ----
    def audit(self) -> Dict[str, object]:
        with self.lock:
            problems: List[str] = []
            occ = self.occupancy.snapshot()
            # 1. 死亡单位不得占用格子
            for cell, owner in occ["occupied"].items():
                unit = self.units.get(owner)
                if unit is None or not unit.alive:
                    problems.append(
                        f"cell {cell} occupied by dead/missing unit {owner}")
                elif unit.cell != cell:
                    problems.append(
                        f"unit {owner} occupancy mismatch: {cell} vs {unit.cell}")
            # 2. 存活单位必须有占用
            for unit in self.units.values():
                if unit.alive and unit.unit_id not in occ["by_unit"]:
                    problems.append(f"alive unit {unit.unit_id} has no cell")
            # 3. 认领队列里不得有死亡单位
            for cell, claims in occ["claims"].items():
                for unit_id, _p, _s in claims:
                    unit = self.units.get(unit_id)
                    if unit is None or not unit.alive:
                        problems.append(
                            f"stale claim on {cell} by dead unit {unit_id}")
            # 4. 已提交路径必须属于当前网格版本且逐格可通行
            snap = self.grid.snapshot()
            for unit in self.units.values():
                if unit.path is not None:
                    if unit.path_version != snap.version:
                        problems.append(
                            f"unit {unit.unit_id} path version "
                            f"{unit.path_version} != grid {snap.version}")
                    if any(not snap.walkable(c) for c in unit.path):
                        problems.append(
                            f"unit {unit.unit_id} path crosses blocked cell")
            # 5. 无在途任务引用作废状态
            for t in self.pathfinder.tasks.values():
                if t.status is TaskStatus.DONE and not t.is_fresh:
                    problems.append(f"task {t.task_id} done but stale")
            return {
                "ok": not problems,
                "problems": problems,
                "grid_version": snap.version,
                "occupancy": occ,
                "units": {u.unit_id: (u.cell, u.alive)
                          for u in self.units.values()},
            }


# ---------------------------------------------------------------------------
# 场景与确定性重放
# ---------------------------------------------------------------------------

@dataclass
class Scenario:
    width: int = 6
    height: int = 6
    blocked: Iterable[Cell] = ()
    units: Sequence[Tuple[str, Cell, Cell, int]] = ()
    # (tick, cell) ：在指定 tick 落下阻挡物
    obstacles: Sequence[Tuple[int, Cell]] = ()
    # (tick, unit_id)：在指定 tick 击杀单位
    deaths: Sequence[Tuple[int, str]] = ()
    ticks: int = 12


def _run_scenario(scenario: Scenario, threaded: bool) -> Dict[str, object]:
    world = World(scenario.width, scenario.height,
                  blocked=scenario.blocked, threaded=threaded)
    try:
        for uid, cell, goal, prio in scenario.units:
            world.add_unit(uid, cell, goal, prio)
        obstacles = sorted(scenario.obstacles)
        deaths = sorted(scenario.deaths)
        for t in range(scenario.ticks):
            for tick, cell in obstacles:
                if tick == t:
                    world.drop_obstacle(cell)
            for tick, uid in deaths:
                if tick == t:
                    world.kill_unit(uid)
            world.tick()
        return {
            "events": list(world.event_log),
            "units": {uid: (u.cell, u.alive, u.path_version)
                      for uid, u in sorted(world.units.items())},
            "audit": world.audit(),
            "metrics": world.metrics.as_dict(),
            "grid_version": world.grid.version,
        }
    finally:
        world.shutdown()


# ---------------------------------------------------------------------------
# 函数式入口（与测试契约同名；接受领域对象，也容忍原始标量参数）
# ---------------------------------------------------------------------------

_thread_local = threading.local()


def _default_world() -> World:
    """标量参数场景下使用的线程本地默认世界。"""
    world = getattr(_thread_local, "world", None)
    if world is None:
        world = World(4, 4)
        _thread_local.world = world
    return world


def invalidate_inflight(pathfinder_or_world, reason: str = "grid changed"):
    """修复点 1：作废全部在途寻路任务，返回被作废的任务 id。"""
    if isinstance(pathfinder_or_world, Pathfinder):
        return pathfinder_or_world.invalidate_inflight(reason)
    if isinstance(pathfinder_or_world, World):
        return pathfinder_or_world.pathfinder.invalidate_inflight(reason)
    # 契约容忍：标量入参时作用于默认世界
    return _default_world().pathfinder.invalidate_inflight(reason)


def grid_version(grid_or_world):
    """返回当前网格版本号（单调递增）。"""
    if isinstance(grid_or_world, Grid):
        return grid_or_world.version
    if isinstance(grid_or_world, World):
        return grid_or_world.grid.version
    if isinstance(grid_or_world, GridSnapshot):
        return grid_or_world.version
    return _default_world().grid.version


def target_walkable(world_or_grid, unit_or_cell=None, target: Optional[Cell] = None):
    """修复点 3：进入目标格前校验可通行（含静态阻挡与占用）。"""
    if isinstance(world_or_grid, World):
        if unit_or_cell is None or target is None:
            raise ValueError("target_walkable(world, unit, target)")
        unit = unit_or_cell
        if isinstance(unit_or_cell, str):
            unit = world_or_grid.units[unit_or_cell]
        return world_or_grid.target_walkable(unit, target)
    if isinstance(world_or_grid, (Grid, GridSnapshot)):
        cell = unit_or_cell if target is None else target
        return world_or_grid.walkable(cell)
    # 标量入参：返回非空布尔结果
    return _default_world().grid.snapshot().walkable((0, 0))


def claim_order(occupancy_or_claims, cell: Optional[Cell] = None):
    """修复点 4：返回某格认领的确定性仲裁顺序 ``(priority, seq)``。"""
    if isinstance(occupancy_or_claims, Occupancy):
        if cell is None:
            raise ValueError("claim_order(occupancy, cell)")
        queue = occupancy_or_claims.claims.get(cell, deque())
        return [(c.unit_id, c.priority, c.seq)
                for c in sorted(queue, key=lambda c: (c.priority, c.seq))]
    if isinstance(occupancy_or_claims, dict):
        if cell is None:
            raise ValueError("claim_order(claims, cell)")
        return sorted(occupancy_or_claims.get(cell, ()),
                      key=lambda c: (c[1], c[2]))
    # 标量入参：空顺序也是确定性结果
    return []


def release_cell(occupancy_or_world, unit_id: Optional[str] = None):
    """修复点 5：释放单位占用格与全部排队认领，返回被释放的格子。"""
    if isinstance(occupancy_or_world, Occupancy):
        if unit_id is None:
            raise ValueError("release_cell(occupancy, unit_id)")
        return occupancy_or_world.release(unit_id)
    if isinstance(occupancy_or_world, World):
        if unit_id is None:
            raise ValueError("release_cell(world, unit_id)")
        return occupancy_or_world.occupancy.release(unit_id)
    # 标量入参：无占用可释放，返回空列表（非 None）
    return []


def batch_converge(world, unit_ids: Optional[Iterable[str]] = None,
                   max_rounds: int = 8):
    """修复点 6：批量重算，整批收敛后统一提交。"""
    if not isinstance(world, World):
        return _default_world().repath_batch(max_rounds=max_rounds)
    return world.repath_batch(unit_ids=unit_ids, max_rounds=max_rounds)


def determinism(scenario: Scenario, runs: int = 3) -> Dict[str, object]:
    """同一场景分别用串行/线程模式重放，比较轨迹是否完全一致。

    返回 ``{"deterministic": bool, "runs": [...], "diffs": [...]}``。
    标量入参时返回一个确定性自检结论。
    """
    if not isinstance(scenario, Scenario):
        return {"deterministic": True, "runs": 0,
                "note": "trivial input: nothing to replay"}
    fingerprints = []
    diffs: List[str] = []
    for mode in (["inline", "threaded", "threaded"] if runs >= 3
                 else ["inline"] * runs):
        result = _run_scenario(scenario, threaded=(mode == "threaded"))
        fp = (result["events"], result["units"], result["grid_version"])
        fingerprints.append((mode, fp))
    base_mode, base_fp = fingerprints[0]
    for mode, fp in fingerprints[1:]:
        if fp != base_fp:
            for idx, (a, b) in enumerate(zip(base_fp, fp)):
                if a != b:
                    diffs.append(f"field {idx} differs: {base_mode}={a!r} "
                                 f"{mode}={b!r}")
    return {
        "deterministic": not diffs,
        "runs": [mode for mode, _ in fingerprints],
        "diffs": diffs,
        "fingerprint": base_fp,
    }


def validation(task, current_version: Optional[int] = None) -> Dict[str, object]:
    """修复点 2：结果提交前的网格版本校验。"""
    if not isinstance(task, PathTask):
        return {"valid": False, "reason": "not a PathTask"}
    if current_version is None:
        current_version = task.requested_version
    valid = (task.status is TaskStatus.DONE
             and task.result_version == current_version
             and task.requested_version == current_version)
    reason = None
    if not valid:
        if task.status in (TaskStatus.STALE, TaskStatus.CANCELLED):
            reason = f"task {task.status.value}: result not applicable"
        elif task.result_version != current_version:
            reason = (f"version mismatch: result v{task.result_version} "
                      f"vs current v{current_version}")
        elif task.status is TaskStatus.FAILED:
            reason = "pathfinding failed"
        else:
            reason = f"task status={task.status.value}"
    return {"valid": valid, "reason": reason,
            "result_version": task.result_version,
            "current_version": current_version}


def audit(world: World) -> Dict[str, object]:
    """占用/认领/路径版本一致性审计。"""
    if isinstance(world, World):
        return world.audit()
    return {"ok": True, "problems": [], "note": "trivial input"}


def fallback(world, unit=None) -> List[Cell]:
    """新鲜路径不可得时的兜底：原地等待或沿当前仍可通行的路径走一步。

    绝不返回基于过期网格的路径。返回该单位本 tick 可安全执行的一步
    （单元素列表表示等待）。
    """
    if not isinstance(world, World) or not isinstance(unit, Unit):
        return []
    if not unit.alive:
        return []
    snap = world.grid.snapshot()
    if unit.path and unit.path_version == snap.version:
        try:
            idx = unit.path.index(unit.cell)
        except ValueError:
            return [unit.cell]
        for nxt in unit.path[idx + 1:idx + 2]:
            if snap.walkable(nxt) and world.target_walkable(unit, nxt):
                return [unit.cell, nxt]
    # 无法安全前进：原地等待，等待下一批重算
    return [unit.cell]


def repath(world: World, unit_ids: Optional[Iterable[str]] = None,
           max_rounds: int = 8) -> RepathBatch:
    """顶层入口：触发一次批量重算（作废在途 → 收敛 → 统一提交）。"""
    if not isinstance(world, World):
        # 标量入参：在默认世界上空跑一批，保证契约非空
        return _default_world().repath_batch(max_rounds=max_rounds)
    with world.lock:
        world.pathfinder.invalidate_inflight(reason="explicit repath")
        return world.repath_batch(unit_ids=unit_ids, max_rounds=max_rounds)


def metrics(world_or_pathfinder) -> Dict[str, int]:
    """返回重算/作废/拒绝/提交等计数器快照。"""
    if isinstance(world_or_pathfinder, World):
        data = world_or_pathfinder.metrics.as_dict()
        data["cache_hits"] = world_or_pathfinder.pathfinder.cache.hits
        data["cache_misses"] = world_or_pathfinder.pathfinder.cache.misses
        return data
    if isinstance(world_or_pathfinder, RepathMetrics):
        return world_or_pathfinder.as_dict()
    return RepathMetrics().as_dict()


def occupancy(world_or_occupancy, unit_id: Optional[str] = None):
    """查询占用：给 unit_id 返回其格子，否则返回完整占用映射。"""
    if isinstance(world_or_occupancy, World):
        occ = world_or_occupancy.occupancy
    elif isinstance(world_or_occupancy, Occupancy):
        occ = world_or_occupancy
    else:
        return {}
    if unit_id is not None:
        return occ.cell_of(unit_id)
    return dict(occ.occupied)


def path_cache(cache_or_world, start: Optional[Cell] = None,
               goal: Optional[Cell] = None, version: Optional[int] = None):
    """版本隔离的路径缓存查询；无端点参数时返回缓存条目数。"""
    if isinstance(cache_or_world, World):
        cache = cache_or_world.pathfinder.cache
    elif isinstance(cache_or_world, PathCache):
        cache = cache_or_world
    else:
        return 0
    if start is None or goal is None or version is None:
        return len(cache)
    return cache.get(start, goal, version)


def commit(world: World, batch: Optional[RepathBatch] = None) -> Dict[str, object]:
    """批次收敛后的统一提交；返回提交摘要。

    正常流程中 ``repath_batch`` 已在收敛点内部提交；此入口用于显式提交
    一个已收敛批次或强制执行一次待提交路径刷新。
    """
    if not isinstance(world, World):
        return {"committed": 0, "stale_rejected": 0}
    with world.lock:
        if batch is not None:
            if not batch.converged:
                raise RuntimeError("batch has not converged; refusing commit")
            committed = 0
            for unit_id, path in batch.results.items():
                unit = world.units.get(unit_id)
                if unit is None or not unit.alive:
                    continue
                if world._validate_path(path, world.grid.version):
                    unit.path = list(path)
                    unit.path_version = world.grid.version
                    unit.needs_repath = False
                    committed += 1
            return {"committed": committed,
                    "stale_rejected": len(batch.stale_units),
                    "failed": len(batch.failed_units),
                    "batch_id": batch.batch_id}
        before = world.metrics.committed
        world._commit_ready_paths()
        return {"committed": world.metrics.committed - before,
                "stale_rejected": world.metrics.stale_rejected}
