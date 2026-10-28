"""Small graph helpers: topological ordering with cycle detection."""
from collections import defaultdict, deque
from typing import Iterable, List


def topo_order(task_ids: Iterable[str], relations) -> List[str]:
    """Kahn's algorithm. Raises ValueError if the relation graph has a cycle."""
    ids = list(task_ids)
    indeg = {t: 0 for t in ids}
    out = defaultdict(set)
    for r in relations:
        for s in r.sources:
            if r.target not in out[s]:
                out[s].add(r.target)
                indeg[r.target] += 1
    queue = deque(t for t in ids if indeg[t] == 0)
    order: List[str] = []
    while queue:
        t = queue.popleft()
        order.append(t)
        for nxt in out[t]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                queue.append(nxt)
    if len(order) != len(ids):
        raise ValueError("Task relations contain a cycle; dependencies must form a DAG.")
    return order
