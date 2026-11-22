"""
Tests for Commune AI Multi-Agent Swarm Team.
"""

from pathlib import Path

import pytest

from engine.memory.store import MemoryStore
from engine.swarm.roles import CoderMember, PlannerMember, ReviewerMember
from engine.swarm.team import SwarmTeam


def test_swarm_member_roles():
    p = PlannerMember()
    c = CoderMember()
    r = ReviewerMember()

    assert p.manifest().name == "planner"
    assert c.manifest().name == "coder"
    assert r.manifest().name == "reviewer"

    assert p.avatar == "🧠"
    assert c.avatar == "💻"
    assert r.avatar == "🔍"


@pytest.mark.asyncio
async def test_swarm_team_code_generation(tmp_path: Path):
    store = MemoryStore(tmp_path / "memory")
    team = SwarmTeam(memory_store=store)

    target_script = str(tmp_path / "calc_test.py")
    query = f"Создай скрипт {target_script} для сложения чисел"

    turn = await team.chat(query)

    # 1. Check internal discussion happened
    assert len(turn.internal_discussion) == 3
    senders = [m.sender for m in turn.internal_discussion]
    assert senders == ["planner", "coder", "reviewer"]

    # 2. Check coder wrote the file
    target_p = Path(target_script)
    assert target_p.exists()
    assert "def main" in target_p.read_text()

    # 3. Check reviewer validated the file
    assert "ошибок нет" in turn.internal_discussion[2].content

    # 4. Check planner formatted user response
    assert turn.final_answer.recipient == "user"
    assert "Всё готово" in turn.final_answer.content

    # 5. Check memory saved the turn
    active_records = store.all_active()
    assert any("swarm_chat_" in r.key for r in active_records)


@pytest.mark.asyncio
async def test_swarm_team_general_discussion(tmp_path: Path):
    team = SwarmTeam(memory_store=None)
    turn = await team.chat("Объясни, как работает конвейер в двух словах")

    # General questions are routed to direct architect answer without dragging Coder/Reviewer
    assert len(turn.internal_discussion) == 1
    assert turn.internal_discussion[0].sender == "planner"
    assert turn.final_answer.recipient == "user"
    assert len(turn.final_answer.content) > 0


@pytest.mark.asyncio
async def test_swarm_team_selective_review(tmp_path: Path):
    test_file = tmp_path / "valid.py"
    test_file.write_text("print('hello')\n")
    team = SwarmTeam(memory_store=None)
    turn = await team.chat(f"Проверь и протестируй {test_file}")

    # Selective invocation: Planner + Reviewer (Coder is not called)
    senders = [m.sender for m in turn.internal_discussion]
    assert "planner" in senders
    assert "reviewer" in senders
    assert "coder" not in senders
    assert turn.final_answer.recipient == "user"
