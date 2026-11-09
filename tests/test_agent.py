"""エージェントの配線テスト。

TestModel は実際のLLMを呼ばない偽モデルで、登録されたツールを全部呼んでから
ダミーの回答を返す。APIキーも利用枠も使わずに、
「ツールが登録され、呼ばれ、結果がモデルに戻るか」を確かめられる。
"""

import pytest
from pydantic_ai import DeferredToolRequests, DeferredToolResults
from pydantic_ai.messages import ToolCallPart, ToolReturnPart
from pydantic_ai.models.test import TestModel

from app.agent import agent
from app import tools

BASE_TOOLS = {"search_glossary", "get_lecture_summary"}


def _parts(result, kind):
    return [p for m in result.all_messages() for p in m.parts if isinstance(p, kind)]


def test_base_tools_are_registered_and_called():
    model = TestModel()
    with agent.override(model=model):
        result = agent.run_sync("LoRAとは？")

    offered = {t.name for t in model.last_model_request_parameters.function_tools}
    assert BASE_TOOLS <= offered  # 演習でツールを足しても壊れないよう「含む」で判定

    called = {p.tool_name for p in _parts(result, ToolCallPart)}
    assert BASE_TOOLS <= called
    # 承認が必要なツール（requires_approval=True）は、実行されずに承認待ちになるので戻り値が無い
    waiting = len(result.output.approvals) if isinstance(result.output, DeferredToolRequests) else 0
    assert len(_parts(result, ToolReturnPart)) + waiting == len(_parts(result, ToolCallPart))
    assert result.output


def test_glossary_result_reaches_the_model():
    model = TestModel(call_tools=["search_glossary"])
    with agent.override(model=model):
        result = agent.run_sync("RAGとは？")
    returns = _parts(result, ToolReturnPart)
    assert returns and returns[0].tool_name == "search_glossary"


@pytest.mark.parametrize("approved", [True, False])
def test_learning_log_requires_approval(tmp_path, monkeypatch, approved):
    monkeypatch.setattr(tools, "LEARNING_LOG_DIR", tmp_path)
    model = TestModel(call_tools=["save_learning_log"])
    with agent.override(model=model):
        pending = agent.run_sync("この内容を学習記録に保存して: 承認テスト", model=model)

    assert isinstance(pending.output, DeferredToolRequests)
    assert not list(tmp_path.iterdir())
    calls = pending.output.approvals
    assert len(calls) == 1
    call = calls[0]

    with agent.override(model=TestModel(call_tools=[])):
        agent.run_sync(
            None,
            message_history=pending.all_messages(),
            deferred_tool_results=DeferredToolResults(
                approvals={call.tool_call_id: approved}
            ),
        )

    saved_files = list(tmp_path.iterdir())
    assert bool(saved_files) is approved
