"""test_sanitizer.py – Validates _sanitize_messages correctness."""
import sys
sys.path.insert(0, ".")

import ast, pathlib
ast.parse(pathlib.Path("agent/core.py").read_text())
print("syntax OK")

from agent.core import SelfEvolvingAgent
S = SelfEvolvingAgent._sanitize_messages

# Test 1: multi-tool-call group – ALL results present → must be fully preserved
msgs1 = [
    {"role": "system", "content": "sys"},
    {"role": "user", "content": "go"},
    {"role": "assistant", "content": None, "tool_calls": [{"id": "A"}, {"id": "B"}]},
    {"role": "tool", "tool_call_id": "A", "name": "f", "content": "r1"},
    {"role": "tool", "tool_call_id": "B", "name": "f", "content": "r2"},
]
out1 = S(msgs1)
assert [m["role"] for m in out1] == ["system", "user", "assistant", "tool", "tool"], \
    f"Test1 FAIL: {[m['role'] for m in out1]}"
print("Test 1 PASS: multi-call group fully preserved")

# Test 2: assistant trimmed → tool(A) and tool(B) are orphaned → must be dropped
msgs2 = [
    {"role": "system", "content": "sys"},
    {"role": "tool", "tool_call_id": "A", "name": "f", "content": "r1"},
    {"role": "tool", "tool_call_id": "B", "name": "f", "content": "r2"},
    {"role": "user", "content": "continue"},
]
out2 = S(msgs2)
assert all(m["role"] != "tool" for m in out2), \
    f"Test2 FAIL: orphans kept {[m['role'] for m in out2]}"
print("Test 2 PASS: orphaned tool results dropped")

# Test 3: assistant+[A,B] present but tool(B) trimmed → incomplete group dropped atomically
msgs3 = [
    {"role": "system", "content": "sys"},
    {"role": "user", "content": "go"},
    {"role": "assistant", "content": None, "tool_calls": [{"id": "A"}, {"id": "B"}]},
    {"role": "tool", "tool_call_id": "A", "name": "f", "content": "r1"},
    {"role": "user", "content": "continue"},
]
out3 = S(msgs3)
assert all(m["role"] != "tool" for m in out3), \
    f"Test3 FAIL: partial result kept {[m['role'] for m in out3]}"
print("Test 3 PASS: incomplete group fully dropped")

# Test 4: two complete single-call groups in sequence
msgs4 = [
    {"role": "system", "content": "sys"},
    {"role": "user", "content": "go"},
    {"role": "assistant", "content": None, "tool_calls": [{"id": "A"}]},
    {"role": "tool", "tool_call_id": "A", "name": "f", "content": "r1"},
    {"role": "assistant", "content": None, "tool_calls": [{"id": "B"}]},
    {"role": "tool", "tool_call_id": "B", "name": "g", "content": "r2"},
]
out4 = S(msgs4)
assert [m["role"] for m in out4] == ["system", "user", "assistant", "tool", "assistant", "tool"], \
    f"Test4 FAIL: {[m['role'] for m in out4]}"
print("Test 4 PASS: two complete sequential groups preserved")

print("\nAll sanitizer tests passed!")
