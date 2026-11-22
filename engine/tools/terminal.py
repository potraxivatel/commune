"""
Terminal execution tools for Commune AI.
Provides RunCommandComponent for executing shell commands with timeouts.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from engine.communication.envelope import ContentType, Envelope, MessageMetadata, MessageStatus
from engine.core.component import Component, ComponentKind, ComponentManifest


class RunCommandComponent(Component):
    """Executes a shell command asynchronously with timeout, capturing stdout/stderr."""

    def manifest(self) -> ComponentManifest:
        return ComponentManifest(
            name="run_command",
            kind=ComponentKind.CODE,
            description="Executes a shell command asynchronously with a timeout, capturing exit code and output",
            version="0.2.0",
            accepts=["text", "json"],
            produces=["json", "text"],
            tags=["terminal", "shell", "tool", "execution"],
        )

    async def process(self, envelope: Envelope, context: dict[str, Any] | None = None) -> Envelope:
        start_time = time.time()
        task_id = envelope.task_id

        command = ""
        timeout_sec = 30.0
        cwd: str | None = None

        if isinstance(envelope.content, dict):
            command = str(envelope.content.get("command", "")).strip()
            timeout_sec = float(envelope.content.get("timeout", 30.0))
            cwd = envelope.content.get("cwd")
        elif isinstance(envelope.content, str):
            command = envelope.content.strip()

        if not command:
            return Envelope.error(task_id, "run_command", "Empty command provided")

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )

            timed_out = False
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=timeout_sec,
                )
            except asyncio.TimeoutError:
                timed_out = True
                try:
                    proc.kill()
                    await proc.wait()
                except ProcessLookupError:
                    pass
                stdout_bytes, stderr_bytes = b"", b"Command timed out after %.1fs" % timeout_sec

            duration_ms = (time.time() - start_time) * 1000.0
            stdout_str = stdout_bytes.decode("utf-8", errors="replace")
            stderr_str = stderr_bytes.decode("utf-8", errors="replace")
            exit_code = -1 if timed_out else (proc.returncode or 0)

            status = MessageStatus.ERROR if timed_out else MessageStatus.OK

            payload = {
                "command": command,
                "exit_code": exit_code,
                "stdout": stdout_str,
                "stderr": stderr_str,
                "timed_out": timed_out,
                "duration_ms": duration_ms,
            }

            approx_tokens = (len(stdout_str) + len(stderr_str)) // 4

            return Envelope(
                task_id=task_id,
                sender="run_command",
                status=status,
                content_type=ContentType.JSON,
                content=payload,
                metadata=MessageMetadata(
                    token_count=approx_tokens,
                    latency_ms=duration_ms,
                ),
            )
        except Exception as e:
            return Envelope.error(task_id, "run_command", f"Execution error: {e}")
