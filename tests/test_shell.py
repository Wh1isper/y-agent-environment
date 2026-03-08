"""Tests for Shell class."""

from pathlib import Path

from y_agent_environment import Shell


class ConcreteShell(Shell):
    """Concrete Shell implementation for testing."""

    async def execute(
        self,
        command: str,
        *,
        timeout: float | None = None,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> tuple[int, str, str]:
        return (0, "", "")


async def test_shell_default_cwd_none() -> None:
    """Shell with default_cwd=None should have empty allowed_paths."""
    shell = ConcreteShell(default_cwd=None)
    assert shell._default_cwd is None
    assert shell._allowed_paths == []


async def test_shell_default_cwd_none_with_allowed_paths(tmp_path: Path) -> None:
    """Shell with default_cwd=None should accept explicit allowed_paths."""
    shell = ConcreteShell(default_cwd=None, allowed_paths=[tmp_path])
    assert shell._default_cwd is None
    assert len(shell._allowed_paths) == 1
    assert shell._allowed_paths[0] == tmp_path.resolve()


async def test_shell_default_cwd_none_context_instructions() -> None:
    """Shell with default_cwd=None should omit default-working-directory."""
    shell = ConcreteShell(default_cwd=None)
    instructions = await shell.get_context_instructions()
    assert instructions is not None
    assert "default-working-directory" not in instructions
    assert "allowed-working-directories" not in instructions
    assert "default-timeout" in instructions


async def test_shell_default_cwd_none_with_allowed_paths_instructions(tmp_path: Path) -> None:
    """Shell with default_cwd=None but allowed_paths should show paths but no default."""
    shell = ConcreteShell(default_cwd=None, allowed_paths=[tmp_path])
    instructions = await shell.get_context_instructions()
    assert instructions is not None
    assert "default-working-directory" not in instructions
    assert "allowed-working-directories" in instructions
    assert str(tmp_path.resolve()) in instructions


async def test_shell_with_default_cwd_context_instructions(tmp_path: Path) -> None:
    """Shell with default_cwd should include both default and allowed in instructions."""
    shell = ConcreteShell(default_cwd=tmp_path)
    instructions = await shell.get_context_instructions()
    assert instructions is not None
    assert "default-working-directory" in instructions
    assert "allowed-working-directories" in instructions
    assert str(tmp_path.resolve()) in instructions


async def test_shell_skip_instructions() -> None:
    """Shell with skip_instructions should return None."""
    shell = ConcreteShell(default_cwd=None, skip_instructions=True)
    instructions = await shell.get_context_instructions()
    assert instructions is None


async def test_shell_default_cwd_set() -> None:
    """Shell with default_cwd should work as before."""
    shell = ConcreteShell(default_cwd=Path("/tmp/test"))
    assert shell._default_cwd == Path("/tmp/test").resolve()
    assert shell._default_cwd in shell._allowed_paths


async def test_shell_default_cwd_always_in_allowed(tmp_path: Path) -> None:
    """default_cwd should be added to allowed_paths even if not explicitly listed."""
    other = tmp_path / "other"
    other.mkdir()
    shell = ConcreteShell(default_cwd=tmp_path, allowed_paths=[other])
    assert tmp_path.resolve() in shell._allowed_paths
    assert other.resolve() in shell._allowed_paths
