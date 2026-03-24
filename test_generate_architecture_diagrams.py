"""
Tests for scripts/generate_architecture_diagrams.py

These tests exercise all non-network logic (file reading, prompt building,
argument parsing, config completeness) using mocks for the LLM API call.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import textwrap
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Import the script as a module
# ---------------------------------------------------------------------------
_SCRIPT_PATH = Path(__file__).parent / "scripts" / "generate_architecture_diagrams.py"
_SPEC = importlib.util.spec_from_file_location("generate_architecture_diagrams", _SCRIPT_PATH)
_MOD = importlib.util.module_from_spec(_SPEC)  # type: ignore[arg-type]
_SPEC.loader.exec_module(_MOD)  # type: ignore[union-attr]
g = _MOD  # shorthand


# ---------------------------------------------------------------------------
# Config completeness
# ---------------------------------------------------------------------------


class TestDiagramConfigs(unittest.TestCase):
    """The _DIAGRAM_CONFIGS list must cover all architecture documents."""

    def test_seven_configs(self) -> None:
        self.assertEqual(len(g._DIAGRAM_CONFIGS), 7)

    def test_all_configs_have_required_keys(self) -> None:
        for cfg in g._DIAGRAM_CONFIGS:
            with self.subTest(cfg=cfg.get("filename")):
                self.assertIn("filename", cfg)
                self.assertIn("source_files", cfg)
                self.assertIn("description", cfg)
                self.assertIsInstance(cfg["source_files"], list)
                self.assertTrue(cfg["source_files"])

    def test_filenames_match_architecture_docs(self) -> None:
        arch_dir = Path(__file__).parent / "docs" / "architecture"
        existing = {p.name for p in arch_dir.glob("*.md") if p.name != "README.md"}
        configured = {cfg["filename"] for cfg in g._DIAGRAM_CONFIGS}
        self.assertEqual(configured, existing, "Configs must cover every docs/architecture/*.md")


# ---------------------------------------------------------------------------
# _read_source_file
# ---------------------------------------------------------------------------


class TestReadSourceFile(unittest.TestCase):
    def test_returns_error_sentinel_for_missing_file(self) -> None:
        result = g._read_source_file(Path("/nonexistent/file.py"))
        self.assertIn("not found", result)

    def test_reads_real_file(self) -> None:
        real = Path(__file__)
        content = g._read_source_file(real)
        self.assertIn("generate_architecture_diagrams", content)

    def test_truncates_large_files(self, tmp_path: Path = None) -> None:  # type: ignore[assignment]
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as fh:
            fh.write("x" * 1000)
            tmp = Path(fh.name)
        try:
            result = g._read_source_file(tmp, max_chars=100)
            self.assertLessEqual(len(result.split("[truncated")[0]), 110)
            self.assertIn("truncated", result)
        finally:
            tmp.unlink()


# ---------------------------------------------------------------------------
# _collect_source_files
# ---------------------------------------------------------------------------


class TestCollectSourceFiles(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).parent

    def test_collects_individual_files(self) -> None:
        files = g._collect_source_files(
            self.repo_root, ["msr_mcp_server.py", "msr_mcp_server_main.py"]
        )
        self.assertIn("msr_mcp_server.py", files)
        self.assertIn("msr_mcp_server_main.py", files)
        self.assertGreater(len(files["msr_mcp_server.py"]), 100)

    def test_expands_directory(self) -> None:
        files = g._collect_source_files(
            self.repo_root, ["use_cases/physical_ai/"]
        )
        # Should include the 12 use-case .md files
        md_keys = [k for k in files if k.endswith(".md")]
        self.assertGreaterEqual(len(md_keys), 12)

    def test_missing_file_returns_sentinel(self) -> None:
        files = g._collect_source_files(self.repo_root, ["does_not_exist.py"])
        self.assertIn("not found", files["does_not_exist.py"])


# ---------------------------------------------------------------------------
# _build_user_prompt
# ---------------------------------------------------------------------------


class TestBuildUserPrompt(unittest.TestCase):
    def test_includes_description(self) -> None:
        prompt = g._build_user_prompt(
            description="Test description",
            source_files={"foo.py": "print('hello')"},
            existing_content="# Existing doc\n",
        )
        self.assertIn("Test description", prompt)

    def test_includes_source_filename(self) -> None:
        prompt = g._build_user_prompt(
            description="desc",
            source_files={"bar.py": "x = 1"},
            existing_content="# doc",
        )
        self.assertIn("`bar.py`", prompt)

    def test_includes_existing_content(self) -> None:
        prompt = g._build_user_prompt(
            description="desc",
            source_files={"a.py": "pass"},
            existing_content="## Existing section\nsome text",
        )
        self.assertIn("## Existing section", prompt)
        self.assertIn("some text", prompt)

    def test_python_fenced_as_python(self) -> None:
        prompt = g._build_user_prompt(
            description="desc",
            source_files={"module.py": "def f(): pass"},
            existing_content="",
        )
        self.assertIn("```python\ndef f(): pass", prompt)

    def test_yaml_fenced_as_yaml(self) -> None:
        prompt = g._build_user_prompt(
            description="desc",
            source_files={"template.yaml": "AWSTemplateFormatVersion: 2010-09-09"},
            existing_content="",
        )
        self.assertIn("```yaml", prompt)


# ---------------------------------------------------------------------------
# _resolve_api_config
# ---------------------------------------------------------------------------


class TestResolveApiConfig(unittest.TestCase):
    def _run_with_env(self, env: dict) -> tuple:
        with patch.dict("os.environ", env, clear=True):
            return g._resolve_api_config()

    def test_uses_openai_key_first(self) -> None:
        api_key, base_url, model = self._run_with_env(
            {
                "MSR_OPENAI_API_KEY": "sk-test",
                "GITHUB_TOKEN": "ghu_test",
            }
        )
        self.assertEqual(api_key, "sk-test")
        self.assertIn("openai.com", base_url)

    def test_falls_back_to_github_token(self) -> None:
        api_key, base_url, _ = self._run_with_env({"GITHUB_TOKEN": "ghu_test"})
        self.assertEqual(api_key, "ghu_test")
        self.assertIn("inference.ai.azure.com", base_url)

    def test_accepts_msr_github_token(self) -> None:
        api_key, _, _ = self._run_with_env({"MSR_GITHUB_TOKEN": "ghu_msr"})
        self.assertEqual(api_key, "ghu_msr")

    def test_exits_when_no_credentials(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(SystemExit):
                g._resolve_api_config()

    def test_model_override(self) -> None:
        _, _, model = self._run_with_env({"GITHUB_TOKEN": "t"})
        self.assertEqual(model, g._DEFAULT_MODEL)

        _, _, model = self._run_with_env({"GITHUB_TOKEN": "t", "MSR_OPENAI_MODEL": "gpt-4-turbo"})
        self.assertEqual(model, "gpt-4-turbo")

    def test_cli_model_override_takes_precedence(self) -> None:
        with patch.dict("os.environ", {"GITHUB_TOKEN": "t", "MSR_OPENAI_MODEL": "env-model"}):
            _, _, model = g._resolve_api_config(model_override="cli-model")
        self.assertEqual(model, "cli-model")


# ---------------------------------------------------------------------------
# _call_llm — success and failure paths (mocked network)
# ---------------------------------------------------------------------------


class TestCallLlm(unittest.TestCase):
    def _fake_response(self, content: str) -> MagicMock:
        body = json.dumps(
            {"choices": [{"message": {"content": content}}]}
        ).encode()
        mock_resp = MagicMock()
        mock_resp.read.return_value = body
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        return mock_resp

    def test_returns_stripped_content(self) -> None:
        with patch("urllib.request.urlopen", return_value=self._fake_response("  hello  ")):
            result = g._call_llm(
                [{"role": "user", "content": "test"}],
                api_key="key",
                base_url="https://example.com",
                model="gpt-4o",
            )
        self.assertEqual(result, "hello")

    def test_raises_on_http_error(self) -> None:
        http_err = urllib.error.HTTPError(
            url="https://example.com",
            code=401,
            msg="Unauthorized",
            hdrs=MagicMock(),  # type: ignore[arg-type]
            fp=MagicMock(**{"read.return_value": b"Unauthorized"}),
        )
        with patch("urllib.request.urlopen", side_effect=http_err):
            with self.assertRaises(urllib.error.HTTPError):
                g._call_llm(
                    [{"role": "user", "content": "test"}],
                    api_key="bad",
                    base_url="https://example.com",
                    model="gpt-4o",
                )


# ---------------------------------------------------------------------------
# generate_diagram_doc — end-to-end with mocked LLM
# ---------------------------------------------------------------------------


class TestGenerateDiagramDoc(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).parent

    def test_returns_llm_output_with_trailing_newline(self) -> None:
        mock_output = "# Updated doc\n```mermaid\nflowchart TD\n  A --> B\n```"
        with patch.object(g, "_call_llm", return_value=mock_output):
            config = {
                "filename": "03_mcp_server.md",
                "source_files": ["msr_mcp_server.py"],
                "description": "MCP server tools",
            }
            result = g.generate_diagram_doc(
                config,
                self.repo_root,
                api_key="key",
                base_url="https://example.com",
                model="gpt-4o",
            )
        self.assertTrue(result.endswith("\n"))
        self.assertIn("```mermaid", result)

    def test_handles_missing_arch_doc_gracefully(self, tmp_path: Path = None) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            fake_root = Path(tmpdir)
            # Create minimal file tree the script checks
            (fake_root / "msr_mcp_server.py").write_text("class MSRMCPServer: pass")
            (fake_root / "docs" / "architecture").mkdir(parents=True)
            # Note: we do NOT create 99_nonexistent.md

            with patch.object(g, "_call_llm", return_value="# New doc\n"):
                result = g.generate_diagram_doc(
                    {
                        "filename": "99_nonexistent.md",
                        "source_files": ["msr_mcp_server.py"],
                        "description": "test",
                    },
                    fake_root,
                    api_key="key",
                    base_url="https://example.com",
                    model="gpt-4o",
                )
        self.assertIn("New doc", result)


# ---------------------------------------------------------------------------
# Integration: all configs produce non-empty prompts against real repo files
# ---------------------------------------------------------------------------


class TestAllConfigsProducePrompts(unittest.TestCase):
    """Verify every diagram config can build a user prompt without errors."""

    def test_all_configs(self) -> None:
        repo_root = Path(__file__).parent
        for cfg in g._DIAGRAM_CONFIGS:
            with self.subTest(filename=cfg["filename"]):
                files = g._collect_source_files(repo_root, cfg["source_files"])
                self.assertGreater(len(files), 0, "Must collect at least one source file")

                existing_path = repo_root / "docs" / "architecture" / cfg["filename"]
                existing = existing_path.read_text() if existing_path.exists() else ""

                prompt = g._build_user_prompt(cfg["description"], files, existing)
                self.assertGreater(len(prompt), 200)
                self.assertIn(cfg["description"][:40], prompt)


if __name__ == "__main__":
    unittest.main()
