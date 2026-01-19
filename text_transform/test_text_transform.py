#!/usr/bin/env python3
"""
Comprehensive tests for text_transform CLI tool.
"""

import base64
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

# Add the directory to path for importing
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import text_transform


class TestJSONOperations(unittest.TestCase):
    """Test JSON parsing and conversion."""

    def test_parse_json_valid(self):
        """Test parsing valid JSON."""
        data = '{"name": "test", "value": 42}'
        result = text_transform.parse_json(data)
        self.assertEqual(result, {"name": "test", "value": 42})

    def test_parse_json_array(self):
        """Test parsing JSON array."""
        data = '[1, 2, 3]'
        result = text_transform.parse_json(data)
        self.assertEqual(result, [1, 2, 3])

    def test_parse_json_invalid(self):
        """Test parsing invalid JSON exits with code 1."""
        with self.assertRaises(SystemExit) as cm:
            text_transform.parse_json('{invalid json}')
        self.assertEqual(cm.exception.code, text_transform.EXIT_PARSE_ERROR)

    def test_to_json(self):
        """Test converting to JSON."""
        data = {"name": "test", "value": 42}
        result = text_transform.to_json(data)
        self.assertIn('"name": "test"', result)
        self.assertIn('"value": 42', result)


class TestYAMLOperations(unittest.TestCase):
    """Test YAML parsing and conversion."""

    def setUp(self):
        """Skip tests if yaml not available."""
        if text_transform.yaml is None:
            self.skipTest("pyyaml not installed")

    def test_parse_yaml_valid(self):
        """Test parsing valid YAML."""
        data = "name: test\nvalue: 42"
        result = text_transform.parse_yaml(data)
        self.assertEqual(result, {"name": "test", "value": 42})

    def test_parse_yaml_list(self):
        """Test parsing YAML list."""
        data = "- one\n- two\n- three"
        result = text_transform.parse_yaml(data)
        self.assertEqual(result, ["one", "two", "three"])

    def test_to_yaml(self):
        """Test converting to YAML."""
        data = {"name": "test", "value": 42}
        result = text_transform.to_yaml(data)
        self.assertIn("name: test", result)
        self.assertIn("value: 42", result)


class TestTOMLOperations(unittest.TestCase):
    """Test TOML parsing and conversion."""

    def setUp(self):
        """Skip tests if toml not available."""
        if text_transform.toml is None:
            self.skipTest("toml not installed")

    def test_parse_toml_valid(self):
        """Test parsing valid TOML."""
        data = 'name = "test"\nvalue = 42'
        result = text_transform.parse_toml(data)
        self.assertEqual(result, {"name": "test", "value": 42})

    def test_parse_toml_nested(self):
        """Test parsing nested TOML."""
        data = '[section]\nkey = "value"'
        result = text_transform.parse_toml(data)
        self.assertEqual(result, {"section": {"key": "value"}})

    def test_to_toml(self):
        """Test converting to TOML."""
        data = {"name": "test", "value": 42}
        result = text_transform.to_toml(data)
        self.assertIn('name = "test"', result)
        self.assertIn("value = 42", result)


class TestCSVOperations(unittest.TestCase):
    """Test CSV parsing and conversion."""

    def test_parse_csv_with_headers(self):
        """Test parsing CSV with headers."""
        data = "name,age\nAlice,30\nBob,25"
        result = text_transform.parse_csv(data, has_headers=True)
        self.assertEqual(result, [
            {"name": "Alice", "age": "30"},
            {"name": "Bob", "age": "25"}
        ])

    def test_parse_csv_without_headers(self):
        """Test parsing CSV without headers."""
        data = "Alice,30\nBob,25"
        result = text_transform.parse_csv(data, has_headers=False)
        self.assertEqual(result, [
            {"row": ["Alice", "30"]},
            {"row": ["Bob", "25"]}
        ])

    def test_parse_csv_empty(self):
        """Test parsing empty CSV."""
        result = text_transform.parse_csv("", has_headers=True)
        self.assertEqual(result, [])

    def test_to_csv_with_headers(self):
        """Test converting to CSV with headers."""
        data = [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25}
        ]
        result = text_transform.to_csv(data, include_headers=True)
        lines = result.strip().split('\n')
        self.assertEqual(len(lines), 3)
        self.assertIn("name", lines[0])
        self.assertIn("age", lines[0])

    def test_to_csv_without_headers(self):
        """Test converting to CSV without headers."""
        data = [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25}
        ]
        result = text_transform.to_csv(data, include_headers=False)
        lines = result.strip().split('\n')
        self.assertEqual(len(lines), 2)


class TestBase64Operations(unittest.TestCase):
    """Test base64 encoding and decoding."""

    def test_base64_encode_decode_roundtrip(self):
        """Test base64 encode/decode roundtrip."""
        original = b"Hello, World!"
        encoded = base64.b64encode(original).decode('ascii')
        decoded = base64.b64decode(encoded)
        self.assertEqual(original, decoded)

    def test_base64_encode_binary(self):
        """Test base64 encoding binary data."""
        data = bytes(range(256))
        encoded = base64.b64encode(data).decode('ascii')
        decoded = base64.b64decode(encoded)
        self.assertEqual(data, decoded)


class TestURLEncoding(unittest.TestCase):
    """Test URL encoding and decoding."""

    def test_url_encode(self):
        """Test URL encoding."""
        import urllib.parse
        result = urllib.parse.quote("hello world", safe='')
        self.assertEqual(result, "hello%20world")

    def test_url_encode_special_chars(self):
        """Test URL encoding special characters."""
        import urllib.parse
        result = urllib.parse.quote("a=1&b=2", safe='')
        self.assertEqual(result, "a%3D1%26b%3D2")

    def test_url_decode(self):
        """Test URL decoding."""
        import urllib.parse
        result = urllib.parse.unquote("hello%20world")
        self.assertEqual(result, "hello world")


class TestJMESPath(unittest.TestCase):
    """Test JMESPath queries."""

    def setUp(self):
        """Skip tests if jmespath not available."""
        if text_transform.jmespath is None:
            self.skipTest("jmespath not installed")

    def test_simple_query(self):
        """Test simple JMESPath query."""
        data = {"name": "test", "value": 42}
        result = text_transform.jmespath.search("name", data)
        self.assertEqual(result, "test")

    def test_nested_query(self):
        """Test nested JMESPath query."""
        data = {"data": {"users": [{"name": "Alice"}, {"name": "Bob"}]}}
        result = text_transform.jmespath.search("data.users[*].name", data)
        self.assertEqual(result, ["Alice", "Bob"])

    def test_filter_query(self):
        """Test filter JMESPath query."""
        data = {"users": [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 17}
        ]}
        result = text_transform.jmespath.search("users[?age > `18`].name", data)
        self.assertEqual(result, ["Alice"])


class TestTemplateRendering(unittest.TestCase):
    """Test Jinja2 template rendering."""

    def setUp(self):
        """Skip tests if jinja2 not available."""
        if text_transform.jinja2 is None:
            self.skipTest("jinja2 not installed")

    def test_simple_template(self):
        """Test simple template rendering."""
        env = text_transform.jinja2.Environment()
        template = env.from_string("Hello, {{ name }}!")
        result = template.render(name="World")
        self.assertEqual(result, "Hello, World!")

    def test_template_with_loop(self):
        """Test template with loop."""
        env = text_transform.jinja2.Environment()
        template = env.from_string("{% for item in items %}{{ item }}{% endfor %}")
        result = template.render(items=["a", "b", "c"])
        self.assertEqual(result, "abc")

    def test_template_with_condition(self):
        """Test template with condition."""
        env = text_transform.jinja2.Environment()
        template = env.from_string("{% if show %}visible{% endif %}")
        result = template.render(show=True)
        self.assertEqual(result, "visible")


class TestInputOutput(unittest.TestCase):
    """Test input/output operations."""

    def test_read_input_from_file(self):
        """Test reading input from file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("test content")
            f.flush()
            try:
                result = text_transform.read_input(f.name)
                self.assertEqual(result, "test content")
            finally:
                os.unlink(f.name)

    def test_read_input_file_not_found(self):
        """Test reading non-existent file."""
        with self.assertRaises(SystemExit) as cm:
            text_transform.read_input("/nonexistent/file.txt")
        self.assertEqual(cm.exception.code, text_transform.EXIT_PARSE_ERROR)

    def test_write_output_to_file(self):
        """Test writing output to file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            temp_path = f.name

        try:
            text_transform.write_output("test content", temp_path)
            with open(temp_path, 'r') as f:
                result = f.read()
            self.assertEqual(result, "test content")
        finally:
            os.unlink(temp_path)


class TestExitCodes(unittest.TestCase):
    """Test exit codes."""

    def test_exit_code_constants(self):
        """Test exit code constants are defined correctly."""
        self.assertEqual(text_transform.EXIT_SUCCESS, 0)
        self.assertEqual(text_transform.EXIT_PARSE_ERROR, 1)
        self.assertEqual(text_transform.EXIT_TRANSFORM_ERROR, 2)
        self.assertEqual(text_transform.EXIT_INVALID_ARGS, 3)


class TestCLIIntegration(unittest.TestCase):
    """Integration tests running the actual CLI."""

    def setUp(self):
        """Set up test fixtures."""
        self.script_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'text_transform.py'
        )

    def run_cli(self, args, stdin_data=None):
        """Run the CLI with given arguments and stdin data."""
        cmd = [sys.executable, self.script_path] + args
        result = subprocess.run(
            cmd,
            input=stdin_data,
            capture_output=True,
            text=True
        )
        return result

    def test_no_command_shows_help(self):
        """Test running without command shows help."""
        result = self.run_cli([])
        self.assertIn('usage:', result.stdout.lower() + result.stderr.lower())

    def test_help_flag(self):
        """Test --help flag."""
        result = self.run_cli(['--help'])
        self.assertIn('text_transform', result.stdout)
        self.assertEqual(result.returncode, 0)

    def test_json_to_yaml_integration(self):
        """Test json-to-yaml command."""
        if text_transform.yaml is None:
            self.skipTest("pyyaml not installed")

        result = self.run_cli(['json-to-yaml'], '{"name": "test"}')
        self.assertEqual(result.returncode, 0)
        self.assertIn('name: test', result.stdout)

    def test_yaml_to_json_integration(self):
        """Test yaml-to-json command."""
        if text_transform.yaml is None:
            self.skipTest("pyyaml not installed")

        result = self.run_cli(['yaml-to-json'], 'name: test')
        self.assertEqual(result.returncode, 0)
        output = json.loads(result.stdout)
        self.assertEqual(output, {"name": "test"})

    def test_csv_to_json_integration(self):
        """Test csv-to-json command."""
        result = self.run_cli(['csv-to-json', '--headers'], 'name,age\nAlice,30')
        self.assertEqual(result.returncode, 0)
        output = json.loads(result.stdout)
        self.assertEqual(output, [{"name": "Alice", "age": "30"}])

    def test_json_to_csv_integration(self):
        """Test json-to-csv command."""
        result = self.run_cli(['json-to-csv'], '[{"name": "Alice", "age": 30}]')
        self.assertEqual(result.returncode, 0)
        self.assertIn('Alice', result.stdout)
        self.assertIn('30', result.stdout)

    def test_jq_integration(self):
        """Test jq command."""
        if text_transform.jmespath is None:
            self.skipTest("jmespath not installed")

        result = self.run_cli(['jq', 'name'], '{"name": "test"}')
        self.assertEqual(result.returncode, 0)
        self.assertIn('test', result.stdout)

    def test_jq_array_query(self):
        """Test jq command with array query."""
        if text_transform.jmespath is None:
            self.skipTest("jmespath not installed")

        result = self.run_cli(['jq', 'data[*].name'], '{"data": [{"name": "a"}, {"name": "b"}]}')
        self.assertEqual(result.returncode, 0)
        output = json.loads(result.stdout)
        self.assertEqual(output, ["a", "b"])

    def test_base64_encode_integration(self):
        """Test base64-encode command."""
        result = self.run_cli(['base64-encode'], 'Hello, World!')
        self.assertEqual(result.returncode, 0)
        self.assertIn('SGVsbG8sIFdvcmxkIQ==', result.stdout)

    def test_base64_decode_integration(self):
        """Test base64-decode command."""
        result = subprocess.run(
            [sys.executable, self.script_path, 'base64-decode'],
            input=b'SGVsbG8sIFdvcmxkIQ==',
            capture_output=True
        )
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, b'Hello, World!')

    def test_url_encode_integration(self):
        """Test url-encode command."""
        result = self.run_cli(['url-encode', 'hello world'])
        self.assertEqual(result.returncode, 0)
        self.assertIn('hello%20world', result.stdout)

    def test_url_decode_integration(self):
        """Test url-decode command."""
        result = self.run_cli(['url-decode', 'hello%20world'])
        self.assertEqual(result.returncode, 0)
        self.assertIn('hello world', result.stdout)

    def test_template_integration(self):
        """Test template command."""
        if text_transform.jinja2 is None:
            self.skipTest("jinja2 not installed")

        with tempfile.NamedTemporaryFile(mode='w', suffix='.j2', delete=False) as f:
            f.write("Hello, {{ name }}!")
            f.flush()
            try:
                result = self.run_cli(['template', f.name, '--vars', '{"name": "World"}'])
                self.assertEqual(result.returncode, 0)
                self.assertEqual(result.stdout, "Hello, World!")
            finally:
                os.unlink(f.name)

    def test_template_with_vars_file(self):
        """Test template command with vars file."""
        if text_transform.jinja2 is None:
            self.skipTest("jinja2 not installed")

        with tempfile.NamedTemporaryFile(mode='w', suffix='.j2', delete=False) as tf:
            tf.write("Hello, {{ name }}!")
            tf.flush()
            template_path = tf.name

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as vf:
            vf.write('{"name": "Universe"}')
            vf.flush()
            vars_path = vf.name

        try:
            result = self.run_cli(['template', template_path, '--vars-file', vars_path])
            self.assertEqual(result.returncode, 0)
            self.assertEqual(result.stdout, "Hello, Universe!")
        finally:
            os.unlink(template_path)
            os.unlink(vars_path)

    def test_invalid_json_error(self):
        """Test invalid JSON returns error code 1."""
        result = self.run_cli(['json-to-yaml'], '{invalid}')
        self.assertEqual(result.returncode, text_transform.EXIT_PARSE_ERROR)

    def test_invalid_jmespath_query(self):
        """Test invalid JMESPath query returns error code 2."""
        if text_transform.jmespath is None:
            self.skipTest("jmespath not installed")

        result = self.run_cli(['jq', '[[invalid'], '{"data": 1}')
        self.assertEqual(result.returncode, text_transform.EXIT_TRANSFORM_ERROR)

    def test_file_input_output(self):
        """Test using -i and -o flags."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as inf:
            inf.write('{"test": "value"}')
            inf.flush()
            input_path = inf.name

        output_path = tempfile.mktemp(suffix='.yaml')

        try:
            if text_transform.yaml is None:
                self.skipTest("pyyaml not installed")

            result = self.run_cli(['json-to-yaml', '-i', input_path, '-o', output_path])
            self.assertEqual(result.returncode, 0)

            with open(output_path, 'r') as f:
                content = f.read()
            self.assertIn('test: value', content)
        finally:
            os.unlink(input_path)
            if os.path.exists(output_path):
                os.unlink(output_path)


class TestFormatConversions(unittest.TestCase):
    """Test format conversion roundtrips."""

    def setUp(self):
        """Set up test data."""
        self.test_data = {
            "string": "hello",
            "number": 42,
            "float": 3.14,
            "boolean": True,
            "list": [1, 2, 3],
            "nested": {"key": "value"}
        }

    def test_json_yaml_roundtrip(self):
        """Test JSON -> YAML -> JSON roundtrip."""
        if text_transform.yaml is None:
            self.skipTest("pyyaml not installed")

        json_str = text_transform.to_json(self.test_data)
        parsed_json = text_transform.parse_json(json_str)
        yaml_str = text_transform.to_yaml(parsed_json)
        parsed_yaml = text_transform.parse_yaml(yaml_str)
        final_json = text_transform.to_json(parsed_yaml)
        final_data = text_transform.parse_json(final_json)

        self.assertEqual(self.test_data, final_data)

    def test_json_toml_roundtrip(self):
        """Test JSON -> TOML -> JSON roundtrip."""
        if text_transform.toml is None:
            self.skipTest("toml not installed")

        json_str = text_transform.to_json(self.test_data)
        parsed_json = text_transform.parse_json(json_str)
        toml_str = text_transform.to_toml(parsed_json)
        parsed_toml = text_transform.parse_toml(toml_str)
        final_json = text_transform.to_json(parsed_toml)
        final_data = text_transform.parse_json(final_json)

        self.assertEqual(self.test_data, final_data)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def test_empty_json(self):
        """Test handling empty JSON object."""
        result = text_transform.parse_json('{}')
        self.assertEqual(result, {})

    def test_empty_array(self):
        """Test handling empty JSON array."""
        result = text_transform.parse_json('[]')
        self.assertEqual(result, [])

    def test_unicode_handling(self):
        """Test Unicode character handling."""
        data = {"emoji": "Hello!", "chinese": "Test", "arabic": "Test"}
        json_str = text_transform.to_json(data)
        parsed = text_transform.parse_json(json_str)
        self.assertEqual(data, parsed)

    def test_null_json(self):
        """Test handling null JSON."""
        result = text_transform.parse_json('null')
        self.assertIsNone(result)

    def test_json_special_chars(self):
        """Test JSON with special characters."""
        data = {"text": "line1\nline2\ttabbed\"quoted\""}
        json_str = text_transform.to_json(data)
        parsed = text_transform.parse_json(json_str)
        self.assertEqual(data, parsed)


if __name__ == '__main__':
    unittest.main(verbosity=2)
