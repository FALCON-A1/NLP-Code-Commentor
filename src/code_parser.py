"""
Code Parser Module
==================
Uses Python's AST (Abstract Syntax Tree) module to parse source code
and extract functions, classes, and their associated docstrings.
"""

import ast
import re
import logging
from typing import List, Dict, Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CodeParser:
    """
    Parses Python source code using AST to extract functions/classes
    along with their docstrings for training data generation.
    """

    def __init__(self):
        self.supported_languages = ["python"]

    def parse_python_code(self, source_code: str) -> List[Dict]:
        """
        Parse Python source code and extract function/class definitions
        with their docstrings.

        Args:
            source_code (str): Raw Python source code string.

        Returns:
            List[Dict]: List of extracted code-docstring pairs.
        """
        extracted = []

        try:
            tree = ast.parse(source_code)
        except SyntaxError as e:
            logger.warning(f"SyntaxError while parsing code: {e}")
            return extracted

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                pair = self._extract_function(node, source_code)
                if pair:
                    extracted.append(pair)

            elif isinstance(node, ast.ClassDef):
                pair = self._extract_class(node, source_code)
                if pair:
                    extracted.append(pair)

        return extracted

    def _extract_function(self, node: ast.FunctionDef, source: str) -> Optional[Dict]:
        """
        Extract a function definition and its docstring.

        Args:
            node: AST FunctionDef node.
            source: Original source code.

        Returns:
            Dict with code body and docstring, or None if no docstring.
        """
        docstring = ast.get_docstring(node)
        if not docstring:
            return None

        # Get function code without the docstring
        func_code = self._get_function_code(node, source)

        return {
            "type": "function",
            "name": node.name,
            "code": func_code,
            "docstring": docstring.strip(),
            "args": [arg.arg for arg in node.args.args],
            "line_number": node.lineno,
            "decorators": [self._get_decorator_name(d) for d in node.decorator_list],
        }

    def _extract_class(self, node: ast.ClassDef, source: str) -> Optional[Dict]:
        """
        Extract a class definition and its docstring.

        Args:
            node: AST ClassDef node.
            source: Original source code.

        Returns:
            Dict with class code and docstring, or None if no docstring.
        """
        docstring = ast.get_docstring(node)
        if not docstring:
            return None

        return {
            "type": "class",
            "name": node.name,
            "code": ast.get_source_segment(source, node) or "",
            "docstring": docstring.strip(),
            "methods": [
                m.name for m in node.body
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
            ],
            "line_number": node.lineno,
            "bases": [self._get_base_name(b) for b in node.bases],
        }

    def _get_function_code(self, node: ast.FunctionDef, source: str) -> str:
        """
        Extract the function's source code, removing the docstring portion
        to create the 'input' for the model.
        """
        full_code = ast.get_source_segment(source, node)
        if full_code is None:
            return ""

        # Remove the docstring from the function body for model input
        lines = full_code.split('\n')
        cleaned_lines = []
        in_docstring = False
        docstring_removed = False

        for line in lines:
            stripped = line.strip()
            if not docstring_removed:
                if stripped.startswith('"""') or stripped.startswith("'''"):
                    if in_docstring:
                        in_docstring = False
                        docstring_removed = True
                        continue
                    elif stripped.count('"""') >= 2 or stripped.count("'''") >= 2:
                        docstring_removed = True
                        continue
                    else:
                        in_docstring = True
                        continue
                elif in_docstring:
                    continue

            cleaned_lines.append(line)

        return '\n'.join(cleaned_lines)

    @staticmethod
    def _get_decorator_name(decorator_node) -> str:
        """Extract decorator name from AST node."""
        if isinstance(decorator_node, ast.Name):
            return decorator_node.id
        elif isinstance(decorator_node, ast.Attribute):
            return f"{decorator_node.value.id}.{decorator_node.attr}" if hasattr(decorator_node.value, 'id') else decorator_node.attr
        elif isinstance(decorator_node, ast.Call):
            if isinstance(decorator_node.func, ast.Name):
                return decorator_node.func.id
            elif isinstance(decorator_node.func, ast.Attribute):
                return decorator_node.func.attr
        return "unknown"

    @staticmethod
    def _get_base_name(base_node) -> str:
        """Extract base class name from AST node."""
        if isinstance(base_node, ast.Name):
            return base_node.id
        elif isinstance(base_node, ast.Attribute):
            return f"{base_node.value.id}.{base_node.attr}" if hasattr(base_node.value, 'id') else base_node.attr
        return "unknown"


class CodeTokenizer:
    """
    Custom tokenizer for source code that handles code-specific tokens
    like operators, brackets, and identifiers.
    """

    # Common Python keywords
    PYTHON_KEYWORDS = {
        'False', 'None', 'True', 'and', 'as', 'assert', 'async', 'await',
        'break', 'class', 'continue', 'def', 'del', 'elif', 'else', 'except',
        'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is',
        'lambda', 'nonlocal', 'not', 'or', 'pass', 'raise', 'return', 'try',
        'while', 'with', 'yield'
    }

    def __init__(self):
        # Regex pattern for splitting code into tokens
        self.token_pattern = re.compile(
            r'(\w+|[^\w\s]|\s+)'
        )
        # Pattern for splitting camelCase and snake_case identifiers
        self.identifier_pattern = re.compile(
            r'[a-z]+|[A-Z][a-z]*|[0-9]+|_'
        )

    def tokenize_code(self, code: str, split_identifiers: bool = True) -> List[str]:
        """
        Tokenize source code into a list of tokens.

        Args:
            code (str): Source code string.
            split_identifiers (bool): Whether to split camelCase/snake_case
                                      identifiers into sub-tokens.

        Returns:
            List[str]: List of code tokens.
        """
        # Basic tokenization
        raw_tokens = self.token_pattern.findall(code)

        tokens = []
        for token in raw_tokens:
            token = token.strip()
            if not token:
                continue

            if split_identifiers and token not in self.PYTHON_KEYWORDS:
                # Split identifiers like 'getUserName' -> ['get', 'User', 'Name']
                sub_tokens = self._split_identifier(token)
                tokens.extend(sub_tokens)
            else:
                tokens.append(token)

        return tokens

    def _split_identifier(self, identifier: str) -> List[str]:
        """
        Split a camelCase or snake_case identifier into sub-tokens.

        Args:
            identifier (str): The identifier string.

        Returns:
            List[str]: Sub-tokens of the identifier.
        """
        if '_' in identifier:
            # Handle snake_case
            parts = [p.lower() for p in identifier.split('_') if p]
            return parts if parts else [identifier]

        # Handle camelCase / PascalCase
        sub_tokens = self.identifier_pattern.findall(identifier)
        if sub_tokens:
            return [t.lower() for t in sub_tokens if t != '_']

        return [identifier.lower()]


def extract_pairs_from_file(filepath: str) -> List[Dict]:
    """
    Convenience function to extract code-docstring pairs from a Python file.

    Args:
        filepath (str): Path to a Python source file.

    Returns:
        List[Dict]: Extracted code-documentation pairs.
    """
    parser = CodeParser()

    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            source_code = f.read()
    except (IOError, OSError) as e:
        logger.error(f"Error reading file {filepath}: {e}")
        return []

    return parser.parse_python_code(source_code)


if __name__ == "__main__":
    # Demo: Parse this file itself
    import os
    pairs = extract_pairs_from_file(os.path.abspath(__file__))
    print(f"Extracted {len(pairs)} code-docstring pairs from this file:\n")
    for pair in pairs:
        print(f"  [{pair['type']}] {pair['name']}")
        print(f"    Docstring: {pair['docstring'][:80]}...")
        print()
