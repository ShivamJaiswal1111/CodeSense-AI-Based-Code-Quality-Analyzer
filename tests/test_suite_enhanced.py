"""
CodeSense - Comprehensive Test Suite
80%+ coverage across all major modules.
Run with: python -m pytest tests/ -v --tb=short
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────────────

PYTHON_GOOD = """
def binary_search(arr: list, target: int) -> int:
    \"\"\"Search for target in sorted array using binary search.\"\"\"
    low, high = 0, len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1


def bubble_sort(arr: list) -> list:
    \"\"\"Sort array using bubble sort.\"\"\"
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr
"""

PYTHON_BAD = """
import os
import hashlib

def badFunc(x,y,z,a,b,c,d):
    eval(x)
    password = "hardcoded123"
    h = hashlib.md5(b"data").hexdigest()
    for i in range(100):
        for j in range(100):
            for k in range(100):
                if i+j+k > 200:
                    print(i,j,k)
"""

PYTHON_SYNTAX_ERROR = """
def broken(
    x = [1,2,3
    return x
"""

JAVA_CODE = """
public class Solution {
    public int binarySearch(int[] arr, int target) {
        int low = 0, high = arr.length - 1;
        while (low <= high) {
            int mid = low + (high - low) / 2;
            if (arr[mid] == target) return mid;
            else if (arr[mid] < target) low = mid + 1;
            else high = mid - 1;
        }
        return -1;
    }
}
"""

CPP_CODE = """
#include <iostream>
#include <vector>
using namespace std;

int binarySearch(vector<int>& arr, int target) {
    int low = 0, high = arr.size() - 1;
    while (low <= high) {
        int mid = low + (high - low) / 2;
        if (arr[mid] == target) return mid;
        else if (arr[mid] < target) low = mid + 1;
        else high = mid - 1;
    }
    return -1;
}
"""


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATORS
# ─────────────────────────────────────────────────────────────────────────────

class TestValidators:
    def test_validate_code_empty(self):
        from validators import validate_code
        ok, msg = validate_code("")
        assert not ok
        assert "empty" in msg.lower()

    def test_validate_code_too_short(self):
        from validators import validate_code
        ok, msg = validate_code("x = 1")
        assert not ok

    def test_validate_code_valid(self):
        from validators import validate_code
        ok, msg = validate_code(PYTHON_GOOD)
        assert ok
        assert msg == ""

    def test_validate_email_valid(self):
        from validators import validate_email
        ok, _ = validate_email("user@example.com")
        assert ok

    def test_validate_email_invalid(self):
        from validators import validate_email
        ok, msg = validate_email("not-an-email")
        assert not ok

    def test_validate_password_weak(self):
        from validators import validate_password
        ok, msg = validate_password("short")
        assert not ok

    def test_validate_password_strong(self):
        from validators import validate_password
        ok, msg = validate_password("Strong@123!")
        assert ok

    def test_validate_language_valid(self):
        from validators import validate_language
        ok, _ = validate_language("python")
        assert ok

    def test_validate_language_invalid(self):
        from validators import validate_language
        ok, msg = validate_language("ruby")
        assert not ok

    def test_sanitize_code(self):
        from validators import sanitize_code
        code = "x = 1\r\n\x00y = 2\r"
        result = sanitize_code(code)
        assert "\r" not in result
        assert "\x00" not in result


# ─────────────────────────────────────────────────────────────────────────────
# SYNTAX ANALYZER
# ─────────────────────────────────────────────────────────────────────────────

class TestSyntaxAnalyzer:
    def setup_method(self):
        from syntax_analyzer import SyntaxAnalyzer
        self.analyzer = SyntaxAnalyzer()

    def test_python_valid_code(self):
        result = self.analyzer.analyze(PYTHON_GOOD, "python")
        assert not result["has_errors"]
        assert result["is_valid"]

    def test_python_syntax_error(self):
        result = self.analyzer.analyze(PYTHON_SYNTAX_ERROR, "python")
        assert result["has_errors"]
        assert len(result["issues"]) > 0

    def test_java_valid_code(self):
        result = self.analyzer.analyze(JAVA_CODE, "java")
        assert isinstance(result, dict)
        assert "issues" in result

    def test_cpp_valid_code(self):
        result = self.analyzer.analyze(CPP_CODE, "cpp")
        assert isinstance(result, dict)

    def test_unsupported_language(self):
        result = self.analyzer.analyze("x = 1", "ruby")
        assert not result["has_errors"]

    def test_syntax_error_has_suggestion(self):
        result = self.analyzer.analyze(PYTHON_SYNTAX_ERROR, "python")
        for issue in result["issues"]:
            assert "suggestion" in issue
            assert len(issue["suggestion"]) > 0


# ─────────────────────────────────────────────────────────────────────────────
# SEMANTIC ANALYZER
# ─────────────────────────────────────────────────────────────────────────────

class TestSemanticAnalyzer:
    def setup_method(self):
        from semantic_analyzer import SemanticAnalyzer
        self.analyzer = SemanticAnalyzer()

    def test_detects_unused_variable(self):
        code = "def fn():\n    unused = 42\n    return 1\n"
        result = self.analyzer.analyze(code, "python")
        types = [i["type"] for i in result["issues"]]
        assert "UnusedVariable" in types

    def test_detects_mutable_default(self):
        code = "def fn(items=[]):\n    items.append(1)\n    return items\n"
        result = self.analyzer.analyze(code, "python")
        types = [i["type"] for i in result["issues"]]
        assert "MutableDefault" in types

    def test_detects_bare_except(self):
        code = "try:\n    x = 1\nexcept:\n    pass\n"
        result = self.analyzer.analyze(code, "python")
        types = [i["type"] for i in result["issues"]]
        assert "BareExcept" in types

    def test_detects_unused_import(self):
        code = "import os\nimport sys\n\ndef fn():\n    return sys.argv\n"
        result = self.analyzer.analyze(code, "python")
        types = [i["type"] for i in result["issues"]]
        assert "UnusedImport" in types

    def test_clean_code_has_no_issues(self):
        code = "def add(a: int, b: int) -> int:\n    return a + b\n"
        result = self.analyzer.analyze(code, "python")
        errors = [i for i in result["issues"] if i["severity"] == "ERROR"]
        assert len(errors) == 0

    def test_summary_has_counts(self):
        result = self.analyzer.analyze(PYTHON_GOOD, "python")
        assert "summary" in result
        assert "total" in result["summary"]


# ─────────────────────────────────────────────────────────────────────────────
# STATIC ANALYZER
# ─────────────────────────────────────────────────────────────────────────────

class TestStaticAnalyzer:
    def setup_method(self):
        from analyzer import StaticAnalyzer
        self.analyzer = StaticAnalyzer()

    def test_python_basic_metrics(self):
        result = self.analyzer.analyze(PYTHON_GOOD, "python")
        m = result["metrics"]
        assert m["total_lines"] > 0
        assert m["comment_lines"] >= 0

    def test_security_detects_eval(self):
        result = self.analyzer.analyze(PYTHON_BAD, "python")
        types  = [f["type"] for f in result["security"]["findings"]]
        assert "eval_usage" in types

    def test_security_detects_hardcoded_password(self):
        result = self.analyzer.analyze(PYTHON_BAD, "python")
        types  = [f["type"] for f in result["security"]["findings"]]
        assert "hardcoded_password" in types

    def test_security_detects_weak_hash(self):
        result = self.analyzer.analyze(PYTHON_BAD, "python")
        types  = [f["type"] for f in result["security"]["findings"]]
        assert "weak_hash" in types

    def test_complexity_python(self):
        result = self.analyzer.analyze(PYTHON_GOOD, "python")
        cx = result["complexity"]
        assert "avg_complexity" in cx
        assert cx["avg_complexity"] >= 1

    def test_java_analysis(self):
        result = self.analyzer.analyze(JAVA_CODE, "java")
        assert "metrics" in result
        assert "security" in result

    def test_overall_health(self):
        result = self.analyzer.analyze(PYTHON_GOOD, "python")
        assert "overall_health" in result
        health = result["overall_health"]
        assert "strengths" in health
        assert "issues" in health


# ─────────────────────────────────────────────────────────────────────────────
# DSA DETECTOR
# ─────────────────────────────────────────────────────────────────────────────

class TestDSADetector:
    def setup_method(self):
        from dsa_detector import DSADetector
        self.detector = DSADetector()

    def test_detects_binary_search(self):
        result = self.detector.detect(PYTHON_GOOD, "python")
        names  = [a["name"] for a in result["algorithms"]]
        assert "binary_search" in names

    def test_detects_bubble_sort(self):
        result = self.detector.detect(PYTHON_GOOD, "python")
        names  = [a["name"] for a in result["algorithms"]]
        assert "bubble_sort" in names

    def test_algo_has_complexity(self):
        result = self.detector.detect(PYTHON_GOOD, "python")
        for algo in result["algorithms"]:
            assert "complexity" in algo
            assert "avg" in algo["complexity"]

    def test_algo_has_reason(self):
        result = self.detector.detect(PYTHON_GOOD, "python")
        for algo in result["algorithms"]:
            assert len(algo["reason"]) > 0

    def test_detects_data_structures(self):
        code = "from collections import deque\nqueue = deque()\nqueue.append(1)\nqueue.popleft()\n"
        result = self.detector.detect(code, "python")
        names  = [d["name"] for d in result["data_structures"]]
        assert "deque" in names or "queue" in names

    def test_summary_fields(self):
        result = self.detector.detect(PYTHON_GOOD, "python")
        s = result["summary"]
        assert "algorithm_count" in s
        assert "data_structure_count" in s
        assert "complexity_score" in s

    def test_detects_dfs(self):
        dfs_code = """
def dfs(graph, node, visited=None):
    if visited is None:
        visited = set()
    visited.add(node)
    for neighbor in graph[node]:
        if neighbor not in visited:
            dfs(graph, neighbor, visited)
    return visited
"""
        result = self.detector.detect(dfs_code, "python")
        names  = [a["name"] for a in result["algorithms"]]
        assert "dfs" in names

    def test_detects_merge_sort(self):
        merge_code = """
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)
"""
        result = self.detector.detect(merge_code, "python")
        names  = [a["name"] for a in result["algorithms"]]
        assert "merge_sort" in names


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTOR
# ─────────────────────────────────────────────────────────────────────────────

class TestFeatureExtractor:
    def setup_method(self):
        from features import FeatureExtractor
        from analyzer import StaticAnalyzer
        from dsa_detector import DSADetector
        self.extractor = FeatureExtractor()
        self.static    = StaticAnalyzer()
        self.dsa       = DSADetector()

    def _extract(self, code, lang="python"):
        analysis = self.static.analyze(code, lang)
        dsa      = self.dsa.detect(code, lang)
        return self.extractor.extract(code, lang, analysis, dsa)

    def test_returns_correct_count(self):
        from constants import NUM_FEATURES
        feats = self._extract(PYTHON_GOOD)
        assert len(feats) == NUM_FEATURES

    def test_all_numeric(self):
        feats = self._extract(PYTHON_GOOD)
        for k, v in feats.items():
            assert isinstance(v, (int, float)), f"{k} is not numeric"

    def test_no_nan(self):
        feats = self._extract(PYTHON_GOOD)
        for k, v in feats.items():
            assert not (v != v), f"{k} is NaN"

    def test_to_array_shape(self):
        from constants import NUM_FEATURES
        feats = self._extract(PYTHON_GOOD)
        arr   = self.extractor.to_array(feats)
        assert arr.shape == (NUM_FEATURES,)

    def test_duplication_score_clean(self):
        feats = self._extract(PYTHON_GOOD)
        assert feats["code_duplication_score"] < 0.5

    def test_security_features(self):
        feats = self._extract(PYTHON_BAD)
        assert feats["security_issue_count"] > 0
        assert feats["critical_security_count"] > 0


# ─────────────────────────────────────────────────────────────────────────────
# ML MODEL
# ─────────────────────────────────────────────────────────────────────────────

class TestMLModel:
    def test_generate_data_shape(self):
        from train_model import generate_training_data
        X, y = generate_training_data(n=100)
        from constants import NUM_FEATURES
        assert X.shape == (100, NUM_FEATURES)
        assert y.shape == (100,)

    def test_scores_in_range(self):
        from train_model import generate_training_data
        _, y = generate_training_data(n=500)
        assert y.min() >= 0
        assert y.max() <= 100

    def test_score_to_grade(self):
        from train_model import score_to_grade
        assert score_to_grade(95) == "A+"
        assert score_to_grade(85) == "A-"
        assert score_to_grade(75) == "B"
        assert score_to_grade(50) == "D"
        assert score_to_grade(30) == "F"

    def test_score_to_label(self):
        from train_model import score_to_label
        assert score_to_label(95) == "Excellent"
        assert score_to_label(80) == "Good"
        assert score_to_label(65) == "Average"

    def test_contextual_adjustments_bounded(self):
        from train_model import calculate_contextual_adjustments
        from constants import MAX_SCORE_ADJUSTMENT
        analysis = {"security": {"counts": {"CRITICAL": 5}, "total_issues": 10},
                    "documentation": {"docstring_ratio": 0.8}}
        dsa = {"summary": {"complexity_score": 90}}
        adj = calculate_contextual_adjustments(analysis, dsa, "python")
        assert abs(adj) <= MAX_SCORE_ADJUSTMENT


# ─────────────────────────────────────────────────────────────────────────────
# CONTEXT ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class TestContextEngine:
    def setup_method(self):
        from context_engine import ContextEngine
        self.engine = ContextEngine()

    def test_detects_test_file_by_content(self):
        code = "import pytest\n\ndef test_add():\n    assert 1 + 1 == 2\n"
        result = self.engine.analyze(code, "python")
        assert result["file_type"] == "test"

    def test_detects_web_domain(self):
        code = "from flask import Flask, request\napp = Flask(__name__)\n@app.route('/')\ndef index(): return 'hi'\n"
        result = self.engine.analyze(code, "python")
        assert result["domain"] == "web"

    def test_detects_algorithm_domain(self):
        result = self.engine.analyze(PYTHON_GOOD, "python")
        assert result["domain"] in ("algorithms", "general")

    def test_relaxed_rules_for_test_files(self):
        code = "import pytest\ndef test_eval():\n    eval('1+1')\n"
        result = self.engine.analyze(code, "python", "test_something.py")
        assert "security_checks" in result["relaxed_rules"]

    def test_returns_all_fields(self):
        result = self.engine.analyze(PYTHON_GOOD, "python")
        for field in ("file_type", "domain", "relaxed_rules", "strict_rules",
                      "intent_notes", "adjustment_hints"):
            assert field in result


# ─────────────────────────────────────────────────────────────────────────────
# CODE FIXER
# ─────────────────────────────────────────────────────────────────────────────

class TestCodeFixer:
    def setup_method(self):
        from code_fixer import CodeFixer
        self.fixer = CodeFixer()

    def test_fix_weak_hash(self):
        code = "import hashlib\nh = hashlib.md5(b'data').hexdigest()\n"
        issues = [{"type": "weak_hash", "line": 2}]
        fixes  = self.fixer.suggest_fixes(code, "python", [], issues)
        assert any("sha256" in (f.get("after") or "") for f in fixes)

    def test_fix_yaml_load(self):
        code = "import yaml\ndata = yaml.load(f)\n"
        issues = [{"type": "yaml_unsafe_load", "line": 2}]
        fixes  = self.fixer.suggest_fixes(code, "python", [], issues)
        assert any("safe_load" in (f.get("after") or "") for f in fixes)

    def test_fix_bare_except(self):
        code = "try:\n    x = 1\nexcept:\n    pass\n"
        issues = [{"type": "BareExcept", "line": 3}]
        fixes  = self.fixer.suggest_fixes(code, "python", issues, [])
        assert len(fixes) > 0

    def test_no_fixes_for_clean_code(self):
        code = "def add(a, b):\n    return a + b\n"
        fixes = self.fixer.suggest_fixes(code, "python", [], [])
        assert len(fixes) == 0


# ─────────────────────────────────────────────────────────────────────────────
# CACHE
# ─────────────────────────────────────────────────────────────────────────────

class TestCache:
    def setup_method(self):
        from cache import LRUCache
        self.cache = LRUCache(max_size=5, ttl=60)

    def test_set_and_get(self):
        self.cache.set("key1", {"value": 42})
        assert self.cache.get("key1") == {"value": 42}

    def test_miss_returns_none(self):
        assert self.cache.get("nonexistent") is None

    def test_lru_eviction(self):
        for i in range(6):
            self.cache.set(f"k{i}", i)
        # First key should be evicted
        assert self.cache.get("k0") is None
        assert self.cache.get("k5") == 5

    def test_stats(self):
        self.cache.set("x", 1)
        self.cache.get("x")
        self.cache.get("y")  # miss
        stats = self.cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1


# ─────────────────────────────────────────────────────────────────────────────
# UTILS
# ─────────────────────────────────────────────────────────────────────────────

class TestUtils:
    def test_detect_python(self):
        from utils import detect_language_from_code
        code = "def hello():\n    print('hi')\n"
        assert detect_language_from_code(code) == "python"

    def test_detect_java(self):
        from utils import detect_language_from_code
        assert detect_language_from_code(JAVA_CODE) == "java"

    def test_detect_cpp(self):
        from utils import detect_language_from_code
        assert detect_language_from_code(CPP_CODE) == "cpp"

    def test_code_hash_consistent(self):
        from utils import code_hash
        h1 = code_hash("hello")
        h2 = code_hash("hello")
        assert h1 == h2

    def test_timer(self):
        from utils import Timer
        import time
        with Timer() as t:
            time.sleep(0.01)
        assert t.elapsed_ms >= 10

    def test_results_to_markdown(self):
        from utils import results_to_markdown
        result = {
            "score": 85.0, "grade": "A-", "language": "python",
            "feedback": {
                "opening": "Great job!",
                "strengths": ["Clean code"],
                "items": [],
                "next_steps": [],
                "learning_path": [],
            }
        }
        md = results_to_markdown(result)
        assert "85" in md
        assert "A-" in md
        assert "Great job!" in md

    def test_truncate(self):
        from utils import truncate
        assert truncate("hello world", 5) == "he..."
        assert truncate("hi", 10) == "hi"

    def test_format_duration(self):
        from utils import format_duration
        assert "ms" in format_duration(500)
        assert "s" in format_duration(2000)


# ─────────────────────────────────────────────────────────────────────────────
# INTEGRATION TEST
# ─────────────────────────────────────────────────────────────────────────────

class TestIntegration:
    """End-to-end pipeline test without DB."""

    def test_full_pipeline_python(self):
        from analyzer import StaticAnalyzer
        from dsa_detector import DSADetector
        from features import FeatureExtractor
        from syntax_analyzer import SyntaxAnalyzer
        from semantic_analyzer import SemanticAnalyzer
        from context_engine import ContextEngine
        from train_model import generate_training_data
        from constants import NUM_FEATURES
        import numpy as np

        code = PYTHON_GOOD
        lang = "python"

        syntax   = SyntaxAnalyzer().analyze(code, lang)
        semantic = SemanticAnalyzer().analyze(code, lang)
        analysis = StaticAnalyzer().analyze(code, lang)
        dsa      = DSADetector().detect(code, lang)
        context  = ContextEngine().analyze(code, lang)
        feats    = FeatureExtractor().extract(code, lang, analysis, dsa)
        arr      = FeatureExtractor().to_array(feats)

        assert not syntax["has_errors"]
        assert len(feats) == NUM_FEATURES
        assert arr.shape == (NUM_FEATURES,)
        assert dsa["summary"]["algorithm_count"] >= 1

    def test_full_pipeline_java(self):
        from analyzer import StaticAnalyzer
        from dsa_detector import DSADetector
        from syntax_analyzer import SyntaxAnalyzer

        syntax   = SyntaxAnalyzer().analyze(JAVA_CODE, "java")
        analysis = StaticAnalyzer().analyze(JAVA_CODE, "java")
        dsa      = DSADetector().detect(JAVA_CODE, "java")

        assert "metrics" in analysis
        assert "security" in analysis
        assert "algorithms" in dsa


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])