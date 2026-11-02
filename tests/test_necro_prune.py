"""Comprehensive Unit and Integration Test Suite for Necro-Prune AST Engine."""

import ast
from importlib.machinery import SourceFileLoader
import os
from pathlib import Path
import tempfile
import types
import unittest

ROOT = Path(__file__).resolve().parent.parent
BIN_PATH = ROOT / "bin" / "necro-prune"

loader = SourceFileLoader("necro_prune_engine", str(BIN_PATH))
necro_mod = types.ModuleType(loader.name)
loader.exec_module(necro_mod)

DeadCodeKind = necro_mod.DeadCodeKind
DeadItem = necro_mod.DeadItem
FileAnalysisReport = necro_mod.FileAnalysisReport
CallGraph = necro_mod.CallGraph
CallGraphBuilder = necro_mod.CallGraphBuilder
DeadCodeAnalyzer = necro_mod.DeadCodeAnalyzer
NecroTransformer = necro_mod.NecroTransformer
IncineratorEngine = necro_mod.IncineratorEngine
run_verification_suite = necro_mod.run_verification_suite


class TestCallGraphAndReachability(unittest.TestCase):
    def test_transitive_reachability(self):
        graph = CallGraph()
        graph.add_node("entry", ast.Pass(), 1, 1)
        graph.add_node("a", ast.Pass(), 2, 2)
        graph.add_node("b", ast.Pass(), 3, 3)
        graph.add_node("dead", ast.Pass(), 4, 4)

        graph.add_edge("entry", "a")
        graph.add_edge("a", "b")
        graph.mark_entrypoint("entry")

        reachable = graph.compute_reachability()
        self.assertEqual(reachable, {"entry", "a", "b"})
        self.assertNotIn("dead", reachable)

    def test_cycle_handling(self):
        graph = CallGraph()
        graph.add_node("f1", ast.Pass(), 1, 1)
        graph.add_node("f2", ast.Pass(), 2, 2)
        graph.add_edge("f1", "f2")
        graph.add_edge("f2", "f1")
        graph.mark_entrypoint("f1")

        reachable = graph.compute_reachability()
        self.assertEqual(reachable, {"f1", "f2"})


class TestDeadCodeAnalyzer(unittest.TestCase):
    def test_clean_file_is_clean(self):
        code = """
import sys

def main():
    print(sys.version)

if __name__ == '__main__':
    main()
"""
        rep = DeadCodeAnalyzer(code).analyze()
        self.assertTrue(rep.is_clean)
        self.assertEqual(rep.dead_count, 0)

    def test_unused_import_detection(self):
        code = """
import os
import sys
from math import sqrt, pi

def compute():
    return sqrt(25)

if __name__ == '__main__':
    compute()
"""
        rep = DeadCodeAnalyzer(code).analyze()
        self.assertFalse(rep.is_clean)
        dead_imports = {item.name for item in rep.items if item.kind == DeadCodeKind.UNUSED_IMPORT}
        self.assertEqual(dead_imports, {"os", "sys", "pi"})

    def test_dead_function_and_class_detection(self):
        code = """
class LiveService:
    def __init__(self):
        self.active = True
    def execute(self):
        return self.active

class ForgottenService:
    def do_work(self):
        return 'dead'

def dead_helper():
    return 42

def run():
    s = LiveService()
    return s.execute()

if __name__ == '__main__':
    run()
"""
        rep = DeadCodeAnalyzer(code).analyze()
        dead_items = {item.name: item.kind for item in rep.items}
        self.assertIn("ForgottenService", dead_items)
        self.assertEqual(dead_items["ForgottenService"], DeadCodeKind.UNREACHABLE_CLASS)
        self.assertIn("dead_helper", dead_items)
        self.assertEqual(dead_items["dead_helper"], DeadCodeKind.UNREACHABLE_FUNCTION)
        self.assertNotIn("LiveService", dead_items)
        self.assertNotIn("run", dead_items)

    def test_export_all_preserves_public_api(self):
        code = """
__all__ = ['public_api_function', 'PublicClass']

def public_api_function():
    return 'accessible'

class PublicClass:
    pass

def unexported_dead_func():
    return 'dead'
"""
        rep = DeadCodeAnalyzer(code).analyze()
        dead_names = {item.name for item in rep.items}
        self.assertNotIn("public_api_function", dead_names)
        self.assertNotIn("PublicClass", dead_names)
        self.assertIn("unexported_dead_func", dead_names)

    def test_post_terminator_detection(self):
        code = """
def branch(val):
    if val > 10:
        return val
        dead_assign = 123
        print('never reached')
    return -1

if __name__ == '__main__':
    branch(20)
"""
        rep = DeadCodeAnalyzer(code).analyze()
        kinds = [item.kind for item in rep.items]
        self.assertIn(DeadCodeKind.DEAD_POST_TERMINATOR, kinds)

    def test_dead_branch_detection(self):
        code = """
def run():
    if False:
        print('unreachable branch')
    return True

if __name__ == '__main__':
    run()
"""
        rep = DeadCodeAnalyzer(code).analyze()
        dead_branches = [item for item in rep.items if item.kind == DeadCodeKind.DEAD_BRANCH]
        self.assertEqual(len(dead_branches), 1)


class TestIncineratorEngine(unittest.TestCase):
    def test_surgical_incineration_and_compilation(self):
        code = """
import math
import unused_module

def live_main():
    return math.sqrt(100)

def dead_zombie():
    return 'incinerate me'

if __name__ == '__main__':
    live_main()
"""
        rep = DeadCodeAnalyzer(code).analyze()
        new_code, diff = IncineratorEngine.transform(code, rep)
        self.assertNotIn("dead_zombie", new_code)
        self.assertNotIn("unused_module", new_code)
        self.assertIn("live_main", new_code)
        self.assertIn("math", new_code)
        self.assertIn("-", diff)  # unified diff exists

        # Verify bytecode compilation
        compiled = compile(new_code, "<test>", "exec")
        self.assertIsNotNone(compiled)

    def test_file_incineration_in_place(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "sample.py"
            original_content = """
import os
def keep_alive():
    return 1
def dead_func():
    return 2
if __name__ == '__main__':
    keep_alive()
"""
            test_file.write_text(original_content, encoding="utf-8")
            rep = DeadCodeAnalyzer(original_content, filepath=test_file).analyze()
            new_code, _ = IncineratorEngine.transform(original_content, rep)
            test_file.write_text(new_code, encoding="utf-8")

            re_inspected = DeadCodeAnalyzer(test_file.read_text(), filepath=test_file).analyze()
            self.assertTrue(re_inspected.is_clean)


class TestVerificationSuiteRunner(unittest.TestCase):
    def test_full_suite(self):
        result = run_verification_suite()
        self.assertTrue(result)


if __name__ == '__main__':
    unittest.main()
