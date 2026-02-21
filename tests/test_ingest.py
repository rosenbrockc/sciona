"""Tests for CLRS-30 and coq-100-theorems ingestion."""

from pathlib import Path
from textwrap import dedent

from ageom.architect.models import ConceptType


class TestIngestCLRS:
    """Test CLRS ingestion with mock data."""

    def _create_mock_clrs_repo(self, tmp_path: Path) -> Path:
        """Create a minimal mock of the CLRS repo structure."""
        clrs_root = tmp_path / "clrs"
        src_dir = clrs_root / "clrs" / "_src"
        algo_dir = src_dir / "algorithms"
        algo_dir.mkdir(parents=True)

        # Create a minimal specs.py with inline constants (no imports)
        specs_py = dedent('''\
            """Algorithm specs for CLRS-30."""

            class Stage:
                INPUT = "input"
                OUTPUT = "output"
                HINT = "hint"

            class Location:
                NODE = "node"
                EDGE = "edge"
                GRAPH = "graph"

            class Type:
                SCALAR = "SCALAR"
                MASK = "MASK"
                POINTER = "POINTER"
                CATEGORICAL = "CATEGORICAL"
                MASK_ONE = "MASK_ONE"
                PERMUTATION_POINTER = "PERMUTATION_POINTER"
                SHOULD_BE_PERMUTATION = "SHOULD_BE_PERMUTATION"

            SPECS = {
                "insertion_sort": {
                    "pos": (Stage.INPUT, Location.NODE, Type.SCALAR),
                    "key": (Stage.INPUT, Location.NODE, Type.SCALAR),
                    "pred": (Stage.OUTPUT, Location.NODE, Type.SHOULD_BE_PERMUTATION),
                },
                "bubble_sort": {
                    "pos": (Stage.INPUT, Location.NODE, Type.SCALAR),
                    "key": (Stage.INPUT, Location.NODE, Type.SCALAR),
                    "pred": (Stage.OUTPUT, Location.NODE, Type.SHOULD_BE_PERMUTATION),
                },
                "heapsort": {
                    "pos": (Stage.INPUT, Location.NODE, Type.SCALAR),
                    "key": (Stage.INPUT, Location.NODE, Type.SCALAR),
                    "pred": (Stage.OUTPUT, Location.NODE, Type.SHOULD_BE_PERMUTATION),
                },
                "bfs": {
                    "adj": (Stage.INPUT, Location.EDGE, Type.MASK),
                    "s": (Stage.INPUT, Location.NODE, Type.MASK_ONE),
                    "pi": (Stage.OUTPUT, Location.NODE, Type.POINTER),
                },
                "dijkstra": {
                    "adj": (Stage.INPUT, Location.EDGE, Type.SCALAR),
                    "s": (Stage.INPUT, Location.NODE, Type.MASK_ONE),
                    "d": (Stage.OUTPUT, Location.NODE, Type.SCALAR),
                },
            }
        ''')
        (src_dir / "specs.py").write_text(specs_py)

        # Create algorithm source files
        sorting_py = dedent('''\
            """Sorting algorithms."""

            def insertion_sort(A):
                """Sort array A using insertion sort."""
                pass

            def bubble_sort(A):
                """Sort array A using bubble sort."""
                pass

            def heapsort(A):
                """Sort array A using heapsort with a max-heap."""
                pass
        ''')
        (algo_dir / "sorting.py").write_text(sorting_py)

        graphs_py = dedent('''\
            """Graph algorithms."""

            def bfs(adj, s):
                """Breadth-first search from source node s."""
                pass

            def dijkstra(adj, s):
                """Dijkstra single-source shortest paths."""
                pass
        ''')
        (algo_dir / "graphs.py").write_text(graphs_py)

        # __init__.py so the dir is treated as a package (not strictly needed but realistic)
        (algo_dir / "__init__.py").write_text("")
        (src_dir / "__init__.py").write_text("")
        (clrs_root / "clrs" / "__init__.py").write_text("")

        return clrs_root

    def test_ingest_produces_primitives(self, tmp_path):
        from ageom.architect.ingest_clrs import ingest_clrs

        repo = self._create_mock_clrs_repo(tmp_path)
        catalog = ingest_clrs(repo)

        # Should have primitives from specs
        assert catalog.size >= 5

    def test_sorting_algorithms_categorized(self, tmp_path):
        from ageom.architect.ingest_clrs import ingest_clrs

        repo = self._create_mock_clrs_repo(tmp_path)
        catalog = ingest_clrs(repo)

        insertion = catalog.get("insertion_sort")
        assert insertion is not None
        assert insertion.source == "clrs-30"
        # Category comes from the module file or name heuristic
        assert insertion.category in (ConceptType.SORTING, ConceptType.CUSTOM)

    def test_graph_algorithms_categorized(self, tmp_path):
        from ageom.architect.ingest_clrs import ingest_clrs

        repo = self._create_mock_clrs_repo(tmp_path)
        catalog = ingest_clrs(repo)

        bfs = catalog.get("bfs")
        assert bfs is not None
        assert bfs.source == "clrs-30"

    def test_io_specs_extracted(self, tmp_path):
        from ageom.architect.ingest_clrs import ingest_clrs

        repo = self._create_mock_clrs_repo(tmp_path)
        catalog = ingest_clrs(repo)

        dijkstra = catalog.get("dijkstra")
        assert dijkstra is not None
        # Should have inputs (adj, s) and outputs (d)
        assert len(dijkstra.inputs) >= 1 or len(dijkstra.outputs) >= 1

    def test_empty_path_returns_empty_catalog(self, tmp_path):
        from ageom.architect.ingest_clrs import ingest_clrs

        catalog = ingest_clrs(tmp_path / "nonexistent")
        assert catalog.size == 0


class TestIngestCoq100:
    """Test coq-100-theorems ingestion with mock data."""

    def _create_mock_coq100_repo(self, tmp_path: Path) -> Path:
        """Create a minimal mock of the coq-100-theorems repo."""
        repo = tmp_path / "coq-100-theorems"
        repo.mkdir()

        theorems_yml = dedent("""\
            1: Irrationality of sqrt(2)
            2: Fundamental Theorem of Algebra
            3: Denumerability of the Rationals
            4: Pythagorean Theorem
            5: Prime Number Theorem
            11: The Infinitude of Primes
            22: Area of a Circle
            44: The Binomial Theorem
            60: Bezout's Theorem
            74: The Principle of Mathematical Induction
        """)
        (repo / "theorems.yml").write_text(theorems_yml)

        statements_yml = dedent("""\
            1: "forall p q : Z, q <> 0 -> p * p <> 2 * (q * q)"
            2: "forall p : C[X], 0 < degree p -> exists z : C, p.[z] = 0"
            3: "Countable Q"
            4: "forall a b c : R, a^2 + b^2 = c^2"
            5: "prime_counting_function ~ (fun n => n / ln n)"
            11: "forall n, exists p, n < p /\\ prime p"
            22: "forall r : R, area_circle r = PI * r^2"
            44: "forall n a b : nat, (a + b)^n = sum_binomial n a b"
            60: "forall a b, exists u v, a*u + b*v = gcd a b"
            74: "forall P : nat -> Prop, P 0 -> (forall n, P n -> P (S n)) -> forall n, P n"
        """)
        (repo / "statements.yml").write_text(statements_yml)

        return repo

    def test_ingest_produces_primitives(self, tmp_path):
        from ageom.architect.ingest_coq100 import ingest_coq100

        repo = self._create_mock_coq100_repo(tmp_path)
        catalog = ingest_coq100(repo)

        assert catalog.size == 10

    def test_theorem_classification(self, tmp_path):
        from ageom.architect.ingest_coq100 import ingest_coq100

        repo = self._create_mock_coq100_repo(tmp_path)
        catalog = ingest_coq100(repo)

        # Pythagorean theorem → GEOMETRY (keyword "pythagor")
        pyth = catalog.get("thm_4")
        assert pyth is not None
        assert pyth.category == ConceptType.GEOMETRY

        # Infinitude of Primes → NUMBER_THEORY (keyword "prime")
        primes = catalog.get("thm_11")
        assert primes is not None
        assert primes.category == ConceptType.NUMBER_THEORY

        # Binomial Theorem → COMBINATORICS (keyword "binomial")
        binom = catalog.get("thm_44")
        assert binom is not None
        assert binom.category == ConceptType.COMBINATORICS

    def test_type_signatures_populated(self, tmp_path):
        from ageom.architect.ingest_coq100 import ingest_coq100

        repo = self._create_mock_coq100_repo(tmp_path)
        catalog = ingest_coq100(repo)

        thm1 = catalog.get("thm_1")
        assert thm1 is not None
        assert thm1.type_signature != ""
        assert "forall" in thm1.type_signature

    def test_source_is_coq_100(self, tmp_path):
        from ageom.architect.ingest_coq100 import ingest_coq100

        repo = self._create_mock_coq100_repo(tmp_path)
        catalog = ingest_coq100(repo)

        for prim in catalog.all_primitives():
            assert prim.source == "coq-100-theorems"

    def test_empty_repo_returns_empty(self, tmp_path):
        from ageom.architect.ingest_coq100 import ingest_coq100

        repo = tmp_path / "empty_repo"
        repo.mkdir()
        catalog = ingest_coq100(repo)
        assert catalog.size == 0
