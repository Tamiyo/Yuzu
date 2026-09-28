"""Functions are declared, called from a query, and evaluated away."""

from support import error_of, rows, sorted_rows


def test_function_without_parameters():
    query = """
        def cap() -> int64 { return 2 }
        from employees
        |> where level > cap()
        |> select name
    """
    assert rows(query) == [("carol",)]


def test_function_with_a_parameter():
    query = """
        def twice(n: int64) -> int64 { return n * 2 }
        from employees
        |> where level == twice(1)
        |> select name
    """
    assert rows(query) == sorted_rows(("bob",), ("dan",))


def test_function_calling_another_function():
    query = """
        def one() -> int64 { return 1 }
        def two() -> int64 { return one() + one() }
        from employees
        |> where level > two()
        |> select name
    """
    assert rows(query) == [("carol",)]


def test_local_bindings_and_assignment():
    query = """
        def threshold() -> int64 { let x = 10
        let mut y = 5
        y = 2
        return y }
        from employees
        |> where level > threshold()
        |> select name
    """
    assert rows(query) == [("carol",)]


def test_function_applied_to_a_column():
    query = """
        def double(n: int64) -> int64 { return n * 2 }
        from employees
        |> where name == "carol"
        |> select double(level) as doubled
    """
    assert rows(query) == [(6,)]


def test_overloads_are_picked_by_argument_count():
    query = """
        def scale(n: int64) -> int64 { return n * 10 }
        def scale(n: int64, factor: int64) -> int64 { return n * factor }
        from employees
        |> where name == "carol"
        |> select scale(level) as ten, scale(level, 2) as two
    """
    assert rows(query) == [(30, 6)]


def test_a_call_no_overload_takes_is_rejected():
    query = """
        def scale(n: int64) -> int64 { return n * 10 }
        def scale(n: int64, factor: int64) -> int64 { return n * factor }
        from employees
        |> select scale() as v
    """
    assert error_of(query) == "error: `scale` expects 1 or 2 argument(s), found 0"
