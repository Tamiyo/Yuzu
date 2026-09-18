"""Functions are declared, called from a query, and evaluated away."""

from support import rows, sorted_rows


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
