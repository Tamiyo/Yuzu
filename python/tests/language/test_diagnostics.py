"""What the language rejects, as the error reaches the Python API."""

from support import error_of


def test_ambiguous_bare_column():
    query = """
        from employees e
        |> join departments d on e.dept_id == d.dept_id
        |> select id
    """
    assert (
        error_of(query)
        == "error: column `id` is ambiguous; qualify it with a relation alias"
    )


def test_unknown_column():
    query = """
        from employees
        |> select nope
    """
    assert error_of(query) == "error: unresolved identifier `nope`"


def test_unknown_table():
    query = """
        from nope
        |> select level
    """
    assert error_of(query) == "error: `nope` is not a relation"


def test_non_bool_predicate():
    query = """
        from employees
        |> where level
    """
    assert (
        error_of(query)
        == "error: expected the `where` predicate to be `bool`, found `int64`"
    )


def test_non_bool_join_condition():
    query = """
        from employees e
        |> join departments d on e.dept_id
    """
    assert (
        error_of(query)
        == "error: expected the `on` condition to be `bool`, found `int64`"
    )


def test_using_column_missing_from_a_side():
    assert (
        error_of("""
            from employees e
            |> join departments d using (nope)
        """)
        == "error: column nope not present in both relations"
    )
    assert (
        error_of("""
            from employees e
            |> join departments d using (salary)
        """)
        == "error: column salary not present in both relations"
    )


def test_operator_without_a_substrait_equivalent():
    query = """
        from employees
        |> select level ** 2 as p
    """
    assert (
        error_of(query)
        == "error: `**` is not supported by the datafusion target"
    )


def test_unbounded_recursion_is_rejected_at_compile_time():
    query = """
        def f(n: int64) -> int64 { return f(n - 1) }
        from employees
        |> select f(3) as v
    """
    assert (
        error_of(query)
        == "error: expanding `f` did not finish within 1000 calls; a function that reaches itself has to reduce to stop"
    )
