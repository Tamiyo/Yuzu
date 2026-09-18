"""`external fn`: body-less declarations the target provides."""

from support import error_of, rows, sorted_rows


def test_external_scalar():
    query = """
        external def upper(s: str) -> str
        from employees
        |> where name == "alice"
        |> select upper(name) as shout
    """
    assert rows(query) == [("ALICE",)]


def test_external_scalar_composes():
    query = """
        external def char_length(s: str) -> int64
        from employees
        |> select char_length(name) + 1 as n
        |> where n == 4
        |> select n
    """
    assert rows(query) == sorted_rows((4,), (4,))


def test_external_agg():
    query = """
        external agg def median(x: int64) -> float64
        from employees
        |> aggregate median(salary) as mid
        group by dept_id
    """
    assert rows(query) == sorted_rows((1, 180000.0), (2, 90000.0), (9, 60000.0))


def test_external_scalar_in_a_select():
    query = """
        external def abs(x: int64) -> int64
        from employees
        |> where name == "alice"
        |> select abs(0 - level) as v
    """
    assert rows(query) == [(1,)]


def test_external_with_a_body_is_rejected():
    query = """
        external def nope(x: int64) -> int64 { return x }
        from employees
        |> select level
    """
    assert error_of(query) == "error: an external function cannot have a body"


def test_fn_without_a_body_is_rejected():
    query = """
        def nope(x: int64) -> int64
        from employees
        |> select level
    """
    assert error_of(query) == "error: function is missing its body"
