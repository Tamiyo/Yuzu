"""The standard library, which every program can import."""

import yuzu
from support import SCHEMA, rows


def test_the_engine_constant_names_the_target():
    query = """
        from yuzu.engine import ENGINE
        from employees
        |> where name == "alice"
        |> select ENGINE as engine
    """
    assert rows(query) == [("datafusion",)]


def test_an_engine_function_can_be_called_by_its_own_name():
    query = """
        from yuzu.target.datafusion import power
        from employees
        |> where name == "carol"
        |> select power(level, 3) as cube
    """
    assert rows(query) == [(27,)]


def test_the_engine_can_be_named():
    options = yuzu.CompileOptions(target="datafusion")
    assert yuzu.compile(SCHEMA + "from employees", options)


def test_an_engine_the_compiler_does_not_know_is_reported():
    options = yuzu.CompileOptions(target="postgres")
    try:
        yuzu.compile(SCHEMA + "from employees", options)
    except ValueError as failure:
        assert "`postgres` is not a supported engine" in str(failure)
    else:
        raise AssertionError("postgres compiled")


def test_a_compile_error_is_a_value_error_with_the_diagnostics():
    try:
        yuzu.compile("let x: i64 = 1\n")
    except yuzu.CompileError as failure:
        assert isinstance(failure, ValueError)
        assert "unknown type `i64`" in str(failure)
    else:
        raise AssertionError("i64 compiled")
