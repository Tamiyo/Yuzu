"""The standard library, which every program can import."""

from support import rows


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
