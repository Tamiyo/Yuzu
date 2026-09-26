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
