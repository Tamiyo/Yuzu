"""Everything evaluates at compile time, so a folded value has to equal what
the expression would have computed."""

from support import rows, sorted_rows


def test_folded_call_matches_the_literal():
    folded = rows(
        """
            def twice(n: int64) -> int64 { return n * 2 }
            from employees
            |> where level == twice(1)
            |> select name
        """
    )
    assert folded == rows("""
        from employees
        |> where level == 2
        |> select name
    """)
    assert folded == sorted_rows(("bob",), ("dan",))


def test_folded_let_matches_the_literal():
    folded = rows("""
        let cap = 1 + 1
        from employees
        |> where level > cap
        |> select name
    """)
    assert folded == rows("""
        from employees
        |> where level > 2
        |> select name
    """)
    assert folded == [("carol",)]


def test_assignment_folds_to_the_last_value():
    query = """
        def threshold() -> int64 { let mut x = 10
        x = 2
        return x }
        from employees
        |> where level > threshold()
        |> select name
    """
    assert rows(query) == [("carol",)]


def test_folded_arithmetic_in_a_projection():
    query = """
        from employees
        |> where name == "alice"
        |> select 2 * 3 + 1 as folded
    """
    assert rows(query) == [(7,)]


def test_membership_over_a_constant_list_folds():
    query = """
        let ids = [1, 3]
        from employees
        |> where level in ids
        |> select name
    """
    assert rows(query) == sorted_rows(("alice",), ("carol",))
