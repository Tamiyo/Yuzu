"""Operators compute the values they should."""

from support import rows, sorted_rows

CAROL = 'from employees |> where name == "carol" |> '
ALICE = 'from employees |> where name == "alice" |> '


def test_integer_arithmetic():
    assert rows(
        CAROL + "select level + 2 as add, level - 2 as sub, level * 2 as mul, level / 2 as div"
    ) == [(5, 1, 6, 1)]


def test_float_arithmetic():
    assert rows(ALICE + "select rating * 2.0 as doubled") == [(3.0,)]


def test_negation():
    assert rows(ALICE + "select -level as neg") == [(-1,)]


def test_comparisons_select_the_right_rows():
    assert rows("""
        from employees
        |> where level > 2
        |> select name
    """) == [("carol",)]
    assert rows("""
        from employees
        |> where level >= 2
        |> select name
    """) == sorted_rows(
        ("bob",), ("carol",), ("dan",)
    )
    assert rows("""
        from employees
        |> where level < 2
        |> select name
    """) == [("alice",)]
    assert rows("""
        from employees
        |> where level <= 2
        |> select name
    """) == sorted_rows(
        ("alice",), ("bob",), ("dan",)
    )
    assert rows("""
        from employees
        |> where level == 2
        |> select name
    """) == sorted_rows(
        ("bob",), ("dan",)
    )
    assert rows("""
        from employees
        |> where level != 2
        |> select name
    """) == sorted_rows(
        ("alice",), ("carol",)
    )


def test_string_equality():
    query = """
        from employees
        |> where name == "bob"
        |> select level
    """
    assert rows(query) == [(2,)]


def test_boolean_operators():
    assert rows("""
        from employees
        |> where active
        |> select name
    """) == sorted_rows(
        ("alice",), ("carol",)
    )
    assert rows("""
        from employees
        |> where not active
        |> select name
    """) == sorted_rows(
        ("bob",), ("dan",)
    )
    assert rows("""
        from employees
        |> where active and level > 1
        |> select name
    """) == [("carol",)]
    assert rows("""
        from employees
        |> where active or level == 2
        |> select name
    """) == sorted_rows(
        ("alice",), ("bob",), ("carol",), ("dan",)
    )


def test_membership():
    assert rows("""
        from employees
        |> where level in [1, 3]
        |> select name
    """) == sorted_rows(
        ("alice",), ("carol",)
    )
    assert rows("""
        from employees
        |> where level not in [1, 3]
        |> select name
    """) == sorted_rows(
        ("bob",), ("dan",)
    )
    assert rows("""
        from employees
        |> where name in ["alice", "dan"]
        |> select level
    """) == sorted_rows(
        (1,), (2,)
    )


def test_shifts_and_powers_reach_the_engine():
    query = """
        from employees
        |> where name == "carol"
        |> select level << 2 as shl, level >> 1 as shr, level ** 2 as sq
    """
    assert rows(query) == [(12, 1, 9)]
