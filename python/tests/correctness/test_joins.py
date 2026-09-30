"""Joins match the right rows, keep the right columns, and resolve each
reference to the side it names."""

from support import error_of, rows, sorted_rows

# alice and carol are in dept 1, bob in dept 2, dan in dept 9 (no department).
# Department `ops` (dept 3) has no employee.
MATCHED = sorted_rows(("alice", "eng"), ("bob", "sales"), ("carol", "eng"))
# The same employees joined to `grades`, whose codes mirror the department ids.
GRADED = sorted_rows(("alice", "gold"), ("bob", "silver"), ("carol", "gold"))


def test_inner_join_keeps_only_matching_rows():
    query = """
        from employees e
        |> join departments d on e.dept_id == d.dept_id
        |> select e.name as who, d.name as dept
    """
    assert (
        rows(query)
        == MATCHED
    )


def test_inner_join_without_matches_is_empty():
    query = """
        from employees e
        |> join departments d on e.dept_id == d.dept_id
        |> where e.salary > 500000
        |> select e.name as who
    """
    assert rows(query) == []


def test_left_join_keeps_unmatched_left_rows():
    query = """
        from employees e
        |> left join departments d on e.dept_id == d.dept_id
        |> select e.name as who, d.name as dept
    """
    assert rows(query) == sorted_rows(*MATCHED, ("dan", None))


def test_right_join_keeps_unmatched_right_rows():
    query = """
        from employees e
        |> right join departments d on e.dept_id == d.dept_id
        |> select e.name as who, d.name as dept
    """
    assert rows(query) == sorted_rows(*MATCHED, (None, "ops"))


def test_full_join_keeps_both_sides():
    query = """
        from employees e
        |> full join departments d on e.dept_id == d.dept_id
        |> select e.name as who, d.name as dept
    """
    assert rows(query) == sorted_rows(*MATCHED, ("dan", None), (None, "ops"))


# --- each reference resolves to the side it names ---


def test_qualified_columns_come_from_their_own_side():
    query = """
        from employees e
        |> join departments d on e.dept_id == d.dept_id
        |> select e.id as employee, d.id as department
    """
    assert rows(query) == sorted_rows(("e1", "d1"), ("e2", "d2"), ("e3", "d1"))


def test_bare_column_from_one_side_only():
    query = """
        from employees e
        |> join grades g on e.dept_id == g.code
        |> select e.name as who, label
    """
    assert rows(query) == GRADED


def test_four_way_join_resolves_every_qualified_column():
    query = """
        from employees e
        |> join departments d on e.dept_id == d.dept_id
        |> join projects p on p.dept_id == d.dept_id
        |> join grades g on g.code == d.dept_id
        |> select e.id as ei, d.id as di, p.id as pi, g.label as label
    """
    assert rows(query) == sorted_rows(
        ("e1", "d1", "p1", "gold"),
        ("e2", "d2", "p2", "silver"),
        ("e3", "d1", "p1", "gold"),
    )


def test_self_join_keeps_the_two_sides_apart():
    query = """
        from employees x
        |> join employees y on x.dept_id == y.dept_id
        |> where x.name == "alice"
        |> select x.id as xi, y.id as yi
    """
    assert rows(query) == sorted_rows(("e1", "e1"), ("e1", "e3"))


def test_qualified_rename_names_one_side():
    query = """
        from employees e
        |> join departments d on e.dept_id == d.dept_id
        |> rename e.id as employee, d.id as department
        |> select employee, department
    """
    assert rows(query) == sorted_rows(("e1", "d1"), ("e2", "d2"), ("e3", "d1"))


# --- `using` carries one copy of each key ---


def test_using_keeps_each_sides_copy_under_its_name():
    """Each side's own key stays reachable through its alias."""
    query = """
        from employees e
        |> join departments d using (dept_id)
        |> select e.dept_id as left_key, d.dept_id as right_key, e.name as who
    """
    assert rows(query) == sorted_rows((1, 1, "alice"), (2, 2, "bob"), (1, 1, "carol"))


def test_a_bare_using_key_is_the_one_merged_column():
    query = """
        from employees e
        |> join departments d using (dept_id)
        |> select dept_id, e.name as who
    """
    assert rows(query) == sorted_rows((1, "alice"), (2, "bob"), (1, "carol"))


def test_using_puts_the_merged_key_first():
    """The key once, then the other left columns, then the other right ones;
    the sides' own copies of the key are not output."""
    query = """
        from employees e
        |> join departments d using (dept_id)
        |> where e.name == "alice"
    """
    assert rows(query) == [(1, "e1", "alice", 1, 120000, True, 1.5, "d1", "eng")]


def test_using_with_several_keys():
    query = """
        from employees a
        |> join employees b using (id, dept_id)
        |> select a.id as id, b.dept_id as dept_id
    """
    assert rows(query) == sorted_rows(("e1", 1), ("e2", 2), ("e3", 1), ("e4", 9))


def test_outer_join_using_preserves_unmatched_rows():
    query = """
        from employees e
        |> left join departments d using (dept_id)
        |> select e.name as who, d.name as dept
    """
    assert rows(query) == sorted_rows(*MATCHED, ("dan", None))


def test_a_right_join_using_takes_the_right_key():
    query = """
        from employees e
        |> right join departments d using (dept_id)
        |> select dept_id, e.name as who
    """
    assert rows(query) == sorted_rows((1, "alice"), (1, "carol"), (2, "bob"), (3, None))


def test_a_full_join_using_takes_the_key_of_either_side():
    query = """
        from employees e
        |> full join departments d using (dept_id)
        |> select dept_id, e.name as who, d.name as dept
    """
    assert rows(query) == sorted_rows(
        (1, "alice", "eng"),
        (1, "carol", "eng"),
        (2, "bob", "sales"),
        (9, "dan", None),
        (3, None, "ops"),
    )


# --- later stages see the joined row ---


def test_stages_after_a_join():
    query = """
        from employees e
        |> join departments d on e.dept_id == d.dept_id
        |> where e.salary > 100000
        |> extend e.salary / 12 as monthly
        |> select d.name as dept, monthly
    """
    assert rows(query) == sorted_rows(("eng", 10000), ("eng", 20000))


def test_distinct_after_a_join():
    query = """
        from employees e
        |> join departments d on e.dept_id == d.dept_id
        |> select d.name as dept
        |> distinct
    """
    assert rows(query) == sorted_rows(("eng",), ("sales",))


def test_drop_after_a_join():
    query = """
        from employees e
        |> join grades g on e.dept_id == g.code
        |> select e.name as who, label, code
        |> drop code
    """
    assert rows(query) == GRADED
