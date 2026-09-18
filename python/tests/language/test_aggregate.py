"""Every aggregate form is writable: each builtin, grouped and full-table,
composite items, and aggregation over a join."""

from support import rows, sorted_rows


def test_count_rows():
    query = """
        from employees
        |> aggregate count() as n
    """
    assert rows(query) == [(4,)]


def test_each_builtin_over_a_column():
    query = """
        from employees
        |> aggregate sum(salary) as total, min(salary) as low, max(salary) as high, avg(rating) as r
    """
    assert rows(query) == [(510000, 60000, 240000, 2.5)]


def test_group_by_one_key():
    query = """
        from employees
        |> aggregate count() as n
        group by dept_id
    """
    assert rows(query) == sorted_rows((1, 2), (2, 1), (9, 1))


def test_group_key_alias_names_the_output():
    query = """
        from employees
        |> aggregate count() as n
        group by level as tier
        |> where tier == 2
        |> select tier, n
    """
    assert rows(query) == [(2, 2)]


def test_composite_item():
    query = """
        from employees
        |> aggregate max(salary) - min(salary) as spread
        group by dept_id
    """
    assert rows(query) == sorted_rows((1, 120000), (2, 0), (9, 0))


def test_aggregate_over_a_join():
    query = """
        from employees e
        |> join departments d on e.dept_id == d.dept_id
        |> aggregate count() as n
        group by d.name
    """
    assert rows(query) == sorted_rows(("eng", 2), ("sales", 1))


def test_aggregate_then_pipeline_continues():
    query = """
        from employees
        |> aggregate sum(salary) as total
        group by dept_id
        |> where total > 100000
        |> select total
    """
    assert rows(query) == [(360000,)]


def test_agg_fn_composes_builtins():
    query = """
        agg def spread(x: int64) -> int64 { return max(x) - min(x) }
        from employees
        |> aggregate spread(salary) as v
        group by dept_id
    """
    assert rows(query) == sorted_rows((1, 120000), (2, 0), (9, 0))
