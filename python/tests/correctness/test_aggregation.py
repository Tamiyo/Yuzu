"""Aggregates compute the values they should."""

from support import rows, sorted_rows


def test_sum_per_group():
    query = """
        from employees
        |> aggregate sum(salary) as total
        group by dept_id
    """
    assert rows(query) == sorted_rows((1, 360000), (2, 90000), (9, 60000))


def test_min_and_max_per_group():
    query = """
        from employees
        |> aggregate min(level) as low, max(level) as high
        group by dept_id
    """
    assert rows(query) == sorted_rows((1, 1, 3), (2, 2, 2), (9, 2, 2))


def test_avg_of_floats():
    query = """
        from employees
        |> aggregate avg(rating) as r
        group by dept_id
    """
    assert rows(query) == sorted_rows((1, 2.5), (2, 2.5), (9, 2.5))


def test_count_counts_rows_not_values():
    query = """
        from employees
        |> aggregate count() as n
        group by active
    """
    assert rows(query) == sorted_rows((True, 2), (False, 2))


def test_measure_argument_is_computed_per_row():
    query = """
        from employees
        |> aggregate sum(salary / 1000) as scaled
        group by dept_id
    """
    assert rows(query) == sorted_rows((1, 360), (2, 90), (9, 60))


def test_composite_reuses_a_repeated_measure():
    query = """
        from employees
        |> aggregate max(salary) - min(salary) + max(salary) as v
        group by dept_id
    """
    assert rows(query) == sorted_rows((1, 360000), (2, 90000), (9, 60000))


def test_group_keys_come_first_in_the_output():
    query = """
        from employees
        |> aggregate count() as n
        group by dept_id, active
    """
    assert rows(query) == sorted_rows(
        (1, True, 2), (2, False, 1), (9, False, 1)
    )


def test_count_distinct():
    query = """
        from employees
        |> aggregate count_distinct(level) as kinds, count(level) as values
    """
    assert rows(query) == [(3, 4)]


def test_group_level_expressions_over_keys():
    query = """
        from employees
        |> aggregate sum(salary) as total, level * 2 as double_level
        group by active, level
    """
    assert rows(query) == sorted_rows(
        (True, 1, 120000, 2),
        (False, 2, 150000, 4),
        (True, 3, 240000, 6),
    )


def test_chained_aggregates_reaggregate_the_previous_output():
    query = """
        from employees
        |> aggregate sum(salary) as dept_total
        group by dept_id, active
        |> aggregate max(dept_total) as biggest
        group by active
    """
    assert rows(query) == sorted_rows((True, 360000), (False, 90000))
