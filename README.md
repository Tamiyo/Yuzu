# Yuzu

Yuzu is a small, statically-typed **relational query language**. You write
queries with pipe syntax (`|>`), and Yuzu compiles them to
[Substrait](https://substrait.io) — a portable query-plan format. Because the
output is Substrait, a Yuzu query can run on any engine that consumes it; today
we execute plans with [Apache DataFusion](https://datafusion.apache.org).

> **Status:** Yuzu is early and under active development. The compiler is
> written in Rust; traits and generic functions from the original design have
> not been ported yet.

```
struct Employee { id: str, department: str, salary: int64 }
table employees = Employee

from employees e
|> where e.salary > 100000
|> extend e.salary / 12 as monthly
|> select e.id, e.department, monthly
```

A query is an ordinary expression of type `Relation[Row]`. Each `|>` stage
transforms the relation flowing through it, and only sees the columns of its
immediate input — just like SQL's pipe syntax.

## Language tour

### Structs and tables

```
struct Employee { id: str, department: str, salary: int64 }
table employees = Employee
```

A `struct` is a named row type; a `table` registers a relation you can query
with `from`.

### Pipe operators

```
from employees e
|> where e.salary > 100000      // keep rows matching a predicate
|> extend e.salary * 2 as bonus // add computed columns
|> rename department as dept    // rename columns
|> drop id                      // remove columns
|> select dept, bonus           // project columns
|> distinct                     // remove duplicate rows
```

| operator          | effect                                             |
| ----------------- | -------------------------------------------------- |
| `from t e`        | the pipe source (the alias `e` is optional)        |
| `select …`        | project a new set of columns                       |
| `where p`         | keep rows where the boolean predicate `p` holds    |
| `extend e as n`   | append computed columns, keeping the existing ones |
| `drop a, b`       | remove named columns                               |
| `rename a as b`   | rename columns                                     |
| `distinct`        | drop duplicate rows                                |
| `set a = e`       | replace a column's value, keeping its place        |
| `limit n`         | keep at most `n` rows, after an optional `offset`  |
| `as t`            | name the whole row, so its columns qualify as `t.` |
| `join t d on p`   | combine rows with a second relation                |

Columns are referenced by bare name against the current stage's row (`salary`),
or through the source alias (`e.salary`). The set of operators and their
semantics are still evolving.

`set` keeps the row's shape — the named column takes a new value where it
stands — while `extend` appends. `limit` takes a row count that must reduce to a
constant, like everything else. `as` renames the whole row and replaces any
alias it had, which is how a computed row gets qualified names again:

```
from employees e
|> join departments d on e.dept_id == d.dept_id
|> select e.id as eid, d.name as dept
|> as joined
|> select joined.eid, joined.dept
```

### Joins

```
from employees e
|> join departments d on e.dept_id == d.id
|> select e.id, d.name
```

The input is the left side and the joined relation is the right; the stage's
output row is the two concatenated, left first. `inner` (the default), `left`,
`right` and `full` all work as modifiers — `|> left join departments d on …`.

When both sides share a key name, `using` matches on it and keeps a single copy
in the output row:

```
from employees |> join departments using (dept_id) |> select dept_id, id, name
```

A join concatenates the two rows, so both sides may carry the same column name.
That is only a problem where a name is *used* without saying which side it came
from — a qualified reference always resolves:

```
from employees e
|> join students s on e.id == s.id
|> rename e.id as eid, s.id as sid   // `rename` takes a qualified column too
|> select eid, sid, e.salary
```

A bare `id` there is `column `id` is ambiguous; qualify it with a relation
alias`, while a bare name only one side has keeps working.

### Functions

```
fn monthly(salary: int64) -> int64 {
    return salary / 12
}

from employees e |> select monthly(e.salary) as monthly
```

Functions close over top-level bindings, and calls to known functions are
inlined and constant-folded at compile time — a call that survives to the plan
(e.g. unbounded recursion) is a compile error, not a runtime one.

### Values and lists

```
let cutoff = 10 + 5
let ids: List[int32] = [1, 2, 3]

from employees e |> where e.department in ["eng", "sales"]
```

Primitive types are the fixed-width integers (`int8`…`int64`,
`uint8`…`uint64`), floats (`float32`, `float64`), `bool`, and `str`, plus
`List[T]`, structs, and relations. Membership (`in`, `not in`) over constant
lists folds at compile time — down to structural equality over lists and
structs — and otherwise becomes part of the plan.

## How it works

A Yuzu program flows through a fixed pipeline:

```
source → tokens → syntax tree → yzl → yzr → Substrait plan
```

- The **syntax tree** is a lossless CST (rowan); the AST is a typed view over
  it, so diagnostics can always point back at real source ranges.
- **yzl** is an MLIR dialect that holds the program as written: functions,
  local variables and query stages. The passes over it check mutability,
  promote local variables to values, infer types, check the aggregate rules
  and inline every call.
- **yzr** is an MLIR dialect of relations. It holds only what a plan can
  express, and MLIR's canonicalization simplifies it.
- The **Substrait** translation maps yzr onto Substrait relations (protobuf,
  with JSON for debugging). Anything a plan cannot express is reported as a
  source-level diagnostic.

## Everything reduces at compile time

Yuzu rests on one core constraint: **every program must fully evaluate at
compile time** into a finite relational plan. The compiler inlines every
function call and folds constants until nothing dynamic remains, and only
then emits Substrait.

This means anything that *can't* be fully evaluated or unrolled — unbounded
loops, unbounded or dynamic recursion — is rejected rather than deferred to
runtime. The payoff is that a compiled query is a fixed plan (a DAG with no
runtime control flow), which is exactly what Substrait expresses, and what lets
the same query run on any engine that consumes it.

## Building and running

```
cargo build                                   # build everything
cargo test                                    # the Rust test suite
cargo e2e                                     # queries executed against DataFusion
cargo run -p yuzu_driver -- query.yuzu        # compile one file
```

`cargo e2e` rebuilds the Python wheel and runs [python/tests/](python/tests/)
against it, so it always exercises the current compiler rather than whatever
was installed last. It needs the venv from
[Running queries from Python](#running-queries-from-python); `cargo test` does
not, and skips it.

Each pipeline stage can be dumped:

```
cargo run -p yuzu_driver -- query.yuzu --debug-yzl        # the module as the frontend lowered it
cargo run -p yuzu_driver -- query.yuzu --debug-yzr        # the module lowered to relations
cargo run -p yuzu_driver -- query.yuzu --debug-substrait  # the Substrait plan
cargo run -p yuzu_driver -- query.yuzu --debug            # all of the above
```

### Running queries from Python

The Python package compiles Yuzu source to Substrait bytes, which DataFusion
executes. It builds as a wheel with [maturin](https://www.maturin.rs):

```
python3 -m venv .venv && source .venv/bin/activate
pip install maturin
maturin develop -m crates/yuzu_python/Cargo.toml   # build + install into the venv
pip install -r python/requirements.txt             # datafusion, for running plans
python python/example.py
```

See [python/example.py](python/example.py) for the whole flow — a query, a
table of data, and results — in ~30 lines, and
[python/tests/](python/tests/) for queries checked against their result rows.
Those split two ways over one shared schema in `support.py`: `language/` covers
whether a construct can be written and run at all, `correctness/` whether the
query computes the right answer. For a distributable wheel:

```
maturin build --release -m crates/yuzu_python/Cargo.toml   # -> target/wheels/
```

The package ships type stubs (generated from the Rust definitions via
`cargo run -p yuzu_python --bin stub_gen`), so `yuzu.compile(...)` gets full
IDE completion out of the box.

## Roadmap

A guiding goal is to make relational plans **first-class, inspectable values**:

- **Relational graphs** — expose a query's plan as a graph you can traverse in
  Yuzu code, not just compile away.
- **First-class transformation APIs** — operate on relational plans directly
  (match, rewrite, inject stages), so passes like privacy transforms
  (e.g. k-anonymity filters over quasi-identifiers) can be written in Yuzu
  itself rather than in the compiler.
- **Traits and generic functions** — port the trait bounds and generics from
  the original design.
