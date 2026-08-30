# The rows edge

`rowfilter`, the one node of a network that is not a module. It keeps or drops
the rows travelling with the frames, by a predicate written as JSON. The
compiler builds this spelling.

## The spelling

    [a]rowfilter=pred=<json>[b]

`pred` is its only option, and it is required. The value is a JSON predicate,
escaped like any other option value - twice, once for the option list and once
for the graph. The option level escapes `\ ' :`; the graph level escapes
`\ ' [ ] , ;` over the top of it.

This predicate:

    {"ge":[{"field":"shot"},{"lit":1}]}

is written into a `-filter_complex` as:

    [0:v]shots[a];[a]rowfilter=pred={"ge"\\:\[{"field"\\:"shot"}\,{"lit"\\:1}\]}[out0]

That is the `-filter_complex` argument itself, before whatever quoting the
shell wants on top.

`rowfilter` is a reserved node name. No `-m` binds it, and a `-m rowfilter=...`
is refused; the host provides it. A network built only of `rowfilter` nodes
binds nothing, and is not refused for having no `-m`.

## The predicate

An operand is a field of the row or a literal:

    {"field": "<name>"}
    {"lit": <string | number | bool>}

A comparison takes two operands. The six are `eq`, `ne`, `lt`, `le`, `gt`,
`ge`:

    {"eq": [{"field": "class"}, {"lit": "person"}]}
    {"ge": [{"field": "shot"}, {"lit": 1}]}

Logic composes them. `and` and `or` take a list, `not` takes one predicate:

    {"and": [<pred>, <pred>, ...]}
    {"or":  [<pred>, <pred>, ...]}
    {"not": <pred>}

An empty `and` keeps every row; an empty `or` keeps none.

## What a comparison means

Numbers compare numerically, strings lexically, booleans as false below true.
A row where the two sides are of different types is **dropped**, and the field
is named once on stderr:

    [rowfilter] field 'score' compares a string with a number; those rows are dropped

Once per field per run, not once per row: rows are runtime data, and a stream
that disagrees with the predicate about a type is a stream to notice, not a
run to stop.

A row that does not carry a field the predicate names is dropped silently.
Absent is not an error - it is how a consumer already skips the rows of a
module it was not written for.

A row that is not a JSON object is dropped.

## What passes through

Frames pass through untouched, and are not copied where the graph allows it.
Only the rows change. Trailing rows - the ones a module had no frame left to
put them on - go through the same predicate.

`rowfilter` reads one stream, window 1, stride 1, pure and one-to-one. It sits
anywhere in a chain: between two modules, straight after an input, or last
before a `-map`. It reads rows and passes on the ones it keeps, so a module
downstream of it still sees rows, and `-annotations in` reaches through it.

It is not a module, so `--describe` never names it and it has no world,
version or params schema.

## Refusals

A predicate that is not valid JSON, or is JSON of a shape the grammar does not
have, is refused at startup naming the node and what was wrong:

    rowfilter: pred is not valid JSON: EOF while parsing an object at line 1 column 12
    rowfilter: pred has no operator; a predicate is one of eq, ne, lt, le, gt, ge, and, or, not
    rowfilter: pred: 'eq' takes two operands, got 3
    rowfilter: pred: an operand is {"field": "<name>"} or {"lit": <value>}

A `rowfilter` node given no `pred`, or given an option that is not `pred`:

    rowfilter takes one option, pred=<json>, and was given none
    rowfilter has no option 'threshold'; it takes pred=<json>
