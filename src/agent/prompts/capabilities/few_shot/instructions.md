## Tool-use examples

The inventory examples in tool descriptions demonstrate tool selection and
argument structure. Invoke the named tool through the native tool-call interface,
sending the JSON under `Arguments` as its arguments object. Keep nested argument
objects, including `ref`, as objects rather than JSON-encoded strings.

Use identifiers discovered in the current conversation and references returned
by actual tool calls. Follow the user's requested final-answer format; example
explanations do not override it. Tool calls belong in the tool-call interface,
not in the final answer.
