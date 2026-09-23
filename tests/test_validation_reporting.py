from src.validation.reporting import _readable_analyses, _structured_results


def test_readable_analyses_converts_table_rows_to_named_records() -> None:
    analyses = [
        {
            "type": "table",
            "title": "Example",
            "description": "Readable table",
            "columns": ["Category", "Task", "Pass@1"],
            "rows": [["data discovery", "task_1", 0.75]],
        },
        {
            "type": "scalar",
            "title": "Score",
            "description": None,
            "value": 0.75,
            "unit": None,
        },
    ]

    assert _readable_analyses(analyses) == [
        {
            "type": "table",
            "title": "Example",
            "description": "Readable table",
            "records": [
                {
                    "Category": "data discovery",
                    "Task": "task_1",
                    "Pass@1": 0.75,
                }
            ],
        },
        analyses[1],
    ]


def test_structured_results_builds_overall_category_task_hierarchy() -> None:
    analyses = [
        {
            "type": "scalar",
            "title": "Average accuracy",
            "value": 0.75,
        },
        {
            "type": "scalar",
            "title": "Stratified average pass@1",
            "value": 0.8,
        },
        {
            "type": "table",
            "title": "Token usage and estimated cost",
            "records": [
                {"Metric": "Input tokens", "Value": 100, "Unit": "tokens"},
                {"Metric": "Estimated cost", "Value": 0.01, "Unit": "USD"},
            ],
        },
        {
            "type": "table",
            "title": "Average pass@k",
            "records": [{"k": 1, "Tasks": 1, "Average Pass@k": 0.75}],
        },
        {
            "type": "table",
            "title": "Average pass^k",
            "records": [{"k": 1, "Tasks": 1, "Average Pass^k": 0.75}],
        },
        {
            "type": "table",
            "title": "Per-task statistical summary",
            "records": [
                {
                    "Case ID": "case-a",
                    "Task": "task-a",
                    "Category": "data discovery",
                    "Correct": 3,
                    "Samples": 4,
                    "Pass@1": 0.75,
                }
            ],
        },
        {
            "type": "table",
            "title": "Per-task pass@k",
            "records": [
                {
                    "Category": "data discovery",
                    "Case ID": "case-a",
                    "Task": "task-a",
                    "k": 1,
                    "Correct": 3,
                    "Samples": 4,
                    "Pass@k": 0.75,
                }
            ],
        },
        {
            "type": "table",
            "title": "Per-task pass^k",
            "records": [
                {
                    "Category": "data discovery",
                    "Case ID": "case-a",
                    "Task": "task-a",
                    "k": 1,
                    "Correct": 3,
                    "Samples": 4,
                    "Pass^k": 0.75,
                }
            ],
        },
        {
            "type": "table",
            "title": "Per-category pass metrics",
            "records": [
                {
                    "Category": "data discovery",
                    "k": 1,
                    "Tasks": 1,
                    "Average Pass@k": 0.75,
                    "Average Pass^k": 0.75,
                }
            ],
        },
        {
            "type": "table",
            "title": "Per-category token usage and estimated cost",
            "records": [
                {
                    "Category": "data discovery",
                    "Attempts With Usage": 4,
                    "Requests": 5,
                    "Tool Calls": 7,
                    "Input Tokens": 100,
                    "Output Tokens": 20,
                    "Total Tokens": 120,
                    "Estimated Cost (USD)": 0.01,
                    "Pricing Source": "openai:gpt-5-mini",
                }
            ],
        },
    ]

    cases = [
        {
            "name": "task-a:1",
            "source_case_name": "task-a",
            "metadata": {
                "case_id": "case-a",
                "category": "data discovery",
            },
            "inputs": "Question",
            "expected_output": "Expected",
            "output": "Answer",
            "metrics": {"input_tokens": 25, "output_tokens": 5},
            "assertions": {"correct": {"value": True}},
            "evaluator_failures": [],
            "task_duration": 1.5,
            "trace_id": "trace-1",
            "span_id": "shared-span-id",
        }
    ]
    failures = [
        {
            "name": "task-a:2",
            "source_case_name": "task-a",
            "metadata": {
                "case_id": "case-a",
                "category": "data discovery",
            },
            "inputs": "Question",
            "expected_output": "Expected",
            "error_message": "model unavailable",
            "error_stacktrace": "traceback",
            "trace_id": "trace-2",
            "span_id": "shared-span-id",
        }
    ]

    assert _structured_results(analyses, cases, failures) == {
        "overall": {
            "tasks": 1,
            "attempts": 4,
            "average_pass_at_1": 0.75,
            "stratified_average_pass_at_1": 0.8,
            "pass_at_k": {"1": 0.75},
            "pass_power_k": {"1": 0.75},
            "usage": {
                "input_tokens": 100,
                "estimated_cost_usd": 0.01,
            },
        },
        "categories": {
            "data discovery": {
                "summary": {
                    "tasks": 1,
                    "pass_at_1": 0.75,
                    "attempts": 4,
                    "attempts_with_usage": 4,
                    "requests": 5,
                    "tool_calls": 7,
                    "input_tokens": 100,
                    "output_tokens": 20,
                    "total_tokens": 120,
                    "estimated_cost_usd": 0.01,
                    "pricing_source": "openai:gpt-5-mini",
                },
                "pass_at_k": {"1": 0.75},
                "pass_power_k": {"1": 0.75},
                "tasks": {
                    "case-a": {
                        "name": "task-a",
                        "summary": {
                            "correct": 3,
                            "attempts": 4,
                            "pass_at_1": 0.75,
                        },
                        "pass_at_k": {"1": 0.75},
                        "pass_power_k": {"1": 0.75},
                        "attempts": [
                            {
                                "name": "task-a:1",
                                "case_id": "case-a",
                                "trace_id": "trace-1",
                                "span_id": "shared-span-id",
                                "status": "completed",
                                "passed": True,
                                "output": "Answer",
                                "metrics": {
                                    "input_tokens": 25,
                                    "output_tokens": 5,
                                    "tool_calls": 0,
                                },
                                "assertions": {"correct": {"value": True}},
                                "evaluator_failures": [],
                                "task_duration": 1.5,
                            },
                            {
                                "name": "task-a:2",
                                "case_id": "case-a",
                                "trace_id": "trace-2",
                                "span_id": "shared-span-id",
                                "status": "failed",
                                "passed": False,
                                "error_message": "model unavailable",
                                "error_stacktrace": "traceback",
                            },
                        ],
                        "input": "Question",
                        "expected_output": "Expected",
                    }
                },
            }
        },
    }
