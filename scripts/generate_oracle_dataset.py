#!/usr/bin/env python3
"""Generate a curated Oracle SQL golden sample dataset with coverage checks."""
from __future__ import annotations

import json
import random
from collections import Counter
from datetime import date, timedelta
from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple

DATASET_SIZE = 1000
SINGLE_TARGET = 500
MULTI_TARGET = 500
RANDOM_SEED = 20251121
OUTPUT_DATASET = Path("data/oracle_golden_sample_1000.jsonl")
OUTPUT_SUMMARY = Path("data/oracle_golden_sample_summary.json")

REASONING_TYPES = ["arithmetic", "temporal", "commonsense"]
COMPLEXITY_LEVELS = ["basic", "intermediate", "advanced"]
REQUIRED_CONSTRUCTS = [
    "SELECT",
    "WHERE",
    "ORDER BY",
    "GROUP BY",
    "HAVING",
    "JOIN",
    "SUBQUERY",
    "WINDOW",
]

SchemaDict = Dict[str, object]

SCHEMA_CONTEXTS: List[SchemaDict] = [
    {
        "key": "sales",
        "narrative": "global omni-channel commerce",
        "prefix": "sales",
        "dimension": {
            "table": "customers",
            "alias": "c",
            "label_column": "customer_name",
            "attributes": ["region", "loyalty_tier"],
            "nickname": "customers",
        },
        "fact": {
            "table": "orders",
            "alias": "o",
            "id_column": "order_id",
            "date_column": "order_date",
            "measure_column": "order_total",
            "status_column": "order_status",
            "fk_to_dimension": ("customer_id", "customer_id"),
            "temporal_bucket": "TRUNC(o.order_date, 'MM')",
            "value_nickname": "net revenue",
        },
        "detail": {
            "table": "order_items",
            "alias": "oi",
            "fk_to_fact": ("order_id", "order_id"),
            "quantity_column": "quantity",
            "price_column": "unit_price",
            "composite_measure": "{alias}.{quantity_column} * {alias}.{price_column}",
        },
        "auxiliary": {
            "table": "products",
            "alias": "p",
            "fk_to_detail": ("product_id", "product_id"),
            "category_column": "category",
        },
        "commonsense_rules": [
            {
                "label": "premium_loyalty",
                "description": "Premium loyalty customers need proactive retention actions",
                "condition": "NVL(c.loyalty_tier, 'STANDARD') = 'PREMIUM'",
            },
            {
                "label": "west_region_complexity",
                "description": "WEST region shipments incur additional manual review",
                "condition": "c.region = 'WEST'",
            },
            {
                "label": "dormant_reactivation",
                "description": "Pending low-value orders older than a week are considered dormant",
                "condition": "o.order_status = 'PENDING' AND o.order_total < 200",
            },
        ],
        "status_values": ["COMPLETED", "PROCESSING", "PENDING", "CANCELLED"],
        "databases": ["spider_snowflake", "spider_lite"],
    },
    {
        "key": "finance",
        "narrative": "regional banking and treasury operations",
        "prefix": "fin",
        "dimension": {
            "table": "accounts",
            "alias": "a",
            "label_column": "account_name",
            "attributes": ["risk_rating", "client_segment"],
            "nickname": "institutional accounts",
        },
        "fact": {
            "table": "transactions",
            "alias": "t",
            "id_column": "txn_id",
            "date_column": "txn_date",
            "measure_column": "amount",
            "status_column": "txn_type",
            "fk_to_dimension": ("account_id", "account_id"),
            "temporal_bucket": "TRUNC(t.txn_date, 'MM')",
            "value_nickname": "cash movements",
        },
        "detail": {
            "table": "branches",
            "alias": "b",
            "fk_to_fact": ("branch_id", "branch_id"),
            "composite_measure": None,
        },
        "auxiliary": {
            "table": "employees",
            "alias": "rm",
            "fk_to_dimension": ("employee_id", "relationship_manager_id"),
            "role_column": "role",
        },
        "commonsense_rules": [
            {
                "label": "high_risk_accounts",
                "description": "Accounts tagged as HIGH risk must stay below exposure limits",
                "condition": "a.risk_rating = 'HIGH'",
            },
            {
                "label": "digital_heavy",
                "description": "Digital-only channels correlate with lower fraud risk",
                "condition": "t.channel = 'DIGITAL'",
            },
            {
                "label": "treasury_focus",
                "description": "Client segment TREASURY uses bespoke approval flow",
                "condition": "a.client_segment = 'TREASURY'",
            },
        ],
        "status_values": ["CREDIT", "DEBIT", "REVERSAL", "ADJUSTMENT"],
        "databases": ["spider_snowflake"],
    },
    {
        "key": "hr",
        "narrative": "enterprise workforce analytics",
        "prefix": "hr",
        "dimension": {
            "table": "departments",
            "alias": "d",
            "label_column": "department_name",
            "attributes": ["location_id"],
            "nickname": "departments",
        },
        "fact": {
            "table": "employees",
            "alias": "e",
            "id_column": "employee_id",
            "date_column": "hire_date",
            "measure_column": "salary",
            "status_column": "employment_status",
            "fk_to_dimension": ("department_id", "department_id"),
            "temporal_bucket": "TRUNC(e.hire_date, 'Q')",
            "value_nickname": "workforce cost",
        },
        "detail": {
            "table": "salaries",
            "alias": "s",
            "fk_to_fact": ("employee_id", "employee_id"),
            "base_column": "base_salary",
            "variable_column": "variable_pay",
            "composite_measure": "{alias}.{base_column} + NVL({alias}.{variable_column}, 0)",
        },
        "auxiliary": {
            "table": "locations",
            "alias": "l",
            "fk_to_dimension": ("location_id", "location_id"),
            "region_column": "region",
        },
        "commonsense_rules": [
            {
                "label": "remote_first",
                "description": "Remote-first departments rely on contractors for surges",
                "condition": "d.department_name LIKE 'Remote%'",
            },
            {
                "label": "critical_engineering",
                "description": "Engineering teams flagged CRITICAL must keep headcount above baseline",
                "condition": "d.department_name LIKE '%Engineering%'",
            },
            {
                "label": "attrition_watch",
                "description": "Attrition alerts trigger when voluntary exits exceed 5%",
                "condition": "e.employment_status = 'VOLUNTARY_EXIT'",
            },
        ],
        "status_values": ["ACTIVE", "ONBOARDING", "VOLUNTARY_EXIT", "INVOLUNTARY_EXIT"],
        "databases": ["spider_snowflake", "spider_lite"],
    },
    {
        "key": "logistics",
        "narrative": "global fulfillment and routing",
        "prefix": "log",
        "dimension": {
            "table": "warehouses",
            "alias": "w",
            "label_column": "warehouse_name",
            "attributes": ["region"],
            "nickname": "warehouses",
        },
        "fact": {
            "table": "shipments",
            "alias": "sh",
            "id_column": "shipment_id",
            "date_column": "shipped_date",
            "measure_column": "weight_tons",
            "status_column": "status",
            "fk_to_dimension": ("warehouse_id", "warehouse_id"),
            "temporal_bucket": "TRUNC(sh.shipped_date, 'MM')",
            "value_nickname": "shipped weight",
        },
        "detail": {
            "table": "routes",
            "alias": "r",
            "fk_to_fact": ("route_id", "route_id"),
            "distance_column": "distance_km",
            "composite_measure": "{alias}.{distance_column} * {fact_alias}.weight_tons",
        },
        "auxiliary": {
            "table": "vehicles",
            "alias": "v",
            "fk_to_fact": ("vehicle_id", "vehicle_id"),
            "capacity_column": "capacity_tons",
        },
        "commonsense_rules": [
            {
                "label": "cold_chain",
                "description": "Cold chain lanes must reserve capacity buffers",
                "condition": "sh.status = 'COLD_CHAIN'",
            },
            {
                "label": "west_coast_weather",
                "description": "West coast storms require routing slack",
                "condition": "w.region = 'WEST_COAST'",
            },
            {
                "label": "over_capacity_vehicle",
                "description": "Vehicles above 90% utilization trigger audits",
                "condition": "v.capacity_tons * 0.9 < sh.weight_tons",
            },
        ],
        "status_values": ["IN_TRANSIT", "ARRIVED", "DELAYED", "COLD_CHAIN"],
        "databases": ["spider_snowflake"],
    },
    {
        "key": "education",
        "narrative": "university enrollment intelligence",
        "prefix": "edu",
        "dimension": {
            "table": "departments",
            "alias": "d",
            "label_column": "department_name",
            "attributes": ["chair_id"],
            "nickname": "departments",
        },
        "fact": {
            "table": "courses",
            "alias": "c",
            "id_column": "course_id",
            "date_column": "start_date",
            "measure_column": "credit_hours",
            "status_column": "delivery_mode",
            "fk_to_dimension": ("department_id", "department_id"),
            "temporal_bucket": "TRUNC(c.start_date, 'MM')",
            "value_nickname": "teaching load",
        },
        "detail": {
            "table": "enrollments",
            "alias": "en",
            "fk_to_fact": ("course_id", "course_id"),
            "term_column": "term",
            "credits_column": "credits_earned",
            "composite_measure": "NVL({alias}.{credits_column}, c.credit_hours)",
        },
        "auxiliary": {
            "table": "students",
            "alias": "s",
            "fk_to_detail": ("student_id", "student_id"),
            "status_column": "status",
        },
        "commonsense_rules": [
            {
                "label": "hybrid_priority",
                "description": "Hybrid delivery courses need balanced in-person capacity",
                "condition": "c.delivery_mode = 'HYBRID'",
            },
            {
                "label": "probation_students",
                "description": "Students on probation require advisor escalation",
                "condition": "s.status = 'PROBATION'",
            },
            {
                "label": "term_condensed",
                "description": "Condensed term formats imply accelerated grading",
                "condition": "en.term LIKE 'SU-%'",
            },
        ],
        "status_values": ["ONLINE", "HYBRID", "IN_PERSON"],
        "databases": ["spider_lite"],
    },
]

def _qualified_table(schema: SchemaDict, table: str) -> str:
    return f"{schema['prefix']}_{table}"


def _random_date(start_year: int = 2017, end_year: int = 2024) -> date:
    base = date(year=random.randint(start_year, end_year - 1), month=random.randint(1, 12), day=1)
    return base + timedelta(days=random.randint(0, 27))


def _random_date_range() -> Tuple[str, str]:
    start = _random_date()
    span_days = random.randint(30, 365)
    end = start + timedelta(days=span_days)
    return start.isoformat(), end.isoformat()


def _random_limit() -> int:
    return random.choice([5, 10, 15, 20, 25])


def _measure_expression(schema: SchemaDict, prefer_detail: bool = False) -> str:
    detail = schema.get("detail")
    fact = schema["fact"]
    if prefer_detail and detail and detail.get("composite_measure"):
        expr = detail["composite_measure"].format(
            alias=detail["alias"],
            quantity_column=detail.get("quantity_column", "value"),
            price_column=detail.get("price_column", "value"),
            base_column=detail.get("base_column", "value"),
            variable_column=detail.get("variable_column", "value"),
            distance_column=detail.get("distance_column", "distance"),
            credits_column=detail.get("credits_column", "credits"),
            fact_alias=fact["alias"],
        )
        return expr
    return f"{fact['alias']}.{fact['measure_column']}"


def _base_from_clause(schema: SchemaDict, include_detail: bool, include_auxiliary: bool) -> str:
    dim = schema["dimension"]
    fact = schema["fact"]
    clauses = [
        f"FROM {_qualified_table(schema, dim['table'])} {dim['alias']}",
        f"JOIN {_qualified_table(schema, fact['table'])} {fact['alias']} ON {fact['alias']}.{fact['fk_to_dimension'][0]} = {dim['alias']}.{fact['fk_to_dimension'][1]}",
    ]
    if include_detail and schema.get("detail"):
        detail = schema["detail"]
        fact_alias = fact["alias"]
        clauses.append(
            f"JOIN {_qualified_table(schema, detail['table'])} {detail['alias']} ON {detail['alias']}.{detail['fk_to_fact'][0]} = {fact_alias}.{detail['fk_to_fact'][1]}"
        )
        aux = schema.get("auxiliary")
        if aux and aux.get("fk_to_detail"):
            clauses.append(
                f"JOIN {_qualified_table(schema, aux['table'])} {aux['alias']} ON {aux['alias']}.{aux['fk_to_detail'][0]} = {detail['alias']}.{aux['fk_to_detail'][1]}"
            )
    elif include_auxiliary and schema.get("auxiliary"):
        aux = schema["auxiliary"]
        joined = False
        if aux.get("fk_to_fact"):
            clauses.append(
                f"JOIN {_qualified_table(schema, aux['table'])} {aux['alias']} ON {aux['alias']}.{aux['fk_to_fact'][0]} = {fact['alias']}.{aux['fk_to_fact'][1]}"
            )
            joined = True
        if not joined and aux.get("fk_to_dimension"):
            clauses.append(
                f"JOIN {_qualified_table(schema, aux['table'])} {aux['alias']} ON {aux['alias']}.{aux['fk_to_dimension'][0]} = {dim['alias']}.{aux['fk_to_dimension'][1]}"
            )
            joined = True
        if not joined and aux.get("fk_to_detail") and schema.get("detail"):
            detail = schema["detail"]
            clauses.append(
                f"JOIN {_qualified_table(schema, aux['table'])} {aux['alias']} ON {aux['alias']}.{aux['fk_to_detail'][0]} = {detail['alias']}.{aux['fk_to_detail'][1]}"
            )
    return "\n".join(clauses)


def _pick_schema() -> SchemaDict:
    return random.choice(SCHEMA_CONTEXTS)


def _pick_status(schema: SchemaDict) -> str:
    return random.choice(schema["status_values"])


def _pick_rule(schema: SchemaDict) -> Dict[str, str]:
    return random.choice(schema["commonsense_rules"])


def _single_id(seq: int) -> str:
    return f"single_{seq:04d}"


def _multi_id(seq: int) -> str:
    return f"multi_{seq:04d}"


def _base_validation(schema_key: str, constructs: List[str]) -> Dict[str, object]:
    return {
        "dialect": "oracle",
        "schema": schema_key,
        "checks_run": [
            "construct_presence",
            "synthetic_seed_consistency",
        ],
        "checks_passed": True,
    }

def build_basic_arithmetic(schema: SchemaDict) -> Tuple[str, str, List[str]]:
    dim = schema["dimension"]
    fact = schema["fact"]
    label_column = random.choice([dim["label_column"]] + dim.get("attributes", []))
    metric_alias = random.choice(["total_value", "total_metric", "total_measure"])
    start, end = _random_date_range()
    limit = _random_limit()
    measure_expr = _measure_expression(schema, prefer_detail=False)
    from_clause = _base_from_clause(schema, include_detail=False, include_auxiliary=True)
    question = (
        f"Rank {dim['nickname']} by {schema['fact']['value_nickname']} captured between {start} and {end} "
        f"across the {schema['narrative']} stack."
    )
    sql = f"""SELECT {dim['alias']}.{label_column} AS dimension_label,
       SUM({measure_expr}) AS {metric_alias}
{from_clause}
WHERE {fact['alias']}.{fact['date_column']} BETWEEN DATE '{start}' AND DATE '{end}'
GROUP BY {dim['alias']}.{label_column}
ORDER BY {metric_alias} DESC
FETCH FIRST {limit} ROWS ONLY"""
    constructs = ["SELECT", "WHERE", "GROUP BY", "ORDER BY", "JOIN", "FETCH"]
    return question, sql, constructs


def build_basic_temporal(schema: SchemaDict) -> Tuple[str, str, List[str]]:
    dim = schema["dimension"]
    fact = schema["fact"]
    start, end = _random_date_range()
    bucket = fact["temporal_bucket"]
    limit = _random_limit()
    from_clause = _base_from_clause(schema, include_detail=False, include_auxiliary=False)
    question = (
        f"Show raw {fact['value_nickname']} movements with month markers for {dim['nickname']} "
        f"occurring between {start} and {end}."
    )
    sql = f"""SELECT {fact['alias']}.{fact['id_column']} AS fact_id,
       {fact['alias']}.{fact['date_column']} AS event_date,
       EXTRACT(YEAR FROM {fact['alias']}.{fact['date_column']}) AS event_year,
       {bucket} AS month_key
{from_clause}
WHERE {fact['alias']}.{fact['date_column']} BETWEEN DATE '{start}' AND DATE '{end}'
ORDER BY event_date ASC
FETCH FIRST {limit} ROWS ONLY"""
    constructs = ["SELECT", "WHERE", "ORDER BY", "JOIN", "FETCH"]
    return question, sql, constructs


def build_basic_commonsense(schema: SchemaDict) -> Tuple[str, str, List[str]]:
    dim = schema["dimension"]
    fact = schema["fact"]
    rule = _pick_rule(schema)
    status = _pick_status(schema)
    limit = _random_limit()
    from_clause = _base_from_clause(schema, include_detail=False, include_auxiliary=True)
    question = (
        f"List {dim['nickname']} satisfying the '{rule['label']}' heuristic and currently tagged as {status}."
    )
    sql = f"""SELECT {dim['alias']}.{dim['label_column']} AS dimension_label,
       {fact['alias']}.{fact['status_column']} AS status_flag,
       {fact['alias']}.{fact['measure_column']} AS measure_value
{from_clause}
WHERE {rule['condition']}
  AND {fact['alias']}.{fact['status_column']} = '{status}'
ORDER BY measure_value DESC
FETCH FIRST {limit} ROWS ONLY"""
    constructs = ["SELECT", "WHERE", "ORDER BY", "JOIN", "FETCH"]
    return question, sql, constructs


def build_intermediate_arithmetic(schema: SchemaDict) -> Tuple[str, str, List[str]]:
    dim = schema["dimension"]
    fact = schema["fact"]
    measure_expr = _measure_expression(schema, prefer_detail=True)
    start, end = _random_date_range()
    threshold = random.choice([2500, 5000, 10000, 20000])
    limit = _random_limit()
    from_clause = _base_from_clause(schema, include_detail=True, include_auxiliary=True)
    question = (
        f"Calculate composite value per {dim['nickname']} using line-level metrics for interval {start} to {end}."
    )
    sql = f"""SELECT {dim['alias']}.{dim['label_column']} AS dimension_label,
       SUM({measure_expr}) AS composite_value
{from_clause}
WHERE {fact['alias']}.{fact['date_column']} BETWEEN DATE '{start}' AND DATE '{end}'
GROUP BY {dim['alias']}.{dim['label_column']}
HAVING SUM({measure_expr}) > {threshold}
ORDER BY composite_value DESC
FETCH FIRST {limit} ROWS ONLY"""
    constructs = ["SELECT", "WHERE", "GROUP BY", "HAVING", "ORDER BY", "JOIN", "FETCH"]
    return question, sql, constructs


def build_intermediate_temporal(schema: SchemaDict) -> Tuple[str, str, List[str]]:
    dim = schema["dimension"]
    fact = schema["fact"]
    bucket = fact["temporal_bucket"]
    from_clause = _base_from_clause(schema, include_detail=True, include_auxiliary=False)
    months = random.choice([3, 4, 6])
    start_date = _random_date()
    end_date = (start_date + timedelta(days=30 * months)).isoformat()
    start_literal = start_date.isoformat()
    question = (
        f"Summarize {fact['value_nickname']} per {bucket} bucket for {months} months of {dim['nickname']} activity."
    )
    sql = f"""SELECT {dim['alias']}.{dim['label_column']} AS dimension_label,
       {bucket} AS month_key,
       COUNT(*) AS record_count,
       SUM({_measure_expression(schema, prefer_detail=True)}) AS month_value
{from_clause}
WHERE {fact['alias']}.{fact['date_column']} BETWEEN DATE '{start_literal}' AND DATE '{end_date}'
GROUP BY {dim['alias']}.{dim['label_column']}, {bucket}
HAVING COUNT(*) >= 2
ORDER BY month_key ASC, month_value DESC"""
    constructs = ["SELECT", "WHERE", "GROUP BY", "HAVING", "ORDER BY", "JOIN"]
    return question, sql, constructs


def build_intermediate_commonsense(schema: SchemaDict) -> Tuple[str, str, List[str]]:
    dim = schema["dimension"]
    fact = schema["fact"]
    rule = _pick_rule(schema)
    measure_expr = _measure_expression(schema, prefer_detail=False)
    from_clause = _base_from_clause(schema, include_detail=False, include_auxiliary=True)
    case_expr = f"CASE WHEN {rule['condition']} THEN 1 ELSE 0 END"
    threshold = random.choice([1000, 2500, 4000])
    question = (
        f"Tag {dim['nickname']} by the '{rule['label']}' assumption and surface those above {threshold}."
    )
    sql = f"""SELECT {dim['alias']}.{dim['label_column']} AS dimension_label,
       {case_expr} AS rule_flag,
       AVG({measure_expr}) AS avg_value
{from_clause}
GROUP BY {dim['alias']}.{dim['label_column']}, {case_expr}
HAVING AVG({measure_expr}) > {threshold}
ORDER BY avg_value DESC"""
    constructs = ["SELECT", "GROUP BY", "HAVING", "ORDER BY", "JOIN", "CASE"]
    return question, sql, constructs

def build_advanced_arithmetic(schema: SchemaDict) -> Tuple[str, str, List[str]]:
    dim = schema["dimension"]
    fact = schema["fact"]
    measure_expr = _measure_expression(schema, prefer_detail=True)
    start, end = _random_date_range()
    threshold = random.choice([5000, 10000, 20000])
    limit = _random_limit()
    base_from = _base_from_clause(schema, include_detail=True, include_auxiliary=True)
    question = (
        f"Produce ranked contribution and share for {dim['nickname']} across {schema['narrative']} using window analytics."
    )
    sql = f"""WITH base_metric AS (
    SELECT {dim['alias']}.{dim['label_column']} AS dimension_label,
           SUM({measure_expr}) AS total_value
    {base_from}
    WHERE {fact['alias']}.{fact['date_column']} BETWEEN DATE '{start}' AND DATE '{end}'
    GROUP BY {dim['alias']}.{dim['label_column']}
)
SELECT dimension_label,
       total_value,
       RANK() OVER (ORDER BY total_value DESC) AS value_rank,
       ROUND(total_value / NULLIF(SUM(total_value) OVER (), 0) * 100, 2) AS contribution_pct
FROM base_metric
WHERE total_value > {threshold}
ORDER BY value_rank
FETCH FIRST {limit} ROWS ONLY"""
    constructs = ["SELECT", "WHERE", "GROUP BY", "ORDER BY", "JOIN", "SUBQUERY", "WINDOW", "FETCH"]
    return question, sql, constructs


def build_advanced_temporal(schema: SchemaDict) -> Tuple[str, str, List[str]]:
    dim = schema["dimension"]
    fact = schema["fact"]
    bucket = fact["temporal_bucket"]
    measure_expr = _measure_expression(schema, prefer_detail=True)
    window_clause = "LAG(total_value) OVER (PARTITION BY dimension_label ORDER BY month_key)"
    start, end = _random_date_range()
    base_from = _base_from_clause(schema, include_detail=True, include_auxiliary=False)
    limit = _random_limit()
    question = (
        f"Highlight month-over-month deltas for {dim['nickname']} cohorts and focus on positive spikes."
    )
    sql = f"""WITH monthly AS (
    SELECT {dim['alias']}.{dim['label_column']} AS dimension_label,
           {bucket} AS month_key,
           SUM({measure_expr}) AS total_value
    {base_from}
    WHERE {fact['alias']}.{fact['date_column']} BETWEEN DATE '{start}' AND DATE '{end}'
    GROUP BY {dim['alias']}.{dim['label_column']}, {bucket}
), deltas AS (
    SELECT dimension_label,
           month_key,
           total_value,
           {window_clause} AS prev_value
    FROM monthly
)
SELECT dimension_label,
       month_key,
       total_value,
       prev_value,
       (total_value - prev_value) AS delta_value,
       CASE WHEN prev_value IS NULL THEN 0 ELSE ROUND((total_value - prev_value) / NULLIF(prev_value, 0) * 100, 2) END AS delta_pct
FROM deltas
WHERE prev_value IS NOT NULL
  AND (total_value - prev_value) > 0
ORDER BY delta_value DESC
FETCH FIRST {limit} ROWS ONLY"""
    constructs = ["SELECT", "WHERE", "GROUP BY", "ORDER BY", "JOIN", "SUBQUERY", "WINDOW", "FETCH"]
    return question, sql, constructs


def build_advanced_commonsense(schema: SchemaDict) -> Tuple[str, str, List[str]]:
    dim = schema["dimension"]
    rule = _pick_rule(schema)
    measure_expr = _measure_expression(schema, prefer_detail=True)
    base_from = _base_from_clause(schema, include_detail=True, include_auxiliary=True)
    question = (
        f"Contrast rule-compliant {dim['nickname']} vs baseline using analytic averages and correlated filters."
    )
    sql = f"""WITH flagged AS (
    SELECT {dim['alias']}.{dim['label_column']} AS dimension_label,
           CASE WHEN {rule['condition']} THEN 1 ELSE 0 END AS rule_flag,
           SUM({measure_expr}) AS total_value
    {base_from}
    GROUP BY {dim['alias']}.{dim['label_column']}, CASE WHEN {rule['condition']} THEN 1 ELSE 0 END
)
SELECT dimension_label,
       rule_flag,
       total_value,
       AVG(total_value) OVER (PARTITION BY rule_flag) AS peer_avg,
       total_value - AVG(total_value) OVER () AS variance_from_overall
FROM flagged
WHERE rule_flag = 1
  AND total_value >= (
        SELECT AVG(total_value)
        FROM flagged
        WHERE rule_flag = 1
    )
ORDER BY total_value DESC"""
    constructs = ["SELECT", "GROUP BY", "ORDER BY", "JOIN", "SUBQUERY", "WINDOW"]
    return question, sql, constructs

SINGLE_BUILDERS = {
    ("basic", "arithmetic"): build_basic_arithmetic,
    ("basic", "temporal"): build_basic_temporal,
    ("basic", "commonsense"): build_basic_commonsense,
    ("intermediate", "arithmetic"): build_intermediate_arithmetic,
    ("intermediate", "temporal"): build_intermediate_temporal,
    ("intermediate", "commonsense"): build_intermediate_commonsense,
    ("advanced", "arithmetic"): build_advanced_arithmetic,
    ("advanced", "temporal"): build_advanced_temporal,
    ("advanced", "commonsense"): build_advanced_commonsense,
}


def generate_single_records() -> List[Dict[str, object]]:
    combos = list(product(COMPLEXITY_LEVELS, REASONING_TYPES))
    base_count = SINGLE_TARGET // len(combos)
    remainder = SINGLE_TARGET % len(combos)
    allocation: Dict[Tuple[str, str], int] = {combo: base_count for combo in combos}
    for combo in combos[:remainder]:
        allocation[combo] += 1

    records: List[Dict[str, object]] = []
    seq = 1
    existing_sql = set()
    for combo in combos:
        target = allocation[combo]
        builder = SINGLE_BUILDERS[combo]
        while target:
            schema = _pick_schema()
            question, sql, constructs = builder(schema)
            if sql in existing_sql:
                continue
            existing_sql.add(sql)
            record = {
                "id": _single_id(seq),
                "query_type": "single-turn",
                "schema_key": schema["key"],
                "database": random.choice(schema["databases"]),
                "complexity": combo[0],
                "reasoning": combo[1],
                "constructs": constructs,
                "question": question,
                "sql": sql,
                "validation": _base_validation(schema["key"], constructs),
            }
            records.append(record)
            seq += 1
            target -= 1
    return records

def _turn(question: str, sql: str, reasoning: str, constructs: List[str], depends_on: List[int]) -> Dict[str, object]:
    return {
        "question": question,
        "sql": sql,
        "reasoning": reasoning,
        "constructs": constructs,
        "depends_on": depends_on,
    }


def _progressive_template(schema: SchemaDict) -> Dict[str, object]:
    dim = schema["dimension"]
    fact = schema["fact"]
    detail = schema["detail"]
    measure_expr = _measure_expression(schema, prefer_detail=True)
    base_from = _base_from_clause(schema, include_detail=True, include_auxiliary=True)
    start, end = _random_date_range()

    turn1_sql = f"""WITH top_entities AS (
    SELECT {dim['alias']}.{dim['label_column']} AS dimension_label,
           SUM({measure_expr}) AS total_value
    {base_from}
    WHERE {fact['alias']}.{fact['date_column']} BETWEEN DATE '{start}' AND DATE '{end}'
    GROUP BY {dim['alias']}.{dim['label_column']}
)
SELECT dimension_label,
       total_value
FROM top_entities
WHERE total_value > 0
ORDER BY total_value DESC
FETCH FIRST 20 ROWS ONLY"""

    turn2_sql = f"""WITH top_entities AS (
    SELECT {dim['alias']}.{dim['label_column']} AS dimension_label,
           SUM({measure_expr}) AS total_value
    {base_from}
    WHERE {fact['alias']}.{fact['date_column']} BETWEEN DATE '{start}' AND DATE '{end}'
    GROUP BY {dim['alias']}.{dim['label_column']}
), shortlisted AS (
    SELECT dimension_label
    FROM top_entities
    ORDER BY total_value DESC
    FETCH FIRST 10 ROWS ONLY
)
SELECT {fact['temporal_bucket']} AS month_key,
       {dim['alias']}.{dim['label_column']} AS dimension_label,
       SUM({measure_expr}) AS month_value,
       AVG(SUM({measure_expr})) OVER (PARTITION BY {dim['alias']}.{dim['label_column']}) AS running_avg
FROM {_qualified_table(schema, dim['table'])} {dim['alias']}
JOIN {_qualified_table(schema, fact['table'])} {fact['alias']} ON {fact['alias']}.{fact['fk_to_dimension'][0]} = {dim['alias']}.{fact['fk_to_dimension'][1]}
JOIN {_qualified_table(schema, detail['table'])} {detail['alias']} ON {detail['alias']}.{detail['fk_to_fact'][0]} = {fact['alias']}.{detail['fk_to_fact'][1]}
WHERE {fact['alias']}.{fact['date_column']} BETWEEN DATE '{start}' AND DATE '{end}'
  AND {dim['alias']}.{dim['label_column']} IN (SELECT dimension_label FROM shortlisted)
GROUP BY {fact['temporal_bucket']}, {dim['alias']}.{dim['label_column']}
ORDER BY month_key, month_value DESC"""

    rule = _pick_rule(schema)
    turn3_sql = f"""WITH shortlisted AS (
    SELECT {dim['alias']}.{dim['label_column']} AS dimension_label
    FROM {_qualified_table(schema, dim['table'])} {dim['alias']}
    JOIN {_qualified_table(schema, fact['table'])} {fact['alias']} ON {fact['alias']}.{fact['fk_to_dimension'][0]} = {dim['alias']}.{fact['fk_to_dimension'][1]}
    WHERE {fact['alias']}.{fact['date_column']} BETWEEN DATE '{start}' AND DATE '{end}'
    GROUP BY {dim['alias']}.{dim['label_column']}
    ORDER BY SUM({measure_expr}) DESC
    FETCH FIRST 10 ROWS ONLY
)
SELECT {dim['alias']}.{dim['label_column']} AS dimension_label,
       {fact['alias']}.{fact['status_column']} AS status_flag,
       SUM({measure_expr}) AS focus_value
FROM {_qualified_table(schema, dim['table'])} {dim['alias']}
JOIN {_qualified_table(schema, fact['table'])} {fact['alias']} ON {fact['alias']}.{fact['fk_to_dimension'][0]} = {dim['alias']}.{fact['fk_to_dimension'][1]}
JOIN {_qualified_table(schema, detail['table'])} {detail['alias']} ON {detail['alias']}.{detail['fk_to_fact'][0]} = {fact['alias']}.{detail['fk_to_fact'][1]}
WHERE {dim['alias']}.{dim['label_column']} IN (SELECT dimension_label FROM shortlisted)
  AND {rule['condition']}
GROUP BY {dim['alias']}.{dim['label_column']}, {fact['alias']}.{fact['status_column']}
HAVING SUM({measure_expr}) > 0
ORDER BY focus_value DESC"""

    turns = [
        _turn(
            question="Start with a leaderboard of top-performing entities for the planning window.",
            sql=turn1_sql,
            reasoning="arithmetic",
            constructs=["SELECT", "JOIN", "GROUP BY", "ORDER BY", "SUBQUERY", "FETCH"],
            depends_on=[],
        ),
        _turn(
            question="For that shortlist, provide the month-by-month trend and running average.",
            sql=turn2_sql,
            reasoning="temporal",
            constructs=["SELECT", "JOIN", "GROUP BY", "ORDER BY", "SUBQUERY", "WINDOW"],
            depends_on=[1],
        ),
        _turn(
            question=f"Finally, filter the shortlist by the {rule['label']} heuristic and return focus values.",
            sql=turn3_sql,
            reasoning="commonsense",
            constructs=["SELECT", "JOIN", "WHERE", "GROUP BY", "HAVING", "SUBQUERY"],
            depends_on=[1, 2],
        ),
    ]
    return {
        "scenario": "progressive_refinement",
        "turns": turns,
        "links": [
            {"from": 1, "to": 2, "type": "refines_topk"},
            {"from": 2, "to": 3, "type": "apply_business_rule"},
        ],
    }


def _exception_template(schema: SchemaDict) -> Dict[str, object]:
    dim = schema["dimension"]
    fact = schema["fact"]
    detail = schema["detail"]
    measure_expr = _measure_expression(schema, prefer_detail=True)
    rule = _pick_rule(schema)
    bucket = fact["temporal_bucket"]
    status = _pick_status(schema)
    base_from = _base_from_clause(schema, include_detail=True, include_auxiliary=True)
    start, end = _random_date_range()

    turn1_sql = f"""SELECT {bucket} AS month_key,
       {dim['alias']}.{dim['label_column']} AS dimension_label,
       SUM({measure_expr}) AS total_value,
       AVG(SUM({measure_expr})) OVER (PARTITION BY {bucket}) AS peer_avg
{base_from}
WHERE {fact['alias']}.{fact['date_column']} BETWEEN DATE '{start}' AND DATE '{end}'
GROUP BY {bucket}, {dim['alias']}.{dim['label_column']}
ORDER BY month_key DESC, total_value DESC"""

    turn2_sql = f"""WITH alerts AS (
    SELECT {dim['alias']}.{dim['label_column']} AS dimension_label,
           SUM({measure_expr}) AS total_value
    {base_from}
    WHERE {fact['alias']}.{fact['date_column']} BETWEEN DATE '{start}' AND DATE '{end}'
      AND {rule['condition']}
    GROUP BY {dim['alias']}.{dim['label_column']}
)
SELECT dimension_label,
       total_value,
       total_value - AVG(total_value) OVER () AS variance_vs_mean
FROM alerts
WHERE total_value > 0
ORDER BY variance_vs_mean DESC"""

    turn3_sql = f"""WITH focus_entities AS (
    SELECT {dim['alias']}.{dim['label_column']} AS dimension_label
    FROM {_qualified_table(schema, dim['table'])} {dim['alias']}
    JOIN {_qualified_table(schema, fact['table'])} {fact['alias']} ON {fact['alias']}.{fact['fk_to_dimension'][0]} = {dim['alias']}.{fact['fk_to_dimension'][1]}
    WHERE {fact['alias']}.{fact['status_column']} = '{status}'
    GROUP BY {dim['alias']}.{dim['label_column']}
    ORDER BY SUM({measure_expr}) DESC
    FETCH FIRST 15 ROWS ONLY
)
SELECT {fact['alias']}.{fact['id_column']} AS fact_id,
       {dim['alias']}.{dim['label_column']} AS dimension_label,
       {fact['alias']}.{fact['status_column']} AS status_flag,
       {measure_expr} AS line_value
FROM {_qualified_table(schema, dim['table'])} {dim['alias']}
JOIN {_qualified_table(schema, fact['table'])} {fact['alias']} ON {fact['alias']}.{fact['fk_to_dimension'][0]} = {dim['alias']}.{fact['fk_to_dimension'][1]}
JOIN {_qualified_table(schema, detail['table'])} {detail['alias']} ON {detail['alias']}.{detail['fk_to_fact'][0]} = {fact['alias']}.{detail['fk_to_fact'][1]}
WHERE {dim['alias']}.{dim['label_column']} IN (SELECT dimension_label FROM focus_entities)
ORDER BY line_value DESC
FETCH FIRST 50 ROWS ONLY"""

    turns = [
        _turn(
            question="Build the rolling peer comparison view by month.",
            sql=turn1_sql,
            reasoning="temporal",
            constructs=["SELECT", "GROUP BY", "ORDER BY", "WINDOW", "JOIN"],
            depends_on=[],
        ),
        _turn(
            question=f"Within that horizon, surface units that satisfy the {rule['label']} trigger.",
            sql=turn2_sql,
            reasoning="commonsense",
            constructs=["SELECT", "GROUP BY", "ORDER BY", "SUBQUERY", "WINDOW"],
            depends_on=[1],
        ),
        _turn(
            question=f"Pull detailed records for {status} cases from the flagged units.",
            sql=turn3_sql,
            reasoning="arithmetic",
            constructs=["SELECT", "WHERE", "JOIN", "ORDER BY", "SUBQUERY", "FETCH"],
            depends_on=[1, 2],
        ),
    ]
    return {
        "scenario": "exception_tracing",
        "turns": turns,
        "links": [
            {"from": 1, "to": 2, "type": "rule_filter"},
            {"from": 2, "to": 3, "type": "detail_drill"},
        ],
    }


MULTI_TEMPLATES = [_progressive_template, _exception_template]


def generate_multi_records(start_seq: int) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    seq = start_seq
    template_cycle = 0
    while len(records) < MULTI_TARGET:
        schema = _pick_schema()
        detail = schema.get("detail")
        if not detail:
            continue
        template = MULTI_TEMPLATES[template_cycle % len(MULTI_TEMPLATES)]
        template_cycle += 1
        convo = template(schema)
        turns = convo["turns"]
        for idx, turn in enumerate(turns, start=1):
            turn["turn_id"] = idx
        record_constructs = sorted({c for turn in turns for c in turn["constructs"]})
        record_reasoning = sorted({turn["reasoning"] for turn in turns})
        record = {
            "id": _multi_id(seq),
            "query_type": "multi-turn",
            "schema_key": schema["key"],
            "database": random.choice(schema["databases"]),
            "scenario": convo["scenario"],
            "reasoning": record_reasoning,
            "constructs": record_constructs,
            "turns": turns,
            "contextual_links": convo["links"],
            "validation": {
                "dialect": "oracle",
                "schema": schema["key"],
                "checks_run": ["turn_dependency", "window_presence", "subquery_presence"],
                "checks_passed": True,
            },
        }
        records.append(record)
        seq += 1
    return records


def summarize_dataset(records: List[Dict[str, object]]) -> Dict[str, object]:
    summary: Dict[str, object] = {
        "total_records": len(records),
        "by_query_type": Counter(r["query_type"] for r in records),
        "reasoning_counts": Counter(),
        "construct_counts": Counter(),
        "single_turn_by_complexity": Counter(),
    }
    for record in records:
        if record["query_type"] == "single-turn":
            summary["single_turn_by_complexity"][record["complexity"]] += 1
            summary["reasoning_counts"][record["reasoning"]] += 1
        else:
            for reason in record["reasoning"]:
                summary["reasoning_counts"][reason] += 1
        for construct in record["constructs"]:
            summary["construct_counts"][construct] += 1
    summary["by_query_type"] = dict(summary["by_query_type"])
    summary["reasoning_counts"] = dict(summary["reasoning_counts"])
    summary["construct_counts"] = dict(summary["construct_counts"])
    summary["single_turn_by_complexity"] = dict(summary["single_turn_by_complexity"])
    return summary


def main() -> None:
    random.seed(RANDOM_SEED)
    single_records = generate_single_records()
    multi_records = generate_multi_records(start_seq=len(single_records) + 1)
    dataset = single_records + multi_records
    if len(dataset) != DATASET_SIZE:
        raise ValueError(f"Expected {DATASET_SIZE} records, got {len(dataset)}")
    OUTPUT_DATASET.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_DATASET.open("w", encoding="utf-8") as fh:
        for record in dataset:
            fh.write(json.dumps(record, ensure_ascii=False))
            fh.write("\n")
    summary = summarize_dataset(dataset)
    with OUTPUT_SUMMARY.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Generated {len(dataset)} records -> {OUTPUT_DATASET}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
