"""Build the canonical portable technical report artifact from audited CSVs."""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = (
    ROOT
    / "artifacts"
    / "planning_benchmarks"
    / "evolutionary-afl-uav-experiments-analysis-v1"
)
OUTPUT = (
    ROOT
    / "artifacts"
    / "planning_benchmarks"
    / "evolutionary-afl-uav-technical-report-v1"
)


def _rows(name: str) -> list[dict[str, str]]:
    with (ANALYSIS / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(value: str) -> float:
    return float(value)


def _int(value: str) -> int:
    return int(value)


def _arm_index(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["arm_id"]: row for row in rows}


def build() -> Path:
    generated_at = datetime.now(UTC).isoformat()
    arm_rows = _rows("arm_statistics.csv")
    paired_rows = _rows("paired_vs_full.csv")
    rooms_rows = _rows("rooms_maze_comparison.csv")
    operator_rows = _rows("operator_diagnostics.csv")
    sensitivity_rows = _rows("sensitivity_summary.csv")
    arm = _arm_index(arm_rows)
    paired = _arm_index(paired_rows)
    rooms = _arm_index(rooms_rows)

    full = arm["deepseek_v4pro_evo"]
    frozen = arm["deepseek_v4pro_strict"]
    frozen_pair = paired["deepseek_v4pro_strict"]
    ablation_ids = [
        "evo_no_qd_archive",
        "evo_no_crossover",
        "evo_move_only",
        "evo_no_rooms_strategy",
        "evo_fixed_length",
    ]
    ablation_data = [
        {
            "arm_id": arm_id,
            "variant": paired[arm_id]["label"],
            "candidate_minus_full_cost": _float(
                paired[arm_id]["median_candidate_minus_full_cost"]
            ),
            "ci_low": _float(paired[arm_id]["median_delta_ci_low"]),
            "ci_high": _float(paired[arm_id]["median_delta_ci_high"]),
            "candidate_map_wins": _int(paired[arm_id]["candidate_map_wins"]),
            "candidate_map_ties": _int(paired[arm_id]["candidate_map_ties"]),
            "candidate_map_losses": _int(paired[arm_id]["candidate_map_losses"]),
            "trusted_feasible_rate": _float(arm[arm_id]["trusted_feasible_rate"]),
            "effective_timeouts": _int(arm[arm_id]["effective_timeouts"]),
            "median_cost": _float(arm[arm_id]["median_cost"]),
            "maps_with_multiple_paths": _int(
                arm[arm_id]["maps_with_multiple_paths"]
            ),
        }
        for arm_id in ablation_ids
    ]
    rooms_data = [
        {
            "arm_id": arm_id,
            "variant": rooms[arm_id]["label"],
            "candidate_wins": _int(rooms[arm_id]["candidate_wins"]),
            "ties": _int(rooms[arm_id]["ties"]),
            "candidate_losses": _int(rooms[arm_id]["candidate_losses"]),
            "candidate_minus_full_cost": _float(
                rooms[arm_id]["median_candidate_minus_full_cost"]
            ),
            "maps": _int(rooms[arm_id]["rooms_maze_maps"]),
        }
        for arm_id in ablation_ids
    ]

    baseline = {
        "median_cost": _float(full["median_cost"]),
        "trusted_feasible_rate": _float(full["trusted_feasible_rate"]),
        "effective_timeouts": _int(full["effective_timeouts"]),
        "maps_with_multiple_paths": _int(full["maps_with_multiple_paths"]),
    }
    sensitivity_index = _arm_index(sensitivity_rows)
    sensitivity_data = [
        {
            "factor": "种群",
            "setting": "16",
            "arm_id": "evo_population_16",
            "median_cost": _float(arm["evo_population_16"]["median_cost"]),
            "trusted_feasible_rate": _float(arm["evo_population_16"]["trusted_feasible_rate"]),
            "candidate_minus_full_cost": _float(sensitivity_index["evo_population_16"]["median_candidate_minus_full_cost"]),
            "effective_timeouts": _int(arm["evo_population_16"]["effective_timeouts"]),
        },
        {
            "factor": "种群",
            "setting": "24",
            "arm_id": "evo_population_24",
            "median_cost": _float(arm["evo_population_24"]["median_cost"]),
            "trusted_feasible_rate": _float(arm["evo_population_24"]["trusted_feasible_rate"]),
            "candidate_minus_full_cost": _float(sensitivity_index["evo_population_24"]["median_candidate_minus_full_cost"]),
            "effective_timeouts": _int(arm["evo_population_24"]["effective_timeouts"]),
        },
        {
            "factor": "种群",
            "setting": "32（v1）",
            "arm_id": "deepseek_v4pro_evo",
            "median_cost": baseline["median_cost"],
            "trusted_feasible_rate": baseline["trusted_feasible_rate"],
            "candidate_minus_full_cost": 0.0,
            "effective_timeouts": baseline["effective_timeouts"],
        },
        {
            "factor": "档案",
            "setting": "4",
            "arm_id": "evo_archive_4",
            "median_cost": _float(arm["evo_archive_4"]["median_cost"]),
            "trusted_feasible_rate": _float(arm["evo_archive_4"]["trusted_feasible_rate"]),
            "candidate_minus_full_cost": _float(sensitivity_index["evo_archive_4"]["median_candidate_minus_full_cost"]),
            "effective_timeouts": _int(arm["evo_archive_4"]["effective_timeouts"]),
        },
        {
            "factor": "档案",
            "setting": "8（v1）",
            "arm_id": "deepseek_v4pro_evo",
            "median_cost": baseline["median_cost"],
            "trusted_feasible_rate": baseline["trusted_feasible_rate"],
            "candidate_minus_full_cost": 0.0,
            "effective_timeouts": baseline["effective_timeouts"],
        },
        {
            "factor": "档案",
            "setting": "12",
            "arm_id": "evo_archive_12",
            "median_cost": _float(arm["evo_archive_12"]["median_cost"]),
            "trusted_feasible_rate": _float(arm["evo_archive_12"]["trusted_feasible_rate"]),
            "candidate_minus_full_cost": _float(sensitivity_index["evo_archive_12"]["median_candidate_minus_full_cost"]),
            "effective_timeouts": _int(arm["evo_archive_12"]["effective_timeouts"]),
        },
        {
            "factor": "代数",
            "setting": "6",
            "arm_id": "evo_generations_6",
            "median_cost": _float(arm["evo_generations_6"]["median_cost"]),
            "trusted_feasible_rate": _float(arm["evo_generations_6"]["trusted_feasible_rate"]),
            "candidate_minus_full_cost": _float(sensitivity_index["evo_generations_6"]["median_candidate_minus_full_cost"]),
            "effective_timeouts": _int(arm["evo_generations_6"]["effective_timeouts"]),
        },
        {
            "factor": "代数",
            "setting": "12",
            "arm_id": "evo_generations_12",
            "median_cost": _float(arm["evo_generations_12"]["median_cost"]),
            "trusted_feasible_rate": _float(arm["evo_generations_12"]["trusted_feasible_rate"]),
            "candidate_minus_full_cost": _float(sensitivity_index["evo_generations_12"]["median_candidate_minus_full_cost"]),
            "effective_timeouts": _int(arm["evo_generations_12"]["effective_timeouts"]),
        },
        {
            "factor": "代数",
            "setting": "20（v1）",
            "arm_id": "deepseek_v4pro_evo",
            "median_cost": baseline["median_cost"],
            "trusted_feasible_rate": baseline["trusted_feasible_rate"],
            "candidate_minus_full_cost": 0.0,
            "effective_timeouts": baseline["effective_timeouts"],
        },
    ]
    time_data = [
        {
            "time_budget": "0.25 秒",
            "seconds": 0.25,
            "median_cost": _float(arm["evo_time_025"]["median_cost"]),
            "trusted_feasible_rate": _float(arm["evo_time_025"]["trusted_feasible_rate"]),
            "effective_timeouts": _int(arm["evo_time_025"]["effective_timeouts"]),
            "median_objective_evaluations": _float(arm["evo_time_025"]["median_objective_evaluations"]),
        },
        {
            "time_budget": "0.5 秒",
            "seconds": 0.5,
            "median_cost": _float(arm["evo_time_050"]["median_cost"]),
            "trusted_feasible_rate": _float(arm["evo_time_050"]["trusted_feasible_rate"]),
            "effective_timeouts": _int(arm["evo_time_050"]["effective_timeouts"]),
            "median_objective_evaluations": _float(arm["evo_time_050"]["median_objective_evaluations"]),
        },
        {
            "time_budget": "1 秒（v1）",
            "seconds": 1.0,
            "median_cost": baseline["median_cost"],
            "trusted_feasible_rate": baseline["trusted_feasible_rate"],
            "effective_timeouts": baseline["effective_timeouts"],
            "median_objective_evaluations": _float(full["median_objective_evaluations"]),
        },
    ]
    full_operator_data = [
        {
            "operator": row["operator"],
            "attempts": _int(row["attempts"]),
            "structure_changes": _int(row["structure_changes"]),
            "structure_change_rate": _float(row["structure_change_rate"]),
            "accepted_offspring_all_operators": _int(row["accepted_offspring_all_operators"]),
            "interpretation": "结构变化率，不是成本改善率",
        }
        for row in operator_rows
        if row["arm_id"] == "deepseek_v4pro_evo"
    ]
    headline_data = [
        {
            "full_median_cost": baseline["median_cost"],
            "frozen_median_cost": _float(frozen["median_cost"]),
            "map_wins_vs_frozen": _int(frozen_pair["candidate_map_losses"]),
            "map_ties_vs_frozen": _int(frozen_pair["candidate_map_ties"]),
            "map_losses_vs_frozen": _int(frozen_pair["candidate_map_wins"]),
            "maps_with_multiple_paths": baseline["maps_with_multiple_paths"],
            "validation_maps": 60,
            "validation_runs": 300,
        }
    ]

    sources = [
        {
            "id": "source-arm-statistics",
            "label": "Validation arm statistics",
            "path": "artifacts/planning_benchmarks/evolutionary-afl-uav-experiments-analysis-v1/arm_statistics.csv",
            "query": {
                "sql": "SELECT * FROM read_csv_auto('artifacts/planning_benchmarks/evolutionary-afl-uav-experiments-analysis-v1/arm_statistics.csv')",
                "engine": "duckdb",
                "description": "Trusted feasibility, cost, runtime, evaluations, and final-path diversity by arm.",
                "language": "sql",
                "tables_used": ["arm_statistics.csv"],
                "filters": ["split=validation", "effective timeout excludes elapsed >= advertised limit"],
                "metric_definitions": [
                    "median_cost = median trusted total cost over feasible non-timeout runs",
                    "trusted_feasible_rate = feasible non-timeout runs / all runs",
                ],
            },
        },
        {
            "id": "source-paired",
            "label": "Paired map comparison against full v1",
            "path": "artifacts/planning_benchmarks/evolutionary-afl-uav-experiments-analysis-v1/paired_vs_full.csv",
            "query": {
                "sql": "SELECT * FROM read_csv_auto('artifacts/planning_benchmarks/evolutionary-afl-uav-experiments-analysis-v1/paired_vs_full.csv')",
                "engine": "duckdb",
                "description": "Exact-map, shared-seed comparison summarized at the per-map median grain.",
                "language": "sql",
                "tables_used": ["paired_vs_full.csv"],
                "filters": ["60 validation maps", "5 shared seeds per arm", "Test excluded"],
                "metric_definitions": [
                    "candidate_minus_full_cost = candidate per-map median cost - full-v1 per-map median cost",
                    "positive delta means full v1 has lower cost",
                ],
            },
        },
        {
            "id": "source-rooms",
            "label": "rooms_maze paired comparison",
            "path": "artifacts/planning_benchmarks/evolutionary-afl-uav-experiments-analysis-v1/rooms_maze_comparison.csv",
            "query": {
                "sql": "SELECT * FROM read_csv_auto('artifacts/planning_benchmarks/evolutionary-afl-uav-experiments-analysis-v1/rooms_maze_comparison.csv')",
                "engine": "duckdb",
                "description": "The paired comparison restricted to the ten Validation rooms_maze maps.",
                "language": "sql",
                "tables_used": ["rooms_maze_comparison.csv"],
                "filters": ["difficulty=rooms_maze", "10 maps", "5 shared seeds per arm"],
            },
        },
        {
            "id": "source-operators",
            "label": "Evolutionary operator diagnostics",
            "path": "artifacts/planning_benchmarks/evolutionary-afl-uav-experiments-analysis-v1/operator_diagnostics.csv",
            "query": {
                "sql": "SELECT * FROM read_csv_auto('artifacts/planning_benchmarks/evolutionary-afl-uav-experiments-analysis-v1/operator_diagnostics.csv')",
                "engine": "duckdb",
                "description": "Aggregated operator attempts and valid structural changes from planner diagnostics.",
                "language": "sql",
                "tables_used": ["operator_diagnostics.csv"],
                "metric_definitions": [
                    "structure_change_rate = structurally changed proposals / operator attempts",
                    "this is not a per-operation objective improvement rate",
                ],
            },
        },
        {
            "id": "source-ablation",
            "label": "Joined ablation comparison and execution statistics",
            "path": "artifacts/planning_benchmarks/evolutionary-afl-uav-experiments-analysis-v1/paired_vs_full.csv",
            "query": {
                "sql": "WITH p AS (SELECT * FROM read_csv_auto('artifacts/planning_benchmarks/evolutionary-afl-uav-experiments-analysis-v1/paired_vs_full.csv')), a AS (SELECT * FROM read_csv_auto('artifacts/planning_benchmarks/evolutionary-afl-uav-experiments-analysis-v1/arm_statistics.csv')) SELECT p.*, a.trusted_feasible_rate, a.effective_timeouts, a.median_cost, a.maps_with_multiple_paths FROM p JOIN a USING (arm_id) WHERE p.experiment_group = 'ablation'",
                "engine": "duckdb",
                "language": "sql",
                "description": "Paired ablation effects joined to trusted execution statistics.",
                "tables_used": ["paired_vs_full.csv", "arm_statistics.csv"],
                "filters": ["experiment_group=ablation", "split=validation"],
            },
        },
        {
            "id": "source-sensitivity",
            "label": "Joined OFAT sensitivity comparison and execution statistics",
            "path": "artifacts/planning_benchmarks/evolutionary-afl-uav-experiments-analysis-v1/sensitivity_summary.csv",
            "query": {
                "sql": "WITH p AS (SELECT * FROM read_csv_auto('artifacts/planning_benchmarks/evolutionary-afl-uav-experiments-analysis-v1/sensitivity_summary.csv')), a AS (SELECT * FROM read_csv_auto('artifacts/planning_benchmarks/evolutionary-afl-uav-experiments-analysis-v1/arm_statistics.csv')) SELECT p.*, a.trusted_feasible_rate, a.effective_timeouts, a.median_cost FROM p JOIN a USING (arm_id)",
                "engine": "duckdb",
                "language": "sql",
                "description": "OFAT paired effects joined to trusted execution statistics; v1 baselines are reused by the report transformation.",
                "tables_used": ["sensitivity_summary.csv", "arm_statistics.csv"],
                "filters": ["experiment_group=sensitivity", "split=validation"],
            },
        },
        {
            "id": "source-audit",
            "label": "Validation data-quality audit",
            "path": "artifacts/planning_benchmarks/evolutionary-afl-uav-experiments-analysis-v1/validation_audit.json",
            "query": {
                "description": "Record-count, key, budget, path-hash, split, and offline-planning audit.",
                "language": "python",
                "filters": ["Validation only", "zero online LLM calls"],
            },
        },
    ]

    artifact: dict[str, Any] = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "Evolutionary AFL-UAV v1 冻结、消融与敏感性报告",
            "description": "uav2d-v1 Validation-only technical report",
            "generatedAt": generated_at,
            "sources": sources,
            "cards": [
                {
                    "id": "card-full-cost",
                    "dataset": "headline",
                    "sourceId": "source-arm-statistics",
                    "description": "完整 v1 的统一可信总成本中位数；越低越好。",
                    "metrics": [
                        {"label": "完整 v1 成本中位数", "field": "full_median_cost", "format": "number"},
                        {"label": "原始冻结 AFL", "field": "frozen_median_cost", "format": "number"},
                    ],
                },
                {
                    "id": "card-map-wins",
                    "dataset": "headline",
                    "sourceId": "source-paired",
                    "description": "按每张地图的五种子成本中位数比较。",
                    "metrics": [
                        {"label": "相对原始 AFL 胜图", "field": "map_wins_vs_frozen", "format": "number"},
                        {"label": "平", "field": "map_ties_vs_frozen", "format": "number"},
                        {"label": "负", "field": "map_losses_vs_frozen", "format": "number"},
                    ],
                },
                {
                    "id": "card-diversity",
                    "dataset": "headline",
                    "sourceId": "source-arm-statistics",
                    "description": "五个种子产生多个不同最终路径哈希的地图数。",
                    "metrics": [
                        {"label": "有多路径的地图", "field": "maps_with_multiple_paths", "format": "number"},
                        {"label": "Validation 地图", "field": "validation_maps", "format": "number"},
                    ],
                },
            ],
            "charts": [
                {
                    "id": "chart-ablation-delta",
                    "title": "消融臂相对完整 v1 的每图成本中位数差",
                    "subtitle": "60 张 Validation 地图；正值表示完整 v1 成本更低，0 为等效基线",
                    "showDescription": True,
                    "intent": "comparison",
                    "question": "去掉每个机制后，路径成本如何变化？",
                    "rationale": "水平条形图适合比较五个长标签的有符号差值。",
                    "comparisonContext": {"baseline": "完整 Evolutionary AFL-UAV v1", "grain": "per-map median", "unit": "trusted total cost"},
                    "type": "horizontalBar",
                    "dataset": "ablation",
                    "sourceId": "source-ablation",
                    "encodings": {
                        "x": {"field": "variant", "type": "nominal", "label": "消融设置"},
                        "y": {"field": "candidate_minus_full_cost", "type": "quantitative", "label": "候选 - 完整 v1", "format": "number"},
                        "tooltip": [
                            {"field": "candidate_map_wins", "type": "quantitative", "label": "候选胜图"},
                            {"field": "candidate_map_losses", "type": "quantitative", "label": "候选负图"},
                            {"field": "trusted_feasible_rate", "type": "quantitative", "label": "可信可行率", "format": "percent"},
                        ],
                    },
                    "valueFormat": "number",
                    "unit": "cost",
                    "layout": "full",
                    "maxRows": 5,
                    "referenceLines": [{"axis": "y", "value": 0, "label": "完整 v1", "color": "neutral", "lineStyle": "dashed"}],
                    "surface": {"surface": "card", "showControls": False, "viewMode": "both"},
                },
                {
                    "id": "chart-time-cost",
                    "title": "不同时间预算下的路径成本中位数",
                    "subtitle": "60 张 Validation 地图 × 5 种子；0.25 秒含 1 条有效超时",
                    "showDescription": True,
                    "intent": "comparison",
                    "question": "缩短规划时间后成本退化多少？",
                    "rationale": "三个离散预算点使用条形图，不暗示连续时间趋势。",
                    "comparisonContext": {"baseline": "1 second v1", "grain": "run", "unit": "trusted total cost"},
                    "type": "bar",
                    "dataset": "time_sensitivity",
                    "sourceId": "source-arm-statistics",
                    "encodings": {
                        "x": {"field": "time_budget", "type": "ordinal", "label": "时间预算"},
                        "y": {"field": "median_cost", "type": "quantitative", "label": "成本中位数", "format": "number"},
                        "tooltip": [
                            {"field": "trusted_feasible_rate", "type": "quantitative", "label": "可信可行率", "format": "percent"},
                            {"field": "effective_timeouts", "type": "quantitative", "label": "有效超时"},
                            {"field": "median_objective_evaluations", "type": "quantitative", "label": "评价次数中位数"},
                        ],
                    },
                    "valueFormat": "number",
                    "unit": "cost",
                    "layout": "full",
                    "maxRows": 3,
                    "surface": {"surface": "card", "showControls": False, "viewMode": "both"},
                },
            ],
            "tables": [
                {
                    "id": "table-ablation",
                    "title": "消融结果明细",
                    "subtitle": "候选臂相对完整 v1 的 60 图成对比较；正成本差表示完整 v1 更好",
                    "showDescription": True,
                    "dataset": "ablation",
                    "sourceId": "source-ablation",
                    "defaultSort": {"field": "candidate_minus_full_cost", "direction": "desc"},
                    "density": "spacious",
                    "layout": "full",
                    "columns": [
                        {"field": "variant", "label": "消融设置", "type": "text"},
                        {"field": "candidate_map_wins", "label": "候选胜", "format": "number"},
                        {"field": "candidate_map_ties", "label": "平", "format": "number"},
                        {"field": "candidate_map_losses", "label": "候选负", "format": "number"},
                        {"field": "candidate_minus_full_cost", "label": "候选-完整成本", "format": "number", "movement": True},
                        {"field": "trusted_feasible_rate", "label": "可信可行率", "format": "percent"},
                        {"field": "effective_timeouts", "label": "有效超时", "format": "number"},
                    ],
                },
                {
                    "id": "table-rooms",
                    "title": "rooms_maze 消融比较",
                    "subtitle": "10 张 rooms_maze Validation 地图；每图使用五种子中位数",
                    "showDescription": True,
                    "dataset": "rooms",
                    "sourceId": "source-rooms",
                    "defaultSort": {"field": "candidate_minus_full_cost", "direction": "desc"},
                    "density": "spacious",
                    "layout": "full",
                    "columns": [
                        {"field": "variant", "label": "消融设置", "type": "text"},
                        {"field": "candidate_wins", "label": "候选胜", "format": "number"},
                        {"field": "ties", "label": "平", "format": "number"},
                        {"field": "candidate_losses", "label": "候选负", "format": "number"},
                        {"field": "candidate_minus_full_cost", "label": "候选-完整成本", "format": "number", "movement": True},
                    ],
                },
                {
                    "id": "table-sensitivity",
                    "title": "算法参数单因素敏感性",
                    "subtitle": "种群、档案、代数分别只改一个参数；v1 基线复用而不重跑调参",
                    "showDescription": True,
                    "dataset": "sensitivity",
                    "sourceId": "source-sensitivity",
                    "defaultSort": {"field": "factor", "direction": "asc"},
                    "density": "spacious",
                    "layout": "full",
                    "columns": [
                        {"field": "factor", "label": "因素", "type": "text"},
                        {"field": "setting", "label": "设置", "type": "text"},
                        {"field": "median_cost", "label": "运行级成本中位数", "format": "number"},
                        {"field": "candidate_minus_full_cost", "label": "每图中位差", "format": "number", "movement": True},
                        {"field": "trusted_feasible_rate", "label": "可信可行率", "format": "percent"},
                        {"field": "effective_timeouts", "label": "有效超时", "format": "number"},
                    ],
                },
                {
                    "id": "table-operators",
                    "title": "完整 v1 算子有效变更率",
                    "subtitle": "结构发生有效变化的比例；不能解释为成本改善概率",
                    "showDescription": True,
                    "dataset": "operators",
                    "sourceId": "source-operators",
                    "defaultSort": {"field": "structure_change_rate", "direction": "desc"},
                    "density": "spacious",
                    "layout": "full",
                    "columns": [
                        {"field": "operator", "label": "算子", "type": "text"},
                        {"field": "attempts", "label": "尝试", "format": "number"},
                        {"field": "structure_changes", "label": "结构变化", "format": "number"},
                        {"field": "structure_change_rate", "label": "有效变更率", "format": "percent"},
                    ],
                },
            ],
            "blocks": [
                {"id": "title", "type": "markdown", "layout": "full", "body": "# Evolutionary AFL-UAV v1 冻结、消融与敏感性报告"},
                {
                    "id": "technical-summary",
                    "type": "markdown",
                    "layout": "full",
                    "body": (
                        "## 完整 v1 明显优于原始 AFL，但不是每个进化组件都得到单独支持\n\n"
                        "完整 Evolutionary AFL-UAV v1 在 60 张 Validation 地图、每图 5 个共享种子上全部可信可行，成本中位数为 **117.363**；原始冻结 AFL 为 **120.810**。按每图五种子中位数，完整 v1 对原始 AFL 为 **53 胜、7 平、0 负**。\n\n"
                        "消融显示：移除 rooms_maze 专用策略和改成固定长度，在 rooms_maze 上通常变差；但去掉质量—多样性档案、只允许移动或去掉交叉，并未稳定导致更高成本。结论应写成“完整组合有效，但部分组件的独立成本贡献尚不明确”，不能宣称每个算子都被证明必要。"
                    ),
                },
                {"id": "headline-strip", "type": "metric-strip", "layout": "full", "cardIds": ["card-full-cost", "card-map-wins", "card-diversity"]},
                {
                    "id": "ablation-finding",
                    "type": "markdown",
                    "layout": "full",
                    "body": (
                        "## rooms 策略与可变长度有针对性价值，档案和交叉的成本贡献不清晰\n\n"
                        "正的“候选−完整 v1”表示去掉组件后更差。去 rooms 策略在全部地图上更常输（8 胜、23 平、29 负），固定长度为 24 胜、6 平、30 负；两者在 rooms_maze 的成本中位差分别为 **+0.094** 和 **+0.162**。\n\n"
                        "相反，去掉质量—多样性档案和仅移动航点的中位差略为负，且多数置信区间触及 0。它们没有证明完整 v1 更差，但明确说明：当前 Validation 不足以把成本提升归因给档案或交叉。档案仍可能服务于搜索多样性，不过最终五种子路径哈希多样性与完整 v1 接近。"
                    ),
                },
                {"id": "ablation-chart-block", "type": "chart", "layout": "full", "chartId": "chart-ablation-delta"},
                {"id": "ablation-table-block", "type": "table", "layout": "full", "tableId": "table-ablation"},
                {
                    "id": "rooms-finding",
                    "type": "markdown",
                    "layout": "full",
                    "body": (
                        "## rooms_maze 改善仍然存在，但机制证据是局部而非普遍\n\n"
                        "去掉 rooms_maze 专用策略后，候选在 10 张 rooms_maze 地图上为 4 胜、0 平、6 负；固定长度为 2 胜、0 平、8 负。这个方向与设计目标一致：窄门洞需要针对性的局部松弛，可变长度允许在复杂通道增加航点、在开阔处删除冗余航点。\n\n"
                        "不过仅移动、去交叉和去档案在 rooms_maze 的一些地图上反而更好，说明完整策略还存在搜索预算分配和算子干扰，不能把 rooms_maze 的全部收益归因给单一组件。"
                    ),
                },
                {"id": "rooms-table-block", "type": "table", "layout": "full", "tableId": "table-rooms"},
                {
                    "id": "sensitivity-finding",
                    "type": "markdown",
                    "layout": "full",
                    "body": (
                        "## 算法参数附近稳定，时间预算主要改变解质量而非可行性\n\n"
                        "种群 16/24/32、档案 4/8/12、代数 6/12/20 的 9 个 OFAT 设置全部达到 100% 可信可行，运行级成本中位数集中在 **117.303–117.495**。这说明 v1 在所测邻域没有脆弱崩溃，但不代表 32/8/20 是最优参数。\n\n"
                        "预算缩短到 0.5 秒时成本中位数为 **117.998**，0.25 秒时为 **118.314**；后者有 1/300 条有效超时。时间越短，成本温和退化，符合 anytime 搜索预期。"
                    ),
                },
                {"id": "time-chart-block", "type": "chart", "layout": "full", "chartId": "chart-time-cost"},
                {"id": "sensitivity-table-block", "type": "table", "layout": "full", "tableId": "table-sensitivity"},
                {
                    "id": "scope-definitions",
                    "type": "markdown",
                    "layout": "full",
                    "body": (
                        "## 比较口径锁定为 Validation 地图 × 共享 seed\n\n"
                        "数据范围仅为 **uav2d-v1 Validation 60 图**，每个随机臂 5 个共享 seed。可信可行要求：统一硬约束验收通过，且没有达到或越过该臂的墙钟时间边界。最终排名先看可信可行率，再看可信可行路径的统一总成本。\n\n"
                        "成本由统一 `PathEvaluator` 计算，包含长度、风险、平滑度和航点惩罚；实验臂不能用自报成本替代。所有运行最多 2,000 次目标评价。"
                    ),
                },
                {
                    "id": "methodology",
                    "type": "markdown",
                    "layout": "full",
                    "body": (
                        "## 消融采用成对地图比较，敏感性采用单因素变化\n\n"
                        "每个候选臂与冻结完整 v1 按完全相同的 map_id 和 seed 配对；主统计先在每张地图内取五种子成本中位数，再统计 60 图胜/平/负。成本差的 95% 区间使用以 map_id 为簇的 2,000 次 bootstrap。\n\n"
                        "敏感性不做参数搜索：每次只改变种群、档案、代数或时间中的一个因素；32/8/20/1 秒基线直接复用冻结结果，任何敏感性结果都不回写 v1。"
                    ),
                },
                {
                    "id": "operator-finding",
                    "type": "markdown",
                    "layout": "full",
                    "body": (
                        "## 算子计数只能回答“是否产生变化”，质量贡献依赖消融\n\n"
                        "完整 v1 中插入算子的有效变更率为 100%，移动约 96%，交叉约 92%，删除约 69%，交换约 40%。这些计数说明算子实现实际被触发并产生了不同结构，但没有记录每一步对最终成本的边际归因。\n\n"
                        "因此，成本贡献只能结合消融臂解释；当前证据不支持把高有效变更率直接等同于高质量贡献。"
                    ),
                },
                {"id": "operator-table-block", "type": "table", "layout": "full", "tableId": "table-operators"},
                {
                    "id": "limitations",
                    "type": "markdown",
                    "layout": "full",
                    "body": (
                        "## 结论可用于方法说明，但不能替代隐藏终测\n\n"
                        "审计核验了 **4,500** 条记录：记录数和唯一键完整、路径哈希一致、评价次数未超过 2,000，规划阶段在线 LLM 调用为 0。消融历史数据中有 3 条越过 1 秒边界，其中 2 条旧记录仍写为 success；本报告原样保留并统一按有效超时处理。\n\n"
                        "Validation 已被用于方法开发和解释，继续据此修改 v1 会增加选择偏差。因此 artifact 明确禁止冻结后 Validation 调参。Test 未读取，最终泛化结论仍需新的隐藏终测集。"
                    ),
                },
                {
                    "id": "recommended-next",
                    "type": "markdown",
                    "layout": "full",
                    "body": (
                        "## 下一步应先保持 v1 不动，再设计隐藏终测\n\n"
                        "1. 保留完整 v1 作为主方法，同时把交叉与质量—多样性档案的独立贡献写成未证实结果。\n"
                        "2. 不再查看 Validation 调参；另建未参与开发的隐藏 Test-v2，一次性比较完整 v1、原始 AFL、关键消融和传统基线。\n"
                        "3. 若后续研究算子信用分配，应在新版本中增加逐操作成本变化日志，并将其预先注册为 v2 指标，不能回填本次结果。"
                    ),
                },
                {
                    "id": "further-questions",
                    "type": "markdown",
                    "layout": "full",
                    "body": (
                        "## 仍需回答的问题\n\n"
                        "- 质量—多样性档案是否在更长预算、更难地图或跨分布地图上才体现收益？\n"
                        "- 固定长度的劣势是否主要集中在门洞数量、转角数或通道宽度较高的 rooms_maze 子集？\n"
                        "- 交叉是否被移动/插入算子覆盖，还是当前交叉实现缺少适合路径几何的拼接与修复？"
                    ),
                },
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "headline": headline_data,
                "ablation": ablation_data,
                "rooms": rooms_data,
                "sensitivity": sensitivity_data,
                "time_sensitivity": time_data,
                "operators": full_operator_data,
            },
            "accessIssues": [],
        },
        "sources": sources,
    }

    OUTPUT.mkdir(parents=True, exist_ok=True)
    artifact_path = OUTPUT / "artifact.json"
    artifact_path.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    chart_map = {
        "delivery_surface": "portable HTML report",
        "charts": [
            {
                "section": "ablation",
                "question": "cost change after removing each mechanism",
                "family": "comparison and ranking",
                "type": "horizontalBar",
                "fields": ["variant", "candidate_minus_full_cost"],
                "takeaway": "component effects are mixed; rooms/fixed-length are clearest degradations",
                "palette_policy": "single-root with neutral zero reference",
            },
            {
                "section": "time sensitivity",
                "question": "cost change across three discrete time budgets",
                "family": "comparison",
                "type": "bar",
                "fields": ["time_budget", "median_cost"],
                "takeaway": "shorter budgets mildly increase cost",
                "palette_policy": "single-root",
            },
        ],
        "tables": ["ablation exact results", "rooms_maze exact results", "OFAT sensitivity", "operator structure-change diagnostics"],
        "qa_target": "portable report.html desktop and narrow verifier",
    }
    (OUTPUT / "chart_map.json").write_text(
        json.dumps(chart_map, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return artifact_path


if __name__ == "__main__":
    print(build())
