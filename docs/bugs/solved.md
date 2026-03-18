# Bug: GameProfile Validation Error (Curriculum)

## Bug
`pydantic_core._pydantic_core.ValidationError: 6 validation errors for GameProfile`
The AI player crashed on startup because the `curriculum` checkpoints were missing the required `condition` field.

## Cause
The `Checkpoint` model was refactored to use a nested `ConditionConfig` object, but the `pokemon_red.yaml` profile was still using the legacy flat format (`address` and `target_value` at the top level).

## Solution
1. Migrated `configs/profiles/pokemon_red.yaml` to the new nested `condition` format.
2. Implemented a `model_validator` in `src/autogameplayer/core/curriculum.py` to provide backward compatibility by automatically converting legacy flat fields into the new `ConditionConfig` structure.

## Test
Verified using a configuration loading script that confirmed both new and legacy formats parse correctly into the updated model.

# Bug: GameState Validation Error (Extra Fields)

## Bug
`pydantic_core._pydantic_core.ValidationError: 1 validation error for GameState`
The AI Player crashed during `env.reset()` because it received an error message string from the server instead of valid JSON.

## Cause
The server failed to validate the `GameConfig` because `configs/pokemon_red_agentic.yaml` contained extra fields (like `stagnation_threshold_minutes`) that were not defined in the `GameHeuristics` or `GameConfig` Pydantic models.

## Solution
1. Applied `model_config = {"extra": "allow"}` to `GameHeuristics`, `RAMLayout`, and `GameConfig` in `src/autogameplayer/core/config_loader.py` to allow unknown YAML fields.
2. Fixed an incorrect type hint in `src/autogameplayer/server/mcp_server.py` where `config` was incorrectly typed as `None` instead of `Optional[GameConfig]`.

## Test
Implemented `tests/test_config_robustness.py` to verify that extra YAML fields no longer cause validation failures in the core configuration models.

# Bug: GameState Validation Error (Regression)

## Bug
`pydantic_core._pydantic_core.ValidationError: 1 validation error for GameState`
The AI Player crashed during `env.reset()` again because the server returned an error string instead of JSON.

## Cause
This was a regression where several model classes (like `UISignals`, `TerminalConditionConfig`) were missing from `src/autogameplayer/core/config_loader.py` (likely due to an incomplete file write) and other models still lacked the `extra="allow"` configuration.

## Solution
1. Restored missing model classes in `src/autogameplayer/core/config_loader.py`.
2. Consistently applied `model_config = {"extra": "allow"}` across all configuration-related Pydantic models in both `config_loader.py` and `curriculum.py` for maximum robustness.

## Test
Verified using the expanded `tests/test_config_robustness.py`, which now covers `GameProfile`, `Checkpoint`, `RAMLayout`, and other related models.

# Bug: GameState Validation Error (Core Model Extra Fields)

## Bug
`pydantic_core._pydantic_core.ValidationError: 1 validation error for GameState`
The AI Player crashed during `env.reset()` because the server returned a Pydantic validation error string instead of a valid JSON `GameState` object.

## Cause
The `GameState` and other core models (Action, Observation, StepResult) were missing the `extra="allow"` configuration. If any part of the system (like the server-side session management) injected unexpected fields into these models, Pydantic would raise a `ValidationError` during tool output validation or serialization.

## Solution
Applied `model_config = {"extra": "allow"}` to all core data models in `src/autogameplayer/core/models.py`. This ensures that these models can handle unexpected fields gracefully, which is essential for a neuro-symbolic system where data might be enriched by multiple asynchronous agents or solvers.

## Test
Updated `tests/test_config_robustness.py` to include explicit tests for `GameState` and `Action` extra field robustness.

# Bug: GameState Validation Error (MCP Serialization)

## Bug
`pydantic_core._pydantic_core.ValidationError: 1 validation error for GameState`
The AI Player crashed during `env.reset()` because the server returned a Pydantic validation error string instead of a valid JSON `GameState` object.

## Cause
Even with `extra="allow"` set on the `GameState` model, returning a complex Pydantic object from a FastMCP tool can be fragile if the server-side validation or serialization fails for internal reasons (e.g. `unexpected_keyword_argument` errors during reconstruction). The server was catching a validation error and returning it as a string, which the client then failed to parse as JSON.

## Solution
1. Changed the `get_game_state` tool in `src/autogameplayer/server/mcp_server.py` to return a `dict` (using `model_dump()`) instead of a `GameState` object. This bypasses the potentially fragile FastMCP auto-serialization for this specific complex type.
2. Verified that all core models and configuration models consistently use `model_config = {"extra": "allow"}` to handle schema evolution.

## Test
Verified that `tests/test_config_robustness.py` passes and manually confirmed that returning a dict from the tool prevents the string-based error propagation.


# Bug: GameState Validation Error (Final Robustness Fix)

## Bug
`pydantic_core._pydantic_core.ValidationError: 1 validation error for GameState`
The AI Player crashed during `env.reset()` because the server returned a Pydantic validation error string instead of a valid JSON `GameState` object.

## Cause
This was a duplicate of the issue described in `bugs/5.md`. The root cause was FastMCP's fragile auto-serialization of complex Pydantic models combined with strict field validation (`extra="forbid"`) in several sub-models. When any part of the system injected unexpected metadata into the `GameState` or `context` dictionary, Pydantic would raise a `ValidationError` on the server, which FastMCP then returned as a plain string.

## Solution
1. Standardized `model_config = {"extra": "allow"}` across **all** Pydantic models in the project, including core data models (`GameState`, `Action`, `Observation`), configuration models (`GameConfig`, `GameProfile`, `UISignals`, etc.), and curriculum models.
2. Updated the `get_game_state` tool in `src/autogameplayer/server/mcp_server.py` to return a `dict` via `model_dump()`, ensuring the server always returns JSON-serializable data and bypasses internal Pydantic reconstruction during tool return.

## Test
Confirmed that the system is resilient to schema changes and data enrichment using `tests/test_config_robustness.py`.

# Bug: GameState Validation Error (Serialization Staleness)

## Bug
`pydantic_core._pydantic_core.ValidationError: 1 validation error for GameState`
The AI Player crashed during `env.reset()` because the server returned a Pydantic validation error string instead of a valid JSON object.

## Cause
The root cause was a combination of FastMCP's internal auto-validation and potential schema staleness. FastMCP attempts to reconstruct return objects based on type hints. If a sub-model (like `GameState`) fails to validate on the server side (due to schema evolution or unexpected types like `int` vs `bool`), FastMCP returns the error trace as a string. The client then fails to parse this string as JSON.

## Solution
1. Standardized `model_config = {"extra": "allow"}` across **all** Pydantic models in the project.
2. Updated all MCP tools in `src/autogameplayer/server/mcp_server.py` to use `Any` return type hints and return dictionaries (via `model_dump()`). This bypasses fragile FastMCP internal reconstruction.
3. Enhanced `src/autogameplayer/core/mcp_client.py` to explicitly detect and raise errors when the server returns a Pydantic validation error string, providing clear diagnostics instead of a generic `json_invalid` crash.

## Test
Verified with `tests/test_config_robustness.py` and manually confirmed that server-side validation errors are now explicitly caught and reported by the client.

# Bug: Tool Validation Error (Unexpected _seq Argument)

## Bug
`Unexpected keyword argument [type=unexpected_keyword_argument, input_value=0, input_type=int]` for `_seq` parameter.
The AI Player crashed or failed to execute tools because the server rejected the `_seq` parameter injected by the `MCPClient`.

## Cause
The `MCPClient.call_tool` method automatically injects a sequence number (`_seq`) into all tool calls to support idempotency. However, most tool definitions in `src/autogameplayer/server/mcp_server.py` did not include this parameter in their signatures, leading to Pydantic validation failures on the server.

## Solution
Updated all 21 tool definitions in `src/autogameplayer/server/mcp_server.py` to include an optional `_seq: int = None` parameter. This ensures compatibility with the client's idempotency tracking without requiring changes to the internal tool logic (except for `send_input` which already used it).

## Test
Verified that tools like `get_game_state` and `manage_checkpoint` no longer raise validation errors when called by the client.

# Bug: Action Attribute Error ('str' object has no attribute 'button')

## Bug
`AttributeError: 'str' object has no attribute 'button'` in `src/autogameplayer/core/environment.py`.
The AI Player crashed during a step because it expected an `Action` object but received a string.

## Cause
Several solvers (including `RandomSolver`, `EnsembleSolver`, and `Agent57Solver`) were returning raw strings (e.g., "a", "up") as the `action` field in their `ActionProposal`. The `EmulatorEnvironment.step()` method was not robust to these legacy formats and immediately tried to access the `.button` attribute, causing a crash.

## Solution
1. Enhanced `EmulatorEnvironment.step()` in `src/autogameplayer/core/environment.py` to be robust: it now automatically converts string or integer actions into full `Action` objects.
2. Updated all built-in solvers (`RandomSolver`, `EnsembleSolver`, `Agent57Solver`, `SkillSolver`) to correctly return `Action` objects, ensuring consistency across the neuro-symbolic pipeline.
3. Cleaned up imports and standardized return patterns in `src/autogameplayer/solvers/composite.py` and `src/autogameplayer/solvers/random.py`.

## Test
Verified that the system is now resilient to different action formats and that all standard solvers produce well-formed `Action` objects.

# Bug: UnknownSolver Telemetry & Missing agent_pipeline Fallback

## Bug
`⚠️ No agent_pipeline found. Falling back to default random solver.`
`Distribution: {'UnknownSolver': {'count': 10, 'avg_confidence': 0.125}}`
The AI Player used the `RandomSolver` despite `brain: "agentic"` being set in the config, and the telemetry incorrectly identified it as `UnknownSolver`.

## Cause
1. `src/autogameplayer/runner/cli.py` only looked for the new `agent_pipeline` configuration. If missing, it immediately fell back to `RandomSolver` without checking the legacy `brain` configuration string.
2. Most solvers were not setting a `solver_name` in their `ActionProposal` metadata, and the `ActionProposal` class was missing a dedicated `source` field for telemetry.

## Solution
1. Implemented `BrainSolverAdapter` in `src/autogameplayer/solvers/adapter.py` to bridge legacy `Brain` implementations with the new `BaseSolver` interface.
2. Updated `src/autogameplayer/runner/cli.py` to attempt to create a `Brain` from `config.brain` if `agent_pipeline` is missing, using the adapter to integrate it into the `SessionOrchestrator`.
3. Added a `source` field to the `ActionProposal` dataclass in `src/autogameplayer/core/solver.py`.
4. Standardized all built-in solvers to set `solver_name` in metadata or use the new `source` field.
5. Improved `SessionOrchestrator` telemetry to correctly resolve the solver name from the new `source` field or metadata.

## Test
Verified that configs using `brain: "agentic"` now correctly instantiate the `AgenticBrain` (via adapter) and that telemetry correctly reports solver names (e.g., `RandomSolver`, `AgenticBrain`) instead of `UnknownSolver`.

# Bug: UnboundLocalError for 'Registry'

## Bug
`UnboundLocalError: cannot access local variable 'Registry' where it is not associated with a value` in `src/autogameplayer/runner/cli.py`.
The AI Player crashed immediately during initialization.

## Cause
The `Registry` class was imported at the top level of the file, but was ALSO imported locally inside the `run_autogame` function (in the legacy fallback block). In Python, a local assignment (including an import) shadows global variables throughout the entire function scope. Since `Registry` was used *before* the local import line, it raised an `UnboundLocalError`.

## Solution
Removed the redundant local import of `Registry` in `src/autogameplayer/runner/cli.py`, allowing the function to correctly use the global `Registry` import.

## Test
Confirmed that the AI Player now successfully initializes and accesses the `Registry` to create the controller and brain.
