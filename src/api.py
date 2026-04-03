"""Top-level compile APIs for energy functions."""

from __future__ import annotations

import inspect
import json
import re
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

from src.backend.aries_backend import ARIESBackend, ARIESBackendConfig
from src.backend.precision import Precision, normalize_precision
from src.backend.sparse_formats import SparseFormat, normalize_sparse_format
from src.compiler.autodiff import combine_modules, get_forward_backward_mlir, module_to_text
from src.compiler.frontend import PythonToMLIR, VariableSpec


VariableInput = Union[
    Sequence[str],
    Mapping[str, Union[VariableSpec, Mapping[str, Any], str]],
    None,
]

_DEFAULT_PROBLEM_SESSION = None
_DEFAULT_BOARD_RUNTIME = None


def compile_energy_function(
    func,
    example_args,
    target="aie",
    output_dir="build",
    use_omeinsum: bool = True,
    julia_cmd: str = "julia",
    gradient_mode: str = "auto",
    variables: VariableInput = None,
    sparse_format: Union[str, SparseFormat] = SparseFormat.TCSR,
    precision: Union[str, Precision] = Precision.FP32,
    expected_input_shapes: Optional[Mapping[str, Sequence[int]]] = None,
    auto_aie_config: bool = True,
    shape_policy: str = "exact",
    shape_buckets: Optional[Mapping[str, Sequence[Sequence[int]]]] = None,
    max_shape: Optional[Mapping[str, Sequence[int]]] = None,
    problem_name: Optional[str] = None,
):
    """
    将 Python 能量函数（含自动微分）编译为 AIE/PE 可执行代码。

    - func: Python 函数
    - example_args: 示例输入（用于 JAX tracing），如 `(x,)` 或 `(X, Y)`
    - target: "aie" 或 "pe" 或 "hybrid"
    - output_dir: 输出根目录（将在其下生成 mlir_compile/ 与 aie_compile/）
    """

    gradient_mode = str(gradient_mode).strip().lower()
    if gradient_mode not in {"auto", "manual"}:
        raise ValueError("gradient_mode must be one of: 'auto', 'manual'")

    target = str(target).lower()
    if target not in {"aie", "pe", "hybrid"}:
        raise ValueError("target must be one of: 'aie', 'pe', 'hybrid'")

    precision_value = normalize_precision(precision)
    sparse_format_value = normalize_sparse_format(sparse_format)

    if gradient_mode == "manual":
        return compile_energy_function_legacy(
            func,
            variables,
            target=target,
            output_dir=output_dir,
            sparse_format=sparse_format_value,
            precision=precision_value,
            problem_name=problem_name,
        )

    if not isinstance(example_args, (tuple, list)):
        example_args = (example_args,)
    example_args = tuple(example_args)
    example_args = _cast_example_args_to_precision(example_args, precision_value)

    arg_names = list(inspect.signature(func).parameters.keys())
    actual_input_shapes = _resolve_input_shapes(func, example_args, expected_input_shapes=None)
    compile_input_shapes = _select_compile_shapes(
        arg_names=arg_names,
        actual_shapes=actual_input_shapes,
        shape_policy=shape_policy,
        shape_buckets=shape_buckets,
        max_shape=max_shape,
    )
    trace_args = _build_trace_args_for_shapes(arg_names, example_args, compile_input_shapes)

    aie_config = _build_aie_config(
        func=func,
        example_args=trace_args,
        expected_input_shapes=compile_input_shapes,
        auto_aie_config=auto_aie_config,
    )

    # 1. 自动微分并生成 forward/backward MLIR
    fwd_mod, bwd_mod = get_forward_backward_mlir(
        func,
        trace_args,
        optimize_contractions=use_omeinsum,
        julia_cmd=julia_cmd,
    )
    combined_mod = combine_modules(fwd_mod, bwd_mod)

    fwd_text = module_to_text(fwd_mod)
    bwd_text = module_to_text(bwd_mod)
    combined_text = module_to_text(combined_mod)

    base_problem_name = _normalize_problem_name(problem_name or getattr(func, "__name__", "kernel"))
    compile_tag = _build_compile_tag(base_problem_name, precision_value, compile_input_shapes)
    dirs = _ensure_compile_dirs(output_root=output_dir, compile_tag=compile_tag)
    mlir_dir = dirs["mlir_dir"]
    aie_dir = dirs["aie_dir"]

    fwd_mlir_path = mlir_dir / f"{compile_tag}.forward.mlir"
    bwd_mlir_path = mlir_dir / f"{compile_tag}.backward.mlir"
    combined_mlir_path = mlir_dir / f"{compile_tag}.combined.mlir"
    fwd_mlir_path.write_text(fwd_text, encoding="utf-8")
    bwd_mlir_path.write_text(bwd_text, encoding="utf-8")
    combined_mlir_path.write_text(combined_text, encoding="utf-8")

    # 2. 交给 ARIES 后端优化
    backend = ARIESBackend(
        config=ARIESBackendConfig(
            sparse_tile_rows=int(aie_config["sparse_tile_rows"]),
            sparse_tile_cols=int(aie_config["sparse_tile_cols"]),
        )
    )
    optimized_mlir = backend.optimize(
        combined_text,
        extra_args=["-canonicalize", "-cse"],
        sparse_format=sparse_format_value,
        precision=precision_value,
    )
    opt_mlir_path = mlir_dir / f"{compile_tag}.combined.opt.mlir"
    opt_mlir_path.write_text(optimized_mlir, encoding="utf-8")
    aie_cfg_path = mlir_dir / f"{compile_tag}.aie_config.json"
    aie_cfg_path.write_text(json.dumps(aie_config, indent=2), encoding="utf-8")

    artifacts: Dict[str, str] = {
        "forward_mlir": str(fwd_mlir_path),
        "backward_mlir": str(bwd_mlir_path),
        "combined_mlir": str(combined_mlir_path),
        "optimized_mlir": str(opt_mlir_path),
        "target": target,
        "gradient_mode": gradient_mode,
        "sparse_format": sparse_format_value.value,
        "precision": precision_value.value,
        "aie_config": str(aie_cfg_path),
        "shape_policy": str(shape_policy),
        "actual_input_shapes": json.dumps(actual_input_shapes),
        "compile_input_shapes": json.dumps(compile_input_shapes),
        "problem_name": base_problem_name,
        "compile_tag": compile_tag,
        "mlir_output_dir": str(mlir_dir),
        "aie_output_dir": str(aie_dir),
    }

    # 3. 调用 ARIES 代码生成
    if target == "aie":
        code_path = aie_dir / f"{compile_tag}_aie.cc"
        backend.generate_aie_code(optimized_mlir, output_path=code_path, emit="cc")
        artifacts["code"] = str(code_path)

    elif target == "pe":
        # PE 目标先输出拆分后的 kernel MLIR，后续可接自定义 PE codegen。
        pe_path = aie_dir / f"{compile_tag}_pe.mlir"
        backend.generate_aie_code(optimized_mlir, output_path=pe_path, emit="mlir")
        artifacts["code"] = str(pe_path)

    else:  # hybrid
        # 简化策略：同一份优化后 IR 同时导出 AIE C++ 与 PE kernel MLIR。
        aie_path = aie_dir / f"{compile_tag}_aie.cc"
        pe_path = aie_dir / f"{compile_tag}_pe.mlir"
        backend.generate_aie_code(optimized_mlir, output_path=aie_path, emit="cc")
        backend.generate_aie_code(optimized_mlir, output_path=pe_path, emit="mlir")
        artifacts["aie_code"] = str(aie_path)
        artifacts["pe_code"] = str(pe_path)

    # 4. 保存代码到 output_dir（已在上面完成）
    return artifacts


def compile_energy_function_legacy(
    func,
    variables,
    target="aie",
    output_dir="build/",
    sparse_format: Union[str, SparseFormat] = SparseFormat.TCSR,
    precision: Union[str, Precision] = Precision.FP32,
    expected_input_shapes: Optional[Mapping[str, Sequence[int]]] = None,
    auto_aie_config: bool = True,
    problem_name: Optional[str] = None,
):
    """Legacy entrypoint using handwritten Python AST frontend."""

    target = str(target).lower()
    if target not in {"aie", "pe", "hybrid"}:
        raise ValueError("target must be one of: 'aie', 'pe', 'hybrid'")

    precision_value = normalize_precision(precision)
    sparse_format_value = normalize_sparse_format(sparse_format)

    aie_config = _build_aie_config(
        func=func,
        example_args=(),
        expected_input_shapes=expected_input_shapes,
        auto_aie_config=auto_aie_config,
    )

    var_specs = _normalize_variable_specs(func, variables, precision_value)
    frontend = PythonToMLIR(
        variable_specs=var_specs,
        return_dtype=precision_value.mlir_dtype,
    )
    mlir_module = frontend.lower(func, output="text")

    compile_shapes = {
        k: tuple(int(x) for x in v)
        for k, v in aie_config.get("input_shapes", {}).items()
    }
    base_problem_name = _normalize_problem_name(problem_name or getattr(func, "__name__", "kernel"))
    compile_tag = _build_compile_tag(base_problem_name, precision_value, compile_shapes)
    dirs = _ensure_compile_dirs(output_root=output_dir, compile_tag=compile_tag)
    mlir_dir = dirs["mlir_dir"]
    aie_dir = dirs["aie_dir"]

    input_mlir_path = mlir_dir / f"{compile_tag}.mlir"
    input_mlir_path.write_text(mlir_module, encoding="utf-8")

    backend = ARIESBackend(
        config=ARIESBackendConfig(
            sparse_tile_rows=int(aie_config["sparse_tile_rows"]),
            sparse_tile_cols=int(aie_config["sparse_tile_cols"]),
        )
    )
    optimized_mlir = backend.optimize(
        mlir_module,
        extra_args=["-canonicalize", "-cse"],
        sparse_format=sparse_format_value,
        precision=precision_value,
    )
    opt_mlir_path = mlir_dir / f"{compile_tag}.opt.mlir"
    opt_mlir_path.write_text(optimized_mlir, encoding="utf-8")
    aie_cfg_path = mlir_dir / f"{compile_tag}.aie_config.json"
    aie_cfg_path.write_text(json.dumps(aie_config, indent=2), encoding="utf-8")

    artifacts: Dict[str, str] = {
        "input_mlir": str(input_mlir_path),
        "optimized_mlir": str(opt_mlir_path),
        "target": target,
        "gradient_mode": "manual",
        "sparse_format": sparse_format_value.value,
        "precision": precision_value.value,
        "aie_config": str(aie_cfg_path),
        "problem_name": base_problem_name,
        "compile_tag": compile_tag,
        "mlir_output_dir": str(mlir_dir),
        "aie_output_dir": str(aie_dir),
    }

    if target == "aie":
        code_path = aie_dir / f"{compile_tag}_aie.cc"
        backend.generate_aie_code(optimized_mlir, output_path=code_path, emit="cc")
        artifacts["code"] = str(code_path)
    elif target == "pe":
        pe_path = aie_dir / f"{compile_tag}_pe.mlir"
        backend.generate_aie_code(optimized_mlir, output_path=pe_path, emit="mlir")
        artifacts["code"] = str(pe_path)
    else:
        aie_path = aie_dir / f"{compile_tag}_aie.cc"
        pe_path = aie_dir / f"{compile_tag}_pe.mlir"
        backend.generate_aie_code(optimized_mlir, output_path=aie_path, emit="cc")
        backend.generate_aie_code(optimized_mlir, output_path=pe_path, emit="mlir")
        artifacts["aie_code"] = str(aie_path)
        artifacts["pe_code"] = str(pe_path)

    return artifacts


def _normalize_variable_specs(
    func: Any,
    variables: VariableInput,
    precision: Precision,
) -> Dict[str, Union[VariableSpec, Dict[str, Any]]]:
    arg_names = list(inspect.signature(func).parameters.keys())
    dtype = precision.mlir_dtype

    if variables is None:
        return {name: {"dtype": dtype, "shape": (None,)} for name in arg_names}

    if isinstance(variables, Mapping):
        specs: Dict[str, Union[VariableSpec, Dict[str, Any]]] = {}
        for name in arg_names:
            if name not in variables:
                specs[name] = {"dtype": dtype, "shape": (None,)}
                continue
            value = variables[name]
            if isinstance(value, str):
                specs[name] = {"dtype": dtype, "shape": (None,)}
            elif isinstance(value, VariableSpec):
                specs[name] = VariableSpec(dtype=dtype, shape=value.shape)
            else:
                if isinstance(value, dict):
                    coerced = dict(value)
                    coerced["dtype"] = dtype
                    specs[name] = coerced
                else:
                    specs[name] = value
        return specs

    if isinstance(variables, Sequence) and not isinstance(variables, (str, bytes)):
        declared = set(variables)
        specs = {}
        for name in arg_names:
            if name in declared:
                specs[name] = {"dtype": dtype, "shape": (None,)}
            else:
                specs[name] = {"dtype": dtype, "shape": None}
        return specs

    raise TypeError("variables must be a list/tuple of names or a dict of variable specs")


def _cast_example_args_to_precision(example_args, precision: Precision):
    runtime_dtype = precision.runtime_dtype
    try:
        import numpy as np
    except Exception:
        np = None

    try:
        import jax.numpy as jnp  # type: ignore
    except Exception:
        jnp = None

    if runtime_dtype == "float32":
        target_dtype = "float32"
    elif runtime_dtype == "float16":
        target_dtype = "float16"
    else:
        target_dtype = "int8"

    casted = []
    for value in example_args:
        if hasattr(value, "astype"):
            casted.append(value.astype(target_dtype))
        elif np is not None:
            casted.append(np.asarray(value, dtype=target_dtype))
        elif jnp is not None:
            casted.append(jnp.asarray(value, dtype=target_dtype))
        else:
            casted.append(value)
    return tuple(casted)


def _build_aie_config(
    *,
    func: Any,
    example_args,
    expected_input_shapes: Optional[Mapping[str, Sequence[int]]],
    auto_aie_config: bool,
) -> Dict[str, Any]:
    input_shapes = _resolve_input_shapes(func, example_args, expected_input_shapes)

    if not auto_aie_config:
        return {
            "mode": "manual-default",
            "input_shapes": input_shapes,
            "estimated_intermediate_elements": _estimate_intermediate_elements(input_shapes),
            "sparse_tile_rows": 32,
            "sparse_tile_cols": 32,
        }

    estimated = _estimate_intermediate_elements(input_shapes)
    tile = _select_tile_size(estimated)
    return {
        "mode": "auto",
        "input_shapes": input_shapes,
        "estimated_intermediate_elements": estimated,
        "sparse_tile_rows": tile,
        "sparse_tile_cols": tile,
    }


def _resolve_input_shapes(
    func: Any,
    example_args,
    expected_input_shapes: Optional[Mapping[str, Sequence[int]]],
) -> Dict[str, Tuple[int, ...]]:
    arg_names = list(inspect.signature(func).parameters.keys())
    resolved: Dict[str, Tuple[int, ...]] = {}

    if expected_input_shapes is not None:
        for name in arg_names:
            user_shape = expected_input_shapes.get(name)
            if user_shape is None:
                resolved[name] = (1024,)
            else:
                resolved[name] = tuple(int(x) for x in user_shape)
        return resolved

    for idx, name in enumerate(arg_names):
        if idx < len(example_args):
            shp = getattr(example_args[idx], "shape", None)
            if shp is not None and len(tuple(shp)) > 0:
                resolved[name] = tuple(int(x) for x in tuple(shp))
            else:
                resolved[name] = (1024,)
        else:
            resolved[name] = (1024,)
    return resolved


def _normalize_problem_name(name: str) -> str:
    cleaned = re.sub(r"[^0-9a-zA-Z_]+", "_", str(name).strip().lower())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "kernel"


def _shape_suffix(input_shapes: Mapping[str, Tuple[int, ...]]) -> str:
    for shape in input_shapes.values():
        if len(shape) >= 2:
            return "x".join(str(int(dim)) for dim in shape)

    for shape in input_shapes.values():
        if len(shape) == 1:
            return str(int(shape[0]))

    return "shape_unknown"


def _build_compile_tag(
    problem_name: str,
    precision: Precision,
    input_shapes: Mapping[str, Tuple[int, ...]],
) -> str:
    return f"{problem_name}_{precision.value}_{_shape_suffix(input_shapes)}"


def _ensure_compile_dirs(output_root: Union[str, Path], compile_tag: str) -> Dict[str, Path]:
    root = Path(output_root)
    mlir_dir = root / "mlir_compile" / compile_tag
    aie_dir = root / "aie_compile" / compile_tag
    mlir_dir.mkdir(parents=True, exist_ok=True)
    aie_dir.mkdir(parents=True, exist_ok=True)
    return {
        "mlir_dir": mlir_dir,
        "aie_dir": aie_dir,
    }


def _select_compile_shapes(
    *,
    arg_names: Sequence[str],
    actual_shapes: Mapping[str, Tuple[int, ...]],
    shape_policy: str,
    shape_buckets: Optional[Mapping[str, Sequence[Sequence[int]]]],
    max_shape: Optional[Mapping[str, Sequence[int]]],
) -> Dict[str, Tuple[int, ...]]:
    policy = str(shape_policy).strip().lower()
    if policy not in {"exact", "bucket", "max_shape"}:
        raise ValueError("shape_policy must be one of: 'exact', 'bucket', 'max_shape'")

    if policy == "exact":
        return {k: tuple(v) for k, v in actual_shapes.items()}

    if policy == "max_shape":
        if not max_shape:
            raise ValueError("shape_policy='max_shape' requires max_shape mapping")
        out: Dict[str, Tuple[int, ...]] = {}
        for name in arg_names:
            act = tuple(actual_shapes.get(name, (1024,)))
            lim = tuple(int(x) for x in max_shape.get(name, ()))
            if not lim:
                raise ValueError(f"max_shape is missing shape for argument '{name}'")
            _validate_shape_within(act, lim, context=f"arg '{name}'")
            out[name] = lim
        return out

    # bucket
    if not shape_buckets:
        raise ValueError("shape_policy='bucket' requires shape_buckets mapping")

    out = {}
    for name in arg_names:
        act = tuple(actual_shapes.get(name, (1024,)))
        cands = [tuple(int(x) for x in s) for s in shape_buckets.get(name, ())]
        if not cands:
            raise ValueError(f"shape_buckets is missing candidates for argument '{name}'")
        fitting = [s for s in cands if _shape_fits(act, s)]
        if not fitting:
            raise ValueError(
                f"No bucket can cover input shape {act} for argument '{name}'. "
                f"Candidates: {cands}"
            )
        fitting.sort(key=lambda s: _num_elements(s))
        out[name] = fitting[0]
    return out


def _shape_fits(actual: Tuple[int, ...], limit: Tuple[int, ...]) -> bool:
    if len(actual) != len(limit):
        return False
    return all(int(a) <= int(b) for a, b in zip(actual, limit))


def _validate_shape_within(actual: Tuple[int, ...], limit: Tuple[int, ...], context: str) -> None:
    if not _shape_fits(actual, limit):
        raise ValueError(f"Input shape {actual} exceeds compile shape {limit} for {context}")


def _build_trace_args_for_shapes(
    arg_names: Sequence[str],
    example_args,
    compile_shapes: Mapping[str, Tuple[int, ...]],
):
    try:
        import numpy as np
    except Exception:
        return example_args

    trace = []
    for idx, name in enumerate(arg_names):
        if idx >= len(example_args):
            break
        value = example_args[idx]
        tgt = tuple(compile_shapes.get(name, getattr(value, "shape", (1024,))))
        src = tuple(getattr(value, "shape", ()))

        if src == tgt:
            trace.append(value)
            continue

        dtype = getattr(value, "dtype", np.float32)
        if len(tgt) == 0:
            trace.append(np.array(0, dtype=dtype))
        else:
            trace.append(np.zeros(tgt, dtype=dtype))
    return tuple(trace)


def _estimate_intermediate_elements(input_shapes: Mapping[str, Tuple[int, ...]]) -> int:
    shapes = [tuple(v) for v in input_shapes.values()]
    if not shapes:
        return 1024

    total_inputs = sum(_num_elements(s) for s in shapes)
    max_input = max(_num_elements(s) for s in shapes)

    matmul_intermediates = 0
    for i in range(len(shapes) - 1):
        a = shapes[i]
        b = shapes[i + 1]
        if len(a) >= 2 and len(b) >= 2 and a[-1] == b[0]:
            matmul_intermediates += int(a[0]) * int(b[-1])

    # Heuristic: one copy of inputs + one max scratch + inferred contractions.
    return int(total_inputs + max_input + matmul_intermediates)


def _num_elements(shape: Tuple[int, ...]) -> int:
    n = 1
    for dim in shape:
        n *= max(int(dim), 1)
    return int(n)


def _select_tile_size(estimated_intermediate_elements: int) -> int:
    if estimated_intermediate_elements <= 262144:
        return 16
    if estimated_intermediate_elements <= 1048576:
        return 32
    if estimated_intermediate_elements <= 4194304:
        return 64
    return 128


def get_problem_session(cache_dir: str = "build/problem_cache"):
    """Return a default two-step problem session manager.

    This manager separates:
    1) define/compile (artifact reuse when signature unchanged), and
    2) solve/execute (board-loaded check before execution).
    """

    global _DEFAULT_PROBLEM_SESSION
    if _DEFAULT_PROBLEM_SESSION is None:
        from src.runtime.problem_session import ProblemSessionManager

        _DEFAULT_PROBLEM_SESSION = ProblemSessionManager(cache_dir=cache_dir)
    return _DEFAULT_PROBLEM_SESSION


def get_board_runtime(state_file: str = "build/board_runtime/state.json"):
    """Return a default persistent board runtime implementation."""

    global _DEFAULT_BOARD_RUNTIME
    if _DEFAULT_BOARD_RUNTIME is None:
        from src.runtime.problem_session import FileBoardRuntime

        _DEFAULT_BOARD_RUNTIME = FileBoardRuntime(state_file=state_file)
    return _DEFAULT_BOARD_RUNTIME


def define_problem(*args, **kwargs):
    """Define and compile (or reuse) a problem through the default session."""

    session = get_problem_session()
    return session.define_problem(*args, **kwargs)


def define_problem_with_solver(
    *,
    problem_type: str,
    solver_type: str,
    energy_function,
    example_args,
    **kwargs,
):
    """Define a problem with explicit problem-type and solver-type separation."""

    session = get_problem_session()
    return session.define_problem(
        problem_type=problem_type,
        solver_type=solver_type,
        energy_function=energy_function,
        example_args=example_args,
        **kwargs,
    )


def define_lp_problem(
    energy_function,
    example_args,
    *,
    solver_type: str = "pdlp",
    **kwargs,
):
    """Convenience API for LP problem definitions with pluggable solver type."""

    session = get_problem_session()
    return session.define_problem(
        problem_type="lp",
        solver_type=solver_type,
        energy_function=energy_function,
        example_args=example_args,
        **kwargs,
    )


def solve_problem(*args, **kwargs):
    """Solve a previously-defined problem through the default session."""

    session = get_problem_session()
    return session.solve_problem(*args, **kwargs)
