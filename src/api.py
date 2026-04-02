"""Top-level compile APIs for energy functions."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Union

from src.backend.aries_backend import ARIESBackend
from src.compiler.autodiff import combine_modules, get_forward_backward_mlir, module_to_text
from src.compiler.frontend import PythonToMLIR, VariableSpec


VariableInput = Union[
    Sequence[str],
    Mapping[str, Union[VariableSpec, Mapping[str, Any], str]],
    None,
]


def compile_energy_function(
    func,
    example_args,
    target="aie",
    output_dir="build",
    use_omeinsum: bool = True,
    julia_cmd: str = "julia",
):
    """
    将 Python 能量函数（含自动微分）编译为 AIE/PE 可执行代码。

    - func: Python 函数
    - example_args: 示例输入（用于 JAX tracing），如 `(x,)` 或 `(X, Y)`
    - target: "aie" 或 "pe" 或 "hybrid"
    - output_dir: 输出目录
    """

    target = str(target).lower()
    if target not in {"aie", "pe", "hybrid"}:
        raise ValueError("target must be one of: 'aie', 'pe', 'hybrid'")

    if not isinstance(example_args, (tuple, list)):
        example_args = (example_args,)
    example_args = tuple(example_args)

    # 1. 自动微分并生成 forward/backward MLIR
    fwd_mod, bwd_mod = get_forward_backward_mlir(
        func,
        example_args,
        optimize_contractions=use_omeinsum,
        julia_cmd=julia_cmd,
    )
    combined_mod = combine_modules(fwd_mod, bwd_mod)

    fwd_text = module_to_text(fwd_mod)
    bwd_text = module_to_text(bwd_mod)
    combined_text = module_to_text(combined_mod)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    func_name = getattr(func, "__name__", "kernel")

    fwd_mlir_path = out_dir / f"{func_name}.forward.mlir"
    bwd_mlir_path = out_dir / f"{func_name}.backward.mlir"
    combined_mlir_path = out_dir / f"{func_name}.combined.mlir"
    fwd_mlir_path.write_text(fwd_text, encoding="utf-8")
    bwd_mlir_path.write_text(bwd_text, encoding="utf-8")
    combined_mlir_path.write_text(combined_text, encoding="utf-8")

    # 2. 交给 ARIES 后端优化
    backend = ARIESBackend()
    optimized_mlir = backend.optimize(
        combined_text,
        extra_args=["-canonicalize", "-cse"],
    )
    opt_mlir_path = out_dir / f"{func_name}.combined.opt.mlir"
    opt_mlir_path.write_text(optimized_mlir, encoding="utf-8")

    artifacts: Dict[str, str] = {
        "forward_mlir": str(fwd_mlir_path),
        "backward_mlir": str(bwd_mlir_path),
        "combined_mlir": str(combined_mlir_path),
        "optimized_mlir": str(opt_mlir_path),
        "target": target,
    }

    # 3. 调用 ARIES 代码生成
    if target == "aie":
        code_path = out_dir / f"{func_name}_aie.cc"
        backend.generate_aie_code(optimized_mlir, output_path=code_path, emit="cc")
        artifacts["code"] = str(code_path)

    elif target == "pe":
        # PE 目标先输出拆分后的 kernel MLIR，后续可接自定义 PE codegen。
        pe_path = out_dir / f"{func_name}_pe.mlir"
        backend.generate_aie_code(optimized_mlir, output_path=pe_path, emit="mlir")
        artifacts["code"] = str(pe_path)

    else:  # hybrid
        # 简化策略：同一份优化后 IR 同时导出 AIE C++ 与 PE kernel MLIR。
        aie_path = out_dir / f"{func_name}_aie.cc"
        pe_path = out_dir / f"{func_name}_pe.mlir"
        backend.generate_aie_code(optimized_mlir, output_path=aie_path, emit="cc")
        backend.generate_aie_code(optimized_mlir, output_path=pe_path, emit="mlir")
        artifacts["aie_code"] = str(aie_path)
        artifacts["pe_code"] = str(pe_path)

    # 4. 保存代码到 output_dir（已在上面完成）
    return artifacts


def compile_energy_function_legacy(func, variables, target="aie", output_dir="build/"):
    """Legacy entrypoint using handwritten Python AST frontend."""

    target = str(target).lower()
    if target not in {"aie", "pe", "hybrid"}:
        raise ValueError("target must be one of: 'aie', 'pe', 'hybrid'")

    var_specs = _normalize_variable_specs(func, variables)
    frontend = PythonToMLIR(variable_specs=var_specs)
    mlir_module = frontend.lower(func, output="text")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    func_name = getattr(func, "__name__", "kernel")

    input_mlir_path = out_dir / f"{func_name}.mlir"
    input_mlir_path.write_text(mlir_module, encoding="utf-8")

    backend = ARIESBackend()
    optimized_mlir = backend.optimize(mlir_module, extra_args=["-canonicalize", "-cse"])
    opt_mlir_path = out_dir / f"{func_name}.opt.mlir"
    opt_mlir_path.write_text(optimized_mlir, encoding="utf-8")

    artifacts: Dict[str, str] = {
        "input_mlir": str(input_mlir_path),
        "optimized_mlir": str(opt_mlir_path),
        "target": target,
    }

    if target == "aie":
        code_path = out_dir / f"{func_name}_aie.cc"
        backend.generate_aie_code(optimized_mlir, output_path=code_path, emit="cc")
        artifacts["code"] = str(code_path)
    elif target == "pe":
        pe_path = out_dir / f"{func_name}_pe.mlir"
        backend.generate_aie_code(optimized_mlir, output_path=pe_path, emit="mlir")
        artifacts["code"] = str(pe_path)
    else:
        aie_path = out_dir / f"{func_name}_aie.cc"
        pe_path = out_dir / f"{func_name}_pe.mlir"
        backend.generate_aie_code(optimized_mlir, output_path=aie_path, emit="cc")
        backend.generate_aie_code(optimized_mlir, output_path=pe_path, emit="mlir")
        artifacts["aie_code"] = str(aie_path)
        artifacts["pe_code"] = str(pe_path)

    return artifacts


def _normalize_variable_specs(
    func: Any,
    variables: VariableInput,
) -> Dict[str, Union[VariableSpec, Dict[str, Any]]]:
    arg_names = list(inspect.signature(func).parameters.keys())

    if variables is None:
        return {name: {"dtype": "f64", "shape": (None,)} for name in arg_names}

    if isinstance(variables, Mapping):
        specs: Dict[str, Union[VariableSpec, Dict[str, Any]]] = {}
        for name in arg_names:
            if name not in variables:
                specs[name] = {"dtype": "f64", "shape": (None,)}
                continue
            value = variables[name]
            if isinstance(value, str):
                specs[name] = {"dtype": value, "shape": (None,)}
            else:
                specs[name] = value
        return specs

    if isinstance(variables, Sequence) and not isinstance(variables, (str, bytes)):
        declared = set(variables)
        specs = {}
        for name in arg_names:
            if name in declared:
                specs[name] = {"dtype": "f64", "shape": (None,)}
            else:
                specs[name] = {"dtype": "f64", "shape": None}
        return specs

    raise TypeError("variables must be a list/tuple of names or a dict of variable specs")
