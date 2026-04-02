"""Frontend: lower a subset of Python AST to MLIR text/module.

This module provides `PythonToMLIR`, a lightweight translator that focuses on
numerical kernels like nested `sum` comprehensions.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


Number = Union[int, float]


@dataclass
class VariableSpec:
    """Type/shape information for a user variable.

    `shape=None` means scalar. Otherwise this variable is lowered as a memref.
    Use `None` in a dimension to denote dynamic shape (mapped to `?`).
    """

    dtype: str = "f64"
    shape: Optional[Tuple[Optional[int], ...]] = None

    @classmethod
    def from_user_value(cls, value: Any) -> "VariableSpec":
        if isinstance(value, VariableSpec):
            return value
        if isinstance(value, dict):
            dtype = value.get("dtype", "f64")
            shape = value.get("shape")
            if shape is None:
                return cls(dtype=dtype, shape=None)
            return cls(dtype=dtype, shape=tuple(shape))
        raise TypeError(f"Unsupported variable spec: {value!r}")

    @property
    def is_scalar(self) -> bool:
        return self.shape is None

    def mlir_arg_type(self) -> str:
        if self.is_scalar:
            return self.dtype
        dims = "x".join("?" if d is None else str(d) for d in self.shape)
        return f"memref<{dims}x{self.dtype}>"


class _MLIRBuilder:
    def __init__(self) -> None:
        self.lines: List[str] = []
        self.indent = 0

    def emit(self, line: str = "") -> None:
        self.lines.append(f"{'  ' * self.indent}{line}")

    def push(self) -> None:
        self.indent += 1

    def pop(self) -> None:
        self.indent -= 1

    def text(self) -> str:
        return "\n".join(self.lines) + "\n"


class PythonToMLIR:
    """Lower a subset of Python functions into MLIR.

    Supported focus pattern:
    - `return sum(<expr> for ... in range(...) [for ...])`
    - arithmetic over constants, loop indices, and memref loads like `x[i]`

    Dialects emitted by generated IR: `func`, `arith`, `math`, `scf`, `memref`.
    """

    def __init__(
        self,
        variable_specs: Optional[Dict[str, Union[VariableSpec, Dict[str, Any]]]] = None,
        return_dtype: str = "f64",
    ) -> None:
        self.variable_specs: Dict[str, VariableSpec] = {}
        for name, spec in (variable_specs or {}).items():
            self.variable_specs[name] = VariableSpec.from_user_value(spec)
        self.return_dtype = return_dtype

        self._builder = _MLIRBuilder()
        self._ssa_counter = 0
        self._const_cache: Dict[Tuple[str, Union[int, float]], str] = {}
        self._globals: Dict[str, Any] = {}

    def lower(
        self,
        fn: Any,
        output: str = "text",
    ) -> Any:
        """Lower a Python function.

        Args:
            fn: Python function object.
            output: `"text"` for MLIR text, `"module"` for MLIR Python module.
        """

        # Reset state so one PythonToMLIR instance can lower multiple functions safely.
        self._builder = _MLIRBuilder()
        self._ssa_counter = 0
        self._const_cache = {}

        source = textwrap.dedent(inspect.getsource(fn))
        module_ast = ast.parse(source)
        func_node = self._find_function(module_ast)
        self._globals = getattr(fn, "__globals__", {})

        arg_bindings = self._build_function_signature(func_node)
        ret_ssa, ret_type = self._lower_return(func_node, arg_bindings)
        if ret_type != self.return_dtype:
            ret_ssa = self._cast(ret_ssa, ret_type, self.return_dtype)
            ret_type = self.return_dtype

        self._builder.emit(f"return {ret_ssa} : {ret_type}")
        self._builder.pop()
        self._builder.emit("}")
        self._builder.pop()
        self._builder.emit("}")

        mlir_text = self._builder.text()
        if output == "text":
            return mlir_text
        if output == "module":
            return self._to_mlir_module(mlir_text)
        raise ValueError("output must be 'text' or 'module'")

    def _find_function(self, module_ast: ast.Module) -> ast.FunctionDef:
        for node in module_ast.body:
            if isinstance(node, ast.FunctionDef):
                return node
        raise ValueError("No function definition found in source")

    def _build_function_signature(self, func_node: ast.FunctionDef) -> Dict[str, Tuple[str, str]]:
        self._builder.emit("module {")
        self._builder.push()

        args = []
        arg_bindings: Dict[str, Tuple[str, str]] = {}
        for idx, arg in enumerate(func_node.args.args):
            name = arg.arg
            spec = self.variable_specs.get(name, VariableSpec(dtype="f64", shape=(None,)))
            arg_type = spec.mlir_arg_type()
            ssa = f"%arg{idx}"
            args.append(f"{ssa}: {arg_type}")
            arg_bindings[name] = (ssa, arg_type)

        self._builder.emit(
            f"func.func @{func_node.name}({', '.join(args)}) -> {self.return_dtype} {{"
        )
        self._builder.push()
        return arg_bindings

    def _lower_return(
        self,
        func_node: ast.FunctionDef,
        arg_bindings: Dict[str, Tuple[str, str]],
    ) -> Tuple[str, str]:
        returns = [n for n in func_node.body if isinstance(n, ast.Return)]
        if len(returns) != 1:
            raise ValueError("Function must contain exactly one return statement")
        expr = returns[0].value
        if expr is None:
            raise ValueError("Return expression is required")
        env = {
            "vars": arg_bindings,
            "loop_vars": {},
        }
        return self._lower_expr(expr, env)

    def _lower_expr(self, node: ast.AST, env: Dict[str, Any]) -> Tuple[str, str]:
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "sum":
            if len(node.args) != 1 or not isinstance(node.args[0], ast.GeneratorExp):
                raise ValueError("sum() must receive a single generator expression")
            return self._lower_sum_generator(node.args[0], env)

        if isinstance(node, ast.BinOp):
            lhs_ssa, lhs_t = self._lower_expr(node.left, env)
            rhs_ssa, rhs_t = self._lower_expr(node.right, env)
            result_t = self._promote(lhs_t, rhs_t)
            if lhs_t != result_t:
                lhs_ssa = self._cast(lhs_ssa, lhs_t, result_t)
            if rhs_t != result_t:
                rhs_ssa = self._cast(rhs_ssa, rhs_t, result_t)
            return self._emit_binop(node.op, lhs_ssa, rhs_ssa, result_t)

        if isinstance(node, ast.Name):
            if node.id in env["loop_vars"]:
                return env["loop_vars"][node.id]
            if node.id in env["vars"]:
                ssa, typ = env["vars"][node.id]
                if typ.startswith("memref<"):
                    raise ValueError(
                        f"Variable '{node.id}' is a memref; use subscript load like {node.id}[i]"
                    )
                return ssa, typ
            if node.id in self._globals and isinstance(self._globals[node.id], int):
                return self._emit_constant(self._globals[node.id], "index"), "index"
            raise ValueError(f"Unknown identifier: {node.id}")

        if isinstance(node, ast.Subscript):
            return self._lower_subscript(node, env)

        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool):
                raise ValueError("Boolean constants are not supported")
            if isinstance(node.value, int):
                return self._emit_constant(node.value, "i64"), "i64"
            if isinstance(node.value, float):
                return self._emit_constant(node.value, "f64"), "f64"

        raise ValueError(f"Unsupported expression node: {ast.dump(node)}")

    def _lower_subscript(self, node: ast.Subscript, env: Dict[str, Any]) -> Tuple[str, str]:
        if not isinstance(node.value, ast.Name):
            raise ValueError("Only name-based subscripts are supported")
        var_name = node.value.id
        if var_name not in env["vars"]:
            raise ValueError(f"Unknown subscript base variable: {var_name}")

        base_ssa, base_type = env["vars"][var_name]
        if not base_type.startswith("memref<"):
            raise ValueError(f"Variable '{var_name}' is not a memref")

        index_nodes = self._extract_indices(node.slice)
        index_ssas: List[str] = []
        for idx_node in index_nodes:
            idx_ssa, idx_t = self._lower_expr(idx_node, env)
            if idx_t != "index":
                idx_ssa = self._cast(idx_ssa, idx_t, "index")
            index_ssas.append(idx_ssa)

        elem_type = self._memref_element_type(base_type)
        out_ssa = self._new_ssa()
        idx_text = ", ".join(index_ssas)
        self._builder.emit(f"{out_ssa} = memref.load {base_ssa}[{idx_text}] : {base_type}")
        return out_ssa, elem_type

    def _lower_sum_generator(self, gen: ast.GeneratorExp, env: Dict[str, Any]) -> Tuple[str, str]:
        acc_type = self._infer_expr_type(gen.elt, env)
        if acc_type == "index":
            acc_type = "i64"

        acc_init = self._emit_constant(0.0 if self._is_float(acc_type) else 0, acc_type)

        def lower_gen(level: int, acc_in: str, loop_env: Dict[str, Any]) -> str:
            if level >= len(gen.generators):
                term_ssa, term_type = self._lower_expr(gen.elt, loop_env)
                if term_type != acc_type:
                    term_ssa = self._cast(term_ssa, term_type, acc_type)
                acc_out, _ = self._emit_binop(ast.Add(), acc_in, term_ssa, acc_type)
                return acc_out

            comp = gen.generators[level]
            if comp.ifs:
                raise ValueError("Comprehension if-clauses are not supported yet")
            if not isinstance(comp.target, ast.Name):
                raise ValueError("Only simple loop targets are supported")

            lb_ssa, ub_ssa, step_ssa = self._lower_range_bounds(comp.iter, loop_env)
            iv_ssa = self._new_ssa()
            iter_arg = self._new_ssa("acc")
            loop_result = self._new_ssa("sum")

            self._builder.emit(
                f"{loop_result} = scf.for {iv_ssa} = {lb_ssa} to {ub_ssa} step {step_ssa} "
                f"iter_args({iter_arg} = {acc_in}) -> ({acc_type}) {{"
            )
            self._builder.push()

            nested_env = {
                "vars": loop_env["vars"],
                "loop_vars": dict(loop_env["loop_vars"]),
            }
            nested_env["loop_vars"][comp.target.id] = (iv_ssa, "index")

            yielded = lower_gen(level + 1, iter_arg, nested_env)
            if yielded != iter_arg:
                self._builder.emit(f"scf.yield {yielded} : {acc_type}")
            else:
                self._builder.emit(f"scf.yield {iter_arg} : {acc_type}")
            self._builder.pop()
            self._builder.emit("}")

            return loop_result

        out = lower_gen(0, acc_init, env)
        return out, acc_type

    def _lower_range_bounds(self, node: ast.AST, env: Dict[str, Any]) -> Tuple[str, str, str]:
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.func.id != "range":
            raise ValueError("Only range(...) iterators are supported")

        args = node.args
        if len(args) == 1:
            lb = self._emit_constant(0, "index")
            ub, ub_t = self._lower_expr(args[0], env)
            if ub_t != "index":
                ub = self._cast(ub, ub_t, "index")
            step = self._emit_constant(1, "index")
            return lb, ub, step
        if len(args) == 2:
            lb, lb_t = self._lower_expr(args[0], env)
            ub, ub_t = self._lower_expr(args[1], env)
            if lb_t != "index":
                lb = self._cast(lb, lb_t, "index")
            if ub_t != "index":
                ub = self._cast(ub, ub_t, "index")
            step = self._emit_constant(1, "index")
            return lb, ub, step
        if len(args) == 3:
            lb, lb_t = self._lower_expr(args[0], env)
            ub, ub_t = self._lower_expr(args[1], env)
            step, step_t = self._lower_expr(args[2], env)
            if lb_t != "index":
                lb = self._cast(lb, lb_t, "index")
            if ub_t != "index":
                ub = self._cast(ub, ub_t, "index")
            if step_t != "index":
                step = self._cast(step, step_t, "index")
            return lb, ub, step
        raise ValueError("range() expects 1 to 3 arguments")

    def _emit_binop(self, op: ast.operator, lhs: str, rhs: str, typ: str) -> Tuple[str, str]:
        out = self._new_ssa()
        if isinstance(op, ast.Add):
            opname = "arith.addf" if self._is_float(typ) else "arith.addi"
        elif isinstance(op, ast.Sub):
            opname = "arith.subf" if self._is_float(typ) else "arith.subi"
        elif isinstance(op, ast.Mult):
            opname = "arith.mulf" if self._is_float(typ) else "arith.muli"
        elif isinstance(op, ast.Div):
            opname = "arith.divf" if self._is_float(typ) else "arith.divsi"
        elif isinstance(op, ast.Pow):
            if not self._is_float(typ):
                raise ValueError("Integer power is not supported")
            opname = "math.powf"
        else:
            raise ValueError(f"Unsupported binary op: {type(op).__name__}")
        self._builder.emit(f"{out} = {opname} {lhs}, {rhs} : {typ}")
        return out, typ

    def _infer_expr_type(self, node: ast.AST, env: Dict[str, Any]) -> str:
        if isinstance(node, ast.BinOp):
            lt = self._infer_expr_type(node.left, env)
            rt = self._infer_expr_type(node.right, env)
            if isinstance(node.op, ast.Div):
                return "f64" if "f" in (lt + rt) else lt
            return self._promote(lt, rt)

        if isinstance(node, ast.Constant):
            if isinstance(node.value, float):
                return "f64"
            if isinstance(node.value, int):
                return "i64"

        if isinstance(node, ast.Name):
            if node.id in env["loop_vars"]:
                return env["loop_vars"][node.id][1]
            if node.id in env["vars"]:
                _, typ = env["vars"][node.id]
                return typ
            if node.id in self._globals and isinstance(self._globals[node.id], int):
                return "index"

        if isinstance(node, ast.Subscript):
            base = node.value
            if isinstance(base, ast.Name) and base.id in env["vars"]:
                _, typ = env["vars"][base.id]
                if typ.startswith("memref<"):
                    return self._memref_element_type(typ)

        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "sum":
            return self._infer_expr_type(node.args[0].elt, env)

        raise ValueError(f"Unable to infer type for node: {ast.dump(node)}")

    def _cast(self, ssa: str, src_t: str, dst_t: str) -> str:
        if src_t == dst_t:
            return ssa
        out = self._new_ssa()

        if src_t == "index" and dst_t.startswith("i"):
            self._builder.emit(f"{out} = arith.index_cast {ssa} : index to {dst_t}")
            return out
        if src_t.startswith("i") and dst_t == "index":
            self._builder.emit(f"{out} = arith.index_cast {ssa} : {src_t} to index")
            return out
        if src_t.startswith("i") and dst_t.startswith("f"):
            self._builder.emit(f"{out} = arith.sitofp {ssa} : {src_t} to {dst_t}")
            return out
        if src_t.startswith("f") and dst_t.startswith("i"):
            self._builder.emit(f"{out} = arith.fptosi {ssa} : {src_t} to {dst_t}")
            return out

        raise ValueError(f"Unsupported cast: {src_t} -> {dst_t}")

    def _emit_constant(self, value: Number, typ: str) -> str:
        key = (typ, value)
        if key in self._const_cache:
            return self._const_cache[key]

        ssa = self._new_ssa("c")
        if typ.startswith("f"):
            literal = f"{float(value)}"
        else:
            literal = str(int(value))
        self._builder.emit(f"{ssa} = arith.constant {literal} : {typ}")
        self._const_cache[key] = ssa
        return ssa

    def _extract_indices(self, slice_node: ast.AST) -> Sequence[ast.AST]:
        if isinstance(slice_node, ast.Tuple):
            return list(slice_node.elts)
        return [slice_node]

    def _memref_element_type(self, memref_t: str) -> str:
        inner = memref_t[len("memref<") : -1]
        return inner.split("x")[-1]

    def _promote(self, lhs_t: str, rhs_t: str) -> str:
        if lhs_t == rhs_t:
            return lhs_t
        if self._is_float(lhs_t) or self._is_float(rhs_t):
            return "f64"
        if "index" in (lhs_t, rhs_t):
            return "i64"
        if lhs_t.startswith("i") and rhs_t.startswith("i"):
            return lhs_t if int(lhs_t[1:]) >= int(rhs_t[1:]) else rhs_t
        return self.return_dtype

    def _is_float(self, typ: str) -> bool:
        return typ.startswith("f")

    def _new_ssa(self, prefix: str = "v") -> str:
        ssa = f"%{prefix}{self._ssa_counter}"
        self._ssa_counter += 1
        return ssa

    def _to_mlir_module(self, mlir_text: str) -> Any:
        try:
            from mlir import ir  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "MLIR Python bindings are not installed. Use output='text' or install bindings."
            ) from exc

        with ir.Context() as ctx:
            # The context keeps dialect registrations and parsing settings.
            return ir.Module.parse(mlir_text, ctx)
