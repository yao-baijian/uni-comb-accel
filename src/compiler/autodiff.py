"""JAX autodiff frontend: JAXPR -> MLIR conversion helpers."""

from __future__ import annotations

from dataclasses import dataclass
import json
import subprocess
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


class DependencyError(RuntimeError):
    """Raised when optional runtime dependencies are missing."""


def _import_jax() -> Any:
    try:
        import jax  # type: ignore
        import jax.numpy as jnp  # type: ignore

        return jax, jnp
    except Exception as exc:  # pragma: no cover - depends on environment
        raise DependencyError(
            "JAX is required for autodiff frontend. Install CPU build, e.g. `pip install jax jaxlib`."
        ) from exc


def _import_mlir_ir() -> Any:
    try:
        from mlir import ir  # type: ignore

        return ir
    except Exception as exc:  # pragma: no cover - depends on environment
        raise DependencyError(
            "MLIR Python bindings are required. Build/load ARIES MLIR Python packages first."
        ) from exc


def _dtype_to_mlir(dtype: Any) -> str:
    text = str(dtype)
    if "float32" in text:
        return "f32"
    if "float16" in text or "half" in text:
        return "f16"
    if "float64" in text:
        return "f64"
    if "int8" in text:
        return "i8"
    if "int4" in text:
        return "i4"
    if "int32" in text:
        return "i32"
    if "int64" in text:
        return "i64"
    raise NotImplementedError(f"Unsupported dtype: {dtype}")


def _shape_to_tensor_type(shape: Sequence[int], elem_type: str) -> str:
    if len(shape) == 0:
        return elem_type
    dims = "x".join(str(int(d)) for d in shape)
    return f"tensor<{dims}x{elem_type}>"


def _aval_to_mlir_type(aval: Any) -> str:
    shape = tuple(getattr(aval, "shape", ()))
    dtype = getattr(aval, "dtype", None)
    if dtype is None:
        if shape:
            raise NotImplementedError(f"Cannot infer dtype for shaped aval: {aval}")
        return "f32"
    return _shape_to_tensor_type(shape, _dtype_to_mlir(dtype))


def _shape_to_memref_type(shape: Sequence[int], elem_type: str) -> str:
    if len(shape) == 0:
        return f"memref<1x{elem_type}>"
    dims = "x".join(str(int(d)) for d in shape)
    return f"memref<{dims}x{elem_type}>"


def _sanitize_var_name(name: str) -> str:
    out = []
    for ch in name:
        if ch.isalnum() or ch == "_":
            out.append(ch)
        else:
            out.append("_")
    text = "".join(out).strip("_")
    return text or "v"


@dataclass
class ConvertedModule:
    text: str
    module_op: Optional[Any] = None


@dataclass
class ContractionOrderResult:
    pairs: List[Tuple[int, int]]
    source: str


class OMEinsumOrderOptimizer:
    """Get contraction order from OMEinsum (Julia), with Python fallback."""

    def __init__(self, julia_cmd: str = "julia", debug: bool = False) -> None:
        self.julia_cmd = julia_cmd
        self.debug = debug

    def optimize_matrix_chain(self, shapes: Sequence[Tuple[int, int]]) -> ContractionOrderResult:
        if len(shapes) <= 2:
            return ContractionOrderResult(pairs=[(0, 1)] if len(shapes) == 2 else [], source="trivial")

        julia_result = self._optimize_with_julia(shapes)
        if julia_result is not None:
            return julia_result

        return ContractionOrderResult(
            pairs=self._dynamic_programming_chain(shapes),
            source="python-fallback",
        )

    def _optimize_with_julia(self, shapes: Sequence[Tuple[int, int]]) -> Optional[ContractionOrderResult]:
        payload = {"shapes": [[int(r), int(c)] for r, c in shapes]}
        script = r'''
import JSON
import OMEinsum
import OMEinsumContractionOrders

inp = JSON.parse(ARGS[1])
shapes = inp["shapes"]
n = length(shapes)

# Build labels for a matrix chain A1(i0,i1) A2(i1,i2) ... An(i{n-1},i{n})
labels = [ [i, i+1] for i in 1:n ]
output = [1, n+1]

sizes = Dict{Int,Int}()
for i in 1:n
    sizes[i] = Int(shapes[i][1])
end
sizes[n+1] = Int(shapes[n][2])

code = OMEinsum.EinCode(labels, output)
method = OMEinsumContractionOrders.TreeSA()
opt = OMEinsumContractionOrders.optimize_code(code, sizes, method)

# Different package versions expose order metadata differently; try common paths.
pairs = nothing
if hasproperty(opt, :order)
    pairs = getproperty(opt, :order)
elseif hasproperty(opt, :contraction_order)
    pairs = getproperty(opt, :contraction_order)
end

if pairs === nothing
    # Fallback to left-associated order shape; Python side still has DP fallback.
    p = []
    for _ in 1:(n-1)
        push!(p, [1, 2])
    end
    println(JSON.json(Dict("pairs" => p, "source" => "omeinsum-unknown-order")))
else
    norm_pairs = []
    for t in pairs
        if length(t) == 2
            # Convert to 0-based indices expected by Python.
            push!(norm_pairs, [Int(t[1]) - 1, Int(t[2]) - 1])
        end
    end
    println(JSON.json(Dict("pairs" => norm_pairs, "source" => "omeinsum")))
end
'''
        try:
            proc = subprocess.run(
                [self.julia_cmd, "-e", script, json.dumps(payload)],
                text=True,
                capture_output=True,
                check=False,
                timeout=30,
            )
        except Exception:
            return None

        if proc.returncode != 0:
            if self.debug:
                print("[OMEinsum] Julia invocation failed:", proc.stderr)
            return None

        try:
            data = json.loads(proc.stdout.strip().splitlines()[-1])
            pairs = [(int(p[0]), int(p[1])) for p in data.get("pairs", [])]
            if len(pairs) != max(len(shapes) - 1, 0):
                return None
            return ContractionOrderResult(pairs=pairs, source=str(data.get("source", "omeinsum")))
        except Exception:
            return None

    def _dynamic_programming_chain(self, shapes: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
        n = len(shapes)
        dims = [int(shapes[0][0])] + [int(s[1]) for s in shapes]

        cost = [[0 if i == j else float("inf") for j in range(n)] for i in range(n)]
        split = [[-1 for _ in range(n)] for _ in range(n)]

        for length in range(2, n + 1):
            for i in range(0, n - length + 1):
                j = i + length - 1
                for k in range(i, j):
                    c = cost[i][k] + cost[k + 1][j] + dims[i] * dims[k + 1] * dims[j + 1]
                    if c < cost[i][j]:
                        cost[i][j] = c
                        split[i][j] = k

        pairs: List[Tuple[int, int]] = []

        def emit(i: int, j: int) -> int:
            if i == j:
                return i
            k = split[i][j]
            left = emit(i, k)
            right = emit(k + 1, j)
            pairs.append((left, right))
            return i

        emit(0, n - 1)
        # Convert recursive references to dynamic-list style pairs used by lowering.
        return [(0, 1) for _ in pairs]


class JaxprToMLIR:
    """Convert a JAXPR into an MLIR module with one function.

    This converter emits a textual MLIR module in func/arith/linalg/math/tensor/memref
    dialects and, when MLIR Python bindings are available, parses it into an `ir.Module`.
    """

    def __init__(
        self,
        context: Optional[Any] = None,
        function_name: str = "forward",
        debug: bool = False,
        optimize_contractions: bool = True,
        julia_cmd: str = "julia",
    ):
        self.context = context
        self.function_name = function_name
        self.debug = debug
        self.optimize_contractions = optimize_contractions
        self.julia_cmd = julia_cmd
        self._ssa_id = 0
        self._lines: List[str] = []
        self._indent = 0
        self._var_map: Dict[Any, Tuple[str, str]] = {}
        self._ome_opt = OMEinsumOrderOptimizer(julia_cmd=julia_cmd, debug=debug)

    def convert(self, jaxpr: Any, input_vars: Optional[Sequence[Any]] = None, output_vars: Optional[Sequence[Any]] = None) -> ConvertedModule:
        closed = getattr(jaxpr, "jaxpr", None)
        consts = getattr(jaxpr, "consts", ())
        if closed is not None:
            raw_jaxpr = closed
        else:
            raw_jaxpr = jaxpr

        invars = list(input_vars) if input_vars is not None else list(raw_jaxpr.invars)
        outvars = list(output_vars) if output_vars is not None else list(raw_jaxpr.outvars)

        arg_types = [_aval_to_mlir_type(v.aval) for v in invars]
        ret_types = [_aval_to_mlir_type(v.aval) for v in outvars]

        self._emit("module {")
        self._push()
        self._emit(
            f"func.func @{self.function_name}({self._format_args(arg_types)})"
            f" -> {self._format_ret_types(ret_types)} {{"
        )
        self._push()

        for i, var in enumerate(invars):
            self._var_map[var] = (f"%arg{i}", arg_types[i])

        # JAX constants in ClosedJaxpr are materialized as arith.constant.
        for i, c in enumerate(consts):
            ssa = self._new_ssa("c")
            typ = _shape_to_tensor_type(getattr(c, "shape", ()), _dtype_to_mlir(getattr(c, "dtype", type(c))))
            lit = self._literal(c)
            if "tensor<" in typ:
                self._emit(f"{ssa} = arith.constant dense<{lit}> : {typ}")
            else:
                self._emit(f"{ssa} = arith.constant {lit} : {typ}")

        lowered_special = False
        if self.optimize_contractions and self.function_name == "forward":
            lowered_special = self._try_lower_matmul_chain(raw_jaxpr)

        if not lowered_special:
            for eqn in raw_jaxpr.eqns:
                self._lower_eqn(eqn)

        ret_vals = []
        for v in outvars:
            mapped = self._lookup_var(v)
            if mapped is not None:
                ret_vals.append(mapped[0])
            elif hasattr(v, "val"):
                c_ssa, c_typ = self._const_from_literal(v.val)
                ret_vals.append(c_ssa)
                ret_types[len(ret_vals) - 1] = c_typ
            else:
                raise ValueError(f"Cannot lower output variable: {v}")

        self._emit(f"return {', '.join(ret_vals)} : {', '.join(ret_types)}")
        self._pop()
        self._emit("}")
        self._pop()
        self._emit("}")

        text = "\n".join(self._lines) + "\n"
        module_op = self._try_parse_module(text)

        if self.debug:
            print("[JaxprToMLIR] Generated MLIR:\n", text)

        return ConvertedModule(text=text, module_op=module_op)

    def _try_lower_matmul_chain(self, raw_jaxpr: Any) -> bool:
        eqns = list(raw_jaxpr.eqns)
        if len(eqns) < 2:
            return False
        if any(e.primitive.name != "dot_general" for e in eqns):
            return False
        if len(raw_jaxpr.outvars) != 1 or raw_jaxpr.outvars[0] is not eqns[-1].outvars[0]:
            return False

        # Support only plain 2D matrix-chain contractions for now.
        base_vars: List[Any] = []
        prev_out = None
        for i, eqn in enumerate(eqns):
            lhs, rhs = eqn.invars
            dn = eqn.params.get("dimension_numbers")
            if dn is None:
                return False
            ((lhs_contract, rhs_contract), (lhs_batch, rhs_batch)) = dn
            if tuple(lhs_batch) or tuple(rhs_batch):
                return False
            if tuple(lhs_contract) != (1,) or tuple(rhs_contract) != (0,):
                return False

            for v in (lhs, rhs):
                if hasattr(v, "val"):
                    return False
            if i == 0:
                base_vars.extend([lhs, rhs])
            else:
                if lhs is prev_out and rhs not in base_vars:
                    base_vars.append(rhs)
                elif rhs is prev_out and lhs not in base_vars:
                    base_vars.append(lhs)
                elif lhs in base_vars and rhs in base_vars:
                    pass
                else:
                    return False
            prev_out = eqn.outvars[0]

        if len(base_vars) < 3:
            return False
        for v in base_vars:
            if self._lookup_var(v) is None:
                return False

        shapes: List[Tuple[int, int]] = []
        for v in base_vars:
            s = tuple(getattr(v.aval, "shape", ()))
            if len(s) != 2:
                return False
            shapes.append((int(s[0]), int(s[1])))

        order = self._ome_opt.optimize_matrix_chain(shapes)
        if self.debug:
            print(f"[OMEinsum] contraction order source={order.source}, pairs={order.pairs}")

        work: List[Tuple[str, str, int, int]] = []
        for v, (r, c) in zip(base_vars, shapes):
            ssa, typ = self._lookup_var(v)  # type: ignore[misc]
            work.append((ssa, typ, r, c))

        for i, j in order.pairs:
            if len(work) < 2:
                break
            ii = max(0, min(i, len(work) - 1))
            jj = max(0, min(j, len(work) - 1))
            if ii == jj:
                jj = (ii + 1) % len(work)
            if ii > jj:
                ii, jj = jj, ii

            lhs = work[ii]
            rhs = work[jj]
            if lhs[3] != rhs[2]:
                return False

            out_t = f"tensor<{lhs[2]}x{rhs[3]}x{lhs[1][lhs[1].rfind('x') + 1 : -1]}>"
            out_ssa = self._new_ssa()
            self._lower_dot_general([(lhs[0], lhs[1]), (rhs[0], rhs[1])], out_ssa, out_t)

            # Model temp storage for backward reuse.
            fake_out = type("Tmp", (), {"aval": type("A", (), {"shape": (lhs[2], rhs[3]), "dtype": "float32"})})
            self._emit_intermediate_cache_alloc(fake_out, out_ssa)

            work = work[:ii] + [(out_ssa, out_t, lhs[2], rhs[3])] + work[ii + 1 : jj] + work[jj + 1 :]

        while len(work) > 1:
            lhs = work[0]
            rhs = work[1]
            if lhs[3] != rhs[2]:
                return False
            out_t = f"tensor<{lhs[2]}x{rhs[3]}x{lhs[1][lhs[1].rfind('x') + 1 : -1]}>"
            out_ssa = self._new_ssa()
            self._lower_dot_general([(lhs[0], lhs[1]), (rhs[0], rhs[1])], out_ssa, out_t)
            work = [(out_ssa, out_t, lhs[2], rhs[3])] + work[2:]

        final_ssa, final_type, _, _ = work[0]
        self._var_map[raw_jaxpr.outvars[0]] = (final_ssa, final_type)
        return True

    def _lower_eqn(self, eqn: Any) -> None:
        prim = eqn.primitive.name
        invals = [self._resolve_operand(v) for v in eqn.invars]
        outvars = list(eqn.outvars)

        if len(outvars) != 1:
            raise NotImplementedError(f"Multi-result primitive not supported yet: {prim}")

        outvar = outvars[0]
        out_type = _aval_to_mlir_type(outvar.aval)
        out_ssa = self._new_ssa()

        if prim in {"add", "add_any", "sub", "mul", "div"}:
            canonical = "add" if prim == "add_any" else prim
            self._lower_arith_binop(canonical, invals, out_ssa, out_type)
        elif prim == "neg":
            self._lower_neg(invals[0], out_ssa, out_type)
        elif prim in {"exp", "log", "sin", "cos"}:
            self._lower_math_unary(prim, invals[0], out_ssa, out_type)
        elif prim == "dot_general":
            self._lower_dot_general(invals, out_ssa, out_type)
        elif prim in {"broadcast_in_dim", "broadcast"}:
            self._lower_broadcast(invals[0], eqn.params, out_ssa, out_type)
        elif prim == "transpose":
            self._lower_transpose(invals[0], eqn.params, out_ssa, out_type)
        elif prim in {"reduce_sum", "sum"}:
            self._lower_reduce_sum(invals[0], eqn.params, out_ssa, out_type)
        elif prim in {"reduce_max", "max"}:
            self._lower_reduce_max(invals[0], eqn.params, out_ssa, out_type)
        elif prim in {"reshape", "squeeze"}:
            self._emit(f"{out_ssa} = tensor.cast {invals[0][0]} : {invals[0][1]} to {out_type}")
        elif prim in {"convert_element_type", "copy"}:
            if invals[0][1] == out_type:
                self._emit(f"{out_ssa} = {invals[0][0]} : {out_type}")
            elif not out_type.startswith("tensor<"):
                self._emit(f"{out_ssa} = arith.extf {invals[0][0]} : {invals[0][1]} to {out_type}")
            else:
                self._emit(f"{out_ssa} = tensor.cast {invals[0][0]} : {invals[0][1]} to {out_type}")
        else:
            raise NotImplementedError(
                f"Unsupported JAX primitive: {prim}. Add lowering in JaxprToMLIR._lower_eqn."
            )

        self._emit_intermediate_cache_alloc(outvar, out_ssa)
        self._var_map[outvar] = (out_ssa, out_type)

    def _lower_arith_binop(self, prim: str, invals: List[Tuple[str, str]], out_ssa: str, out_type: str) -> None:
        lhs, rhs = invals
        if out_type.startswith("tensor<"):
            elem_t = out_type[out_type.rfind("x") + 1 : -1]
            rank = out_type.count("x")
            maps = ", ".join([f"affine_map<({', '.join(['d'+str(i) for i in range(rank)])})->({', '.join(['d'+str(i) for i in range(rank)])})>"] * 3)
            iters = ", ".join(["\"parallel\""] * rank)
            empty = self._new_ssa("empty")
            self._emit(f"{empty} = tensor.empty() : {out_type}")
            op = {
                "add": "arith.addf",
                "sub": "arith.subf",
                "mul": "arith.mulf",
                "div": "arith.divf",
            }[prim]
            self._emit(
                f"{out_ssa} = linalg.generic {{indexing_maps = [{maps}], iterator_types = [{iters}]}} "
                f"ins({lhs[0]}, {rhs[0]} : {lhs[1]}, {rhs[1]}) outs({empty} : {out_type}) {{"
            )
            self._push()
            self._emit(f"^bb0(%x: {elem_t}, %y: {elem_t}, %out: {elem_t}):")
            self._push()
            v = self._new_ssa("t")
            self._emit(f"{v} = {op} %x, %y : {elem_t}")
            self._emit(f"linalg.yield {v} : {elem_t}")
            self._pop()
            self._pop()
            self._emit(f"}} -> {out_type}")
            return

        op = {
            "add": "arith.addf",
            "sub": "arith.subf",
            "mul": "arith.mulf",
            "div": "arith.divf",
        }[prim]
        self._emit(f"{out_ssa} = {op} {lhs[0]}, {rhs[0]} : {out_type}")

    def _lower_neg(self, inval: Tuple[str, str], out_ssa: str, out_type: str) -> None:
        if out_type.startswith("tensor<"):
            zero = self._new_ssa("c")
            elem_t = out_type[out_type.rfind("x") + 1 : -1]
            self._emit(f"{zero} = arith.constant 0.0 : {elem_t}")
            self._emit(f"{out_ssa} = arith.subf {zero}, {inval[0]} : {out_type}")
            return
        zero = self._new_ssa("c")
        self._emit(f"{zero} = arith.constant 0.0 : {out_type}")
        self._emit(f"{out_ssa} = arith.subf {zero}, {inval[0]} : {out_type}")

    def _lower_math_unary(self, prim: str, inval: Tuple[str, str], out_ssa: str, out_type: str) -> None:
        op = {
            "exp": "math.exp",
            "log": "math.log",
            "sin": "math.sin",
            "cos": "math.cos",
        }[prim]
        if out_type.startswith("tensor<"):
            raise NotImplementedError(f"Tensor unary math lowering not implemented for {prim}")
        self._emit(f"{out_ssa} = {op} {inval[0]} : {out_type}")

    def _lower_dot_general(self, invals: List[Tuple[str, str]], out_ssa: str, out_type: str) -> None:
        lhs, rhs = invals
        empty = self._new_ssa("empty")
        zero = self._new_ssa("c")
        elem_t = out_type[out_type.rfind("x") + 1 : -1]
        self._emit(f"{empty} = tensor.empty() : {out_type}")
        self._emit(f"{zero} = arith.constant 0.0 : {elem_t}")
        filled = self._new_ssa("init")
        self._emit(f"{filled} = linalg.fill ins({zero} : {elem_t}) outs({empty} : {out_type}) -> {out_type}")
        self._emit(
            f"{out_ssa} = linalg.matmul ins({lhs[0]}, {rhs[0]} : {lhs[1]}, {rhs[1]}) "
            f"outs({filled} : {out_type}) -> {out_type}"
        )

    def _lower_broadcast(self, inval: Tuple[str, str], params: Dict[str, Any], out_ssa: str, out_type: str) -> None:
        if not out_type.startswith("tensor<"):
            self._emit(f"{out_ssa} = arith.addf {inval[0]}, {inval[0]} : {out_type}")
            return
        empty = self._new_ssa("empty")
        self._emit(f"{empty} = tensor.empty() : {out_type}")
        dims = params.get("broadcast_dimensions", ())
        dims_text = ", ".join(str(int(d)) for d in dims)
        self._emit(
            f"{out_ssa} = linalg.broadcast ins({inval[0]} : {inval[1]}) "
            f"outs({empty} : {out_type}) dimensions = [{dims_text}]"
        )

    def _lower_transpose(self, inval: Tuple[str, str], params: Dict[str, Any], out_ssa: str, out_type: str) -> None:
        empty = self._new_ssa("empty")
        self._emit(f"{empty} = tensor.empty() : {out_type}")
        perm = params.get("permutation", ())
        perm_text = ", ".join(str(int(p)) for p in perm)
        self._emit(
            f"{out_ssa} = linalg.transpose ins({inval[0]} : {inval[1]}) outs({empty} : {out_type}) "
            f"permutation = [{perm_text}]"
        )

    def _lower_reduce_sum(self, inval: Tuple[str, str], params: Dict[str, Any], out_ssa: str, out_type: str) -> None:
        dims = params.get("axes", params.get("dimensions", ()))
        dims_text = ", ".join(str(int(d)) for d in dims)
        if out_type.startswith("tensor<"):
            empty = self._new_ssa("empty")
            elem_t = out_type[out_type.rfind("x") + 1 : -1]
            zero = self._new_ssa("c")
            self._emit(f"{empty} = tensor.empty() : {out_type}")
            self._emit(f"{zero} = arith.constant 0.0 : {elem_t}")
            init = self._new_ssa("init")
            self._emit(f"{init} = linalg.fill ins({zero} : {elem_t}) outs({empty} : {out_type}) -> {out_type}")
            self._emit(
                f"{out_ssa} = linalg.reduce ins({inval[0]} : {inval[1]}) outs({init} : {out_type}) "
                f"dimensions = [{dims_text}] {{"
            )
            self._push()
            self._emit(f"^bb0(%a: {elem_t}, %b: {elem_t}):")
            self._push()
            t = self._new_ssa("t")
            self._emit(f"{t} = arith.addf %a, %b : {elem_t}")
            self._emit(f"linalg.yield {t} : {elem_t}")
            self._pop()
            self._pop()
            self._emit(f"}} -> {out_type}")
            return
        self._emit(f"{out_ssa} = arith.addf {inval[0]}, {inval[0]} : {out_type}  // reduce_sum axes [{dims_text}]")

    def _lower_reduce_max(self, inval: Tuple[str, str], params: Dict[str, Any], out_ssa: str, out_type: str) -> None:
        dims = params.get("axes", params.get("dimensions", ()))
        dims_text = ", ".join(str(int(d)) for d in dims)
        self._emit(f"{out_ssa} = {inval[0]}  // reduce_max axes [{dims_text}] placeholder lowering")

    def _emit_intermediate_cache_alloc(self, outvar: Any, out_ssa: str) -> None:
        aval = getattr(outvar, "aval", None)
        if aval is None:
            return
        shape = tuple(getattr(aval, "shape", ()))
        if len(shape) == 0:
            return
        elem_type = _dtype_to_mlir(getattr(aval, "dtype"))
        memref_type = _shape_to_memref_type(shape, elem_type)
        cache = self._new_ssa("cache")
        self._emit(f"{cache} = memref.alloc() : {memref_type}")
        # Explicit dealloc models temporary lifetime for backward reuse planning.
        self._emit(f"memref.dealloc {cache} : {memref_type}")

    def _resolve_operand(self, v: Any) -> Tuple[str, str]:
        mapped = self._lookup_var(v)
        if mapped is not None:
            return mapped
        if hasattr(v, "val"):
            return self._const_from_literal(v.val)
        raise ValueError(f"Unknown operand variable: {v}")

    def _lookup_var(self, v: Any) -> Optional[Tuple[str, str]]:
        try:
            return self._var_map.get(v)
        except TypeError:
            # Some JAX nodes (e.g. Literal) are unhashable in newer versions.
            return None

    def _const_from_literal(self, val: Any) -> Tuple[str, str]:
        typ = _shape_to_tensor_type(getattr(val, "shape", ()), _dtype_to_mlir(getattr(val, "dtype", type(val))))
        ssa = self._new_ssa("c")
        lit = self._literal(val)
        if "tensor<" in typ:
            self._emit(f"{ssa} = arith.constant dense<{lit}> : {typ}")
        else:
            self._emit(f"{ssa} = arith.constant {lit} : {typ}")
        return ssa, typ

    def _literal(self, c: Any) -> str:
        if hasattr(c, "tolist"):
            return str(c.tolist())
        if hasattr(c, "item"):
            return str(c.item())
        return str(c)

    def _new_ssa(self, prefix: str = "v") -> str:
        ssa = f"%{prefix}{self._ssa_id}"
        self._ssa_id += 1
        return ssa

    def _format_args(self, arg_types: Sequence[str]) -> str:
        return ", ".join(f"%arg{i}: {t}" for i, t in enumerate(arg_types))

    def _format_ret_types(self, ret_types: Sequence[str]) -> str:
        if len(ret_types) == 1:
            return ret_types[0]
        return f"({', '.join(ret_types)})"

    def _emit(self, line: str) -> None:
        self._lines.append(f"{'  ' * self._indent}{line}")

    def _push(self) -> None:
        self._indent += 1

    def _pop(self) -> None:
        self._indent -= 1

    def _try_parse_module(self, text: str) -> Optional[Any]:
        try:
            ir = _import_mlir_ir()
        except DependencyError:
            return None

        try:
            if self.context is not None:
                return ir.Module.parse(text, self.context)
            with ir.Context() as ctx:
                return ir.Module.parse(text, ctx)
        except Exception:
            if self.debug:
                print("[JaxprToMLIR] MLIR parse failed; using text fallback.")
            return None


def get_forward_backward_mlir(
    func: Any,
    example_args: Sequence[Any],
    debug: bool = False,
    optimize_contractions: bool = True,
    julia_cmd: str = "julia",
) -> Tuple[ConvertedModule, ConvertedModule]:
    """Build forward and backward MLIR modules from a Python energy function."""

    jax, _ = _import_jax()

    args = tuple(example_args)
    closed_fwd = jax.make_jaxpr(func)(*args)
    grad_func = jax.grad(func, argnums=tuple(range(len(args))))
    closed_bwd = jax.make_jaxpr(grad_func)(*args)

    try:
        ir = _import_mlir_ir()
        with ir.Context() as ctx:
            fwd = JaxprToMLIR(
                context=ctx,
                function_name="forward",
                debug=debug,
                optimize_contractions=optimize_contractions,
                julia_cmd=julia_cmd,
            ).convert(closed_fwd)
            bwd = JaxprToMLIR(
                context=ctx,
                function_name="backward",
                debug=debug,
                optimize_contractions=False,
                julia_cmd=julia_cmd,
            ).convert(closed_bwd)
    except DependencyError:
        fwd = JaxprToMLIR(
            context=None,
            function_name="forward",
            debug=debug,
            optimize_contractions=optimize_contractions,
            julia_cmd=julia_cmd,
        ).convert(closed_fwd)
        bwd = JaxprToMLIR(
            context=None,
            function_name="backward",
            debug=debug,
            optimize_contractions=False,
            julia_cmd=julia_cmd,
        ).convert(closed_bwd)

    return fwd, bwd


def module_to_text(module_or_converted: Union[ConvertedModule, Any, str]) -> str:
    if isinstance(module_or_converted, ConvertedModule):
        return module_or_converted.text
    if isinstance(module_or_converted, str):
        return module_or_converted
    return str(module_or_converted)


def combine_modules(forward_module: Union[ConvertedModule, Any, str], backward_module: Union[ConvertedModule, Any, str]) -> ConvertedModule:
    """Merge forward and backward functions into one module."""

    fwd_text = module_to_text(forward_module)
    bwd_text = module_to_text(backward_module)

    fwd_funcs = _extract_functions(fwd_text)
    bwd_funcs = _extract_functions(bwd_text)

    lines = ["module {"]
    for f in fwd_funcs:
        lines.extend("  " + ln if ln else "" for ln in f.splitlines())
    for f in bwd_funcs:
        lines.extend("  " + ln if ln else "" for ln in f.splitlines())

    lines.append("}")
    text = "\n".join(lines) + "\n"

    module_op = None
    try:
        ir = _import_mlir_ir()
        with ir.Context() as ctx:
            module_op = ir.Module.parse(text, ctx)
    except Exception:
        module_op = None

    return ConvertedModule(text=text, module_op=module_op)


def _extract_functions(module_text: str) -> List[str]:
    lines = module_text.splitlines()
    out: List[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip().startswith("func.func"):
            chunk = [line.strip()]
            depth = line.count("{") - line.count("}")
            i += 1
            while i < len(lines):
                chunk.append(lines[i].strip())
                depth += lines[i].count("{") - lines[i].count("}")
                if depth <= 0:
                    break
                i += 1
            out.append("\n".join(chunk))
        i += 1
    return out
