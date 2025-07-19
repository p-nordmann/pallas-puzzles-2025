import functools
import inspect
import os
import random
import time

import jax
import jax.experimental.pallas as pl
import jax.numpy as jnp
import jax.random as jrandom
from IPython.display import HTML


def load_pups():
    with open("../assets/pups.txt", "r") as f:
        pups = f.read().split(" ")
    random.shuffle(pups)
    return pups


pups = load_pups()
pup_idx = 0


def fetch_a_puppy():
    global pups, pup_idx
    pup = pups[pup_idx % len(pups)]
    pup_idx = pup_idx + 1
    return HTML(
        f"""
        <p><strong>Correct!</strong></p>
        <video alt="test" controls autoplay>
            <source src="https://openpuppies.com/mp4/{pup}.mp4"  type="video/mp4">
        </video>
        """
    )


def test(
    puzzle_fn,
    puzzle_spec_fn,
    nelem: dict[str, int] | None = {},
    B: dict[str, int] | None = {"B0": 32},
):
    interpret = "INTERPRET" in os.environ and os.environ["INTERPRET"] == "False"
    rtol: float = 1e-3
    atol: float = 1e-3

    # find the input sizes and block sizes #########################################################
    nelem = dict() if nelem is None else dict(nelem)
    B = dict() if B is None else dict(B)
    for i in range(3):
        if f"N{i}" in nelem and f"B{i}" not in B:
            B[f"B{i}"] = 32

    signature = inspect.signature(puzzle_spec_fn)
    args = {}
    for n, p in signature.parameters.items():
        print(p)
        args[n + "_ref"] = ([d.size for d in p.annotation.dims], p)
    args["z_ref"] = ([d.size for d in signature.return_annotation.dims], None)

    # generate random inputs #######################################################################
    rand_key = jrandom.PRNGKey(int(time.time()))
    pl_args = []
    float_dtype = jnp.float32
    for _, (v, t) in args.items():
        rand_key, subkey = jrandom.split(rand_key)
        if t is not None:
            dtype = t.annotation.dtypes[0]
            if dtype.startswith("float"):
                float_dtype = jnp.dtype(dtype)
        else:
            dtype = float_dtype
        pl_args.append((jrandom.uniform(subkey, v) - 0.5).astype(jnp.dtype(dtype)))
        if t is not None and t.annotation.dtypes[0] == "int32":
            rand_key, subkey = jrandom.split(rand_key)
            pl_args[-1] = jrandom.randint(
                subkey, v, minval=-100000, maxval=100000
            ).astype(jnp.int32)
    # evaluate reference implementation ############################################################
    z_ = puzzle_spec_fn(*pl_args[:-1])

    # evaluate the pallas implementation ###########################################################
    grid = (
        pl.cdiv(nelem["N0"], B["B0"]),
        pl.cdiv(nelem.get("N1", 1), B.get("B1", 1)),
        pl.cdiv(nelem.get("N2", 1), B.get("B2", 1)),
    )
    out_specs = pl.BlockSpec(z_.shape, lambda i, j, k: (0,) * z_.ndim)
    out_shape = jax.ShapeDtypeStruct(z_.shape, z_.dtype)
    in_specs = [
        pl.BlockSpec(v.shape, lambda i, j, k, _out=(0,) * v.ndim: _out)
        for v in pl_args[:-1]
    ]
    z = pl.pallas_call(
        functools.partial(puzzle_fn, **B),
        grid=grid,
        in_specs=in_specs,
        out_specs=out_specs,
        out_shape=out_shape,
        interpret=interpret,
        debug=False,
    )(*pl_args[:-1])

    # check results ################################################################################
    match = jnp.allclose(z, z_, rtol=rtol, atol=atol)
    if not match:
        print("Results do not match.")
        print("Yours:", z)
        print("Spec:", z_)
        print(jnp.allclose(z, z_))
        print(f"Error: {jnp.linalg.norm(z - z_) / jnp.linalg.norm(z_)}")
        return z, z_

    return fetch_a_puppy()
