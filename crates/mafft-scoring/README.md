# mafft-scoring

> Substitution matrices and gap penalties for [rust-MAFFT](https://github.com/luksgrin/rust-MAFFT).

> **Acknowledgment.** rust-MAFFT is a port of [MAFFT](https://mafft.cbrc.jp/alignment/software/)
> by **Kazutaka Katoh** and colleagues (CBRC). The science is theirs.
> If you use this crate in published work, please cite
> [Katoh & Standley 2013](https://doi.org/10.1093/molbev/mst010);
> full guidance at [the project citation page](https://luksgrin.github.io/rust-MAFFT/citation/).

The MAFFT scoring stack ported to pure Rust: BLOSUM62 (default for
protein), JTT, transmembrane (TM), and DNA matrices, plus the
gap-opening / gap-extension constants that every alignment mode reads.
Also hosts `calcW` — the inner-loop SP-component computation — whose
exact multiply-then-add shape matches the reference C output, a
necessary condition for byte-identity with C MAFFT.

## Install

```sh
cargo add mafft-scoring
```

Most callers want the top-level [`mafft`](https://crates.io/crates/mafft)
crate, which uses these matrices implicitly. Depend on `mafft-scoring`
directly only when you need to compose a custom scoring context.

## Documentation

API reference: [docs.rs/mafft-scoring](https://docs.rs/mafft-scoring)

Project documentation: [luksgrin.github.io/rust-MAFFT](https://luksgrin.github.io/rust-MAFFT/)

## License

MIT for the Rust port; the matrix constants and `calcW` algorithm are
derived from upstream MAFFT under BSD-3-Clause. See `LICENSE-MIT` and
`LICENSE-BSD` in the workspace root.
