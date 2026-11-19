# Vendored taffy

Upstream: taffy 0.13.0 from crates.io (DioxusLabs/taffy at 45a56299d366ddb383e593a1f0372158d00e8530), MIT (`LICENSE`, `rustfmt.toml` from that commit).

Changes: a leaf's measure function may return a `LeafMeasure` (its size and an optional first baseline) instead of a size, and the leaf's layout output reports that baseline as `first_baselines.y`, so text leaves take part in `align-items: baseline` (`src/compute/leaf.rs`, `src/tree/taffy_tree.rs`, and the `LeafMeasure` re-exports in `src/compute/mod.rs` and `src/lib.rs`). The manifest drops `[profile.release]`, which Cargo ignores outside the workspace root.

`compute_root_layout` and `TaffyTree::compute_layout_with_measure` return the root's first vertical baseline; `compute_layout` continues to return `()` (`src/compute/mod.rs`, `src/tree/taffy_tree.rs`).
