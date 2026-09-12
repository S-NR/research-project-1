# Packaged input snapshot

This folder contains the canonical unnumbered input files used by the pipeline.
The uploaded `(1)` annual copies were byte-identical duplicates and are not
packaged. The unnumbered coordinate file was selected because it exactly matches
the coordinates in the prior 210-candidate and DK result tables; the numbered
coordinate copy differed only by floating-point serialization noise.

Run `python run_kodagu_dk_pipeline.py --validate-inputs` to verify dimensions,
keys, completeness, redundant AlphaEarth equality, and SHA-256 identities.
