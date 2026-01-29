I will remove the redundant "5. Compilation 3D" block from `export_and_gen_tikz.py` as it is now integrated into the iterative generation loop.
Then, I will run the `export_and_gen_tikz.py` script for all models with the `--compile` flag to verify that:

1. Both 2D and 3D figures are generated.
2. Overlap detection is performed during the generation process.
3. Parameters are adjusted automatically if overlaps are detected.
4. Final PDFs are produced successfully.

