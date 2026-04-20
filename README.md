# Frame Potential Project

This folder contains CPU and GPU implementations to estimate the frame potential of parameterized quantum circuits and compare it to the Haar reference.

Main use cases:

- Evaluate ansatz expressibility via `F^(t)` and `F/F_Haar`.
- Compare CPU vs GPU runtime.
- Run reproducible experiments from scripts and notebooks.

## Contents

- `frame_potential.py`: CPU implementation (NumPy + Qiskit).
- `frame_potential_gpu.py`: GPU-accelerated implementation (PyTorch + Qiskit).
- `circuit_generation.py`: built-in ansatz constructors.
- `schuster_ansatz.py`: additional ansatz helpers.
- `test_frame_potential.ipynb`: CPU notebook walkthrough.
- `test_frame_potential_gpu.ipynb`: GPU notebook walkthrough.
- `tests.ipynb`: extra experiments.

## Install


### Optional GPU (PyTorch)

Install PyTorch

Here in this porject i coded a version dependant of my machine that is equiped with a intel graphic card that does not work as intended with cuda.

In the future i intend to write this in a cuda frame work in odred to run this code in HPC


## Notebooks

- `test_frame_potential.ipynb`: CPU reference workflow, sanity checks, convergence and plots.
- `test_frame_potential_gpu.ipynb`: GPU transfer, manual einsum block checks, and CPU/GPU comparison.

## Returned Metrics

CPU and GPU pipelines return dictionaries containing:

- `frame_potential`
- `variance`
- `fidelity_error`
- `haar_value`
- `delta`
- `ratio`

The GPU pipeline also reports `device` and `dtype`.

## Troubleshooting

- `Required aspect fp64 is not supported on the device`:
	use `torch.complex64`; the GPU code falls back to float32 accumulation when fp64 is unavailable.
- `ModuleNotFoundError` in notebooks:
	confirm the selected Jupyter kernel uses the same virtual environment where dependencies are installed.
- Very slow runs:
	reduce `n_samples`, reduce qubits/repetitions, or tune `batch_size`.

## Notes

- Sampling unitary matrices uses Qiskit and runs on CPU.
- GPU acceleration applies to the pairwise frame-potential contraction stage.
- For reproducibility, set `seed` in both CPU and GPU high-level functions.
