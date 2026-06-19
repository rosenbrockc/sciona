# Library Deficits (All Resolved)

All primitives previously referenced by the conceptual benchmark CDGs and test cases have been successfully implemented and registered within the provider repositories.

## Grounded Primitives status

| Benchmark Case | Leaf Node | Expected FQDN | Status | Ingestion Details |
| :--- | :--- | :--- | :--- | :--- |
| **Merge Sort** | Split List | `algorithms.split_list_halves` | **RESOLVED** | Implemented in `algorithmic.divide_and_conquer.sorting` |
| | Merge Sorted Halves | `algorithms.merge_sorted_halves` | **RESOLVED** | Implemented in `algorithmic.divide_and_conquer.sorting` |
| **Shortest Path** | Initialize Distances | `algorithms.initialize_distances` | **RESOLVED** | Implemented in `algorithmic.graph.shortest_paths` |
| | Relax Edges | `algorithms.relax_edges` | **RESOLVED** | Implemented in `algorithmic.graph.shortest_paths` |
| **DSP Bandpass Filter** | Design Filter | `algorithms.design_bandpass_filter` | **RESOLVED** | Documented & matched via generic `scipy.signal` atoms |
| | Apply Filter | `algorithms.apply_bandpass_filter` | **RESOLVED** | Documented & matched via generic `scipy.signal` atoms |
| **Binary Search** | Compute Midpoint | `algorithms.binary_search_midpoint` | **RESOLVED** | Implemented in `algorithmic.search` |
| | Compare Target | `algorithms.binary_search_compare` | **RESOLVED** | Implemented in `algorithmic.search` |
| **FFT Spectral Analysis** | Apply Hann Window | `algorithms.apply_hann_window` | **RESOLVED** | Documented & matched via `fft_transform` (spectral window) |
| | Fourier Transform | `algorithms.compute_fft` | **RESOLVED** | Documented & matched via `fft_transform` (rfft) |
| | Magnitude Spectrum | `algorithms.extract_magnitude` | **RESOLVED** | Documented & matched via `welch_power_spectral_density` |
| **Strassen Matrix Multiply**| Split Into Submatrices | `algorithms.split_matrix_quadrants`| **RESOLVED** | Implemented in `algorithmic.divide_and_conquer.matrix` |
| | Assemble Result Matrix | `algorithms.combine_matrix_quadrants`| **RESOLVED** | Implemented in `algorithmic.divide_and_conquer.matrix` |
| **String Edit Distance** | Initialize DP Table | `algorithms.init_edit_distance_table`| **RESOLVED** | Implemented in `algorithmic.dynamic_programming.string_algo` |
| | Fill DP Table | `algorithms.fill_edit_distance_table`| **RESOLVED** | Implemented in `algorithmic.dynamic_programming.string_algo` |

*Note: All new atoms compile cleanly and have positive verification verdicts in their respective review bundles.*
