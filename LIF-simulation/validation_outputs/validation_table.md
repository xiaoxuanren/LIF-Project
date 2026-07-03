# Spontaneous-bursting validation - parameter + results table

One row per run. Build seed is fixed at `7` for every run (identical wiring); the `seed` column is the membrane-noise seed that distinguishes runs.

| run_label | network_source | num_clusters | N | within_prob_base | between_prob_base | realized_within | realized_between | hub_fraction | hub_between_prob | hub_weight_scale | hub_reciprocal_factor | noise_sigma | depressing | delta_q | tau_q | baseline_mean | baseline_sd | adapt_increment_scale | adapt_tau_scale | exc_weight_scale | duration_ms | seed | mean_rate_Hz | burst_freq_Hz | peak_AF_steady | pct_in_burst | IB_rate_Hz | runaway_flag | n_spikes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| validated_config (A) | clustered | 15 | 228 | 0.5 | 0.15 | 0.3702 | 0.000474 | 0.1 | 0.4 | 1.5 | 2.0 | 1.4 | True | 0.8 | 1500 | 0.0 | 0.0 | 1.0 | 3.0 | 4.0 | 30000 | 100 | 2.1789 | 1.8621 | 0.7061 | 56.56 | 0.9843 | False | 15303 |
| noise_off_control (B.2) | clustered | 15 | 228 | 0.5 | 0.15 | 0.3702 | 0.000474 | 0.1 | 0.4 | 1.5 | 2.0 | 0.0 | True | 0.8 | 1500 | 0.0 | 0.0 | 1.0 | 3.0 | 4.0 | 3000 | 100 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | False | 0 |
| seed_B (B.3) | clustered | 15 | 228 | 0.5 | 0.15 | 0.3702 | 0.000474 | 0.1 | 0.4 | 1.5 | 2.0 | 1.4 | True | 0.8 | 1500 | 0.0 | 0.0 | 1.0 | 3.0 | 4.0 | 3000 | 200 | 2.2478 | 1.0 | 0.7632 | 54.15 | 1.0626 | False | 1816 |
| 4AP_incr_0.5 (C) | clustered | 15 | 228 | 0.5 | 0.15 | 0.3702 | 0.000474 | 0.1 | 0.4 | 1.5 | 2.0 | 1.4 | True | 0.8 | 1500 | 0.0 | 0.0 | 0.5 | 3.0 | 4.0 | 6000 | 100 | 2.5754 | 2.2 | 0.614 | 43.6 | 1.5163 | False | 3901 |
| 4AP_incr_0.25 (C) | clustered | 15 | 228 | 0.5 | 0.15 | 0.3702 | 0.000474 | 0.1 | 0.4 | 1.5 | 2.0 | 1.4 | True | 0.8 | 1500 | 0.0 | 0.0 | 0.25 | 3.0 | 4.0 | 6000 | 100 | 2.7737 | 1.8 | 0.5088 | 25.02 | 2.1486 | False | 4324 |
| suppressed_incr_2.0 (C) | clustered | 15 | 228 | 0.5 | 0.15 | 0.3702 | 0.000474 | 0.1 | 0.4 | 1.5 | 2.0 | 1.4 | True | 0.8 | 1500 | 0.0 | 0.0 | 2.0 | 3.0 | 4.0 | 6000 | 100 | 1.8325 | 1.0 | 0.6535 | 73.91 | 0.4939 | False | 2756 |
