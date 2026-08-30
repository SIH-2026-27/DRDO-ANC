# DRDO-ANC

## Evaluation Matrix

GTCRN was evaluated on the manifest benchmark across **60 benchmark cases** in both **offline and streaming modes**, resulting in **120 successful evaluations (120/120)**.

### Overall Performance

| Model | Mode | Cases | Mean SNR (dB) | Mean SI-SDR (dB) | Mean STOI | Mean PESQ | Median RTF | Mean RTF |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **GTCRN** | Offline + Streaming | 120 | **10.8097** | **10.4654** | **0.6243** | **1.5026** | **1.4682** | **5.8803** |

**Evaluation status:** 120/120 successful, 0 failed.

### Performance by Input SNR

| Model | Input SNR | Mean SNR (dB) | Mean SI-SDR (dB) | Mean STOI | Mean PESQ |
|---|---:|---:|---:|---:|---:|
| **GTCRN** | 0 dB | 9.5687 | 9.0732 | 0.5976 | 1.4498 |
| **GTCRN** | +5 dB | 12.0506 | 11.8575 | 0.6509 | 1.5555 |

### Performance by Noise Category

| Model | Noise Category | Mean SNR (dB) | Mean SI-SDR (dB) | Mean STOI | Mean PESQ |
|---|---|---:|---:|---:|---:|
| **GTCRN** | Impulsive Firearms | 10.3880 | 9.9751 | 0.6198 | 1.3654 |
| **GTCRN** | UAV Drone | 11.8250 | 11.6740 | 0.6184 | 1.5342 |
| **GTCRN** | Vehicle Engine | 10.2160 | 9.7470 | 0.6346 | 1.6082 |

### Evaluation Notes

- **Successful evaluations:** 120/120
- **Failed evaluations:** 0/120
- **Benchmark cases:** 60
- **Evaluation modes:** Offline and Streaming
- **Median RTF:** 1.4682
- **Mean RTF:** 5.8803

### Reproduction

```powershell
.venv311\Scripts\python.exe scripts\run_df3_manifest_benchmark.py --model GTCRN