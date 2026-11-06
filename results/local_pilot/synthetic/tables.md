# Result tables

## Model x condition (mean ANLS, 95% cluster bootstrap CI)

| Model | Condition | n | Documents | Mean ANLS | 95% CI |
| :--- | :--- | ---: | ---: | ---: | :--- |
| windows_ocr+qwen2.5:1.5b/Q4_K_M | C0_clean_end2end | 40 | 10 | 0.7333 | [0.7333, 0.7333] |
| windows_ocr+qwen2.5:1.5b/Q4_K_M | C2_oracle_layout_crop | 40 | 10 | 1.0 | [1.0, 1.0] |
| windows_ocr+qwen2.5:1.5b/Q4_K_M | C3_oracle_text_injection | 40 | 10 | 1.0 | [1.0, 1.0] |
| windows_ocr+qwen2.5:1.5b/Q4_K_M | C3b_shuffled_text | 40 | 10 | 1.0 | [1.0, 1.0] |

## Paired model comparisons

| Comparison | n | Mean diff | 95% CI | p (permutation) | p (Holm) |
| :--- | ---: | ---: | :--- | ---: | ---: |
