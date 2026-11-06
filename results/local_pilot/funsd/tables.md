# Result tables

## Model x condition (mean ANLS, 95% cluster bootstrap CI)

| Model | Condition | n | Documents | Mean ANLS | 95% CI |
| :--- | :--- | ---: | ---: | ---: | :--- |
| windows_ocr+qwen2.5:1.5b/Q4_K_M | C0_clean_end2end | 60 | 20 | 0.0733 | [0.0149, 0.1444] |
| windows_ocr+qwen2.5:1.5b/Q4_K_M | C2_oracle_layout_crop | 60 | 20 | 0.0906 | [0.0351, 0.1504] |
| windows_ocr+qwen2.5:1.5b/Q4_K_M | C3_oracle_text_injection | 60 | 20 | 0.0772 | [0.0198, 0.1477] |
| windows_ocr+qwen2.5:1.5b/Q4_K_M | C3b_shuffled_text | 60 | 20 | 0.103 | [0.0419, 0.1623] |

## Paired model comparisons

| Comparison | n | Mean diff | 95% CI | p (permutation) | p (Holm) |
| :--- | ---: | ---: | :--- | ---: | ---: |
