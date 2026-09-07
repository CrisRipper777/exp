# R0-P6.5A: Hybrid Functional Teacher (observed data only)

- dataset=Grocery | n_relations=3998
- hybrid definition: Q_text_h=Q_text_1, Q_visual_h=Q_visual_1, Q_joint_h=Q_joint_1+S_second | max|Q_joint_1-(Q_text_1+Q_visual_1)|=2.22e-15

## Joint utility ranking (exact Q_joint vs estimates)
- spearman_exact_vs_hybrid=+0.935 | kendall_exact_vs_hybrid=+0.895 | spearman_exact_vs_first=+0.967 | spearman_exact_vs_second=+0.893

## Pooled ranking (exact vs estimates over text/visual/joint)
- spearman_exact_vs_hybrid=+0.955 | kendall_exact_vs_hybrid=+0.921 | spearman_exact_vs_first=+0.967 | spearman_exact_vs_second=+0.927

## Action accuracy vs best_exact
- first=0.955 | second=0.892 | hybrid=0.889 | prior=0.480

## Subset action accuracy

| subset | n | first | second | hybrid |
|---|---|---|---|---|
| all | 3998 | 0.955 | 0.892 | 0.889 |
| absS_norm_gt_0.05 | 1179 | 0.922 | 0.725 | 0.720 |
| absS_norm_gt_0.10 | 572 | 0.890 | 0.612 | 0.570 |
| S_exact_gt_0 | 1897 | 0.949 | 0.934 | 0.884 |
| S_exact_lt_0 | 2085 | 0.965 | 0.858 | 0.897 |
| receiver_correct_full_True | 3236 | 0.955 | 0.868 | 0.874 |
| receiver_correct_full_False | 762 | 0.957 | 0.993 | 0.950 |

## Confusion (rows=exact, cols=hybrid)
- [[337, 18, 20, 12], [0, 588, 0, 80], [0, 0, 883, 139], [0, 17, 159, 1745]]

_Observed numbers only._
