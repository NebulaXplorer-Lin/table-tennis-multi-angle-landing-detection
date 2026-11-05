# B new-locator event exploration

Train new B ball locators with a frozen 13-frame TCN, compare no-table/YOLO-geometry/YOLO-visual features and reflect on full-video validation errors.

Same B Global+Local sha, association, thresholds 0.7/0.7, PP114/40 scene split and OpenTT game1-4/game5, 50/50 source sampling, seed, AdamW 3e-4 wd1e-4, effective batch8, 40 epochs patience8.

40 complete PP validation videos; original frames ±1 and original coordinates 50px; per-view macro F1, tie fewer FP. No public/test tuning.

Historical 79.56% uses old locators and old table; historical comparison is not an isolated locator ablation. All new arms share stable motion mask; table arms differ in table features.

YOLO confidence remains0.1; group22 known misses. Compare event by-view/group outcomes before adoption.

Arm: concat
