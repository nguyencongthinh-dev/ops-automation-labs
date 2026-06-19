# SUBMIT.md — Reflection: MLOps Lifecycle Lab

## Câu 1: Drift threshold bạn chọn là bao nhiêu và tại sao?

Threshold là **0.15** (15% features drifted theo Evidently DataDriftPreset).

Cách chọn empirical: chạy `drift_detector.py` trên chính baseline.csv, split 70/30 (3024 rows reference, 1296 rows current). Kết quả drift score = **0.00** — hoàn toàn 0, vì cùng distribution. Sau đó thử với seasonal noise (thêm Gaussian noise nhỏ σ=0.5): score ~0.04. Threshold 0.15 = 3.75× noise floor, đảm bảo không bị false positive từ intraday traffic variation.

Validation với `data/drifted.csv`: drift score đo được = **0.67** (2/3 features bị detect là drifted: latency_p99 và rps — Wasserstein distance >> threshold). Vượt ngưỡng 0.15 bằng 4.5×. Nếu threshold = 0.05: false positive từ seasonal variation. Nếu threshold = 0.50: bỏ sót giai đoạn đầu khi chỉ 1 feature bắt đầu dịch chuyển.

---

## Câu 2: Điều gì xảy ra nếu model v2 sau retrain lại tệ hơn v1?

Pipeline có **3 lớp bảo vệ**:

**Lớp 1 — Holdout validation (trước khi register staging):** `retrain.py --holdout data/holdout.csv` đánh giá v2 trực tiếp trên tập holdout (old pattern) trước khi đẩy lên registry. Nếu precision v2 < precision v1 trên holdout, log cảnh báo vào MLflow và audit_log.jsonl.

**Lớp 2 — Manual approval gate:** ML engineer xem anomaly_rate và holdout metrics của v2 trước khi gõ `y` promote. Output terminal in rõ: `v2 precision: X.XXXX  recall: X.XXXX`. Nếu các chỉ số tệ hơn, gõ `N` → v2 ở lại `@staging`, không ảnh hưởng production.

**Lớp 3 — Post-deploy auto-rollback:** Sau khi v2 được promote, `post_deploy_monitor` chạy 24 cycles đánh giá precision trên `post_deploy_eval.csv` (200 rows có label). Nếu precision < 0.65: `set_registered_model_alias("production", v1_version)` + `POST /reload` ngay lập tức. Toàn bộ < 5 giây. Sự kiện được ghi vào `outputs/audit_log.jsonl` với event `auto_rollback_v2_to_v1`.

---

## Câu 3: Sự khác biệt giữa data drift và concept drift?

**Data drift**: phân phối input thay đổi — P(X) thay đổi, nhưng mối quan hệ X→Y giữ nguyên. Ví dụ trong lab này: latency baseline tăng từ 120ms lên 156ms (+30%) vì thêm 3rd-party integration. Model v1 coi 156ms là bình thường nhưng bây giờ bị IsolationForest coi là nghi ngờ do học trên distribution cũ.

**Concept drift**: mối quan hệ input-output thay đổi — P(Y|X) thay đổi dù P(X) ổn định. Ví dụ: cùng latency 180ms trước đây là anomaly (99th percentile), nhưng sau khi scale up infra thì 180ms là bình thường hoàn toàn. Một số incident từ old pattern được relabel là normal vì SLA mới thay đổi.

Evidently DataDriftPreset trong lab này detect **data drift** bằng statistical tests (Wasserstein distance cho numerical features). Concept drift được detect gián tiếp qua `--check-mode combined`: đánh giá precision/recall của model trên `labeled-current` với nhãn thực tế. Nếu precision < 0.70 dù drift score thấp → đó là concept drift.

---

## Câu 4: Tại sao blue-green swap quan trọng hơn replace file trực tiếp?

**Replace file trực tiếp** tạo ra 3 vấn đề không chấp nhận được trong production:
1. **Race condition**: serve.py đang xử lý request dùng model cũ, đồng thời file bị ghi đè → corrupted pickle deserialization → crash hoặc wrong predictions.
2. **Không có rollback**: version cũ đã bị overwrite, phải retrain lại từ đầu — trong payment system, downtime 30 phút = mất rất nhiều transaction.
3. **Không có audit trail**: không biết lúc nào file được thay và bởi ai.

**Blue-green qua MLflow alias**: alias `production` được swap atomically — một atomic write operation trong PostgreSQL (MLflow backend). Serve.py chỉ gọi `mlflow.sklearn.load_model("models:/anomaly-detector@production")` khi nhận `/reload`. Tất cả in-flight requests trước đó hoàn thành với v1. Swap + reload mất < 5 giây. Cả v1 và v2 tồn tại song song trong registry với immutable version numbers — audit trail đầy đủ.

---

## Câu 5: Nếu automate approval gate, dùng metric gì và threshold nào?

Dùng **precision delta + anomaly rate constraint + holdout validation**. Điều kiện auto-promote khi thỏa cả 3:

1. `precision_v2_holdout >= precision_v1_holdout × 0.97` — v2 không tệ hơn v1 quá 3% trên old pattern
2. `abs(lof_anomaly_rate - if_anomaly_rate) < 0.03` — IsolationForest và LOF đồng thuận (detector agreement rate), nếu lệch nhau quá: 2 detector đang "tranh cãi", cần human review
3. `0.01 <= v2_train_anomaly_rate <= 0.10` — không bị degenerate (flag tất cả hoặc không flag gì)

Ngưỡng 3% delta conservative cho payment domain — 3% trên 1000 RPM = 30 missed anomalies/phút. Thêm điều kiện LOF agreement là điểm phân biệt so với simple threshold: khi cả 2 detector độc lập (tree-based + density-based) đồng ý với nhau, confidence tăng lên đáng kể. Nếu auto-promote fail: đẩy alert cho ML engineer review trong 4h với ML metrics summary đính kèm.
