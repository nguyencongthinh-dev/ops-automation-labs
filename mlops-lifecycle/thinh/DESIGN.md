# DESIGN.md — MLOps Lifecycle: Anomaly Detection Pipeline

## Tổng quan

Pipeline phát hiện drift trong metrics payment gateway (latency_p99, error_rate, rps), trigger retrain model, và swap phiên bản mới qua MLflow Registry alias.

**2 Anomaly Detectors được sử dụng:**
- **IsolationForest** (primary, được đăng ký vào MLflow Registry): tree-based, anomaly score = average path length trong random trees. Ngắn hơn = bất thường hơn.
- **LOF — Local Outlier Factor** (comparison, logged vào cùng MLflow run): density-based, phát hiện outlier local trong từng cluster bằng k-nearest-neighbor density ratio. LOF score >> 1 = mật độ thấp hơn láng giềng = anomaly.

Cả hai mô hình đều được huấn luyện trên baseline.csv. IsolationForest là model chính được serve và register. LOF được so sánh để validate decision của IsolationForest. Detector agreement rate được log vào MLflow.

---

## Sub-checkpoint 1: Drift Threshold

**Giá trị đã chọn: 0.15** (15% features bị drift theo Evidently DataDriftPreset).

**Cách chọn:** Trước tiên chạy drift_detector trên chính baseline.csv, chia 70/30 (2-tháng đầu làm reference, 1-tháng cuối làm current). Kết quả drift score = 0.04 — đây là "noise floor" khi không có drift thực sự. Từ đó chọn threshold = 0.15, tức 3.75× noise floor. Với drifted.csv, score thực đo được là 0.67 (2/3 features drifted), vượt threshold rõ ràng.

**Rủi ro nếu threshold quá thấp (ví dụ 0.05):** false positive — retrain trigger sau mỗi seasonal fluctuation bình thường (sáng/tối traffic khác nhau). Tốn compute và gây alert fatigue.

**Rủi ro nếu threshold quá cao (ví dụ 0.50):** false negative — bỏ sót drift thực, model tiếp tục serve với phân phối không còn phù hợp, precision/recall giảm âm thầm.

---

## Sub-checkpoint 2: Loại Drift

**Loại được detect: Data drift** — P(X) thay đổi, tức phân phối input features (latency_p99, error_rate, rps) đã dịch chuyển so với training data.

**Evidently DataDriftPreset detect:** Statistical test trên từng feature. Mặc định dùng Wasserstein distance cho numerical features. Khi share_of_drifted_columns > threshold → flag.

**Tại sao data drift phù hợp với bài toán này:** Payment gateway anomaly detection cần biết khi nào "bình thường mới" (new normal) khác với "bình thường cũ". Sau campaign, latency baseline tăng lên 156ms — model v1 train với baseline 120ms sẽ coi 156ms là anomaly dù thực ra là normal. Detect data drift cho phép retrain model với distribution mới trước khi precision giảm đáng kể.

**Concept drift (P(Y|X) thay đổi) không được detect trực tiếp** trong pipeline này vì không có ground truth labels trong production. Performance drift (proxy: theo dõi anomaly rate trend) được log vào MLflow mỗi lần drift check để visualize.

---

## Sub-checkpoint 3: Retrain Trigger Configuration

**Trigger type: Manual approval gate** — semi-automatic.

**Cadence:** Không có schedule cố định. Drift check được gọi khi có batch data mới (có thể integrate vào daily batch job). Nhưng promotion từ staging → production luôn yêu cầu human approval.

**Lý do chọn manual:** Model anomaly detection trong payment system ảnh hưởng trực tiếp đến on-call SLA. Một model tệ hơn được promote tự động có thể gây false negatives trên incident thực, hoặc alert storm từ false positives. Approval gate đảm bảo ML engineer review metric (anomaly_rate của v2 vs v1) trước khi cutover.

**Approval timeout:** Không implement timeout trong lab. Trong production, recommend 24h timeout — nếu không có approval trong 24h, staging version bị archive và drift check reset. Tránh trạng thái "staging model treo mãi không ai review".

**Nếu tự động hoàn toàn:** Có thể dùng A/B shadow mode — serve.py gọi cả v1 (production) và v2 (staging) song song trong 24h, so sánh anomaly_rate delta. Nếu delta < 5% và không có false negative trên known incident window → auto-promote. Ngưỡng 5% là conservative cho payment domain.

---

## Sub-checkpoint 4: Versioning và Rollback

**Chiến lược versioning:** MLflow Registry với aliases, không phụ thuộc vào version numbers.

- `production` alias → version đang serve
- `staging` alias → version candidate sau retrain
- Version numbers (1, 2, 3…) là immutable audit trail

**Tại sao alias tốt hơn version number trong code serve.py:** `mlflow.pyfunc.load_model("models:/anomaly-detector@production")` không thay đổi khi swap. Nếu hardcode version number, phải redeploy serve.py mỗi lần retrain.

**Rollback path:**
1. Phát hiện v2 underperform (precision giảm, alert storm): `MlflowClient.set_registered_model_alias("anomaly-detector", "production", "1")` — swap alias về v1.
2. Gọi `POST /reload` trên serve.py — load lại v1 từ registry.
3. Toàn bộ quá trình < 30 giây, không cần redeploy container.

**Ai có quyền rollback:** ML engineer on-call (có MLflow admin access). Trong production, rollback nên được wrap thành Runbook command với audit log.

**Retention policy:** Giữ tất cả registered versions vô thời hạn (artifacts tốn storage nhưng model IsolationForest < 1MB). Không xóa version cũ vì cần cho audit và rollback bất kỳ lúc nào.

---

## Kiến trúc component

```
baseline.csv (reference)
     |
     |──► pipeline.py ──► IsolationForest (primary, registered)
     |                   LOF (comparison, logged)
     |                   Detector Agreement Rate (MLflow metric)
     |                   MLflow Run ──► Registry v1 @production
     |
drifted.csv (current window)
     |
     |──► drift_detector.py
     |         | score=0.67 > threshold=0.15
     |         |
     |         | (DataDriftPreset + DataQualityPreset)
     |         | -> per-feature FeatureDriftDetail
     |         | -> HTML Report + Summary JSON
     |──► retrain.py
               |
               |── train IsoForest trên baseline+drift (sliding window)
               |── LOF comparison metrics logged
               |── MLflow Run -> Registry v2 @staging
               |── [HUMAN APPROVAL]
               |── set alias production -> v2
               |── POST /reload -> serve.py
```

---

## Sub-checkpoint 5: Cơ chế phát hiện drift — tại sao cần combined mode

Chỉ dùng `DataDriftPreset` (data drift) là chưa đủ. Data drift phát hiện khi P(X) thay đổi — tức phân phối input features dịch chuyển. Nhưng trong tình huống payment gateway, có thể xảy ra **concept drift**: P(Y|X) thay đổi mà P(X) vẫn ổn định. Ví dụ cụ thể: sau khi payment processor mới rollout, cùng một mức latency 180ms có thể là "bình thường mới" với processor cũ nhưng là "anomaly thực sự" với processor mới — hoặc ngược lại. Evidently sẽ không phát hiện điều này vì feature distribution không đổi.

`--check-mode combined` chạy song song 2 cơ chế: (1) Evidently `DataDriftPreset` trên feature distribution, và (2) đánh giá precision/recall của model hiện tại trên `holdout.csv` (tập có nhãn từ old pattern). Nếu một trong hai flag — `is_drift = True` hoặc `perf_is_degraded = True` — retrain sẽ được trigger. Ngưỡng performance mặc định là precision ≥ 0.70; nếu model v1 đạt 0.91 trên validation set ban đầu mà chỉ còn 0.62 trên holdout hiện tại, đó là tín hiệu concept drift rõ ràng dù feature score của Evidently vẫn thấp.

---

## Sub-checkpoint 6: Data selection strategy — sliding window vs alternatives

Khi retrain chỉ trên drift window (7 ngày gần nhất), model v2 overfit vào phân phối mới: nó học rằng latency 156ms là "bình thường" nhưng quên rằng hệ thống vẫn phải xử lý các batch job chạy theo pattern cũ. Thực nghiệm: train trên drift window → v2 precision trên `holdout.csv` (old pattern) giảm ~18% so với v1.

**Sliding window strategy** (baseline + drift window concat) cho kết quả tốt hơn vì model thấy cả 2 regime. Với `baseline.csv` (4320 rows) + `drifted.csv` (1008 rows), tổng training set là 5328 rows — đủ để IsolationForest không bị dominated bởi phân phối mới. Acceptance criterion: v2 precision và recall trên `holdout.csv` phải ≥ v1 precision/recall đo trên cùng tập đó.

Các alternative: (a) **Pure drift window** — đơn giản nhưng overfit như phân tích trên; (b) **Weighted sampling** (oversample baseline) — phức tạp hơn, hợp lý khi drift window rất nhỏ; (c) **Full historical concat** — an toàn nhất nhưng tốn compute khi data tích lũy nhiều tháng. Sliding window là trade-off tốt nhất cho lab này.

---

## Sub-checkpoint 7: Auto-rollback — threshold và policy

Sau khi v2 được promote lên `@production`, `post_deploy_monitor` chạy N polling cycles đánh giá precision trên `post_deploy_eval.csv` (200 rows có nhãn rõ ràng: 60% clear-normal, 40% clear-anomaly). Ngưỡng mặc định: `precision < 0.65` → auto-rollback.

Tại sao 0.65? Đây là ngưỡng bảo thủ — thấp hơn baseline 91% nhưng đủ xa để không trigger false rollback do sampling noise trên 200 rows. Tính toán: với 80 anomaly rows (40%), nếu model miss 30 → precision = 50/57 ≈ 0.88; nếu model hoàn toàn confused → precision ≈ 0.40. Ngưỡng 0.65 nằm ở điểm "model rõ ràng đang sai lệch nghiêm trọng".

Rollback flow: `client.set_registered_model_alias(MODEL_NAME, "archived", v2_version)` → `client.set_registered_model_alias(MODEL_NAME, "production", v1_version)` → `POST /reload`. Toàn bộ < 5 giây. Mọi sự kiện được append vào `outputs/audit_log.jsonl` với event key `auto_rollback_v2_to_v1`, bao gồm version bị demote, version được restore, precision trigger value, và cycle number.

---

## Observability: tại sao các metrics này quan trọng trong MLOps

MLOps monitoring khác service monitoring thông thường ở chỗ nguyên nhân degradation không phải lỗi code mà là **sự dịch chuyển của dữ liệu**. Drift score và precision/recall theo thời gian cho phép phát hiện model decay trước khi on-call nhận được complaint. Active version gauge và alias state table giải quyết vấn đề "đang serve version nào?" — câu hỏi thường mất nhiều phút tra cứu trong MLflow UI. Retrain event counter và auto-rollback counter tạo audit trail tối giản: số lần hệ thống tự can thiệp là tín hiệu về độ ổn định của distribution production. Các metrics này không thay thế MLflow experiment tracking mà bổ sung vào: MLflow lưu chi tiết từng run, Grafana visualize trend vận hành theo thời gian thực.

---

## Trade-offs đã chấp nhận

| Quyết định | Được | Mất |
|---|---|---|
| IsolationForest làm PRIMARY model | Train < 1s, explainable, no GPU, scale tốt | Không capture temporal patterns, mỗi row độc lập |
| LOF làm COMPARISON detector | Phát hiện local cluster outlier, validate IF decision | novelty=False nên không dùng được để serve real-time |
| Manual approval gate | An toàn, human oversight | Latency trong retrain loop (hours, không phải minutes) |
| Sliding window data strategy | Giữ cả old và new distribution patterns | Training set lớn hơn, cần cân bằng contamination |
| Local artifact store | Không cần S3 setup | Không scale multi-node, artifacts mất khi volume bị xóa |

## Tại sao IsolationForest và LOF bổ sung cho nhau

Hai detector sử dụng fundamentally different approaches:

- **IsolationForest**: Hoạt động bằng cách isolate anomalies qua random partitioning. Các điểm dễ isolate (path length ngắn) -> anomaly. Xuất sắc trong việc phát hiện **global outliers** — các điểm nằm xa khỏi vector chính.

- **LOF**: Hoạt động bằng cách so sánh local density. Một điểm là outlier khi density của nó thấp hơn đáng kể so với k láng giềng. Xuất sắc trong việc phát hiện **local cluster outliers** — các điểm nằm trong cluster nhỏ bị lấp bởi cluster lớn.

Khi cả 2 đồng ý (agreement rate cao): confidence cao rằng anomaly thực sự. Khi lệch nhau (IF phát hiện nhưng LOF không): có thể là global outlier nhưng không phải local cluster outlier — tín hiệu cần ML engineer review thủ công.

