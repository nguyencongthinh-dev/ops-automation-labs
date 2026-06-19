# Hướng dẫn chạy MLOps Pipeline từ đầu đến cuối

Quy trình chạy hệ thống MLOps phát hiện bất thường bao gồm các bước sau:
1. Khởi động hệ thống (MLflow, Prometheus, Grafana, PostgreSQL, Pushgateway) bằng cách chạy `bash lab-mlops-lifecycle/data-pack/scripts/start_stack.sh` từ thư mục gốc của workspace.
2. Tạo dữ liệu bằng cách chạy `uv run python lab-mlops-lifecycle/data-pack/data/generate_data.py`.
3. Thiết lập biến môi trường MLflow và huấn luyện mô hình v1 trên dữ liệu baseline:
   ```bash
   export MLFLOW_TRACKING_URI=http://localhost:5000
   uv run python thinh/pipeline.py --data lab-mlops-lifecycle/data-pack/data/baseline.csv
   ```
4. Phục vụ mô hình v1 sử dụng FastAPI:
   ```bash
   uv run python thinh/serve.py
   ```
5. Kích hoạt giám sát drift, tái huấn luyện và giám sát sau triển khai (tự động rollback):
   ```bash
   uv run python thinh/retrain.py \
     --reference lab-mlops-lifecycle/data-pack/data/baseline.csv \
     --current lab-mlops-lifecycle/data-pack/data/drifted.csv \
     --holdout lab-mlops-lifecycle/data-pack/data/holdout.csv \
     --post-deploy-eval lab-mlops-lifecycle/data-pack/data/post_deploy_eval.csv \
     --auto-approve
   ```
   *(Lưu ý: Bỏ tham số `--auto-approve` nếu bạn muốn phê duyệt thủ công mô hình trước khi thăng cấp lên production).*
