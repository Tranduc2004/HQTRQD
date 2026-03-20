# GIAI ĐOẠN 6: PHÁT TRIỂN SẢN PHẨM WEB DSS
# Hướng dẫn chạy Web Application

## Cài đặt thư viện

Trước khi chạy ứng dụng, cần cài đặt các thư viện sau:

```bash
pip install streamlit plotly
```

## Chạy ứng dụng

Mở terminal/command prompt trong thư mục dự án và chạy lệnh:

```bash
streamlit run app.py
```

Ứng dụng sẽ tự động mở trong trình duyệt tại địa chỉ: http://localhost:8501

## Cấu trúc Web Application

### 1. 🏠 Trang chủ (Home)
- Giới thiệu tổng quan về hệ thống
- Hiển thị trạng thái DSS và AHP
- Thống kê nhanh về các đánh giá

### 2. 📝 Form đánh giá (Customer Form)
**Chức năng:**
- Chọn hạng vé (Business/Economy/Eco Plus)
- Nhập tên hành khách (tùy chọn)
- Đánh giá 6 tiêu chí dịch vụ (thang điểm 1-5)
- Gửi dữ liệu về backend

**Kết quả nhận được:**
- Dự đoán mức độ hài lòng (Satisfied/Dissatisfied)
- Độ tin cậy của dự đoán
- Điểm rủi ro (Risk Score)
- Phân tích tác động của từng tiêu chí
- Khuyến nghị cải thiện

### 3. 📊 Dashboard quản trị (Admin Dashboard)
**Tab Overview:**
- KPI metrics (tổng đánh giá, tỷ lệ hài lòng, điểm rủi ro trung bình)
- Biểu đồ phân bố hài lòng theo hạng vé
- Phân bố mức độ rủi ro
- Timeline xu hướng theo thời gian

**Tab XGBoost Results:**
- Phân bố độ tin cậy của mô hình
- Điểm rủi ro theo hạng vé
- Ma trận dự đoán

**Tab AHP Weights:**
- Trọng số AHP theo từng hạng vé
- Top 3 ưu tiên cho mỗi hạng
- Consistency Ratio (CR)

**Tab Alerts:**
- Danh sách hành khách có rủi ro cao
- Cảnh báo theo mức độ (HIGH/MEDIUM/LOW)
- Chi tiết đánh giá của từng hành khách

### 4. ℹ️ Giới thiệu hệ thống (About System)
- Kiến trúc hệ thống (6 giai đoạn)
- Hiệu suất mô hình
- Hướng dẫn sử dụng
- Thông tin kỹ thuật

## Tính năng nổi bật

✅ **Real-time Analysis**: Phân tích ngay lập tức khi submit form
✅ **Interactive Dashboard**: Biểu đồ tương tác với Plotly
✅ **Risk Assessment**: Đánh giá rủi ro tự động
✅ **Export Data**: Xuất dữ liệu CSV cho báo cáo
✅ **Session Management**: Lưu trữ đánh giá trong session
✅ **Responsive Design**: Giao diện thân thiện, dễ sử dụng

## Lưu ý quan trọng

1. **Yêu cầu file trước khi chạy:**
   - `airline_dss_system.pkl` - Hệ thống DSS đã train (từ Stage 5)
   - `ahp_weights_all_classes.pkl` - Trọng số AHP (từ Stage 4)

2. **Chạy các notebook trước:**
   - Nếu thiếu file, cần chạy:
     - `index.ipynb` - Business model
     - `ecoclass.ipynb` - Economy model
     - `ecoplusclass.ipynb` - Eco Plus model
     - `ahp_analysis.ipynb` - AHP weights
     - `dss_integration.ipynb` - DSS system

3. **Dữ liệu test:**
   - Cần file `Data/test.csv` để load dữ liệu mẫu

## Quy trình sử dụng đầy đủ

### Cho khách hàng:
1. Mở app → Chọn "Customer Form"
2. Chọn hạng vé của bạn
3. Đánh giá 6 tiêu chí (di chuyển slider)
4. Click "Submit Evaluation"
5. Xem kết quả dự đoán và khuyến nghị

### Cho quản trị viên:
1. Mở app → Chọn "Admin Dashboard"
2. Xem KPI metrics tổng quan
3. Phân tích tab "Overview" - xu hướng
4. Kiểm tra tab "XGBoost Results" - hiệu suất mô hình
5. Xem tab "AHP Weights" - độ ưu tiên tiêu chí
6. Theo dõi tab "Alerts" - hành khách rủi ro cao
7. Download CSV để báo cáo

## Troubleshooting

**Lỗi: "DSS system not found"**
- Chạy `dss_integration.ipynb` để tạo file `airline_dss_system.pkl`

**Lỗi: "AHP weights not found"**
- Chạy `ahp_analysis.ipynb` để tạo file `ahp_weights_all_classes.pkl`

**Lỗi: "No submissions yet"**
- Submit ít nhất 1 đánh giá ở "Customer Form" trước khi vào Dashboard

**App không mở được:**
- Kiểm tra đã cài streamlit: `pip install streamlit`
- Kiểm tra đang ở đúng thư mục chứa `app.py`
- Thử port khác: `streamlit run app.py --server.port 8502`

## Mở rộng trong tương lai

- 🔐 Thêm authentication cho admin
- 💾 Lưu dữ liệu vào database
- 📧 Gửi email cảnh báo tự động
- 📱 Phiên bản mobile-responsive
- 🌐 Multi-language support
- 📊 Thêm biểu đồ nâng cao
- 🤖 Chatbot hỗ trợ

## Kết luận

Web application này hoàn thành Stage 6 của dự án, cung cấp:
- ✅ Form thu thập dữ liệu từ khách hàng
- ✅ Dashboard hỗ trợ ra quyết định cho quản trị viên
- ✅ Tích hợp đầy đủ XGBoost + AHP
- ✅ Giao diện thân thiện, dễ sử dụng
- ✅ Sẵn sàng triển khai thực tế
