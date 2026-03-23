# Tai Lieu Du An Web DSS Danh Gia Hai Long Hanh Khach Hang Khong

## 1) Tong quan du an

Du an xay dung he thong ho tro ra quyet dinh (Decision Support System - DSS) danh gia muc do hai long hanh khach hang khong. Giai phap ket hop:

- Mo hinh hoc may XGBoost de du doan hai long/khong hai long.
- AHP (Analytic Hierarchy Process) de xep uu tien tieu chi va ho tro de xuat phuong an hanh dong.
- Lop ung dung Web cho nguoi dung cuoi (khach hang) va quan tri vien.

Muc tieu chinh:

- Thu thap danh gia dich vu theo hang ve.
- Du doan trang thai hai long theo thoi gian gan thuc.
- Tinh diem rui ro va xac dinh cac tieu chi can cai thien.
- Ho tro quan tri vien phan tich xu huong va de xuat quyet dinh.

## 2) Pham vi va thanh phan

Du an bao gom 2 lop ung dung:

- Ung dung Flask UI: giao dien web chinh, luu feedback vao PostgreSQL, dashboard quan tri.
- Ung dung Streamlit (prototype/bo sung): trinh bay DSS va dashboard theo huong data app.

Thanh phan nghiep vu chinh:

- Phan he du doan hai long theo tung hang ve (Business, Economy, Eco Plus).
- Phan he tinh impact ket hop XGBoost + AHP.
- Phan he goi y khuyen nghi theo muc do hai long va tieu chi uu tien.
- Phan he quan tri thong ke, xem chi tiet feedback, va decision support.

## 3) Cong nghe su dung

- Backend web: Flask
- Data app: Streamlit
- Machine Learning: XGBoost, scikit-learn
- Tinh toan du lieu: pandas, numpy
- Truc quan hoa: Plotly
- Co so du lieu: PostgreSQL (qua psycopg2)

Danh sach thu vien duoc khai bao tai `requirements.txt`.

## 4) Cau truc thu muc

```
HHTRQD/
|- app.py                         # Streamlit app (giao dien thay the/bo sung)
|- recommendation_engine.py       # Logic xep hang va sinh khuyen nghi
|- requirements.txt               # Danh sach thu vien Python
|- database.sql                   # File SQL (hien dang de trong)
|- Data/
|  |- train.csv
|  |- test.csv
|- flask_ui/
|  |- app_flask.py                # Flask app chinh
|  |- static/
|  |  |- styles.css
|  |- templates/
|     |- home.html
|     |- survey.html
|     |- dashboard.html
|     |- admin_login.html
|     |- admin_feedback_list.html
|     |- admin_feedback_detail.html
|     |- admin_decision_support.html
|     |- admin_model_stats.html
|     |- calculation_steps.html
|     |- about.html
|- *.ipynb                        # Notebook huan luyen mo hinh, AHP, tich hop DSS
```

## 5) Kien truc tong the

Luong xu ly tong quan:

1. Nguoi dung nhap danh gia tai trang survey.
2. He thong tao vector dac trung theo hang ve.
3. DSS du doan hai long, tinh confidence va risk score.
4. DSS tinh impact tieu chi (ket hop feature importance + AHP weight).
5. Engine tao danh sach khuyen nghi theo muc do uu tien.
6. Ket qua luu vao bang `feedback_submissions` trong PostgreSQL.
7. Quan tri vien theo doi dashboard va decision support.

## 6) Co so du lieu

Ung dung Flask tu khoi tao bang neu chua ton tai:

- `feedback_submissions`: luu ban ghi danh gia, ket qua du doan, thong tin file upload.
- `admin_users`: tai khoan quan tri.

Tai khoan fallback trong code:

- Username mac dinh: `admin`
- Password mac dinh: `admin123`

Luu y bao mat: can doi thong tin dang nhap bang bien moi truong khi trien khai that.

## 7) Yeu cau dau vao mo hinh

De DSS day du, du an can cac artifact da huan luyen (tao tu notebook):

- `business_xgboost_optimized.pkl`
- `economy_xgboost_optimized.pkl`
- `ecoplus_xgboost_optimized.pkl`
- `ahp_weights_all_classes.pkl`
- `airline_dss_system.pkl` (tuy chon fallback)

Neu thieu artifact, can chay cac notebook huan luyen/tich hop trong du an.

## 8) Huong dan cai dat va chay

### 8.1 Cai dat moi truong

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 8.2 Cau hinh PostgreSQL (khuyen nghi)

Co the dat bien moi truong:

```bash
set PGHOST=localhost
set PGPORT=5432
set PGDATABASE=airline
set PGUSER=postgres
set PGPASSWORD=123
```

Them bien moi truong bao mat cho Flask:

```bash
set FLASK_SECRET_KEY=your_secret_key
set AIRLINE_ADMIN_USER=admin
set AIRLINE_ADMIN_PASS=strong_password
```

### 8.3 Chay Flask UI (khuyen nghi cho san pham web)

```bash
python flask_ui/app_flask.py
```

Truy cap: `http://localhost:8502`

### 8.4 Chay Streamlit app (neu can)

```bash
streamlit run app.py
```

Mac dinh truy cap: `http://localhost:8501`

## 9) Chuc nang theo vai tro

### 9.1 Khach hang

- Truy cap trang `survey`.
- Chon hang ve va nhap diem danh gia cac tieu chi.
- Nhan ket qua:
  - Du doan hai long/khong hai long.
  - Do tin cay du doan.
  - Risk score va muc rui ro.
  - Top tieu chi anh huong.
  - Goi y hanh dong uu tien.

### 9.2 Quan tri vien

- Dang nhap qua trang `admin/login`.
- Theo doi dashboard KPI tong hop.
- Xem danh sach feedback va chi tiet tung feedback.
- Theo doi decision support (PA1/PA2/PA3).
- Xem thong ke model (hardcoded tu notebook).

## 10) Danh sach route Flask chinh

- `/`: Trang chu
- `/survey`: Form danh gia khach hang
- `/about`: Gioi thieu he thong
- `/calculation-steps`: Trinh bay cac buoc tinh AHP
- `/admin/login`: Dang nhap quan tri
- `/admin/logout`: Dang xuat
- `/admin/dashboard`: Dashboard KPI
- `/admin/feedback`: Danh sach feedback
- `/admin/feedback/<id>`: Chi tiet feedback + AHP decision support
- `/admin/decision-support`: Tong hop de xuat quyet dinh
- `/admin/model_stats`: Thong ke mo hinh

## 11) Luong nghiep vu decision support

He thong xay dung goi y theo 3 phuong an (PA):

- `PA1`: Duy tri/Cai thien chat luong dich vu
- `PA2`: Chuong trinh khach hang than thiet
- `PA3`: De xuat nang hang

Trong moi feedback chi tiet, he thong:

1. Chuan hoa diem cac tieu chi.
2. Tao ma tran cap doi AHP.
3. Tinh vector trong so va kiem tra CR (Consistency Ratio).
4. Cham diem PA theo tieu chi.
5. Chon PA toi uu dua tren tong diem co trong so.

## 12) Troubleshooting nhanh

1. Loi khong du doan duoc DSS:
   - Kiem tra cac file `.pkl` da ton tai dung ten.
   - Chay lai cac notebook huan luyen va tich hop.

2. Loi ket noi PostgreSQL:
   - Kiem tra service PostgreSQL dang chay.
   - Kiem tra bien moi truong `PG*`.
   - Kiem tra tai khoan co quyen tao bang.

3. Dashboard trong:
   - Thu submit it nhat 1 feedback tai `/survey`.

4. Khong dang nhap duoc admin:
   - Kiem tra bang `admin_users`.
   - Neu can, su dung fallback account theo bien moi truong.

## 13) Huong phat trien tiep

- Ma hoa mat khau admin (bcrypt/argon2) thay vi luu plain text.
- Tach cau hinh ra file `.env` va bo sung quan ly secret.
- Bo sung migration DB (Alembic/Flyway).
- Viet test tu dong cho route va pipeline DSS.
- Dong bo ten class `Eco`/`Eco Plus` de tranh sai lech trong bao cao.

## 14) Ket luan

Du an da co mot nen tang DSS web kha day du cho bai toan danh gia hai long hanh khach, ket hop mo hinh du doan va AHP de chuyen du lieu phan hoi thanh hanh dong uu tien cho quan tri vien. Tai lieu nay co the dung lam baseline cho bao cao do an, ban giao ky thuat, va mo rong he thong trong cac giai doan tiep theo.
