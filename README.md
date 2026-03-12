# 🚦 Vietnam Traffic Sign Detection with YOLO26n (Ultralytics)

Dự án huấn luyện và triển khai **mô hình phát hiện biển báo giao thông Việt Nam** bằng **YOLO26n (nano)** sử dụng thư viện **Ultralytics**.  
Có kèm **demo Streamlit** cho phép upload **ảnh/video** và hiển thị kết quả detect (bbox + nhãn + confidence).

---

## 1) Bài toán

- **Input**: Ảnh hoặc video có chứa biển báo giao thông.
- **Output**: Bounding boxes + nhãn lớp (class) + confidence cho từng biển báo.
- **Task**: Object Detection.

---

## 2) Model

- **Architecture**: YOLO26n (nano)
- **Framework**: Ultralytics
- **Pretrained**: sử dụng pretrained weights (COCO) rồi fine-tune trên dataset Việt Nam.

**File weights sau huấn luyện**:

- `runs/vietnam_traffic_y26n_w0/weights/best.pt` (khuyến nghị dùng để inference)
- `runs/vietnam_traffic_y26n_w0/weights/last.pt`

---

## 3) Dataset

- Nguồn dataset: Roboflow Universe (Vietnam Traffic Sign)
- Định dạng: YOLO TXT + YAML
- Kích thước ảnh: 640×640
- Số lớp (nc): **54**

**Split (theo export trong project)**:

- Train: ~3201 images (có thể > số ảnh gốc do Roboflow export/augmentation)
- Val: 305 images
- Test: 153 images

> Ghi chú: Trong quá trình làm notebook, `data.yaml` export ban đầu có thể trỏ sai `../train/images`.  
> Project đã tạo `data_fixed.yaml` để đảm bảo path đúng.

---

## 4) Classes (54)

Danh sách tên lớp (theo `data_fixed.yaml`):

1. cam_cac_loai_xe_3_banh
2. cam_do_xe
3. cam_dung_xe_va_do_xe
4. cam_o_to
5. cam_o_to_quay_dau_xe
6. cam_o_to_re_phai
7. cam_o_to_re_trai
8. cam_o_to_tai
9. cam_o_to_tai_trong_luong_qua_10t
10. cam_o_to_tai_trong_luong_qua_2-5t
11. cam_o_to_tai_va_xe_khach
12. cam_quay_dau_xe
13. cam_re_phai
14. cam_re_trai
15. cam_re_trai_va_phai
16. cam_su_dung_coi
17. cam_vuot
18. cam_xe_2_banh_va_xe_3_banh_co_dong_co
19. cam_xe_di_nguoc_chieu
20. cho
21. cho_ngoat_nguy_hiem_ben_phai
22. cho_ngoat_nguy_hiem_ben_trai
23. cho_ngoat_nguy_hiem_lien_tiep
24. cho_quay_xe
25. chu_y_nguoi_di_bo_cat_ngang
26. chu_y_nguoi_di_xe_dap_cat_ngang
27. chu_y_tre_em
28. cong_truong
29. di_cham
30. duong_cho_nguoi_di_bo
31. duong_cho_xe_o_to
32. duong_co_o_ga
33. duong_co_song_map_mo_nhan_tao
34. duong_het_uu_tien
35. duong_mot_chieu
36. duong_nguoi_di_bo_sang_ngang
37. duong_uu_tien
38. giao_nhau_co_tin_hieu_den
39. giao_nhau_voi_duong_khong_uu_tien
40. giao_nhau_voi_duong_uu_tien
41. han_che_chieu_cao
42. huong_di_tren_moi_lan_duong_theo_vach_ke_duong
43. huong_phai_di_vuot_chuong_ngai_vat_phai
44. huong_phai_di_vuot_chuong_ngai_vat_trai
45. khac
46. khu_vuc_quay_xe
47. lan_duong_cho_tung_xe_theo_vach_ke_duong
48. nguy_hiem_khac
49. noi_giao_nhau_chay_theo_vong
50. object
51. toc_do_toi_da_40
52. toc_do_toi_da_50
53. toc_do_toi_da_60
54. toc_do_toi_da_80

---

## 5) Kết quả (Validation)

Ví dụ metrics (val) từ lần train YOLO26n:

- Precision: **0.834**
- Recall: **0.668**
- mAP@50: **0.757**
- mAP@50-95: **0.572**

> Kết quả phụ thuộc seed, cấu hình augmentation và batch size.

---

## 6) Không công khai trọng số model

Bạn phải tự huấn luyện mô hình lại bằng notebook có sẵn để lấy trọng số

## 7) Cài môi trường (Windows / VSCode)

Yêu cầu:

- Python 3.11+
- NVIDIA GPU + CUDA driver
- Khuyến nghị dùng `.venv`

### 7.1 Tạo venv và cài cơ bản

bash
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -U pip

### 7.2 Cài PyTorch (CUDA) / Cài thư viện cho dự án

- pip install --upgrade --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
- pip install ultralytics roboflow opencv-python streamlit pandas matplotlib seaborn pyyaml
- pip install -r requirements.txt

## 8) Run app

.\.venv\Scripts\python.exe -m streamlit run app.py (terminal)
