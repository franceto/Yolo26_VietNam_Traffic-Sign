# Vietnam Traffic Sign Detection with YOLO26n

<div align="center">

<!-- Banner / Screenshot placeholder -->
<img src="assets/demo_result.png" alt="Vietnam Traffic Sign Detection Demo" width="90%" />

<br/>

**Dự án phát hiện biển báo giao thông Việt Nam bằng YOLO26n và Ultralytics, có demo Streamlit hỗ trợ ảnh và video.**

<br/>

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO26n-111827?style=for-the-badge)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Roboflow](https://img.shields.io/badge/Roboflow-Dataset-6706CE?style=for-the-badge&logo=roboflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20Demo-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Image%2FVideo-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)

</div>

---

## 1. Project Title & Catchphrase

**Vietnam Traffic Sign Detection with YOLO26n** là hệ thống phát hiện biển báo giao thông Việt Nam trong ảnh hoặc video.

Dự án tập trung vào bài toán **Object Detection**, trả về **bounding box**, **nhãn lớp** và **confidence** cho từng biển báo được phát hiện.

---

## 2. Quick Demo & Visuals

<div align="center">

[Streamlit Web Demo](#) · [Training Notebook](#) · [Roboflow Dataset Source](#)

<br/><br/>

<img src="assets/demo_result.png" alt="Detection Result" width="90%" />

</div>

> Nếu chưa có ảnh demo, hãy đặt ảnh kết quả detect vào `assets/demo_result.png` hoặc sửa lại đường dẫn ảnh tương ứng.

---

## 3. Tính Năng Nổi Bật

- **Phát hiện biển báo Việt Nam:** hỗ trợ 54 lớp biển báo theo dữ liệu Roboflow Universe.
- **Inference ảnh và video:** demo Streamlit cho phép upload ảnh/video và hiển thị bbox, nhãn, confidence.
- **Fine-tune YOLO26n:** sử dụng pretrained weights rồi huấn luyện lại trên dataset biển báo giao thông Việt Nam.
- **Cấu hình dữ liệu an toàn:** sử dụng `data_fixed.yaml` để tránh lỗi đường dẫn train/val/test từ file export ban đầu.
- **Không public model weights:** người dùng cần tự train lại bằng notebook để tạo `best.pt`.

---

## 4. Công Nghệ Sử Dụng

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO26n-111827?style=for-the-badge)
![Roboflow](https://img.shields.io/badge/Roboflow-Dataset-6706CE?style=for-the-badge&logo=roboflow&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Image%2FVideo-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557C?style=for-the-badge)
![Seaborn](https://img.shields.io/badge/Seaborn-Visualization-4B8BBE?style=for-the-badge)
![YAML](https://img.shields.io/badge/YAML-Config-CB171E?style=for-the-badge&logo=yaml&logoColor=white)

</div>

---

## 5. Triển Khai Nhanh

**Prerequisites**

- Python 3.11+
- NVIDIA GPU và CUDA driver nếu muốn train trên GPU
- Khuyến nghị dùng môi trường ảo `.venv`
- Model weights không được công khai; cần tự train để tạo `best.pt`

```bash
# Clone repository
git clone https://github.com/franceto/Yolo26-Vietnam_Traffic-Sign.git
cd Yolo26-Vietnam_Traffic-Sign

# Tạo và kích hoạt môi trường ảo trên Windows
python -m venv .venv
.\.venv\Scripts\activate

# Cập nhật pip
python -m pip install -U pip

# Cài PyTorch CUDA
pip install --upgrade --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Cài thư viện cho dự án
pip install ultralytics roboflow opencv-python streamlit pandas matplotlib seaborn pyyaml
pip install -r requirements.txt

# Huấn luyện lại model bằng notebook hoặc script của dự án để tạo best.pt
# File khuyến nghị sau train:
# runs/vietnam_traffic_y26n_w0/weights/best.pt

# Chạy demo Streamlit
.\.venv\Scripts\python.exe -m streamlit run app.py
```

---

## 6. Tài Liệu Dự Án

### Bài Toán

| Thành phần | Mô tả |
|---|---|
| Input | Ảnh hoặc video có chứa biển báo giao thông |
| Output | Bounding boxes, class labels, confidence scores |
| Task | Object Detection |
| Ứng dụng demo | Streamlit |
| Framework train/inference | Ultralytics |

### Model

| Hạng mục | Thông tin |
|---|---|
| Architecture | YOLO26n nano |
| Framework | Ultralytics |
| Pretrained | COCO pretrained weights |
| Chiến lược | Fine-tune trên dataset biển báo Việt Nam |
| Weights khuyến nghị | `runs/vietnam_traffic_y26n_w0/weights/best.pt` |
| Weights cuối epoch | `runs/vietnam_traffic_y26n_w0/weights/last.pt` |

> Trọng số model không được công khai trong repository. Người dùng cần tự huấn luyện lại để tạo file weights.

### Dataset

| Hạng mục | Thông tin |
|---|---|
| Nguồn | Roboflow Universe - Vietnam Traffic Sign |
| Định dạng | YOLO TXT + YAML |
| Kích thước ảnh | 640 x 640 |
| Số lớp | 54 |
| Train | Khoảng 3201 images |
| Val | 305 images |
| Test | 153 images |
| File cấu hình đã sửa | `data_fixed.yaml` |

> File `data.yaml` export ban đầu có thể trỏ sai `../train/images`. Dự án sử dụng `data_fixed.yaml` để đảm bảo đường dẫn đúng.

### Classes

<details>
<summary>Danh sách 54 lớp</summary>

```text
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
```

</details>

### Validation Metrics

| Metric | Value |
|---|---:|
| Precision | 0.834 |
| Recall | 0.668 |
| mAP@50 | 0.757 |
| mAP@50-95 | 0.572 |

> Kết quả có thể thay đổi theo seed, cấu hình augmentation, batch size và môi trường huấn luyện.

### Inference Weights

Sau khi huấn luyện, ưu tiên dùng:

```text
runs/vietnam_traffic_y26n_w0/weights/best.pt
```

File `last.pt` chỉ nên dùng khi muốn tiếp tục huấn luyện hoặc kiểm tra epoch cuối:

```text
runs/vietnam_traffic_y26n_w0/weights/last.pt
```

### Gợi Ý Project Structure

```text
.
├── app.py
├── requirements.txt
├── data_fixed.yaml
├── notebooks/
│   └── train_yolo26n.ipynb
├── runs/
│   └── vietnam_traffic_y26n_w0/
│       └── weights/
│           ├── best.pt
│           └── last.pt
├── assets/
│   └── demo_result.png
└── README.md
```

> Cấu trúc trên là cấu trúc gợi ý dựa trên thông tin được cung cấp. Hãy chỉnh lại nếu repository thực tế khác.

### Notes

- Không commit các file weights nặng như `.pt` nếu repository không public model.
- Không commit `.venv`, cache, file tạm, hoặc thư mục `runs/` nếu dung lượng lớn.
- Khi chia sẻ repo demo, nên cung cấp hướng dẫn train lại hoặc link model riêng nếu được phép.
- Với Streamlit, cần đảm bảo `app.py` trỏ đúng đường dẫn tới `best.pt`.

### Author

**franceto (ANH PHAP TO)**  
GitHub: [https://github.com/franceto](https://github.com/franceto)

### Support

Nếu project hữu ích, hãy cho repository một sao.

Made by **Franceto (ANH PHAP TO)**
