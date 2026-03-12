from pathlib import Path
import streamlit as st
import cv2
from ultralytics import YOLO

st.set_page_config(page_title="Vietnam Traffic Sign - YOLO26", layout="wide")
st.title("🚦 Vietnam Traffic Sign Detection (YOLO26n)")
st.caption("Demo cơ bản: upload ảnh hoặc video (.mp4) → detect → hiển thị kết quả")

MODEL_PATH = r"D:\Documents\KimTin\Yolo26_VN_Traffic-Sign\runs\vietnam_traffic_y26n_w0\weights\best.pt"
TMP_DIR = Path("tmp")
OUT_DIR = Path("runs_streamlit")
TMP_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

conf = st.sidebar.slider("Confidence", 0.05, 0.95, 0.25, 0.05)
iou  = st.sidebar.slider("IoU",        0.10, 0.90, 0.45, 0.05)

uploaded = st.file_uploader("Chọn ảnh hoặc video", type=["jpg","jpeg","png","webp","mp4"])

if uploaded is None:
    st.info("Hãy upload 1 ảnh hoặc video để bắt đầu.")
    st.stop()

suffix = Path(uploaded.name).suffix.lower()
input_path = TMP_DIR / f"input{suffix}"
with open(input_path, "wb") as f:
    f.write(uploaded.getbuffer())

col1, col2 = st.columns(2)

if suffix in [".jpg", ".jpeg", ".png", ".webp"]:
    # ===== IMAGE =====
    img_bgr = cv2.imread(str(input_path))
    if img_bgr is None:
        st.error("Không đọc được ảnh. Hãy thử ảnh khác.")
        st.stop()

    with col1:
        st.subheader("Ảnh gốc")
        st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)

    res = model.predict(source=str(input_path), conf=conf, iou=iou, verbose=False)[0]
    annotated = res.plot()  # BGR numpy

    with col2:
        st.subheader("Kết quả detect")
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

else:
    # ===== VIDEO =====
    with col1:
        st.subheader("Video gốc")
        st.video(str(input_path))

    name = "pred_video"
    _ = model.predict(
        source=str(input_path),
        conf=conf,
        iou=iou,
        save=True,
        project=str(OUT_DIR),
        name=name,
        exist_ok=True,
        verbose=False
    )

    out_folder = OUT_DIR / name
    out_video = None
    for ext in [".mp4", ".avi", ".mov", ".mkv"]:
        cand = list(out_folder.glob(f"*{ext}"))
        if cand:
            out_video = cand[0]
            break

    with col2:
        st.subheader("Video kết quả (annotated)")
        if out_video and out_video.exists():
            st.video(str(out_video))
            st.write("Output:", str(out_video))
        else:
            st.warning("Không tìm thấy video output. Kiểm tra folder: " + str(out_folder))
