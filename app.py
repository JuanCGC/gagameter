import streamlit as st
import numpy as np
import mediapipe as mp
import random
from PIL import Image, ImageDraw
import io
import os

st.set_page_config(
    page_title="GAGAMETER",
    page_icon="🥴",
    layout="centered",
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 3rem;
        text-align: center;
        margin-bottom: 0;
    }
    .sub-title {
        text-align: center;
        font-size: 1.1rem;
        color: #888;
        margin-bottom: 2rem;
    }
    .result-card {
        padding: 1.5rem;
        border-radius: 1rem;
        border: 1px solid #333;
        background: #111;
        margin: 1rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<p class="main-title">🥴 GAGAMETER 😵‍💫</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-title">Subí tu foto y descubrí qué tan GAGA está tu cerebro hoy 💀</p>',
    unsafe_allow_html=True,
)

# --- MediaPipe init (new 0.10.x API) ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=1,
    min_face_detection_confidence=0.5,
)
landmarker = FaceLandmarker.create_from_options(options)

# --- Landmark indices ---
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 23
LEFT_EYE_LEFT = 130
LEFT_EYE_RIGHT = 243

RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374
RIGHT_EYE_LEFT = 263
RIGHT_EYE_RIGHT = 362

UPPER_LIP = 13
LOWER_LIP = 14


def ear(landmarks, h, w, top, bottom, left, right):
    p_top = np.array([landmarks[top].x * w, landmarks[top].y * h])
    p_bot = np.array([landmarks[bottom].x * w, landmarks[bottom].y * h])
    p_l = np.array([landmarks[left].x * w, landmarks[left].y * h])
    p_r = np.array([landmarks[right].x * w, landmarks[right].y * h])

    vert = np.linalg.norm(p_top - p_bot)
    horiz = np.linalg.norm(p_l - p_r)
    return vert / horiz if horiz > 0 else 0.5


def mouth_ratio(landmarks, h, w):
    upper = np.array([landmarks[UPPER_LIP].x * w, landmarks[UPPER_LIP].y * h])
    lower = np.array([landmarks[LOWER_LIP].x * w, landmarks[LOWER_LIP].y * h])
    lip_dist = np.linalg.norm(upper - lower)

    nose = np.array([landmarks[6].x * w, landmarks[6].y * h])
    chin = np.array([landmarks[152].x * w, landmarks[152].y * h])
    face_height = np.linalg.norm(nose - chin)

    return lip_dist / face_height if face_height > 0 else 0


def compute_gaga_score(pil_img):
    rgb = np.array(pil_img.convert("RGB"))
    h, w, _ = rgb.shape

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        return None, None

    landmarks = result.face_landmarks[0]

    ear_left = ear(landmarks, h, w, LEFT_EYE_TOP, LEFT_EYE_BOTTOM, LEFT_EYE_LEFT, LEFT_EYE_RIGHT)
    ear_right = ear(landmarks, h, w, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM, RIGHT_EYE_LEFT, RIGHT_EYE_RIGHT)
    avg_ear = (ear_left + ear_right) / 2.0

    ear_score = max(0.0, min(1.0, 1.0 - (avg_ear / 0.30)))
    if avg_ear > 0.35:
        ear_score = 0.05

    mratio = mouth_ratio(landmarks, h, w)
    mouth_score = max(0.0, min(1.0, (mratio - 0.02) / 0.08))

    raw = ear_score * 0.55 + mouth_score * 0.35

    jitter = random.uniform(-0.08, 0.08)
    final = max(0.0, min(1.0, raw + jitter))

    score_pct = round(final * 100)

    # --- Draw landmarks ---
    draw_img = pil_img.convert("RGB").copy()
    draw = ImageDraw.Draw(draw_img)

    indices_to_draw = [
        LEFT_EYE_TOP, LEFT_EYE_BOTTOM, LEFT_EYE_LEFT, LEFT_EYE_RIGHT,
        RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM, RIGHT_EYE_LEFT, RIGHT_EYE_RIGHT,
        UPPER_LIP, LOWER_LIP,
    ]
    for idx in indices_to_draw:
        lm = landmarks[idx]
        cx, cy = int(lm.x * w), int(lm.y * h)
        r = max(2, int(w / 200))
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(0, 255, 255))

    left_pts = [landmarks[i] for i in [LEFT_EYE_LEFT, LEFT_EYE_TOP, LEFT_EYE_RIGHT, LEFT_EYE_BOTTOM]]
    for i in range(4):
        p1 = (int(left_pts[i].x * w), int(left_pts[i].y * h))
        p2 = (int(left_pts[(i + 1) % 4].x * w), int(left_pts[(i + 1) % 4].y * h))
        draw.line([p1, p2], fill=(0, 255, 255), width=max(1, int(w / 400)))

    right_pts = [landmarks[i] for i in [RIGHT_EYE_LEFT, RIGHT_EYE_TOP, RIGHT_EYE_RIGHT, RIGHT_EYE_BOTTOM]]
    for i in range(4):
        p1 = (int(right_pts[i].x * w), int(right_pts[i].y * h))
        p2 = (int(right_pts[(i + 1) % 4].x * w), int(right_pts[(i + 1) % 4].y * h))
        draw.line([p1, p2], fill=(0, 255, 255), width=max(1, int(w / 400)))

    mp1 = (int(landmarks[UPPER_LIP].x * w), int(landmarks[UPPER_LIP].y * h))
    mp2 = (int(landmarks[LOWER_LIP].x * w), int(landmarks[LOWER_LIP].y * h))
    draw.line([mp1, mp2], fill=(0, 255, 255), width=max(2, int(w / 200)))

    return score_pct, draw_img


# --- FILE UPLOADER ---
uploaded = st.file_uploader(
    "Subí tu foto de perfil (o la de tu amigo más zombie)",
    type=["jpg", "jpeg", "png"],
)

if uploaded:
    pil = Image.open(uploaded).convert("RGB")

    with st.spinner("🧠 Analizando nivel de GAGA..."):
        score, annotated = compute_gaga_score(pil)

    col1, col2 = st.columns(2)

    with col1:
        st.image(pil, caption="📸 Tu foto", use_container_width=True)

    with col2:
        if annotated is not None:
            st.image(annotated, caption="🤖 Escaneo neuronal", use_container_width=True)
        else:
            st.image(pil, caption="🤖 Escaneo neuronal", use_container_width=True)

    st.markdown("---")

    if score is None:
        st.error(
            "😅 No se detectó ningún rostro en la imagen. "
            "Probá con otra foto donde se vea bien tu cara."
        )
    else:
        if score <= 20:
            label = "🧠 Totalmente Alerta"
            desc = "Mente al 100%, café al día. Hoy no te gana nadie."
        elif score <= 50:
            label = "✈️ Modo Avión Mental"
            desc = "Mente de viaje, distracción leve. Todavía hay esperanza."
        elif score <= 80:
            label = "⚠️ GAGA Activo — Peligro"
            desc = "Mirada fija en la pared, módem interno reiniciándose..."
        else:
            label = "🌌 GAGA Absoluto — Desconexión Espacial"
            desc = "El alma dejó el cuerpo. Si le hablás, solo responde: '¿Eh?'"

        st.markdown(f'<div class="result-card">', unsafe_allow_html=True)
        st.subheader(f"GAGA-SCORE: {score}%")
        st.markdown(f"### {label}")
        st.markdown(f"_{desc}_")
        st.progress(score / 100.0)
        st.markdown("</div>", unsafe_allow_html=True)

        if annotated is not None:
            buf = io.BytesIO()
            annotated.save(buf, format="PNG")
            buf.seek(0)
            st.download_button(
                label="📥 Descargar veredicto como imagen",
                data=buf,
                file_name="gagameter_veredicto.png",
                mime="image/png",
            )

st.markdown("---")
st.caption(
    "⚠️ Esto es una herramienta humorística. Los resultados no representan "
    "ningún diagnóstico real. Si te sentís identificado, tomate un café. ☕"
)
