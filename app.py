import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import random
from PIL import Image
import io

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

# --- MediaPipe init ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5
)

# --- Landmark indices ---
# MediaPipe Face Mesh (468 landmarks, 0-indexed)

# Left eye: iris center ~468 (not always reliable), use eyelid landmarks
# LEFT EYE top/bottom: 159 (upper), 23 (lower)
# LEFT EYE left/right: 130 (left), 243 (right)
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 23
LEFT_EYE_LEFT = 130
LEFT_EYE_RIGHT = 243

# RIGHT EYE top/bottom: 386 (upper), 374 (lower)
# RIGHT EYE left/right: 359 (right), 463 (left) — mirrored
RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374
RIGHT_EYE_LEFT = 263
RIGHT_EYE_RIGHT = 362

# UPPER LIP center: 13 (top of upper lip)
# LOWER LIP center: 14 (bottom of lower lip)
UPPER_LIP = 13
LOWER_LIP = 14


def ear(landmarks, h, w, top, bottom, left, right):
    """Eye Aspect Ratio simplificado."""
    p_top = np.array([landmarks[top].x * w, landmarks[top].y * h])
    p_bot = np.array([landmarks[bottom].x * w, landmarks[bottom].y * h])
    p_l = np.array([landmarks[left].x * w, landmarks[left].y * h])
    p_r = np.array([landmarks[right].x * w, landmarks[right].y * h])

    vert = np.linalg.norm(p_top - p_bot)
    horiz = np.linalg.norm(p_l - p_r)
    return vert / horiz if horiz > 0 else 0.5


def mouth_ratio(landmarks, h, w):
    """Distancia vertical labios / distancia de referencia (nariz-mentón)."""
    upper = np.array([landmarks[UPPER_LIP].x * w, landmarks[UPPER_LIP].y * h])
    lower = np.array([landmarks[LOWER_LIP].x * w, landmarks[LOWER_LIP].y * h])
    lip_dist = np.linalg.norm(upper - lower)

    # reference: nose bridge (6) to chin (152)
    nose = np.array([landmarks[6].x * w, landmarks[6].y * h])
    chin = np.array([landmarks[152].x * w, landmarks[152].y * h])
    face_height = np.linalg.norm(nose - chin)

    return lip_dist / face_height if face_height > 0 else 0


def compute_gaga_score(image):
    """Process image with MediaPipe and return (score, annotated_image)."""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return None, None

    landmarks = results.multi_face_landmarks[0].landmark
    h, w, _ = image.shape

    # --- Eye metrics ---
    ear_left = ear(landmarks, h, w, LEFT_EYE_TOP, LEFT_EYE_BOTTOM, LEFT_EYE_LEFT, LEFT_EYE_RIGHT)
    ear_right = ear(landmarks, h, w, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM, RIGHT_EYE_LEFT, RIGHT_EYE_RIGHT)
    avg_ear = (ear_left + ear_right) / 2.0

    # Typical EAR ~0.25-0.35 for open eyes, <0.20 for closed
    # Lower EAR → more "gaga" (droopy / vacant stare)
    ear_score = max(0.0, min(1.0, 1.0 - (avg_ear / 0.30)))
    if avg_ear > 0.35:
        ear_score = 0.05  # wide open → very alert

    # --- Mouth metric ---
    mratio = mouth_ratio(landmarks, h, w)
    # Higher mratio → mouth agape → more gaga
    mouth_score = max(0.0, min(1.0, (mratio - 0.02) / 0.08))

    # --- Weighted combination ---
    raw = ear_score * 0.55 + mouth_score * 0.35

    # --- Random factor for dynamism (±8%) ---
    jitter = random.uniform(-0.08, 0.08)
    final = max(0.0, min(1.0, raw + jitter))

    score_pct = round(final * 100)

    # --- Draw landmarks on image ---
    draw = image.copy()
    h2, w2 = draw.shape[:2]
    indices_to_draw = [
        LEFT_EYE_TOP, LEFT_EYE_BOTTOM, LEFT_EYE_LEFT, LEFT_EYE_RIGHT,
        RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM, RIGHT_EYE_LEFT, RIGHT_EYE_RIGHT,
        UPPER_LIP, LOWER_LIP,
    ]
    for idx in indices_to_draw:
        lm = landmarks[idx]
        cx, cy = int(lm.x * w2), int(lm.y * h2)
        cv2.circle(draw, (cx, cy), 3, (0, 255, 255), -1)

    # Thin connections for eyes
    left_pts = [landmarks[i] for i in [LEFT_EYE_LEFT, LEFT_EYE_TOP, LEFT_EYE_RIGHT, LEFT_EYE_BOTTOM]]
    for i in range(4):
        p1 = (int(left_pts[i].x * w2), int(left_pts[i].y * h2))
        p2 = (int(left_pts[(i + 1) % 4].x * w2), int(left_pts[(i + 1) % 4].y * h2))
        cv2.line(draw, p1, p2, (0, 255, 255), 1)
    right_pts = [landmarks[i] for i in [RIGHT_EYE_LEFT, RIGHT_EYE_TOP, RIGHT_EYE_RIGHT, RIGHT_EYE_BOTTOM]]
    for i in range(4):
        p1 = (int(right_pts[i].x * w2), int(right_pts[i].y * h2))
        p2 = (int(right_pts[(i + 1) % 4].x * w2), int(right_pts[(i + 1) % 4].y * h2))
        cv2.line(draw, p1, p2, (0, 255, 255), 1)

    # Mouth line
    mp1 = (int(landmarks[UPPER_LIP].x * w2), int(landmarks[UPPER_LIP].y * h2))
    mp2 = (int(landmarks[LOWER_LIP].x * w2), int(landmarks[LOWER_LIP].y * h2))
    cv2.line(draw, mp1, mp2, (0, 255, 255), 2)

    return score_pct, cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)


# --- FILE UPLOADER ---
uploaded = st.file_uploader(
    "Subí tu foto de perfil (o la de tu amigo más zombie)",
    type=["jpg", "jpeg", "png"],
)

if uploaded:
    pil = Image.open(uploaded).convert("RGB")
    orig = np.array(pil)
    bgr = cv2.cvtColor(orig, cv2.COLOR_RGB2BGR)

    with st.spinner("🧠 Analizando nivel de GAGA..."):
        score, annotated = compute_gaga_score(bgr)

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
        # --- Classification ---
        if score <= 20:
            label = "🧠 Totalmente Alerta"
            desc = "Mente al 100%, café al día. Hoy no te gana nadie."
            bar_color = "normal"
        elif score <= 50:
            label = "✈️ Modo Avión Mental"
            desc = "Mente de viaje, distracción leve. Todavía hay esperanza."
            bar_color = "normal"
        elif score <= 80:
            label = "⚠️ GAGA Activo — Peligro"
            desc = "Mirada fija en la pared, módem interno reiniciándose..."
            bar_color = "warning"
        else:
            label = "🌌 GAGA Absoluto — Desconexión Espacial"
            desc = "El alma dejó el cuerpo. Si le hablás, solo responde: '¿Eh?'"
            bar_color = "error"

        st.markdown(f'<div class="result-card">', unsafe_allow_html=True)
        st.subheader(f"GAGA-SCORE: {score}%")
        st.markdown(f"### {label}")
        st.markdown(f"_{desc}_")

        if bar_color == "normal":
            st.progress(score / 100.0)
        elif bar_color == "warning":
            st.progress(score / 100.0)
        elif bar_color == "error":
            st.progress(score / 100.0)

        st.markdown("</div>", unsafe_allow_html=True)

        # --- Download button ---
        if annotated is not None:
            result_pil = Image.fromarray(annotated)
            buf = io.BytesIO()
            result_pil.save(buf, format="PNG")
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
