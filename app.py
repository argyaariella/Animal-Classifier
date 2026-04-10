import streamlit as st
import torch
import torch.nn as nn
from torchvision.transforms import transforms
from PIL import Image
import time

st.set_page_config(
    page_title="Animal Classifier",
    page_icon="🐾",
    layout="centered"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0a0a0f; color: #f0ede8; }

.hero { text-align: center; padding: 3rem 0 2rem; }
.hero-title { font-family: 'Syne', sans-serif; font-size: 3.2rem; font-weight: 800; letter-spacing: -1px; background: linear-gradient(135deg, #f0ede8 0%, #c9a96e 50%, #f0ede8 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; margin: 0; line-height: 1.1; }
.hero-sub { font-size: 1rem; color: #7a7a8a; margin-top: 0.6rem; font-weight: 300; }

.result-card { background: #13131a; border: 1px solid #1e1e2e; border-radius: 20px; padding: 2rem 2.5rem; margin-top: 1.5rem; position: relative; overflow: hidden; }
.result-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px; background: linear-gradient(90deg, #c9a96e, #e8c98a, #c9a96e); }
.result-label { font-size: 0.75rem; font-weight: 500; letter-spacing: 2px; text-transform: uppercase; color: #7a7a8a; margin-bottom: 0.5rem; }
.result-animal { font-family: 'Syne', sans-serif; font-size: 2.8rem; font-weight: 800; color: #c9a96e; margin: 0; line-height: 1; }
.result-emoji { font-size: 3rem; margin-bottom: 0.5rem; }

.prob-row { margin-top: 1.2rem; }
.prob-item { margin-bottom: 10px; }
.prob-header { display: flex; justify-content: space-between; font-size: 0.82rem; color: #7a7a8a; margin-bottom: 4px; }
.prob-header span:last-child { color: #c9a96e; font-weight: 600; }
.bar-bg { background: #1e1e2e; border-radius: 999px; height: 6px; width: 100%; }
.bar-fill { height: 6px; border-radius: 999px; background: linear-gradient(90deg, #c9a96e, #e8c98a); }
.bar-fill-dim { height: 6px; border-radius: 999px; background: #2a2a3a; }

.classes-row { display: flex; gap: 10px; margin-top: 1.5rem; flex-wrap: wrap; }
.class-pill { background: #1a1a26; border: 1px solid #2a2a3a; border-radius: 999px; padding: 6px 16px; font-size: 0.8rem; color: #7a7a8a; }
.class-pill.active { background: #2a2010; border-color: #c9a96e; color: #c9a96e; }

.animal-info { background: #0f0f18; border: 1px solid #1e1e2e; border-radius: 14px; padding: 1.2rem 1.5rem; margin-top: 1rem; font-size: 0.88rem; color: #a0a0b0; line-height: 1.8; }
.animal-info b { color: #c9a96e; font-family: 'Syne', sans-serif; font-size: 1rem; }

.history-section { margin-top: 2.5rem; }
.history-title { font-family: 'Syne', sans-serif; font-size: 1rem; font-weight: 700; color: #7a7a8a; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 1rem; }
.history-grid { display: flex; gap: 12px; flex-wrap: wrap; }
.history-item { background: #13131a; border: 1px solid #1e1e2e; border-radius: 12px; padding: 10px 14px; font-size: 0.8rem; color: #7a7a8a; display: flex; align-items: center; gap: 8px; }
.history-item span { color: #c9a96e; font-weight: 600; }

.info-box { background: #13131a; border: 1px solid #1e1e2e; border-radius: 12px; padding: 1.2rem 1.5rem; margin-top: 1rem; font-size: 0.85rem; color: #7a7a8a; line-height: 1.7; }
.info-box b { color: #c9a96e; }

.multi-result { background: #13131a; border: 1px solid #1e1e2e; border-radius: 14px; padding: 1rem 1.5rem; margin-bottom: 10px; display: flex; align-items: center; gap: 14px; }
.multi-emoji { font-size: 1.8rem; }
.multi-label { font-family: 'Syne', sans-serif; font-size: 1.1rem; font-weight: 700; color: #c9a96e; }
.multi-conf { font-size: 0.8rem; color: #7a7a8a; }

.loading-bar { height: 3px; background: linear-gradient(90deg, #c9a96e, #e8c98a, #c9a96e); background-size: 200% 100%; animation: shimmer 1.5s infinite; border-radius: 999px; margin: 1rem 0; }
@keyframes shimmer { 0% { background-position: 200% 0; } 100% { background-position: -200% 0; } }

#MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
.stButton > button { background: linear-gradient(135deg, #c9a96e, #e8c98a) !important; color: #0a0a0f !important; border: none !important; border-radius: 10px !important; font-family: 'Syne', sans-serif !important; font-weight: 700 !important; width: 100% !important; padding: 0.65rem !important; }
.stTabs [data-baseweb="tab"] { color: #7a7a8a !important; font-family: 'DM Sans', sans-serif !important; }
.stTabs [aria-selected="true"] { color: #c9a96e !important; }
.stTabs [data-baseweb="tab-highlight"] { background: #c9a96e !important; }
.stTabs [data-baseweb="tab-list"] { background: #13131a !important; border-radius: 10px !important; }
</style>
""", unsafe_allow_html=True)


# ── Model ─────────────────────────────────────────────────
class Net(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.conv1    = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1      = nn.BatchNorm2d(32)
        self.conv2    = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2      = nn.BatchNorm2d(64)
        self.conv3    = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3      = nn.BatchNorm2d(128)
        self.pooling  = nn.MaxPool2d(2, 2)
        self.relu     = nn.ReLU()
        self.dropout  = nn.Dropout(p=0.4)
        self.flatten  = nn.Flatten()
        self.linear   = nn.Linear(128 * 16 * 16, 256)
        self.dropout2 = nn.Dropout(p=0.4)
        self.output   = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pooling(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pooling(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pooling(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.relu(self.linear(x))
        x = self.dropout2(x)
        x = self.output(x)
        return x


@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load("animal_classifier_final.pth", map_location=device, weights_only=False)
    classes = checkpoint['label_encoder_classes'].tolist()
    model = Net(num_classes=checkpoint['num_classes']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, classes, device


transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

EMOJI_MAP = {"cat": "🐱", "dog": "🐶", "wild": "🦁"}

ANIMAL_INFO = {
    "cat": {
        "title": "Domestic Cat",
        "desc": "Kucing domestik (Felis catus) adalah hewan peliharaan yang telah hidup bersama manusia selama lebih dari 10.000 tahun. Dikenal dengan insting berburu yang tajam, kemampuan menyeimbangkan tubuh, dan sifat mandiri namun tetap penuh kasih sayang.",
        "facts": ["Tidur 12–16 jam per hari", "Memiliki 32 otot di setiap telinga", "Dapat berlari hingga 48 km/jam"]
    },
    "dog": {
        "title": "Domestic Dog",
        "desc": "Anjing (Canis lupus familiaris) adalah sahabat setia manusia sejak lebih dari 15.000 tahun lalu. Dikenal dengan loyalitas tinggi, kemampuan mencium yang luar biasa, dan kecerdasan sosial yang membuatnya menjadi hewan peliharaan paling populer di dunia.",
        "facts": ["Indera penciuman 40x lebih kuat dari manusia", "Dapat memahami 250+ kata", "Ada lebih dari 340 ras anjing di dunia"]
    },
    "wild": {
        "title": "Wild Animal",
        "desc": "Hewan liar adalah hewan yang hidup bebas di alam tanpa domestikasi. Mereka memiliki insting bertahan hidup yang kuat, adaptasi luar biasa terhadap lingkungan, dan peran penting dalam menjaga keseimbangan ekosistem.",
        "facts": ["Bagian vital dari rantai makanan", "Adaptasi unik terhadap habitatnya", "Terancam oleh perubahan iklim & perburuan"]
    }
}

# ── Session State ─────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []


def predict(image):
    model, classes, device = load_model()
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        probs  = torch.softmax(output, dim=1)[0]
        pred   = torch.argmax(probs).item()
    return classes[pred], probs.tolist(), classes


# ── Header ────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <p class="hero-title">ANIMAL<br>CLASSIFIER</p>
    <p class="hero-sub">Upload a photo — AI identifies the animal instantly</p>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["  🐾 Single Image  ", "  📁 Multi Upload  ", "  🕐 History  "])

# ════════════════════════════════════════════
# TAB 1 — Single Image
# ════════════════════════════════════════════
with tab1:
    uploaded = st.file_uploader("Upload gambar hewan", type=["jpg", "jpeg", "png"], label_visibility="collapsed", key="single")

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image, use_container_width=True)

        if st.button("✦  IDENTIFY ANIMAL", key="btn_single"):
            # Animasi loading
            progress_placeholder = st.empty()
            progress_placeholder.markdown('<div class="loading-bar"></div>', unsafe_allow_html=True)
            
            status = st.empty()
            messages = ["🔍 Memuat model...", "🧠 Menganalisis fitur...", "⚡ Menghitung probabilitas...", "✅ Selesai!"]
            for msg in messages:
                status.markdown(f"<p style='color:#7a7a8a;font-size:0.85rem;text-align:center'>{msg}</p>", unsafe_allow_html=True)
                time.sleep(0.4)

            progress_placeholder.empty()
            status.empty()

            try:
                label, probs, classes = predict(image)
                emoji   = EMOJI_MAP.get(label.lower(), "🐾")
                conf    = probs[classes.index(label)] * 100
                info    = ANIMAL_INFO.get(label.lower(), {})

                # Simpan ke history
                st.session_state.history.append({
                    "emoji": emoji, "label": label.upper(), "conf": f"{conf:.1f}%"
                })

                # Probability bars semua kelas
                prob_bars = ""
                for i, cls in enumerate(classes):
                    p = probs[i] * 100
                    is_active = cls == label
                    fill_class = "bar-fill" if is_active else "bar-fill-dim"
                    name_color = "#c9a96e" if is_active else "#7a7a8a"
                    prob_bars += f"""
                    <div class="prob-item">
                        <div class="prob-header">
                            <span style="color:{name_color}">{EMOJI_MAP.get(cls,'')} {cls.upper()}</span>
                            <span>{p:.1f}%</span>
                        </div>
                        <div class="bar-bg"><div class="{fill_class}" style="width:{int(p)}%"></div></div>
                    </div>"""

                # Pills
                pills = '<div class="classes-row">'
                for cls in classes:
                    active = "active" if cls == label else ""
                    pills += f'<span class="class-pill {active}">{EMOJI_MAP.get(cls,"")} {cls.upper()}</span>'
                pills += '</div>'

                # Facts
                facts_html = ""
                if info.get("facts"):
                    facts_html = "<ul style='margin:0.5rem 0 0 1rem;padding:0;'>"
                    for f in info["facts"]:
                        facts_html += f"<li style='color:#7a7a8a;font-size:0.82rem;margin-bottom:4px'>{f}</li>"
                    facts_html += "</ul>"

                st.markdown(f"""
                <div class="result-card">
                    <div class="result-emoji">{emoji}</div>
                    <div class="result-label">DETECTED ANIMAL</div>
                    <div class="result-animal">{label.upper()}</div>
                    <div style="margin-top:1.5rem">
                        <div class="result-label">PROBABILITY ALL CLASSES</div>
                        <div class="prob-row">{prob_bars}</div>
                    </div>
                    {pills}
                </div>
                """, unsafe_allow_html=True)

                # Info hewan
                if info:
                    st.markdown(f"""
                    <div class="animal-info">
                        <b>{emoji} {info['title']}</b><br><br>
                        {info['desc']}
                        {facts_html}
                    </div>
                    """, unsafe_allow_html=True)

            except FileNotFoundError:
                st.error("File model tidak ditemukan!")
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.markdown("""
        <div class="info-box">
            <b>Cara pakai:</b><br>
            1. Upload foto hewan (JPG/PNG)<br>
            2. Klik tombol <b>IDENTIFY ANIMAL</b><br>
            3. Lihat hasil prediksi + probabilitas semua kelas + info hewan
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════
# TAB 2 — Multi Upload
# ════════════════════════════════════════════
with tab2:
    multi_files = st.file_uploader("Upload beberapa gambar sekaligus", type=["jpg", "jpeg", "png"],
                                    accept_multiple_files=True, label_visibility="collapsed", key="multi")

    if multi_files:
        if st.button("✦  IDENTIFY ALL", key="btn_multi"):
            st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)
            
            progress = st.progress(0)
            for i, file in enumerate(multi_files):
                image = Image.open(file).convert("RGB")
                label, probs, classes = predict(image)
                emoji = EMOJI_MAP.get(label.lower(), "🐾")
                conf  = probs[classes.index(label)] * 100

                st.session_state.history.append({
                    "emoji": emoji, "label": label.upper(), "conf": f"{conf:.1f}%"
                })

                col_img, col_res = st.columns([1, 2])
                with col_img:
                    st.image(image, use_container_width=True)
                with col_res:
                    prob_bars = ""
                    for j, cls in enumerate(classes):
                        p = probs[j] * 100
                        fill = "bar-fill" if cls == label else "bar-fill-dim"
                        c = "#c9a96e" if cls == label else "#7a7a8a"
                        prob_bars += f"""
                        <div class="prob-item">
                            <div class="prob-header"><span style="color:{c}">{EMOJI_MAP.get(cls,'')} {cls.upper()}</span><span>{p:.1f}%</span></div>
                            <div class="bar-bg"><div class="{fill}" style="width:{int(p)}%"></div></div>
                        </div>"""

                    st.markdown(f"""
                    <div class="multi-result" style="flex-direction:column;align-items:flex-start">
                        <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px">
                            <span style="font-size:2rem">{emoji}</span>
                            <div>
                                <div class="multi-label">{label.upper()}</div>
                                <div class="multi-conf">Confidence: {conf:.1f}%</div>
                            </div>
                        </div>
                        {prob_bars}
                    </div>
                    """, unsafe_allow_html=True)

                progress.progress((i + 1) / len(multi_files))

            st.success(f"✅ {len(multi_files)} gambar berhasil dianalisis!")
    else:
        st.markdown("""
        <div class="info-box">
            <b>Multi Upload:</b><br>
            Upload beberapa foto sekaligus →
            klik <b>IDENTIFY ALL</b> →
            semua gambar diprediksi sekaligus!
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════
# TAB 3 — History
# ════════════════════════════════════════════
with tab3:
    if st.session_state.history:
        st.markdown(f"""
        <div class="history-title">— {len(st.session_state.history)} predictions this session —</div>
        """, unsafe_allow_html=True)

        for i, item in enumerate(reversed(st.session_state.history)):
            st.markdown(f"""
            <div class="multi-result">
                <span style="font-size:1.5rem">{item['emoji']}</span>
                <div>
                    <div class="multi-label">{item['label']}</div>
                    <div class="multi-conf">Confidence: {item['conf']}</div>
                </div>
                <div style="margin-left:auto;font-size:0.75rem;color:#3a3a4a">#{len(st.session_state.history)-i}</div>
            </div>
            """, unsafe_allow_html=True)

        if st.button("🗑️  Clear History", key="clear"):
            st.session_state.history = []
            st.rerun()
    else:
        st.markdown("""
        <div class="info-box">
            <b>History kosong.</b><br>
            Prediksi beberapa gambar dulu di tab Single Image atau Multi Upload!
        </div>
        """, unsafe_allow_html=True)