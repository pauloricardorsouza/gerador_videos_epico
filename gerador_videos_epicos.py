import os
import streamlit as st
from io import BytesIO
from PIL import Image
import requests
from moviepy.editor import ImageClip, CompositeVideoClip, AudioFileClip, TextClip, vfx
from pydub import AudioSegment
from elevenlabs import generate, set_api_key
import numpy as np
import docx

# ============================
# CONFIGURAÇÕES DE API
# ============================
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
ELEVEN_API_KEY = st.secrets.get("ELEVEN_API_KEY")

if not OPENAI_API_KEY or not ELEVEN_API_KEY:
    st.warning("Por favor, configure suas API Keys nos Secrets do Streamlit Cloud.")
    st.stop()

set_api_key(ELEVEN_API_KEY)

PASTA_VIDEOS = "videos_gerados"
os.makedirs(PASTA_VIDEOS, exist_ok=True)

# ============================
# FUNÇÕES PRINCIPAIS
# ============================
def gerar_imagem(texto, tamanho="1024x1024"):
    endpoint = "https://api.openai.com/v1/images/generations"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    data = {"model": "gpt-image-1", "prompt": texto, "size": tamanho}
    resp = requests.post(endpoint, headers=headers, json=data)
    url = resp.json()["data"][0]["url"]
    imagem = Image.open(BytesIO(requests.get(url).content))
    return imagem

def gerar_narracao(texto, voz="Antoni"):
    audio = generate(text=texto, voice=voz, model="eleven_multilingual_v2")
    return AudioSegment.from_file(BytesIO(audio), format="mp3")

def gerar_trilha_epica(duracao_segundos=60, volume_db=-20):
    sr = 44100
    t = np.linspace(0, duracao_segundos, int(sr * duracao_segundos), False)
    pad1 = 0.1 * np.sin(2 * np.pi * 110 * t)
    pad2 = 0.08 * np.sin(2 * np.pi * 132 * t)
    bass = 0.07 * np.sin(2 * np.pi * 55 * t)
    noise = 0.02 * (np.random.rand(len(t)) - 0.5)
    mix = pad1 + pad2 + bass + noise
    mix = mix / (np.max(np.abs(mix)) + 1e-9)
    audio_int16 = np.int16(mix * 32767)
    segment = AudioSegment(data=audio_int16.tobytes(), sample_width=2, frame_rate=sr, channels=1)
    stereo = AudioSegment.from_mono_audiosegments(segment.pan(-0.2), segment.pan(0.2))
    stereo = stereo - abs(volume_db)
    return stereo

def criar_video_cena(cena_texto, duracao=30, voz="Antoni", cena_num=1):
    # Imagem
    img = gerar_imagem(cena_texto)
    img_path = os.path.join(PASTA_VIDEOS, f"cena_{cena_num}.png")
    img.save(img_path)

    # Áudio
    voz_seg = gerar_narracao(cena_texto, voz)
    trilha = gerar_trilha_epica(duracao)
    final_audio = voz_seg.overlay(trilha)
    audio_path = os.path.join(PASTA_VIDEOS, f"audio_{cena_num}.mp3")
    final_audio.export(audio_path, format="mp3")

    # Vídeo
    img_clip = ImageClip(img_path).set_duration(duracao).resize(height=720).set_position("center")
    img_clip = img_clip.fx(vfx.zoom_in, final_scale=1.05)
    legenda_clip = TextClip(txt=cena_texto, fontsize=40, color="white", font="Georgia-Bold",
                            size=(720, None), method="caption", align="center").set_duration(duracao)
    video = CompositeVideoClip([img_clip, legenda_clip]).set_audio(AudioFileClip(audio_path))

    out_path = os.path.join(PASTA_VIDEOS, f"video_{cena_num}.mp4")
    video.write_videofile(out_path, fps=24, codec="libx264", audio_codec="aac", verbose=False, logger=None)
    return out_path

# ============================
# INTERFACE WEB
# ============================
st.title("🎬 Gerador de Vídeos Épicos Web")

arquivo = st.file_uploader("Faça upload do roteiro (.txt ou .docx):", type=["txt", "docx"])
duracao = st.slider("Duração aproximada do vídeo (segundos):", min_value=10, max_value=120, value=30)
voz = st.selectbox("Escolha a voz para narração:", ["Antoni", "Bella", "Elli", "Josh"])

if st.button("Gerar Vídeos") and arquivo:
    st.info("Gerando vídeos, aguarde...")
    cenas = []
    if arquivo.type == "text/plain":
        texto = arquivo.read().decode("utf-8")
        blocos = texto.strip().split('\n\n')
        for bloco in blocos:
            cenas.append(bloco.strip())
    elif arquivo.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(arquivo)
        for p in doc.paragraphs:
            if p.text.strip():
                cenas.append(p.text.strip())

    for i, cena_texto in enumerate(cenas, 1):
        video_path = criar_video_cena(cena_texto, duracao, voz, i)
        st.video(video_path)
        with open(video_path, "rb") as f:
            st.download_button(f"Download Vídeo {i}", f, file_name=f"video_{i}.mp4")
        st.success(f"Vídeo {i} pronto!")
