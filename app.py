import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import cv2
from deepface import DeepFace
import torch
import torch.nn as nn
import numpy as np
import sounddevice as sd
import threading
import queue
import time
import collections
import hashlib
from scipy.signal import butter, lfilter, find_peaks
from pedalboard import Pedalboard, Reverb, Chorus, LowpassFilter, Gain, Delay

# ==========================================
# 1. KONFIGURASI SISTEM & MUSIC THEORY
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LATENT_DIM = 100
EMOTION_DIM = 2
SAMPLE_RATE = 16000
CHECKPOINT_PATH = os.path.join(BASE_DIR, "generator_epoch_500.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

current_raw_emotion = "Calm" 
current_va = [0.0, 0.0]      
is_running = True
is_locked = False
current_ai_seed_hash = "INITIALIZING..."

emotion_map = {
    "happy": {"name": "Happy", "va": [0.8, 0.6]},
    "sad": {"name": "Sad", "va": [-0.6, -0.5]},
    "angry": {"name": "Angry", "va": [-0.7, 0.8]},
    "neutral": {"name": "Calm", "va": [0.0, 0.0]},
    "fear": {"name": "Sad", "va": [-0.5, 0.7]},
    "surprise": {"name": "Happy", "va": [0.4, 0.8]},
    "disgust": {"name": "Angry", "va": [-0.7, 0.2]}
}

chords = {
    "Happy": [261.63, 329.63, 392.00, 493.88, 523.25, 659.25], 
    "Sad":   [261.63, 311.13, 392.00, 587.33, 622.25, 783.99], 
    "Angry": [130.81, 146.83, 185.00, 196.00, 261.63, 293.66], 
    "Calm":  [174.61, 220.00, 261.63, 329.63, 392.00, 440.00]  
}

# ==========================================
# 2. ARSITEKTUR AI
# ==========================================
class WaveGANGenerator(nn.Module):
    def __init__(self, latent_dim, emotion_dim):
        super(WaveGANGenerator, self).__init__()
        self.fc = nn.Linear(latent_dim + emotion_dim, 100 * 128)
        self.conv_blocks = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=25, stride=4, padding=11, output_padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(64, 32, kernel_size=25, stride=5, padding=10, output_padding=0),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(32, 16, kernel_size=25, stride=4, padding=11, output_padding=1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(16, 1, kernel_size=25, stride=4, padding=11, output_padding=1),
            nn.Tanh()
        )
    def forward(self, noise, emotion):
        x = torch.cat([noise, emotion], dim=1)
        x = self.fc(x)
        x = x.view(-1, 128, 100) 
        return self.conv_blocks(x)

print("[SYSTEM] Memuat AI Core...")
generator = WaveGANGenerator(LATENT_DIM, EMOTION_DIM).to(device)
try:
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    if 'generator_state' in checkpoint:
        generator.load_state_dict(checkpoint['generator_state'])
    else:
        generator.load_state_dict(checkpoint)
    generator.eval()
except Exception as e:
    print(f"[ERROR KRITIS] {e}")
    exit()

# ==========================================
# 3. SYNTHESIZER
# ==========================================
def butter_lowpass_filter(data, cutoff, fs, order=3):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, data)

def generate_synth_pluck(frequency, duration, sr):
    t = np.linspace(0, duration, int(sr * duration), False)
    tone = np.sin(2 * np.pi * frequency * t) * 0.7
    tone += np.sign(np.sin(2 * np.pi * (frequency * 1.01) * t)) * 0.15 
    decay = np.exp(-5.0 * t) 
    return tone * decay

# ==========================================
# 4. GAPLESS AUDIO STREAMING ENGINE
# ==========================================
audio_buffer = queue.Queue(maxsize=10) 
audio_remainder = np.zeros(0, dtype=np.float32)

def flush_audio():
    """Membersihkan antrean seketika saat emosi berubah agar langsung transisi"""
    global audio_remainder
    with audio_buffer.mutex:
        audio_buffer.queue.clear()
    audio_remainder = np.zeros(0, dtype=np.float32)

def audio_callback(outdata, frames, time_info, status):
    """Callback ini dipanggil langsung oleh Soundcard. JANGAN masukkan sleep/delay di sini!"""
    global audio_remainder
    needed = frames
    written = 0
    outdata.fill(0.0) # Default silence
    
    while written < needed:
        if len(audio_remainder) > 0:
            take = min(needed - written, len(audio_remainder))
            outdata[written:written+take, 0] = audio_remainder[:take]
            audio_remainder = audio_remainder[take:]
            written += take
        else:
            try:
                # Ambil masakan terbaru dari The Chef (Non-blocking)
                target_emo, next_chunk = audio_buffer.get_nowait()
                audio_remainder = next_chunk.astype(np.float32)
            except queue.Empty:
                break # Jika Chef lambat, diam sejenak agar tidak crash

def generator_worker():
    global current_va, current_raw_emotion, is_running, current_ai_seed_hash
    
    with torch.inference_mode(): 
        current_latent = torch.randn(1, LATENT_DIM).to(device)
        note_index = 0 
        
        # PENTING: Papan efek tetap hidup di luar loop agar Reverb tidak terpotong!
        board = Pedalboard()
        current_board_emo = None
        
        while is_running:
            if audio_buffer.full():
                time.sleep(0.05)
                continue

            target_emo = current_raw_emotion
            target_va = current_va[:] 
            
            # --- UPDATE EFEK HANYA JIKA EMOSI BERUBAH ---
            if target_emo != current_board_emo:
                v_norm = (target_va[0] + 1.0) / 2.0 
                a_norm = (target_va[1] + 1.0) / 2.0
                
                board = Pedalboard()
                filter_cutoff = 800 + (v_norm * 5000) 
                board.append(LowpassFilter(cutoff_frequency_hz=filter_cutoff))
                
                delay_mix = 0.2 + (a_norm * 0.4)
                board.append(Delay(delay_seconds=0.3, feedback=0.4, mix=delay_mix))
                board.append(Chorus(rate_hz=0.5, depth=0.3))
                
                reverb_wet = 0.8 if target_emo in ["Calm", "Sad"] else 0.4
                board.append(Reverb(room_size=0.8, damping=0.5, wet_level=reverb_wet))
                board.append(Gain(gain_db=8.0))
                current_board_emo = target_emo

            gan_coords = emotion_map[[k for k, v in emotion_map.items() if v["name"] == target_emo][0]]["va"]
            labels = torch.tensor([gan_coords], dtype=torch.float32).to(device)
            
            # KITA HANYA MASAK 2 DETIK (Tanpa for-loop) AGAR LEBIH CEPAT
            drift = torch.randn(1, LATENT_DIM).to(device) * 0.15
            current_latent = current_latent + drift
            current_latent = current_latent / torch.norm(current_latent) * np.sqrt(LATENT_DIM)
            
            raw_chunk = generator(current_latent, labels).cpu().numpy()[0, 0, :]
            
            tensor_bytes = current_latent.cpu().numpy().tobytes()
            current_ai_seed_hash = hashlib.md5(tensor_bytes).hexdigest()[:12].upper()
            
            ai_pulse = np.abs(raw_chunk)
            ai_pulse = butter_lowpass_filter(ai_pulse, cutoff=100, fs=SAMPLE_RATE) 
            ai_pulse = ai_pulse / (np.max(ai_pulse) + 1e-8) 
            
            peak_threshold = 0.85 - (target_va[1] * 0.3) 
            peaks, _ = find_peaks(ai_pulse, height=peak_threshold, distance=SAMPLE_RATE//6)
            
            chunk_canvas = np.zeros(int(SAMPLE_RATE * 2.0))
            active_chord = chords[target_emo]
            latent_params = current_latent[0].cpu().numpy()
            
            for idx, peak_index in enumerate(peaks):
                param = latent_params[idx % LATENT_DIM]
                if target_emo == "Happy": note_index = (note_index + 1 + int(abs(param)*2)) % len(active_chord)
                elif target_emo in ["Sad", "Calm"]: note_index = (note_index - 1 - int(abs(param)*2)) % len(active_chord)
                else: note_index = int(abs(param)*10) % len(active_chord)

                freq = active_chord[note_index]
                pluck = generate_synth_pluck(freq, duration=1.0, sr=SAMPLE_RATE)
                
                end_index = min(peak_index + len(pluck), len(chunk_canvas))
                pluck_len = end_index - peak_index
                chunk_canvas[peak_index:end_index] += pluck[:pluck_len]
            
            # Terapkan efek dengan reset=False agar gema tidak terpotong saat chunk bersambung!
            final_audio = board(chunk_canvas, SAMPLE_RATE, reset=False)
            
            if len(final_audio.shape) > 1:
                final_audio = final_audio[0]
                
            audio_buffer.put((target_emo, final_audio))

# ==========================================
# 5. MAIN THREAD (Webcam & Proof Snapshot)
# ==========================================
def main():
    global current_va, current_raw_emotion, is_running, is_locked, current_ai_seed_hash
    
    gen_thread = threading.Thread(target=generator_worker)
    gen_thread.start()
    
    # MEMBUKA KONEKSI LANGSUNG KE SPEAKER (Tanpa thread player terpisah)
    stream = sd.OutputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback)
    stream.start()
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    frame_count = 0
    va_buffer = collections.deque(maxlen=7) 
    frozen_frame = None 
    
    while True:
        ret, live_frame = cap.read()
        if not ret: break
            
        frame_count += 1
        
        if not is_locked:
            display_frame = live_frame.copy() 
            
            if frame_count % 3 == 0:
                try:
                    bright_frame = cv2.convertScaleAbs(live_frame, alpha=1.2, beta=20)
                    results = DeepFace.analyze(
                        bright_frame, actions=['emotion'], enforce_detection=False, 
                        detector_backend='mtcnn', silent=True
                    )
                    
                    dominant_emo = results[0]['dominant_emotion']
                    if dominant_emo in emotion_map:
                        new_emotion = emotion_map[dominant_emo]["name"]
                        
                        emotion_probs = results[0]['emotion']
                        v_calc, a_calc = 0.0, 0.0
                        for emo, prob in emotion_probs.items():
                            if emo in emotion_map:
                                weight = prob / 100.0
                                v_calc += emotion_map[emo]["va"][0] * weight
                                a_calc += emotion_map[emo]["va"][1] * weight
                        
                        va_buffer.append([v_calc, a_calc])
                            
                        if len(va_buffer) > 0:
                            avg_v = sum(v[0] for v in va_buffer) / len(va_buffer)
                            avg_a = sum(v[1] for v in va_buffer) / len(va_buffer)
                            current_va = [round(avg_v, 2), round(avg_a, 2)]
                            
                            if new_emotion != current_raw_emotion:
                                current_raw_emotion = new_emotion
                                flush_audio() # Potong seketika agar langsung ganti lagu!
                except Exception:
                    pass
        else:
            if frozen_frame is not None:
                display_frame = frozen_frame.copy()
            else:
                display_frame = live_frame.copy()

        # --- BIOMETRIC HUD ---
        cv2.rectangle(display_frame, (10, 10), (550, 180), (20, 20, 20), -1)
        
        status_text = "SYSTEM: [LOCKED] - PROOF SNAPSHOT CAPTURED" if is_locked else "SYSTEM: [SCANNING] - LIVE VIDEO"
        text_color = (0, 0, 255) if is_locked else (0, 255, 0)
        cv2.putText(display_frame, status_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2, cv2.LINE_AA)
        
        cv2.putText(display_frame, f"MUSICAL ANCHOR : {current_raw_emotion}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        v_bar_len = int(((current_va[0] + 1.0) / 2.0) * 200)
        a_bar_len = int(((current_va[1] + 1.0) / 2.0) * 200)
        
        cv2.putText(display_frame, f"VALENCE (Filter/Tone): {current_va[0]:.2f}", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1, cv2.LINE_AA)
        cv2.rectangle(display_frame, (270, 95), (470, 110), (50, 50, 50), -1)
        cv2.rectangle(display_frame, (270, 95), (270 + v_bar_len, 110), (255, 200, 0), -1)
        
        cv2.putText(display_frame, f"AROUSAL (Delay/Speed): {current_va[1]:.2f}", (20, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 1, cv2.LINE_AA)
        cv2.rectangle(display_frame, (270, 125), (470, 140), (50, 50, 50), -1)
        cv2.rectangle(display_frame, (270, 125), (270 + a_bar_len, 140), (255, 100, 0), -1)
        
        cv2.putText(display_frame, f"AI TENSOR HASH: 0x{current_ai_seed_hash}", (20, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        
        cv2.imshow('Skripsi: Smart Affective Arpeggiator', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            is_running = False
            break
        elif key == ord(' '): 
            is_locked = not is_locked 
            if is_locked:
                frozen_frame = live_frame.copy()
            else:
                frozen_frame = None

    # Bersihkan memory dan tutup stream
    stream.stop()
    stream.close()
    cap.release()
    cv2.destroyAllWindows()
    gen_thread.join()

if __name__ == "__main__":
    main()