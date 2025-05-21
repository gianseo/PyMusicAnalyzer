# Music Analyzer CLI

A Python application to analyze audio files and extract musical features:

- **BPM** (Beats Per Minute)  
- **Energy** (intensity/loudness)  
- **Valence** (musical positiveness)  
- **Mood** (based on energy and valence)  
- **Danceability** (rhythm regularity and beat strength)  

---

### 📦 Prerequisites

Install the required dependencies:

```bash
pip install librosa numpy matplotlib scikit-learn tabulate mutagen
```

Optionally, pin versions via a `requirements.txt`:

```text
librosa>=0.9.0
numpy
matplotlib
scikit-learn
tabulate
mutagen
```

---

### 🚀 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/gianseo/PyMusicAnalyzer.git
cd PyMusicAnalyzer
pip install -r requirements.txt
```

---

### 🛠️ Usage

Run the analyzer by specifying one or more audio files:

```bash
python PyMusicAnalyzer.py path/to/song.mp3
```

Or batch-process multiple files:

```bash
python PyMusicAnalyzer.py song1.mp3 song2.wav song3.ogg
```

You can also enable PNG visualizations (off by default) with the `-PNG` flag:

```bash
python PyMusicAnalyzer.py -PNG song1.mp3 song2.m4a
```

---

### ⚙️ Optional Flags

- `-PNG`  
  Enable saving the 2×2 feature visualizations (BPM chart, Energy vs Valence, Danceability, Mood quadrant) as PNGs.  
  By default, visualizations are **not** generated—omit this flag if you only want the console output and metadata embedding.

---

### 📊 Output

1. **Embedded‐tag fallback**  
   If your file already contains embedded tags for BPM, Energy, Valence, Mood, and Danceability, the script will read and use those values (skipping re‐analysis).

2. **Console Table**  
   Shows BPM, Energy (0–10), Valence (0–10), Mood (sad/calm/angry/happy), and Danceability (0–100%).

3. **Visualizations**  
   (Only with `-PNG`)  
   Saves a PNG per track to `analysis_results/<track_name>_analysis.png`, containing:  
   - Tempo (BPM) bar chart  
   - Energy vs Valence bar plot  
   - Danceability bar chart  
   - Mood quadrant scatterplot  

4. **Metadata Embedding**  
   For any file that **did not** already have embedded metrics, writes your computed values back into its tags:  
   - **BPM**  
     - MP3: ID3 `TBPM`  
     - M4A/AAC: MP4 `tmpo`  
     - FLAC/OGG/WAV: `BPM` tag  
   - **Energy, Valence, Mood, Danceability**  
     - MP3: four `TXXX` frames (`desc` = Energy, Valence, Mood, Danceability)  
     - M4A/AAC: four freeform atoms `----:com.apple.iTunes:<Key>`  
     - FLAC/OGG/WAV: flat tags `ENERGY`, `VALENCE`, `MOOD`, `DANCEABILITY`  

---

### 🔍 Understanding the Metrics

- **BPM (Tempo)**: Beats per minute; higher = faster.  
- **Energy** *(0–10)*: Loudness/intensity; higher = more energetic.  
- **Valence** *(0–10)*: Positiveness; higher = more happy.  
- **Mood**:  
  - Low E, Low V → **Sad**  
  - Low E, High V → **Calm / Peaceful**  
  - High E, Low V → **Angry / Intense**  
  - High E, High V → **Happy / Joyful**  
- **Danceability** *(0–100%)*: Rhythm regularity & tempo; higher = more danceable.

---

### 📈 Technical Notes

- **Embedded‐tag check**: skips expensive Librosa analysis when tags are already present.  
- **Librosa analysis**:  
  - Uses `beat_track` for tempo.  
  - RMS for energy (normalized).  
  - Composite of spectral stats for valence.  
  - Onset‐autocorrelation + tempo proximity for danceability.  
- **Warnings**: suppresses PySoundFile fallback and deprecation warnings for a cleaner console.

---

### 🎉 Example

```bash
# Just compute & embed tags (no PNGs):
python PyMusicAnalyzer.py mellow.wav upbeat.mp3

# Compute + save visualizations:
python PyMusicAnalyzer.py -PNG mellow.wav upbeat.mp3
```

---

### 🔗 License

MIT © Gianseo, Giulio Tuseo
