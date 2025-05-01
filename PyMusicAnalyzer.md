## Music Analyzer CLI

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
git https://github.com/gianseo/PyMusicAnalyzer.git
cd PyMusicAnalyzer
pip install -r requirements.txt
```

---

### 🛠️ Usage

Run the analyzer by specifying one or more audio files:

```bash
python PyMusicAnalyzer.py path/to/song.mp3
```

Analyze multiple files at once:

```bash
python PyMusicAnalyzer.py song1.mp3 song2.wav song3.ogg
```

#### Optional Flags

- `-d, --duration <seconds>`: Analyze only the first N seconds (default: 60).
- `-o, --output-dir <dir>`: Specify output directory for visualizations (default: `analysis_results`).

---

### 📊 Output

1. **Console Table**: Shows BPM, Energy, Valence, Mood, and Danceability.
2. **Visualizations**: For each track, saves a PNG with:
   - BPM chart
   - Energy vs. Valence bar plot
   - Danceability bar chart
   - Mood quadrant scatter

   Files are saved to `analysis_results/<track_name>_analysis.png`.
3. **Metadata Embedding**: Writes only BPM and custom metrics—no comment fields:
   - **BPM**
     - MP3: ID3 `TBPM`
     - M4A/AAC: MP4 `tmpo`
     - FLAC/OGG/WAV: `BPM` tag (format support varies)
   - **Energy, Valence, Mood, Danceability**
     - MP3: four `TXXX` frames (`desc` = Energy, Valence, Mood, Danceability)
     - M4A/AAC: four freeform atoms `----:com.apple.iTunes:<Key>`
     - FLAC/OGG/WAV: flat tags `ENERGY`, `VALENCE`, `MOOD`, `DANCEABILITY`

Most tag editors and media players will display these fields, and DJ software can use the standard BPM tag.

---

### 🔍 Understanding the Metrics

- **BPM (Tempo)**: Beats per minute; higher = faster.
- **Energy** *(0–10)*: Loudness/intensity; higher = more energetic.
- **Valence** *(0–10)*: Positiveness; higher = more happy.
- **Mood**: Quadrant based on energy/valence:
  - Low Energy, Low Valence → **Sad**
  - Low Energy, High Valence → **Calm/Peaceful**
  - High Energy, Low Valence → **Angry/Intense**
  - High Energy, High Valence → **Happy/Joyful**
- **Danceability** *(0–100%)*: Rhythm regularity & tempo; higher = more danceable.

---

### ⚙️ Technical Notes

- Uses `librosa.beat.beat_track` for tempo extraction.
- Analysis uses first 60 seconds by default.
- Custom metrics written without comment tags for compatibility.

---

### 🎉 Example

```bash
python PyMusicAnalyzer.py happy_song.mp3 sad_ballad.mp3
```

Generates console output, visualizations, and embeds BPM + custom metrics.

---

### 🔗 License

MIT © Gianseo, Giulio Tuseo
