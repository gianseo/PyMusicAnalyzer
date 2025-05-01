#!/usr/bin/env python3
import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os
import argparse
from tabulate import tabulate
import mutagen
from mutagen.id3 import ID3, ID3NoHeaderError, TBPM, TXXX
from mutagen.mp4 import MP4, MP4Tags

class MusicAnalyzer:
    def __init__(self):
        self.mood_mapping = {
            (0, 0): "sad",
            (0, 1): "calm/peaceful",
            (1, 0): "angry/intense",
            (1, 1): "happy/joyful"
        }

    def analyze_track(self, file_path):
        print(f"Analyzing: {os.path.basename(file_path)}")
        try:
            y, sr = librosa.load(file_path, duration=60)
            tempo, _ = self.get_bpm(y, sr)
            energy = float(self.get_energy(y))
            valence = float(self.get_valence(y, sr))
            danceability = float(self.get_danceability(y, sr))
            mood = self.determine_mood(energy, valence)

            return {
                "file": os.path.basename(file_path),
                "bpm": round(tempo, 1),
                "energy": round(energy * 10, 1),
                "valence": round(valence * 10, 1),
                "mood": mood,
                "danceability": round(danceability * 100, 1)
            }
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
            import traceback; traceback.print_exc()
            return None

    def get_bpm(self, y, sr):
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        # Use beat_track to get tempo more reliably
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        return float(tempo), onset_env

    def get_energy(self, y):
        rms = librosa.feature.rms(y=y)[0]
        scaler = MinMaxScaler()
        return np.mean(scaler.fit_transform(rms.reshape(-1, 1)))

    def get_valence(self, y, sr):
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        y_harmonic, _ = librosa.effects.hpss(y)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        factors = [
            np.mean(cent) / (sr / 2),
            np.mean(np.abs(y_harmonic)) * 10,
            np.mean(spectral_contrast) * 0.2,
            np.mean(spectral_rolloff) / (sr / 2)
        ]
        return max(0, min(1, np.mean(factors)))

    def get_danceability(self, y, sr):
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        ac = librosa.autocorrelate(onset_env, max_size=len(onset_env))
        ac_norm = ac / ac.max() if ac.max() > 0 else ac
        try:
            from librosa.util import peak_pick
            peaks = peak_pick(ac_norm, pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=0.5, wait=10)
        except Exception:
            peaks = np.array([i for i in range(1, len(ac_norm) - 1)
                              if ac_norm[i] > ac_norm[i - 1] and ac_norm[i] > ac_norm[i + 1] and ac_norm[i] > 0.5])
        # Ensure peaks is iterable and check length
        if isinstance(peaks, np.ndarray) and peaks.size > 0:
            rhythm_regularity = np.mean(ac_norm[peaks])
        elif isinstance(peaks, (list, tuple)) and len(peaks) > 0:
            rhythm_regularity = np.mean([ac_norm[i] for i in peaks])
        else:
            rhythm_regularity = 0
        beat_strength = np.mean(onset_env) * 10
        dance = 0.6 * rhythm_regularity + 0.4 * min(1.0, beat_strength)
        tempo_factor = 1.0 - min(1.0, abs(tempo - 110) / 40)
        return max(0, min(1, 0.8 * dance + 0.2 * tempo_factor))

    def determine_mood(self, energy, valence):
        e_bin = 1 if energy > 0.5 else 0
        v_bin = 1 if valence > 0.5 else 0
        return self.mood_mapping[(e_bin, v_bin)]

    def visualize_results(self, results):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes[0, 0].bar(results["file"], results["bpm"])
        axes[0, 0].set_title("Tempo (BPM)")
        axes[0, 0].tick_params(axis='x', rotation=45)
        x = np.arange(1); w = 0.35
        axes[0, 1].bar(x - w/2, results["energy"], w, label='Energy')
        axes[0, 1].bar(x + w/2, results["valence"], w, label='Valence')
        axes[0, 1].set_title('Energy vs Valence')
        axes[0, 1].set_xticks(x); axes[0, 1].set_xticklabels([results["file"]])
        axes[0, 1].legend(); axes[0, 1].tick_params(axis='x', rotation=45)
        axes[1, 0].bar(results["file"], results["danceability"])
        axes[1, 0].set_title("Danceability (%)"); axes[1, 0].tick_params(axis='x', rotation=45)
        v_norm = results["valence"] / 10; e_norm = results["energy"] / 10
        axes[1, 1].scatter(v_norm, e_norm, s=200)
        axes[1, 1].set_title("Mood Quadrant")
        axes[1, 1].axhline(0.5, linestyle='--'); axes[1, 1].axvline(0.5, linestyle='--')
        axes[1, 1].annotate(results["mood"], (v_norm, e_norm), xytext=(10, 10), textcoords='offset points', bbox=dict(boxstyle='round', fc='yellow', alpha=0.5))
        plt.tight_layout()
        out_dir = 'analysis_results'; os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, f"{os.path.splitext(results['file'])[0]}_analysis.png")
        plt.savefig(out_file); plt.close(); print(f"Visualization saved to {out_file}")

    def embed_metadata(self, file_path, result):
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.mp3':
            from mutagen.mp3 import MP3
            tags = None
            try:
                tags = ID3(file_path)
            except ID3NoHeaderError:
                tags = MP3(file_path); tags.add_tags(); tags = ID3(file_path)
            tags.add(TBPM(encoding=3, text=str(result['bpm'])))
            tags.add(TXXX(encoding=3, desc='Energy', text=str(result['energy'])))
            tags.add(TXXX(encoding=3, desc='Valence', text=str(result['valence'])))
            tags.add(TXXX(encoding=3, desc='Mood', text=result['mood']))
            tags.add(TXXX(encoding=3, desc='Danceability', text=str(result['danceability'])))
            tags.save()
        elif ext in ('.m4a', '.mp4', '.aac'):
            audio = MP4(file_path)
            if audio.tags is None: audio.tags = MP4Tags()
            audio.tags['tmpo'] = [int(result['bpm'])]
            for key in ('Energy', 'Valence', 'Mood', 'Danceability'):
                atom = f'----:com.apple.iTunes:{key}'
                val = str(result[key.lower()])
                audio.tags[atom] = [val.encode('utf-8')]
            audio.save()
        elif ext in ('.flac', '.ogg'):
            audio = mutagen.File(file_path)
            audio['BPM'] = str(result['bpm'])
            audio['ENERGY'] = str(result['energy'])
            audio['VALENCE'] = str(result['valence'])
            audio['MOOD'] = result['mood']
            audio['DANCEABILITY'] = str(result['danceability'])
            audio.save()
        elif ext == '.wav':
            from mutagen.wave import WAVE
            audio = WAVE(file_path)
            audio['TBPM'] = str(result['bpm'])
            audio['Energy'] = str(result['energy'])
            audio['Valence'] = str(result['valence'])
            audio['Mood'] = result['mood']
            audio['Danceability'] = str(result['danceability'])
            audio.save()
        else:
            print(f"Unsupported format: {ext}")

    def main(self):
        parser = argparse.ArgumentParser(description='Analyze music for BPM, mood, danceability.')
        parser.add_argument('files', nargs='+', help='Audio file paths')
        args = parser.parse_args()
        results = []
        for p in args.files:
            if os.path.isfile(p):
                res = self.analyze_track(p)
                if res: results.append(res); self.visualize_results(res)
            else:
                print(f"File not found: {p}")
        if results:
            print('\nAnalysis Results:')
            table = [[r['file'], r['bpm'], r['energy'], r['valence'], r['mood'], r['danceability']] for r in results]
            print(tabulate(table, headers=['File','BPM','Energy','Valence','Mood','Danceability'], tablefmt='grid'))
            for p in args.files:
                if os.path.isfile(p):
                    m = next((r for r in results if r['file']==os.path.basename(p)), None)
                    if m: self.embed_metadata(p, m)

if __name__ == '__main__':
    MusicAnalyzer().main()
