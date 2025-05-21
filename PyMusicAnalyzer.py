#!/usr/bin/env python3

import os
import argparse
import warnings

# Suppress known warnings from librosa/audioread and NumPy deprecations
warnings.filterwarnings("ignore", message="PySoundFile failed.*")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tabulate import tabulate

import mutagen
from mutagen.id3 import ID3, ID3NoHeaderError, TBPM, TXXX
from mutagen.mp4 import MP4, MP4Tags

class MusicAnalyzer:
    def __init__(self, do_png):
        self.do_png = do_png
        self.mood_mapping = {
            (0, 0): "sad",
            (0, 1): "calm/peaceful",
            (1, 0): "angry/intense",
            (1, 1): "happy/joyful"
        }

    def read_embedded(self, file_path):
        """
        Try to read BPM, Energy, Valence, Danceability, Mood from existing tags.
        Return dict (including 'file_path' and 'embedded'=True) if all present, else None.
        """
        ext = os.path.splitext(file_path)[1].lower()
        try:
            if ext == '.mp3':
                tags = ID3(file_path)
                bpm = float(tags.get('TBPM').text[0])
                data = {'bpm': bpm}
                for key in ('Energy','Valence','Danceability','Mood'):
                    frame = next(f for f in tags.getall('TXXX') if f.desc == key)
                    val = frame.text[0]
                    data[key.lower()] = float(val) if key!='Mood' else val

            elif ext in ('.m4a','.mp4','.aac'):
                audio = MP4(file_path)
                t = audio.tags or {}
                bpm = float(t.get('tmpo',[None])[0])
                data = {'bpm': bpm}
                for key in ('Energy','Valence','Danceability','Mood'):
                    atom = f'----:com.apple.iTunes:{key}'
                    raw = t.get(atom,[None])[0]
                    if raw is None: 
                        raise KeyError
                    val = raw.decode('utf-8')
                    data[key.lower()] = float(val) if key!='Mood' else val

            else:
                audio = mutagen.File(file_path)
                data = {
                    'bpm': float(audio.get('BPM')[0]),
                    'energy':      float(audio.get('ENERGY')[0]),
                    'valence':     float(audio.get('VALENCE')[0]),
                    'danceability':float(audio.get('DANCEABILITY')[0]),
                    'mood':        audio.get('MOOD')[0]
                }

        except Exception:
            return None

        data.update({
            'file':      os.path.basename(file_path),
            'file_path': file_path,
            'embedded':  True
        })
        return data

    def analyze_track(self, file_path):
        print(f"Analyzing: {os.path.basename(file_path)}")
        try:
            y, sr = librosa.load(file_path, duration=60)

            # 1) compute raw values
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

            energy_raw = np.mean(
                MinMaxScaler().fit_transform(
                    librosa.feature.rms(y=y)[0].reshape(-1,1)
                )
            )

            cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            y_harmonic, _ = librosa.effects.hpss(y)
            contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            factors = [
                np.mean(cent)/(sr/2),
                np.mean(np.abs(y_harmonic))*10,
                np.mean(contrast)*0.2,
                np.mean(rolloff)/(sr/2)
            ]
            valence_raw = max(0, min(1, np.mean(factors)))

            danceability_raw = self.get_danceability(y, sr)

            # 2) force Python floats
            tempo        = float(tempo)
            energy       = float(energy_raw)
            valence      = float(valence_raw)
            danceability = float(danceability_raw)

            # 3) mood quadrant
            mood = self.mood_mapping[
                (1 if energy > 0.5 else 0,
                 1 if valence > 0.5 else 0)
            ]

            return {
                "file":         os.path.basename(file_path),
                "file_path":    file_path,
                "bpm":          round(tempo, 1),
                "energy":       round(energy * 10, 1),
                "valence":      round(valence * 10, 1),
                "danceability": round(danceability * 100, 1),
                "mood":         mood,
                "embedded":     False
            }

        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
            return None

    def get_danceability(self, y, sr):
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        ac = librosa.autocorrelate(onset_env, max_size=len(onset_env))
        ac_norm = ac/ac.max() if ac.max()>0 else ac
        try:
            from librosa.util import peak_pick
            peaks = peak_pick(ac_norm)
        except Exception:
            peaks = np.array([
                i for i in range(1, len(ac_norm)-1)
                if ac_norm[i]>ac_norm[i-1]
                and ac_norm[i]>ac_norm[i+1]
                and ac_norm[i]>0.5
            ])

        rhythm = float(np.mean(ac_norm[peaks])) if getattr(peaks,'size',len(peaks))>0 else 0
        beat_strength = float(np.mean(onset_env)) * 10
        dance = 0.6*rhythm + 0.4*min(1.0, beat_strength)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo_factor = 1.0 - min(1.0, abs(tempo - 110)/40)
        return float(max(0, min(1, 0.8*dance + 0.2*tempo_factor)))

    def visualize_results(self, result):
        if not self.do_png:
            return

        fig, axes = plt.subplots(2,2,figsize=(12,10))
        fname = result["file"]

        axes[0,0].bar([fname],[result["bpm"]])
        axes[0,0].set_title("Tempo (BPM)")
        axes[0,0].tick_params(axis='x', rotation=45)

        x, w = np.arange(1), 0.35
        axes[0,1].bar(x-w/2,[result["energy"]],w,label='Energy')
        axes[0,1].bar(x+w/2,[result["valence"]],w,label='Valence')
        axes[0,1].set_title("Energy vs Valence")
        axes[0,1].set_xticks(x)
        axes[0,1].set_xticklabels([fname])
        axes[0,1].legend()
        axes[0,1].tick_params(axis='x', rotation=45)

        axes[1,0].bar([fname],[result["danceability"]])
        axes[1,0].set_title("Danceability (%)")
        axes[1,0].tick_params(axis='x', rotation=45)

        v_norm, e_norm = result["valence"]/10, result["energy"]/10
        axes[1,1].scatter(v_norm,e_norm,s=200)
        axes[1,1].set_title("Mood Quadrant")
        axes[1,1].axhline(0.5, linestyle='--')
        axes[1,1].axvline(0.5, linestyle='--')
        axes[1,1].annotate(
            result["mood"],
            (v_norm,e_norm),
            xytext=(10,10),
            textcoords='offset points',
            bbox=dict(boxstyle='round', fc='yellow', alpha=0.5)
        )

        plt.tight_layout()
        out_dir = 'analysis_results'
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir,
                                f"{os.path.splitext(fname)[0]}_analysis.png")
        plt.savefig(out_file)
        plt.close()
        print(f"Visualization saved to {out_file}")

    def embed_metadata(self, file_path, result):
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.mp3':
            from mutagen.mp3 import MP3
            try:
                tags = ID3(file_path)
            except ID3NoHeaderError:
                mp3 = MP3(file_path)
                mp3.add_tags()
                tags = ID3(file_path)
            tags.add(TBPM(encoding=3, text=str(result['bpm'])))
            for key in ('Energy','Valence','Mood','Danceability'):
                tags.add(TXXX(encoding=3, desc=key,
                              text=str(result[key.lower()])))
            tags.save()

        elif ext in ('.m4a','.mp4','.aac'):
            audio = MP4(file_path)
            if audio.tags is None:
                audio.tags = MP4Tags()
            audio.tags['tmpo'] = [int(result['bpm'])]
            for key in ('Energy','Valence','Mood','Danceability'):
                atom = f'----:com.apple.iTunes:{key}'
                audio.tags[atom] = [str(result[key.lower()]).encode('utf-8')]
            audio.save()

        else:
            audio = mutagen.File(file_path)
            audio['BPM']         = str(result['bpm'])
            audio['ENERGY']      = str(result['energy'])
            audio['VALENCE']     = str(result['valence'])
            audio['MOOD']        = result['mood']
            audio['DANCEABILITY']= str(result['danceability'])
            audio.save()

    def main(self):
        parser = argparse.ArgumentParser(
            description='Analyze music for BPM, mood, danceability.'
        )
        parser.add_argument('files', nargs='+',
                            help='Audio file paths')
        parser.add_argument('-PNG', action='store_true', dest='png',
                            help='Enable saving PNG visualizations')
        args = parser.parse_args()

        analyzer = MusicAnalyzer(do_png=args.png)
        results = []

        for path in args.files:
            if not os.path.isfile(path):
                print(f"File not found: {path}")
                continue

            embedded = analyzer.read_embedded(path)
            if embedded:
                print(f"✔️  Using embedded metadata for {embedded['file']}")
                results.append(embedded)
                analyzer.visualize_results(embedded)
                continue

            res = analyzer.analyze_track(path)
            if res:
                results.append(res)
                analyzer.visualize_results(res)

        if results:
            print('\nAnalysis Results:')
            table = [
                [r['file'], r['bpm'], r['energy'],
                 r['valence'], r['mood'], r['danceability']]
                for r in results
            ]
            print(tabulate(
                table,
                headers=['File','BPM','Energy','Valence','Mood','Danceability'],
                tablefmt='grid'
            ))

            for r in results:
                if not r.get('embedded', False):
                    analyzer.embed_metadata(r['file_path'], r)


if __name__ == '__main__':
    MusicAnalyzer(do_png=False).main()
