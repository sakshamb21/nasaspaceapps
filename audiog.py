from flask import Flask, request, send_file, jsonify
import numpy as np
from scipy.io.wavfile import write
import hashlib
import io

app = Flask(__name__)

def object_sound(object_name, duration=20, sample_rate=44100):
    """Generate deterministic noise for a single object name."""
    seed = int(hashlib.sha256(object_name.encode()).hexdigest(), 16) % (2**32)
    rng = np.random.default_rng(seed)
    samples = rng.normal(0, 0.3, int(sample_rate * duration))

    # Derive tonal shaping from hash
    tone_factor = seed % 100 / 100
    filter_size = int(200 + tone_factor * 1800)
    samples = np.convolve(samples, np.ones(filter_size)/filter_size, mode='same')

    # Normalize
    samples = samples / np.max(np.abs(samples))
    return samples


def combined_sound(object_names, duration=20, sample_rate=44100):
    """Combine multiple object sounds into one normalized track."""
    total_samples = np.zeros(int(sample_rate * duration))
    for name in object_names:
        total_samples += object_sound(name, duration, sample_rate)
    total_samples /= len(object_names)  # average mix
    total_samples = np.int16(total_samples / np.max(np.abs(total_samples)) * 32767)

    # Write WAV to memory
    buffer = io.BytesIO()
    write(buffer, sample_rate, total_samples)
    buffer.seek(0)
    return buffer


@app.route('/generate', methods=['POST'])
def generate():
    """
    POST JSON: { "objects": ["rock", "tree", "metal"] }
    Returns: Combined 20s WAV audio directly
    """
    data = request.get_json()
    if not data or "objects" not in data:
        return jsonify({"error": "Please send JSON with 'objects': [ ... ]"}), 400

    object_names = data["objects"]
    if isinstance(object_names, str):
        object_names = [object_names]

    audio_buffer = combined_sound(object_names)
    filename = "_".join(object_names) + ".wav"

    return send_file(
        audio_buffer,
        mimetype="audio/wav",
        as_attachment=False,
        download_name=filename
    )


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
