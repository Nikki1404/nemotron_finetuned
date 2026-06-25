for f in data/audio_16k/*.wav; do
  base=$(basename "$f" .wav)
  mkdir -p "data/audio_chunks/$base"
  ffmpeg -y -i "$f" -f segment -segment_time 20 -c copy "data/audio_chunks/$base/${base}_%03d.wav"
done
