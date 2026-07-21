use std::fs::File;
use std::path::Path;

use anyhow::{anyhow, bail, Context, Result};
use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

pub const WHISPER_SAMPLE_RATE: u32 = 16_000;

pub const AUDIO_EXTENSIONS: &[&str] = &[
    "wav", "mp3", "m4a", "aac", "flac", "ogg", "oga", "opus", "aiff", "aif", "caf", "wma", "mka",
    "weba", "amr", "ac3", "dts", "ape", "wv", "au", "mp2", "spx", "tta", "mpc", "ra", "gsm", "w64",
];

pub const VIDEO_EXTENSIONS: &[&str] = &[
    "mp4", "mov", "m4v", "mkv", "webm", "avi", "ts", "mts", "m2ts", "3gp", "3g2", "flv", "wmv",
    "mpg", "mpeg", "m2v", "ogv", "vob", "asf", "f4v", "divx", "rm", "rmvb",
];

pub struct DecodedAudio {
    /// 16 kHz mono f32 PCM, ready for whisper
    pub samples: Vec<f32>,
    pub duration_secs: f32,
}

fn has_extension_in(path: &Path, list: &[&str]) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| list.contains(&e.to_ascii_lowercase().as_str()))
        .unwrap_or(false)
}

pub fn is_audio_file(path: &Path) -> bool {
    has_extension_in(path, AUDIO_EXTENSIONS)
}

pub fn is_video_file(path: &Path) -> bool {
    has_extension_in(path, VIDEO_EXTENSIONS)
}

pub fn is_media_file(path: &Path) -> bool {
    is_audio_file(path) || is_video_file(path)
}

fn ffmpeg_available() -> bool {
    std::process::Command::new("ffmpeg")
        .arg("-version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Container duration in seconds via ffprobe; None when ffprobe is
/// missing or the container doesn't declare one (progress goes
/// indeterminate, extraction still works).
fn probe_duration_secs(path: &Path) -> Option<f32> {
    let output = std::process::Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
        ])
        .arg(path)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let secs: f32 = String::from_utf8_lossy(&output.stdout).trim().parse().ok()?;
    (secs > 0.0).then_some(secs)
}

/// Extract/convert any media via ffmpeg, streaming raw 16 kHz mono f32
/// PCM over stdout — no temp files, video streams dropped with -vn.
/// Progress is derived from the PCM byte count against the ffprobe
/// duration (-1 when unknown). Returns None if cancelled.
fn ffmpeg_extract(
    path: &Path,
    cancel: Option<&std::sync::atomic::AtomicBool>,
    on_progress: &mut dyn FnMut(i32),
) -> Result<Option<DecodedAudio>> {
    use std::io::Read;
    use std::sync::atomic::Ordering;

    let total_secs = probe_duration_secs(path);

    let mut child = std::process::Command::new("ffmpeg")
        .args(["-v", "error", "-nostdin", "-i"])
        .arg(path)
        .args([
            "-vn", "-sn", "-map", "0:a:0", "-ac", "1", "-ar", "16000", "-f", "f32le", "-",
        ])
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .context("running ffmpeg")?;

    // Drain stderr on its own thread so a chatty ffmpeg can't deadlock
    // against a full pipe while we read stdout.
    let mut stderr = child.stderr.take().expect("stderr piped");
    let drain = std::thread::spawn(move || {
        let mut buf = String::new();
        let _ = stderr.read_to_string(&mut buf);
        buf
    });

    let mut stdout = child.stdout.take().expect("stdout piped");
    let mut pcm: Vec<u8> = Vec::new();
    let mut chunk = vec![0u8; 256 * 1024];
    let mut last_pct = i32::MIN;
    loop {
        if cancel.map(|c| c.load(Ordering::SeqCst)).unwrap_or(false) {
            let _ = child.kill();
            let _ = child.wait();
            let _ = drain.join();
            return Ok(None);
        }
        let n = stdout.read(&mut chunk).context("reading ffmpeg output")?;
        if n == 0 {
            break;
        }
        pcm.extend_from_slice(&chunk[..n]);
        let pct = match total_secs {
            Some(total) => {
                let done_secs = (pcm.len() / 4) as f32 / WHISPER_SAMPLE_RATE as f32;
                ((done_secs / total * 100.0) as i32).min(100)
            }
            None => -1,
        };
        if pct != last_pct {
            last_pct = pct;
            on_progress(pct);
        }
    }
    let status = child.wait().context("waiting for ffmpeg")?;
    let stderr_text = drain.join().unwrap_or_default();

    if !status.success() {
        bail!("ffmpeg failed: {}", stderr_text.trim());
    }
    if pcm.is_empty() {
        bail!("ffmpeg produced no audio (does the file have an audio track?)");
    }

    let samples: Vec<f32> = pcm
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    let duration_secs = samples.len() as f32 / WHISPER_SAMPLE_RATE as f32;
    Ok(Some(pad_short(DecodedAudio {
        samples,
        duration_secs,
    })))
}

/// Load any audio or video file as 16 kHz mono f32. Video (and anything
/// symphonia can't decode) goes through ffmpeg.
pub fn load_media(path: &Path) -> Result<DecodedAudio> {
    load_media_with(path, None, |_| {})
        .map(|opt| opt.expect("load without a cancel flag cannot be cancelled"))
}

/// `load_media` with a cancel flag and an extraction-progress callback
/// (percent 0–100, or -1 while the total duration is unknown). Returns
/// None if cancelled.
pub fn load_media_with(
    path: &Path,
    cancel: Option<&std::sync::atomic::AtomicBool>,
    mut on_progress: impl FnMut(i32),
) -> Result<Option<DecodedAudio>> {
    let on_progress: &mut dyn FnMut(i32) = &mut on_progress;
    if is_video_file(path) {
        if ffmpeg_available() {
            return ffmpeg_extract(path, cancel, on_progress);
        }
        // No ffmpeg: symphonia can still demux AAC/ALAC out of mp4/mov
        return decode_audio(path, cancel).context(
            "could not extract audio from video (install ffmpeg for full container support: brew install ffmpeg)",
        );
    }
    match decode_audio(path, cancel) {
        Ok(audio) => Ok(audio),
        Err(err) if ffmpeg_available() => ffmpeg_extract(path, cancel, on_progress)
            .map_err(|fferr| anyhow!("{err:#}; ffmpeg fallback: {fferr:#}")),
        Err(err) => Err(err),
    }
}

/// Write 16 kHz mono f32 samples as a PCM16 WAV — the hand-off format for
/// engines that take an audio file rather than in-process samples.
pub fn write_wav_16k_mono(path: &Path, samples: &[f32]) -> Result<()> {
    use std::io::Write;
    let mut f = std::io::BufWriter::new(File::create(path)?);
    let data_len = (samples.len() * 2) as u32;
    f.write_all(b"RIFF")?;
    f.write_all(&(36 + data_len).to_le_bytes())?;
    f.write_all(b"WAVEfmt ")?;
    f.write_all(&16u32.to_le_bytes())?;
    f.write_all(&1u16.to_le_bytes())?; // PCM
    f.write_all(&1u16.to_le_bytes())?; // mono
    f.write_all(&WHISPER_SAMPLE_RATE.to_le_bytes())?;
    f.write_all(&(WHISPER_SAMPLE_RATE * 2).to_le_bytes())?; // byte rate
    f.write_all(&2u16.to_le_bytes())?; // block align
    f.write_all(&16u16.to_le_bytes())?; // bits per sample
    f.write_all(b"data")?;
    f.write_all(&data_len.to_le_bytes())?;
    for s in samples {
        let v = (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        f.write_all(&v.to_le_bytes())?;
    }
    f.flush()?;
    Ok(())
}

fn pad_short(mut audio: DecodedAudio) -> DecodedAudio {
    // whisper.cpp misbehaves on clips shorter than ~1s; pad with silence
    if audio.samples.len() < WHISPER_SAMPLE_RATE as usize {
        audio
            .samples
            .resize(WHISPER_SAMPLE_RATE as usize + 1600, 0.0);
    }
    audio
}

/// Decode any supported container/codec to 16 kHz mono f32.
pub fn load_audio(path: &Path) -> Result<DecodedAudio> {
    decode_audio(path, None)
        .map(|opt| opt.expect("decode without a cancel flag cannot be cancelled"))
}

/// Symphonia decode with an optional cancel flag, checked per packet.
/// Returns None if cancelled.
fn decode_audio(
    path: &Path,
    cancel: Option<&std::sync::atomic::AtomicBool>,
) -> Result<Option<DecodedAudio>> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .context("unrecognized audio format")?;
    let mut format = probed.format;

    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or_else(|| anyhow!("no decodable audio track"))?;
    let track_id = track.id;
    let sample_rate = track
        .codec_params
        .sample_rate
        .ok_or_else(|| anyhow!("unknown sample rate"))?;

    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .context("unsupported codec")?;

    let mut mono: Vec<f32> = Vec::new();
    let mut sample_buf: Option<SampleBuffer<f32>> = None;

    loop {
        if cancel
            .map(|c| c.load(std::sync::atomic::Ordering::SeqCst))
            .unwrap_or(false)
        {
            return Ok(None);
        }
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(SymphoniaError::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                break
            }
            Err(SymphoniaError::ResetRequired) => break,
            Err(e) => return Err(e).context("reading audio packet"),
        };
        if packet.track_id() != track_id {
            continue;
        }
        let decoded = match decoder.decode(&packet) {
            Ok(d) => d,
            // Skip over malformed frames rather than failing the whole file
            Err(SymphoniaError::DecodeError(_)) => continue,
            Err(e) => return Err(e).context("decoding audio"),
        };

        let spec = *decoded.spec();
        let channels = spec.channels.count().max(1);
        let needs_realloc = sample_buf
            .as_ref()
            .map(|b| b.capacity() < decoded.capacity() * channels)
            .unwrap_or(true);
        if needs_realloc {
            sample_buf = Some(SampleBuffer::new(decoded.capacity() as u64, spec));
        }
        let buf = sample_buf.as_mut().unwrap();
        buf.copy_interleaved_ref(decoded);

        for frame in buf.samples().chunks_exact(channels) {
            mono.push(frame.iter().sum::<f32>() / channels as f32);
        }
    }

    if mono.is_empty() {
        bail!("no audio samples decoded");
    }

    let duration_secs = mono.len() as f32 / sample_rate as f32;
    let samples = resample_to_16k(mono, sample_rate)?;

    Ok(Some(pad_short(DecodedAudio {
        samples,
        duration_secs,
    })))
}

fn resample_to_16k(input: Vec<f32>, from_rate: u32) -> Result<Vec<f32>> {
    if from_rate == WHISPER_SAMPLE_RATE {
        return Ok(input);
    }

    let ratio = WHISPER_SAMPLE_RATE as f64 / from_rate as f64;
    let params = SincInterpolationParameters {
        sinc_len: 128,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 128,
        window: WindowFunction::BlackmanHarris2,
    };
    const CHUNK: usize = 1024;
    let mut resampler =
        SincFixedIn::<f32>::new(ratio, 2.0, params, CHUNK, 1).context("creating resampler")?;

    let mut out: Vec<f32> = Vec::with_capacity((input.len() as f64 * ratio) as usize + CHUNK);
    let mut pos = 0;
    while pos + CHUNK <= input.len() {
        let result = resampler
            .process(&[&input[pos..pos + CHUNK]], None)
            .context("resampling")?;
        out.extend_from_slice(&result[0]);
        pos += CHUNK;
    }
    if pos < input.len() {
        let result = resampler
            .process_partial(Some(&[&input[pos..]]), None)
            .context("resampling tail")?;
        out.extend_from_slice(&result[0]);
    }
    // Flush the resampler's internal delay line
    let result = resampler
        .process_partial::<&[f32]>(None, None)
        .context("flushing resampler")?;
    out.extend_from_slice(&result[0]);

    // rubato pads the last partial chunk to a full chunk, appending
    // silence beyond the true signal length — trim to the exact ratio
    let expected = (input.len() as f64 * ratio).round() as usize;
    out.truncate(expected);

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static UNIQUE: AtomicUsize = AtomicUsize::new(0);

    /// Minimal PCM16 WAV writer for test fixtures.
    fn write_wav(path: &Path, sample_rate: u32, channels: u16, samples: &[i16]) {
        let data_len = (samples.len() * 2) as u32;
        let byte_rate = sample_rate * channels as u32 * 2;
        let mut f = std::fs::File::create(path).unwrap();
        f.write_all(b"RIFF").unwrap();
        f.write_all(&(36 + data_len).to_le_bytes()).unwrap();
        f.write_all(b"WAVEfmt ").unwrap();
        f.write_all(&16u32.to_le_bytes()).unwrap();
        f.write_all(&1u16.to_le_bytes()).unwrap(); // PCM
        f.write_all(&channels.to_le_bytes()).unwrap();
        f.write_all(&sample_rate.to_le_bytes()).unwrap();
        f.write_all(&byte_rate.to_le_bytes()).unwrap();
        f.write_all(&(channels * 2).to_le_bytes()).unwrap();
        f.write_all(&16u16.to_le_bytes()).unwrap();
        f.write_all(b"data").unwrap();
        f.write_all(&data_len.to_le_bytes()).unwrap();
        for s in samples {
            f.write_all(&s.to_le_bytes()).unwrap();
        }
    }

    fn temp_wav(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "transcribe-stt-audio-{}-{}-{name}",
            std::process::id(),
            UNIQUE.fetch_add(1, Ordering::SeqCst)
        ))
    }

    fn sine(rate: u32, freq: f32, secs: f32, amp: f32) -> Vec<i16> {
        (0..(rate as f32 * secs) as usize)
            .map(|i| {
                let t = i as f32 / rate as f32;
                (amp * (2.0 * std::f32::consts::PI * freq * t).sin() * i16::MAX as f32) as i16
            })
            .collect()
    }

    #[test]
    fn extension_detection() {
        assert!(is_audio_file(Path::new("A.WAV"))); // case-insensitive
        assert!(is_audio_file(Path::new("x.m4a")));
        assert!(is_audio_file(Path::new("x.mka")));
        assert!(is_audio_file(Path::new("x.amr")));
        assert!(is_video_file(Path::new("x.mp4")));
        assert!(is_video_file(Path::new("x.MOV")));
        assert!(is_video_file(Path::new("x.mpg")));
        assert!(is_video_file(Path::new("x.m2ts")));
        assert!(is_video_file(Path::new("x.vob")));
        assert!(is_video_file(Path::new("x.rmvb")));
        assert!(!is_audio_file(Path::new("x.mp4"))); // video, not audio
        assert!(!is_media_file(Path::new("x.txt")));
        assert!(!is_media_file(Path::new("noext")));
    }

    #[test]
    fn resample_passthrough_at_16k() {
        let input = vec![0.5f32; 16_000];
        let out = resample_to_16k(input.clone(), 16_000).unwrap();
        assert_eq!(out, input);
    }

    #[test]
    fn resample_produces_expected_length() {
        // 2s at 44100 -> ~32000 samples at 16k (sinc delay costs a few)
        let input: Vec<f32> = sine(44_100, 440.0, 2.0, 0.5)
            .iter()
            .map(|&s| s as f32 / i16::MAX as f32)
            .collect();
        let out = resample_to_16k(input, 44_100).unwrap();
        let expected = 32_000.0;
        assert!(
            (out.len() as f32 - expected).abs() / expected < 0.02,
            "got {} samples, expected ~{expected}",
            out.len()
        );
        // Signal energy must survive resampling
        let rms = (out.iter().map(|s| s * s).sum::<f32>() / out.len() as f32).sqrt();
        assert!(rms > 0.2, "rms {rms} too low — signal lost");
    }

    #[test]
    fn short_clips_are_padded_to_one_second() {
        let decoded = pad_short(DecodedAudio {
            samples: vec![0.1; 100],
            duration_secs: 0.006,
        });
        assert!(decoded.samples.len() > WHISPER_SAMPLE_RATE as usize);
    }

    #[test]
    fn decodes_wav_and_downmixes_stereo() {
        let path = temp_wav("stereo.wav");
        // 0.5s stereo at 16k: identical L/R, so downmix must preserve amplitude
        let mono = sine(16_000, 440.0, 0.5, 0.5);
        let stereo: Vec<i16> = mono.iter().flat_map(|&s| [s, s]).collect();
        write_wav(&path, 16_000, 2, &stereo);

        let decoded = load_audio(&path).unwrap();
        assert!((decoded.duration_secs - 0.5).abs() < 0.05);
        let rms = (decoded.samples[..8000].iter().map(|s| s * s).sum::<f32>() / 8000.0).sqrt();
        assert!((rms - 0.35).abs() < 0.05, "rms {rms}, expected ~0.354");
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn decodes_and_resamples_8k_wav() {
        let path = temp_wav("8k.wav");
        write_wav(&path, 8_000, 1, &sine(8_000, 200.0, 1.5, 0.5));

        let decoded = load_audio(&path).unwrap();
        assert!((decoded.duration_secs - 1.5).abs() < 0.05);
        // 1.5s should resample to ~24000 samples at 16k
        assert!((decoded.samples.len() as f32 - 24_000.0).abs() < 500.0);
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn wav_writer_round_trips_through_the_decoder() {
        let path = temp_wav("written.wav");
        let samples: Vec<f32> = sine(16_000, 440.0, 1.5, 0.5)
            .iter()
            .map(|&s| s as f32 / i16::MAX as f32)
            .collect();
        write_wav_16k_mono(&path, &samples).unwrap();

        let decoded = load_audio(&path).unwrap();
        assert_eq!(decoded.samples.len(), samples.len());
        assert!((decoded.duration_secs - 1.5).abs() < 0.01);
        let rms = |xs: &[f32]| (xs.iter().map(|s| s * s).sum::<f32>() / xs.len() as f32).sqrt();
        assert!((rms(&decoded.samples) - rms(&samples)).abs() < 0.01);
        std::fs::remove_file(&path).unwrap();
    }

    /// Render a short mp4 (sine audio + black video) with ffmpeg.
    fn generate_test_mp4(path: &Path, secs: f32) -> bool {
        std::process::Command::new("ffmpeg")
            .args(["-v", "error", "-y", "-f", "lavfi", "-i"])
            .arg(format!("sine=frequency=440:duration={secs}"))
            .args(["-f", "lavfi", "-i"])
            .arg(format!("color=c=black:s=64x64:d={secs}"))
            .args(["-c:v", "mpeg4", "-c:a", "aac", "-shortest"])
            .arg(path)
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    }

    #[test]
    fn extracts_video_audio_with_progress() {
        if !ffmpeg_available() {
            eprintln!("skipping: ffmpeg not installed");
            return;
        }
        let path = temp_wav("gen.mp4");
        assert!(generate_test_mp4(&path, 2.0), "ffmpeg fixture failed");

        let mut reports: Vec<i32> = Vec::new();
        let decoded = load_media_with(&path, None, |p| reports.push(p))
            .unwrap()
            .expect("not cancelled");
        assert!((decoded.duration_secs - 2.0).abs() < 0.3);
        // sine must survive the rip: check signal energy
        let rms = (decoded.samples.iter().map(|s| s * s).sum::<f32>()
            / decoded.samples.len() as f32)
            .sqrt();
        // ffmpeg's sine source generates at ~1/8 full scale (rms ~0.09)
        assert!(rms > 0.05, "rms {rms} too low — audio track lost");
        // progress is known (ffprobe), monotonic, and reaches the end
        assert!(!reports.is_empty());
        assert!(reports.windows(2).all(|w| w[0] <= w[1]));
        assert!(*reports.last().unwrap() >= 90);
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn extraction_honours_cancel() {
        if !ffmpeg_available() {
            eprintln!("skipping: ffmpeg not installed");
            return;
        }
        let path = temp_wav("cancel.mp4");
        assert!(generate_test_mp4(&path, 2.0), "ffmpeg fixture failed");

        let cancel = std::sync::atomic::AtomicBool::new(true);
        let result = load_media_with(&path, Some(&cancel), |_| {}).unwrap();
        assert!(result.is_none(), "pre-cancelled load must return None");
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn rejects_non_audio_bytes() {
        let path = temp_wav("garbage.wav");
        std::fs::write(&path, b"this is not audio at all").unwrap();
        assert!(load_audio(&path).is_err());
        std::fs::remove_file(&path).unwrap();
    }
}
