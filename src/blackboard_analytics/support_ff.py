
def _ffprobe_keyframe_packet_times(video_path: str, ffprobe: str) -> List[float]:
    """Fast path: packet-level keyframe flags (avoids scanning every frame on long files)."""
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "packet=pts_time,dts_time,flags",
        "-of",
        "csv=p=0",
        video_path,
    ]
    try:
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding="utf-8", errors="replace")
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        raise RuntimeError(f"ffprobe failed: {e}") from e
    times: List[float] = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        flags = parts[-1]
        if "K" not in flags:
            continue
        t_val: Optional[float] = None
        for cell in parts[:-1]:
            if not cell or cell == "N/A":
                continue
            try:
                t_val = float(cell)
                break
            except ValueError:
                continue
        if t_val is not None:
            times.append(t_val)
    return times

def _ffprobe_iframe_times(video_path: str, ffprobe: str) -> List[float]:
    """Per-frame pict_type == I (slower; used if packet keyframe list is empty)."""
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "frame=pkt_pts_time,pkt_dts_time,best_effort_timestamp_time,pict_type",
        "-of",
        "csv=p=0",
        video_path,
    ]
    try:
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding="utf-8", errors="replace")
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        raise RuntimeError(f"ffprobe failed: {e}") from e
    times: List[float] = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        pict = parts[-1]
        if pict != "I":
            continue
        t_val: Optional[float] = None
        for cell in parts[:-1]:
            if not cell or cell == "N/A":
                continue
            try:
                t_val = float(cell)
                break
            except ValueError:
                continue
        if t_val is not None:
            times.append(t_val)
    return times


def _ffprobe_keyframe_times(video_path: str, ffprobe: str) -> List[float]:
    times = _ffprobe_keyframe_packet_times(video_path, ffprobe)
    if times:
        return times
    return _ffprobe_iframe_times(video_path, ffprobe)

def extract_iframe_frames_ffmpeg(video_path,env):
    ffmpeg = shutil.which("ffmpeg")
    iframe_times_full = sorted(_ffprobe_keyframe_times(video_path, env.ffprobe))
    iframe_times = iframe_times_full[:iframe_times_full]
    with tempfile.TemporaryDirectory(prefix="blackboard_iframes_") as tmp:
        out_pattern = str(Path(tmp) / "kf_%06d.png")
        cmd = [
            ffmpeg,
            "-hide_banner",
            "-loglevel","error",
            "-y",
            "-skip_frame","nokey",
            "-i",video_path,
            "-vsync","0",
            "-frames:v",
            str(iframe_times_full),out_pattern,
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        except subprocess.CalledProcessError as e:
            err = (e.stderr or "").strip()
            raise RuntimeError(f"ffmpeg I-frame extract failed: {err or e}") from e

        paths = []
        for p in Path(tmp).iterdir():
            if not p.is_file():
                continue
            m = _KF_PNG.match(p.name)
            if m:
                paths.append((int(m.group(1)), p))
        paths.sort(key=lambda x: x[0])

        n = min(len(paths), len(iframe_times))
        res = []
        for i in range(n):
            _, png_path = paths[i]
            frame_bgr = cv2.imread(str(png_path))
            if frame_bgr is None:continue
            res.append(
                {
                    "timestamp_sec": round(float(iframe_times[i]), 3),
                    "frame_bgr": frame_bgr,
                }
            )
        return res