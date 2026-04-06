// Page flow: pick files → POST /api/analyze → fill the panels from the JSON → PDF link uses session_id.
(() => {
  const el = (id) => document.getElementById(id);
  const ui = {
    video: el("videoInput"),
    image: el("imageInput"),
    audio: el("audioInput"),
    videoLabel: el("videoName"),
    imageLabel: el("imageName"),
    audioLabel: el("audioName"),
    run: el("runBtn"),
    zone: el("dropZone"),
    busy: el("statusCard"),
    spin: el("spinner"),
    busyText: el("statusText"),
    out: el("resultsCard"),
    pdf: el("downloadPdf"),
    boardUl: el("boardList"),
    boardEmpty: el("boardEmpty"),
    clarity: el("clarityBox"),
    speech: el("speechText"),
    align: el("alignBox"),
    deepseek: el("deepseekBox"),
    deepseekReason: el("deepseekReason"),
    deepseekEvidence: el("deepseekEvidence"),
    lineFilter: el("deepseekFilterBox"),
    lineFilterReason: el("deepseekFilterReason"),
    lineFilterDroppedLabel: el("deepseekDroppedLabel"),
    lineFilterDroppedUl: el("deepseekDroppedList"),
    errBox: el("errorsBox"),
    errUl: el("errorsList"),
    json: el("rawJson"),
  };

  const verdictPretty = {
    highly_aligned: "Highly aligned",
    partially_related: "Partially related",
    content_mismatch: "Content mismatch",
  };
  const deepseekPretty = {
    highly_relevant: "Highly relevant",
    partially_relevant: "Partially relevant",
    weakly_relevant: "Weakly relevant",
    off_topic: "Off topic",
  };
  const clarityPretty = { clear: "Clear", fair: "Fair", poor: "Poor" };

  const showChosen = (span, file, emptyLabel) => {
    span.textContent = file ? file.name : emptyLabel;
  };

  const mp3Ok = (f) => {
    if (!f?.name || !/\.mp3$/i.test(f.name)) return false;
    const m = (f.type || "").toLowerCase();
    return !m || m === "audio/mpeg" || m.includes("mp3") || m.includes("mpeg");
  };

  const videoOk = (f) => {
    if (!f?.name) return false;
    return /\.(mp4|mov|avi|mkv)$/i.test(f.name);
  };

  const clearImageAudio = () => {
    ui.image.value = "";
    ui.audio.value = "";
    showChosen(ui.imageLabel, null, "None");
    showChosen(ui.audioLabel, null, "None");
  };

  const clearVideo = () => {
    ui.video.value = "";
    showChosen(ui.videoLabel, null, "None");
  };

  const row = (parent, label, value) => {
    if (value === undefined || value === null || value === "") return;
    const d = document.createElement("div");
    d.className = "metric-row";
    d.innerHTML = `<span>${label}</span><strong>${String(value)}</strong>`;
    parent.appendChild(d);
  };

  const showLineFilter = (f) => {
    if (!ui.lineFilter) return;
    ui.lineFilter.replaceChildren();
    if (ui.lineFilterReason) {
      ui.lineFilterReason.classList.add("hidden");
      ui.lineFilterReason.textContent = "";
    }
    if (ui.lineFilterDroppedLabel) ui.lineFilterDroppedLabel.classList.add("hidden");
    if (ui.lineFilterDroppedUl) {
      ui.lineFilterDroppedUl.replaceChildren();
      ui.lineFilterDroppedUl.classList.add("hidden");
    }
    if (!f) {
      row(ui.lineFilter, "Status", "Not run (disabled or no API response object)");
      return;
    }
    if (!f.enabled) {
      row(ui.lineFilter, "Status", "Disabled in config");
      row(ui.lineFilter, "Model", f.model || "—");
      return;
    }
    row(ui.lineFilter, "Status", f.error ? "Used with warning / fallback" : "Applied");
    row(ui.lineFilter, "Model", f.model || "—");
    row(
      ui.lineFilter,
      "Lines kept / dropped",
      `${(f.kept_lines || []).length} / ${(f.dropped_lines || []).length}`
    );
    if (f.error) {
      row(ui.lineFilter, "Note", f.error);
    }
    if (f.reason && ui.lineFilterReason) {
      ui.lineFilterReason.textContent = f.reason;
      ui.lineFilterReason.classList.remove("hidden");
    }
    const dropped = Array.isArray(f.dropped_lines) ? f.dropped_lines.filter(Boolean) : [];
    if (dropped.length && ui.lineFilterDroppedUl && ui.lineFilterDroppedLabel) {
      ui.lineFilterDroppedLabel.classList.remove("hidden");
      ui.lineFilterDroppedUl.classList.remove("hidden");
      for (const t of dropped) {
        const li = document.createElement("li");
        li.textContent = t;
        ui.lineFilterDroppedUl.appendChild(li);
      }
    }
  };

  const showDeepSeek = (result) => {
    const d = result || {};
    ui.deepseek.replaceChildren();
    ui.deepseekEvidence.replaceChildren();
    ui.deepseekReason.classList.add("hidden");
    ui.deepseekEvidence.classList.add("hidden");

    if (!d.enabled) {
      row(ui.deepseek, "Status", "Not enabled");
      row(ui.deepseek, "Model", d.model || "deepseek-chat");
      return;
    }

    row(ui.deepseek, "Status", d.error ? "Failed" : "Success");
    row(ui.deepseek, "Model", d.model);

    if (d.error) {
      row(ui.deepseek, "Error", d.error);
      return;
    }

    row(ui.deepseek, "Overall relevance", deepseekPretty[d.overall_relevance] || d.overall_relevance);
    row(ui.deepseek, "Score", d.score);

    if (d.reason) {
      ui.deepseekReason.textContent = d.reason;
      ui.deepseekReason.classList.remove("hidden");
    }

    const evidence = Array.isArray(d.evidence) ? d.evidence.filter(Boolean) : [];
    if (evidence.length) {
      ui.deepseekEvidence.classList.remove("hidden");
      for (const item of evidence) {
        const li = document.createElement("li");
        li.textContent = item;
        ui.deepseekEvidence.appendChild(li);
      }
    }
  };

  ui.video.addEventListener("change", () => {
    const f = ui.video.files[0];
    if (f && !videoOk(f)) {
      clearVideo();
      alert("Video must be MP4, MOV, AVI, or MKV.");
      return;
    }
    if (f) clearImageAudio();
    showChosen(ui.videoLabel, f, "None");
  });
  ui.image.addEventListener("change", () => {
    const f = ui.image.files[0];
    if (f) clearVideo();
    showChosen(ui.imageLabel, f, "None");
  });
  ui.audio.addEventListener("change", () => {
    const f = ui.audio.files[0];
    if (f && !mp3Ok(f)) {
      ui.audio.value = "";
      showChosen(ui.audioLabel, null, "None");
      alert("Audio must be MP3 (.mp3).");
      return;
    }
    if (f) clearVideo();
    showChosen(ui.audioLabel, f, "None");
  });

  const noDrag = (e) => {
    e.preventDefault();
    ui.zone.classList.toggle("dragover", e.type === "dragenter" || e.type === "dragover");
  };
  ["dragenter", "dragover"].forEach((t) => ui.zone.addEventListener(t, noDrag));
  ["dragleave", "drop"].forEach((t) =>
    ui.zone.addEventListener(t, (e) => {
      e.preventDefault();
      ui.zone.classList.remove("dragover");
    })
  );

  ui.zone.addEventListener("drop", (e) => {
    for (const f of Array.from(e.dataTransfer.files || [])) {
      if (videoOk(f)) {
        const dt = new DataTransfer();
        dt.items.add(f);
        clearImageAudio();
        ui.video.files = dt.files;
        showChosen(ui.videoLabel, f, "None");
        break;
      } else if (f.type.startsWith("image/") && !ui.image.files.length && !ui.video.files.length) {
        const dt = new DataTransfer();
        dt.items.add(f);
        ui.image.files = dt.files;
        showChosen(ui.imageLabel, f, "None");
      } else if (mp3Ok(f) && !ui.audio.files.length && !ui.video.files.length) {
        const dt = new DataTransfer();
        dt.items.add(f);
        ui.audio.files = dt.files;
        showChosen(ui.audioLabel, f, "None");
      }
    }
  });

  const setBusy = (on, msg) => {
    ui.busy.classList.toggle("hidden", !on && !msg);
    ui.spin.classList.toggle("hidden", !on);
    if (msg !== undefined) ui.busyText.textContent = msg;
  };

  const showAnalysisOnPage = (payload) => {
    const r = payload.result || {};
    ui.out.classList.remove("hidden");

    // Board OCR lines
    const lines = r.board_texts || [];
    ui.boardUl.replaceChildren();
    ui.boardEmpty.classList.toggle("hidden", lines.length > 0);
    for (const t of lines) {
      const li = document.createElement("li");
      li.textContent = t;
      ui.boardUl.appendChild(li);
    }

    showLineFilter(r.deepseek_board_line_filter);

    // Legibility score
    const c = r.clarity || {};
    ui.clarity.replaceChildren();
    row(ui.clarity, "Level", clarityPretty[c.clarity] || c.clarity);
    row(ui.clarity, "Score (0–100)", c.score);
    row(ui.clarity, "Suggestion", c.suggestion);
    row(
      ui.clarity,
      "Laplacian variance",
      c.laplacian_variance != null ? c.laplacian_variance.toFixed(2) : "—"
    );
    row(
      ui.clarity,
      "Stroke width variance",
      c.stroke_width_variance != null ? c.stroke_width_variance.toFixed(4) : "—"
    );

    ui.speech.textContent = r.speech_text || "(no transcription)";

    // Does speech match the board?
    const a = r.alignment || {};
    ui.align.replaceChildren();
    row(ui.align, "Semantic similarity", a.semantic_similarity);
    row(ui.align, "Keyword overlap (Jaccard)", a.keyword_overlap_rate);
    row(ui.align, "Verdict", verdictPretty[a.verdict] || a.verdict);

    showDeepSeek(r.deepseek_alignment);

    // Pipeline step failures (we still show whatever succeeded)
    const bad = r.errors || {};
    const keys = Object.keys(bad).filter((k) => bad[k]);
    ui.errBox.classList.toggle("hidden", !keys.length);
    ui.errUl.replaceChildren();
    for (const k of keys) {
      const li = document.createElement("li");
      li.textContent = `${k}: ${bad[k]}`;
      ui.errUl.appendChild(li);
    }

    ui.json.textContent = JSON.stringify(payload, null, 2);

    const sid = payload.session_id;
    if (sid && r.pdf_path) {
      ui.pdf.href = `/api/report/${sid}`;
      ui.pdf.classList.remove("hidden");
    } else {
      ui.pdf.href = "#";
      ui.pdf.classList.add("hidden");
    }
  };

  ui.run.addEventListener("click", async () => {
    const vid = ui.video.files[0];
    const img = ui.image.files[0];
    const aud = ui.audio.files[0];
    if (vid) {
      if (!videoOk(vid)) {
        alert("Video must be MP4, MOV, AVI, or MKV.");
        return;
      }
    } else {
      if (!img || !aud) {
        alert("Select one video file, or select both a blackboard image and an MP3 file.");
        return;
      }
      if (!mp3Ok(aud)) {
        alert("Audio must be MP3 (.mp3).");
        return;
      }
    }

    ui.run.disabled = true;
    ui.out.classList.add("hidden");
    setBusy(true, "Running… First run may download models and take several minutes.");

    const fd = new FormData();
    if (vid) {
      fd.append("video", vid);
    } else {
      fd.append("image", img);
      fd.append("audio", aud);
    }

    try {
      const res = await fetch("/api/analyze", { method: "POST", body: fd });
      const text = await res.text();
      let data;
      try {
        data = JSON.parse(text);
      } catch {
        throw new Error(text.slice(0, 200) || `HTTP ${res.status}`);
      }
      if (!res.ok) {
        const d = data.detail;
        throw new Error(typeof d === "string" ? d : JSON.stringify(d) || `HTTP ${res.status}`);
      }
      ui.busy.classList.add("hidden");
      showAnalysisOnPage(data);
    } catch (e) {
      setBusy(false, "");
      ui.spin.classList.add("hidden");
      ui.busyText.textContent = "";
      alert("Analysis failed: " + (e.message || String(e)));
    } finally {
      ui.run.disabled = false;
    }
  });
})();
