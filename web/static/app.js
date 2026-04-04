// Page flow: pick files → POST /api/analyze → fill the panels from the JSON → PDF link uses session_id.
(() => {
  const el = (id) => document.getElementById(id);
  const ui = {
    image: el("imageInput"),
    audio: el("audioInput"),
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
    errBox: el("errorsBox"),
    errUl: el("errorsList"),
    json: el("rawJson"),
  };

  const verdictPretty = {
    highly_aligned: "Highly aligned",
    partially_related: "Partially related",
    content_mismatch: "Content mismatch",
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

  const row = (parent, label, value) => {
    if (value === undefined || value === null || value === "") return;
    const d = document.createElement("div");
    d.className = "metric-row";
    d.innerHTML = `<span>${label}</span><strong>${String(value)}</strong>`;
    parent.appendChild(d);
  };

  ui.image.addEventListener("change", () =>
    showChosen(ui.imageLabel, ui.image.files[0], "None")
  );
  ui.audio.addEventListener("change", () => {
    const f = ui.audio.files[0];
    if (f && !mp3Ok(f)) {
      ui.audio.value = "";
      showChosen(ui.audioLabel, null, "None");
      alert("Audio must be MP3 (.mp3).");
      return;
    }
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
      if (f.type.startsWith("image/") && !ui.image.files.length) {
        const dt = new DataTransfer();
        dt.items.add(f);
        ui.image.files = dt.files;
        showChosen(ui.imageLabel, f, "None");
      } else if (mp3Ok(f) && !ui.audio.files.length) {
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
    const img = ui.image.files[0];
    const aud = ui.audio.files[0];
    if (!img || !aud) {
      alert("Select both a blackboard image and an MP3 file.");
      return;
    }
    if (!mp3Ok(aud)) {
      alert("Audio must be MP3 (.mp3).");
      return;
    }

    ui.run.disabled = true;
    ui.out.classList.add("hidden");
    setBusy(true, "Running… First run may download models and take several minutes.");

    const fd = new FormData();
    fd.append("image", img);
    fd.append("audio", aud);

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
