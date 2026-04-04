(function () {
  const $ = (id) => document.getElementById(id);

  const imageInput = $("imageInput");
  const audioInput = $("audioInput");
  const configInput = $("configInput");
  const imageName = $("imageName");
  const audioName = $("audioName");
  const configName = $("configName");
  const runBtn = $("runBtn");
  const dropZone = $("dropZone");
  const statusCard = $("statusCard");
  const spinner = $("spinner");
  const statusText = $("statusText");
  const resultsCard = $("resultsCard");
  const downloadPdf = $("downloadPdf");
  const boardList = $("boardList");
  const boardEmpty = $("boardEmpty");
  const clarityBox = $("clarityBox");
  const speechText = $("speechText");
  const alignBox = $("alignBox");
  const errorsBox = $("errorsBox");
  const errorsList = $("errorsList");
  const rawJson = $("rawJson");

  const VERDICT_LABELS = {
    highly_aligned: "Highly aligned",
    partially_related: "Partially related",
    content_mismatch: "Content mismatch",
  };

  const CLARITY_LABELS = {
    clear: "Clear",
    fair: "Fair",
    poor: "Poor",
  };

  function setFileName(el, file, fallback) {
    el.textContent = file ? file.name : fallback;
  }

  function isMp3File(file) {
    if (!file || !file.name) return false;
    if (!/\.mp3$/i.test(file.name)) return false;
    const t = (file.type || "").toLowerCase();
    if (t === "") return true;
    return (
      t === "audio/mpeg" ||
      t === "audio/mp3" ||
      t.includes("mpeg") ||
      t.includes("mp3")
    );
  }

  imageInput.addEventListener("change", () => setFileName(imageName, imageInput.files[0], "None"));
  audioInput.addEventListener("change", () => {
    const f = audioInput.files[0];
    if (f && !isMp3File(f)) {
      audioInput.value = "";
      setFileName(audioName, null, "None");
      alert("Audio must be MP3 (.mp3).");
      return;
    }
    setFileName(audioName, f, "None");
  });
  configInput.addEventListener("change", () =>
    setFileName(configName, configInput.files[0], "Default")
  );

  ["dragenter", "dragover"].forEach((ev) => {
    dropZone.addEventListener(ev, (e) => {
      e.preventDefault();
      dropZone.classList.add("dragover");
    });
  });
  ["dragleave", "drop"].forEach((ev) => {
    dropZone.addEventListener(ev, (e) => {
      e.preventDefault();
      dropZone.classList.remove("dragover");
    });
  });

  dropZone.addEventListener("drop", (e) => {
    const files = Array.from(e.dataTransfer.files || []);
    for (const f of files) {
      if (f.type.startsWith("image/") && !imageInput.files.length) {
        const dt = new DataTransfer();
        dt.items.add(f);
        imageInput.files = dt.files;
        setFileName(imageName, f, "None");
      } else if (isMp3File(f)) {
        if (!audioInput.files.length) {
          const dt = new DataTransfer();
          dt.items.add(f);
          audioInput.files = dt.files;
          setFileName(audioName, f, "None");
        }
      }
    }
  });

  function showStatus(loading, text) {
    statusCard.classList.remove("hidden");
    spinner.classList.toggle("hidden", !loading);
    statusText.textContent = text;
  }

  function hideStatus() {
    statusCard.classList.add("hidden");
  }

  function formatClarity(v) {
    return CLARITY_LABELS[v] || v || "—";
  }

  function formatVerdict(v) {
    return VERDICT_LABELS[v] || v || "—";
  }

  function renderResult(data) {
    const r = data.result || {};
    resultsCard.classList.remove("hidden");

    const texts = r.board_texts || [];
    boardList.innerHTML = "";
    if (texts.length) {
      boardEmpty.classList.add("hidden");
      texts.forEach((t) => {
        const li = document.createElement("li");
        li.textContent = t;
        boardList.appendChild(li);
      });
    } else {
      boardEmpty.classList.remove("hidden");
    }

    const c = r.clarity || {};
    clarityBox.innerHTML = "";
    const clarityRows = [
      ["Level", formatClarity(c.clarity)],
      ["Score (0–100)", c.score],
      ["Suggestion", c.suggestion],
      ["Laplacian variance", c.laplacian_variance != null ? c.laplacian_variance.toFixed(2) : "—"],
      ["Stroke width variance", c.stroke_width_variance != null ? c.stroke_width_variance.toFixed(4) : "—"],
    ];
    clarityRows.forEach(([k, v]) => {
      if (v === undefined || v === null || v === "") return;
      const row = document.createElement("div");
      row.className = "metric-row";
      row.innerHTML = `<span>${k}</span><strong>${String(v)}</strong>`;
      clarityBox.appendChild(row);
    });

    speechText.textContent = r.speech_text || "(no transcription)";

    const a = r.alignment || {};
    alignBox.innerHTML = "";
    [
      ["Semantic similarity", a.semantic_similarity],
      ["Keyword overlap (Jaccard)", a.keyword_overlap_rate],
      ["Verdict", formatVerdict(a.verdict)],
    ].forEach(([k, v]) => {
      if (v === undefined || v === null) return;
      const row = document.createElement("div");
      row.className = "metric-row";
      row.innerHTML = `<span>${k}</span><strong>${String(v)}</strong>`;
      alignBox.appendChild(row);
    });

    const err = r.errors || {};
    const errKeys = Object.keys(err).filter((k) => err[k]);
    if (errKeys.length) {
      errorsBox.classList.remove("hidden");
      errorsList.innerHTML = "";
      errKeys.forEach((k) => {
        const li = document.createElement("li");
        li.textContent = `${k}: ${err[k]}`;
        errorsList.appendChild(li);
      });
    } else {
      errorsBox.classList.add("hidden");
    }

    rawJson.textContent = JSON.stringify(data, null, 2);

    const sid = data.session_id;
    if (sid && r.pdf_path) {
      downloadPdf.href = `/api/report/${sid}`;
      downloadPdf.classList.remove("hidden");
    } else {
      downloadPdf.href = "#";
      downloadPdf.classList.add("hidden");
    }
  }

  runBtn.addEventListener("click", async () => {
    const imgF = imageInput.files[0];
    const audF = audioInput.files[0];
    if (!imgF || !audF) {
      alert("Select both a blackboard image and an MP3 file.");
      return;
    }
    if (!isMp3File(audF)) {
      alert("Audio must be MP3 (.mp3).");
      return;
    }

    runBtn.disabled = true;
    resultsCard.classList.add("hidden");
    showStatus(true, "Running… First run may download models and take several minutes.");

    const fd = new FormData();
    fd.append("image", imgF);
    fd.append("audio", audF);
    const cfgF = configInput.files[0];
    if (cfgF) fd.append("config_yaml", cfgF);

    try {
      const res = await fetch("/api/analyze", {
        method: "POST",
        body: fd,
      });
      const text = await res.text();
      let json;
      try {
        json = JSON.parse(text);
      } catch {
        throw new Error(text.slice(0, 200) || `HTTP ${res.status}`);
      }
      if (!res.ok) {
        const detail = json.detail;
        const msg = typeof detail === "string" ? detail : JSON.stringify(detail);
        throw new Error(msg || `HTTP ${res.status}`);
      }
      hideStatus();
      renderResult(json);
    } catch (e) {
      showStatus(false, "");
      spinner.classList.add("hidden");
      statusText.textContent = "";
      alert("Analysis failed: " + (e.message || String(e)));
    } finally {
      runBtn.disabled = false;
    }
  });
})();
