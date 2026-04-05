const video = document.getElementById("video");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const captureBtn = document.getElementById("captureBtn");
const liveToggle = document.getElementById("liveToggle");
const themeToggle = document.getElementById("themeToggle");
const themeLabel = document.getElementById("themeLabel");

const gestureText = document.getElementById("gesture");
const textOutputText = document.getElementById("textOutput");
const confidenceText = document.getElementById("confidence");
const explanationText = document.getElementById("explanation");
const statusText = document.getElementById("status");
const placeholder = document.getElementById("placeholder");
const handInfo = document.getElementById("handInfo");
const faceInfo = document.getElementById("faceInfo");
const poseInfo = document.getElementById("poseInfo");
const historyList = document.getElementById("historyList");

const API_BASE = "";
const THEME_KEY = "signbridge-theme";
let stream = null;
let liveTimer = null;
let isPredicting = false;
let previewMonitor = null;
const predictionHistory = [];
statusText.innerText = "Camera not started";
applyTheme(localStorage.getItem(THEME_KEY) || "dark");

window.addEventListener("submit", (event) => {
    event.preventDefault();
});

startBtn.addEventListener("mousedown", async (event) => {
    event.preventDefault();
    event.stopPropagation();

    await startCamera(false);
});

stopBtn.addEventListener("mousedown", (event) => {
    event.preventDefault();
    event.stopPropagation();

    stopLiveMode();
    stopPreviewMonitor();

    if (stream) {
        stream.getTracks().forEach((track) => track.stop());
        stream = null;
    }

    video.srcObject = null;
    placeholder.style.display = "block";
    setControlEnabled(captureBtn, false);
    setControlEnabled(stopBtn, false);
    setControlEnabled(startBtn, true);
    statusText.innerText = "Camera stopped.";
});

liveToggle.addEventListener("change", (event) => {
    event.stopPropagation();

    if (!stream) {
        statusText.innerText = "Start the camera before enabling live mode.";
        liveToggle.checked = false;
        return;
    }

    if (liveToggle.checked) {
        startLiveMode();
    } else {
        stopLiveMode();
        statusText.innerText = "Live mode paused. Manual prediction is still available.";
    }
});

themeToggle.addEventListener("change", () => {
    const theme = themeToggle.checked ? "light" : "dark";
    applyTheme(theme);
    localStorage.setItem(THEME_KEY, theme);
});

captureBtn.addEventListener("mousedown", async (event) => {
    event.preventDefault();
    event.stopPropagation();

    if (liveToggle.checked) {
        stopLiveMode();
        liveToggle.checked = false;
        statusText.innerText = "Live mode paused for manual prediction.";
    }

    await predictFrame();
});

function startLiveMode() {
    stopLiveMode();
    statusText.innerText = "Live mode enabled. Predicting every 2 seconds.";
    liveTimer = window.setInterval(() => {
        predictFrame();
    }, 2000);
}

function stopLiveMode() {
    if (liveTimer) {
        window.clearInterval(liveTimer);
        liveTimer = null;
    }
}

async function predictFrame() {
    if (!stream || isPredicting || !video.videoWidth || !video.videoHeight) {
        return;
    }

    isPredicting = true;
    statusText.innerText = "Processing frame...";
    await ensureVideoPreview();

    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0);

    canvas.toBlob(async (blob) => {
        const formData = new FormData();
        formData.append("file", blob, "frame.jpg");

        try {
            const res = await fetch(`${API_BASE}/predict`, {
                method: "POST",
                body: formData
            });

            if (!res.ok) {
                throw new Error("Prediction request failed");
            }

            const data = await res.json();

            gestureText.innerText = data.gesture;
            textOutputText.innerText = data.text;
            confidenceText.innerText = `${Math.round((data.confidence || 0) * 100)}%`;
            explanationText.innerText = data.explanation;

            renderModalities(data.modalities);
            addHistoryEntry(data);
            statusText.innerText = "Prediction updated.";
        } catch (error) {
            statusText.innerText = "Backend error. Make sure FastAPI is running on port 8000.";
        } finally {
            await ensureVideoPreview();
            isPredicting = false;
        }
    }, "image/jpeg");
}

video.addEventListener("pause", async () => {
    await ensureVideoPreview();
});

video.addEventListener("ended", async () => {
    await ensureVideoPreview();
});

async function ensureVideoPreview() {
    if (!stream) {
        return;
    }

    const track = stream.getVideoTracks()[0];
    if (!track || track.readyState !== "live") {
        statusText.innerText = "Camera track stopped. Restart the camera.";
        return;
    }

    if (video.srcObject !== stream) {
        video.srcObject = stream;
    }

    if (video.paused || video.ended || video.readyState < 2) {
        try {
            await video.play();
        } catch (error) {
            statusText.innerText = "Camera preview paused unexpectedly.";
        }
    }
}

function startPreviewMonitor() {
    stopPreviewMonitor();
    previewMonitor = window.setInterval(() => {
        ensureVideoPreview();
    }, 800);
}

function stopPreviewMonitor() {
    if (previewMonitor) {
        window.clearInterval(previewMonitor);
        previewMonitor = null;
    }
}

function renderModalities(modalities = {}) {
    const hand = modalities.hand || {};
    const face = modalities.face || {};
    const pose = modalities.pose || {};

    const raisedFingers = Object.entries(hand.finger_states || {})
        .filter(([, isUp]) => isUp)
        .map(([finger]) => finger)
        .join(", ");

    handInfo.innerText = hand.detected
        ? `Raised fingers: ${raisedFingers || "none"} | Open palm: ${formatBool(hand.open_palm)} | Pinch: ${formatBool(hand.pinch)}`
        : "Hand not detected clearly.";

    faceInfo.innerText = face.detected
        ? `Face detected | Centered: ${formatBool(face.centered)} | Score: ${face.score ?? "--"}`
        : "Face cues unavailable for this frame.";

    poseInfo.innerText = pose.detected
        ? `Shoulders visible: ${formatBool(pose.shoulders_visible)} | Upright: ${formatBool(pose.upright)} | Raised arm: ${formatBool(pose.hand_above_shoulder)}`
        : "Body posture cues unavailable for this frame.";
}

function addHistoryEntry(data) {
    const item = `${data.text} (${Math.round((data.confidence || 0) * 100)}%)`;
    predictionHistory.unshift(item);

    if (predictionHistory.length > 5) {
        predictionHistory.pop();
    }

    historyList.innerHTML = "";
    predictionHistory.forEach((entry) => {
        const li = document.createElement("li");
        li.innerText = entry;
        historyList.appendChild(li);
    });
}

function formatBool(value) {
    return value ? "Yes" : "No";
}

function setControlEnabled(element, enabled) {
    element.classList.toggle("is-disabled", !enabled);
    element.setAttribute("aria-disabled", enabled ? "false" : "true");
    element.tabIndex = enabled ? 0 : -1;
}

async function startCamera(isRestore) {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
        video.muted = true;
        await ensureVideoPreview();
        startPreviewMonitor();

        placeholder.style.display = "none";
        setControlEnabled(captureBtn, true);
        setControlEnabled(stopBtn, true);
        setControlEnabled(startBtn, false);
        statusText.innerText = "Camera live. Ready for prediction.";

        if (liveToggle.checked) {
            startLiveMode();
        }
    } catch (error) {
        statusText.innerText = "Camera access was denied or unavailable.";
    }
}

function applyTheme(theme) {
    const isLight = theme === "light";
    document.body.classList.toggle("light-mode", isLight);
    themeToggle.checked = isLight;
    themeLabel.innerText = isLight ? "Dark Mode" : "Light Mode";
}
