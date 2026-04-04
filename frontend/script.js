const video = document.getElementById("video");
const startBtn = document.getElementById("startBtn");
const captureBtn = document.getElementById("captureBtn");

const gestureText = document.getElementById("gesture");
const confidenceText = document.getElementById("confidence");
const explanationText = document.getElementById("explanation");
const statusText = document.getElementById("status");
const placeholder = document.getElementById("placeholder");


// Start Camera
startBtn.onclick = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;

    placeholder.style.display = "none";
    captureBtn.disabled = false;
};

// Capture & Predict
captureBtn.onclick = async () => {
    statusText.innerText = "Processing...";

    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    ctx.drawImage(video, 0, 0);

    canvas.toBlob(async (blob) => {
        const formData = new FormData();
        formData.append("file", blob, "frame.jpg");

        try {
            const res = await fetch("http://127.0.0.1:8000/predict", {
                method: "POST",
                body: formData
            });

            const data = await res.json();

            gestureText.innerText = data.gesture;
            confidenceText.innerText = data.confidence;
            explanationText.innerText = data.explanation;

            statusText.innerText = "Done ✅";
        } catch (err) {
            statusText.innerText = "Error ❌";
        }
    }, "image/jpeg");
};