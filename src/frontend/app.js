const videoSelect = document.getElementById("videoSelect");
const videoPlayer = document.getElementById("videoPlayer");
const questionForm = document.getElementById("questionForm");
const questionInput = document.getElementById("questionInput");
const askButton = document.getElementById("askButton");
const messages = document.getElementById("messages");
const evidence = document.getElementById("evidence");
const uploadForm = document.getElementById("uploadForm");
const videoFile = document.getElementById("videoFile");
const uploadButton = document.getElementById("uploadButton");
const uploadStatus = document.getElementById("uploadStatus");

let currentVideoId = null;

function addMessage(text, type) {
    const element = document.createElement("div");
    element.className = `message ${type}-message`;
    element.textContent = text;
    messages.appendChild(element);
    messages.scrollTop = messages.scrollHeight;
}

function formatTime(seconds) {
    const minutes = Math.floor(seconds / 60);
    const remaining = Math.floor(seconds % 60);

    return `${minutes}:${String(remaining).padStart(2, "0")}`;
}

async function loadVideos(preferredVideoId = null) {
    try {
        const response = await fetch("/api/videos");

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const data = await response.json();
        const previousVideoId = currentVideoId;

        videoSelect.innerHTML = "";

        for (const video of data.videos) {
            const option = document.createElement("option");
            option.value = video.video_id;
            option.textContent =
                video.source_filename || video.video_id;

            videoSelect.appendChild(option);
        }

        if (data.videos.length === 0) {
            videoSelect.innerHTML =
                '<option value="">No indexed videos</option>';

            currentVideoId = null;
            videoPlayer.removeAttribute("src");
            videoPlayer.load();
            return;
        }

        const availableIds = new Set(
            data.videos.map(video => video.video_id)
        );

        const selectedId =
            (preferredVideoId &&
                availableIds.has(preferredVideoId) &&
                preferredVideoId) ||
            (previousVideoId &&
                availableIds.has(previousVideoId) &&
                previousVideoId) ||
            data.videos[0].video_id;

        if (selectedId !== currentVideoId) {
            selectVideo(selectedId);
        } else {
            videoSelect.value = selectedId;
        }

    } catch (error) {
        addMessage(
            `Cannot load videos: ${error.message}`,
            "ai"
        );
    }
}

function selectVideo(videoId) {
    currentVideoId = videoId;
    videoSelect.value = videoId;

    videoPlayer.src =
        `/api/videos/${encodeURIComponent(videoId)}/stream`;

    videoPlayer.load();

    evidence.innerHTML = "";
    messages.innerHTML =
        '<p class="placeholder">Ask a question about this video.</p>';
}

function showEvidence(frames) {
    evidence.innerHTML = "";

    for (const frame of frames) {
        const button = document.createElement("button");
        button.className = "evidence-item";
        button.type = "button";

        const image = document.createElement("img");
        image.src =
            `/api/videos/${encodeURIComponent(currentVideoId)}` +
            `/frames/${frame.frame_id}`;
        image.alt = `Frame at ${formatTime(frame.timestamp)}`;
        image.loading = "lazy";

        const timestamp = document.createElement("span");
        timestamp.textContent = formatTime(frame.timestamp);

        button.append(image, timestamp);

        button.addEventListener("click", () => {
            videoPlayer.currentTime = frame.timestamp;
            videoPlayer.play().catch(() => {});
        });

        evidence.appendChild(button);
    }
}

videoSelect.addEventListener("change", (event) => {
    selectVideo(event.target.value);
});

questionForm.addEventListener("submit", async (event) => {
    event.preventDefault();

    if (!currentVideoId) return;

    const question = questionInput.value.trim();
    if (!question) return;

    addMessage(question, "user");
    questionInput.value = "";

    askButton.disabled = true;
    askButton.textContent = "Thinking...";

    try {
        const response = await fetch(
            `/api/videos/${encodeURIComponent(currentVideoId)}/query`,
            {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    question,
                    top_k: 2,
                    context_before: 2,
                    context_after: 2
                })
            }
        );

        const result = await response.json();

        if (!response.ok) {
            throw new Error(
                typeof result.detail === "string"
                    ? result.detail
                    : `HTTP ${response.status}`
            );
        }

        addMessage(result.answer, "ai");
        showEvidence(result.frames);

    } catch (error) {
        addMessage(`Error: ${error.message}`, "ai");
    } finally {
        askButton.disabled = false;
        askButton.textContent = "Ask";
    }
});

function setUploadStatus(message, type = "") {
    uploadStatus.textContent = message;
    uploadStatus.className = type;
}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

async function waitForIndexing(videoId) {
    const maxAttempts = 300;

    for (let attempt = 0; attempt < maxAttempts; attempt++) {
        const response = await fetch(
            `/api/videos/${encodeURIComponent(videoId)}/status`
        );

        if (!response.ok) {
            throw new Error(
                `Cannot check indexing status: HTTP ${response.status}`
            );
        }

        const job = await response.json();

        if (job.status === "ready") {
            return job;
        }

        if (job.status === "failed") {
            throw new Error(
                job.error || "Video indexing failed"
            );
        }

        setUploadStatus(
            job.status === "queued"
                ? "Video queued for indexing..."
                : "Processing video: extracting frames and building index..."
        );

        await sleep(2000);
    }

    throw new Error(
        "Indexing is still running. Refresh the page later to check."
    );
}

uploadForm.addEventListener("submit", async event => {
    event.preventDefault();

    const file = videoFile.files[0];

    if (!file) return;

    uploadButton.disabled = true;
    uploadButton.textContent = "Uploading...";
    setUploadStatus(`Uploading ${file.name}...`);

    try {
        const formData = new FormData();
        formData.append("file", file);

        const response = await fetch("/api/videos", {
            method: "POST",
            body: formData
        });

        const job = await response.json();

        if (!response.ok) {
            throw new Error(
                typeof job.detail === "string"
                    ? job.detail
                    : `Upload failed: HTTP ${response.status}`
            );
        }

        uploadButton.textContent = "Indexing...";

        await waitForIndexing(job.video_id);

        await loadVideos(job.video_id);

        setUploadStatus(
            `${file.name} is ready for questions!`,
            "success"
        );

        uploadForm.reset();

    } catch (error) {
        setUploadStatus(error.message, "error");

    } finally {
        uploadButton.disabled = false;
        uploadButton.textContent = "Upload and index";
    }
});

loadVideos();