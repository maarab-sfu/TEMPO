const videoSelect = document.getElementById("videoSelect");
const videoPlayer = document.getElementById("videoPlayer");
const questionForm = document.getElementById("questionForm");
const questionInput = document.getElementById("questionInput");
const askButton = document.getElementById("askButton");
const messages = document.getElementById("messages");
const evidence = document.getElementById("evidence");

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

async function loadVideos() {
    try {
        const response = await fetch("/api/videos");

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const data = await response.json();

        videoSelect.innerHTML = "";

        for (const video of data.videos) {
            const option = document.createElement("option");
            option.value = video.video_id;
            option.textContent = video.source_filename || video.video_id;
            videoSelect.appendChild(option);
        }

        if (data.videos.length > 0) {
            selectVideo(data.videos[0].video_id);
        } else {
            videoSelect.innerHTML =
                '<option value="">No indexed videos</option>';
        }
    } catch (error) {
        addMessage(`Cannot load videos: ${error.message}`, "ai");
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

loadVideos();