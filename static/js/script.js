// script.js - Premium Frontend Logic
// Custom cursor, 3D tilt, canvas drawing, API communication

document.addEventListener("DOMContentLoaded", () => {

    // ── 1. Remove Loader ──────────────────────────────────────
    setTimeout(() => {
        const loader = document.getElementById("loader");
        if (loader) {
            loader.style.opacity = "0";
            setTimeout(() => loader.remove(), 500);
        }
    }, 1500);


    // ── 2. Custom Cursor ──────────────────────────────────────
    const cursorDot     = document.querySelector(".cursor-dot");
    const cursorOutline = document.querySelector(".cursor-outline");

    window.addEventListener("mousemove", (e) => {
        cursorDot.style.left = e.clientX + "px";
        cursorDot.style.top  = e.clientY + "px";
        cursorOutline.animate(
            { left: e.clientX + "px", top: e.clientY + "px" },
            { duration: 400, fill: "forwards" }
        );
    });

    // Cursor grows on hover over buttons
    document.querySelectorAll("button").forEach(btn => {
        btn.addEventListener("mouseenter", () => {
            cursorDot.style.transform =
                "translate(-50%, -50%) scale(2)";
            cursorOutline.style.width  = "50px";
            cursorOutline.style.height = "50px";
        });
        btn.addEventListener("mouseleave", () => {
            cursorDot.style.transform =
                "translate(-50%, -50%) scale(1)";
            cursorOutline.style.width  = "32px";
            cursorOutline.style.height = "32px";
        });
    });


    // ── 3. 3D Tilt Effect on Canvas Card ─────────────────────
    const tiltCard = document.getElementById("canvas-card");
    if (tiltCard) {
        tiltCard.addEventListener("mousemove", (e) => {
            const rect    = tiltCard.getBoundingClientRect();
            const x       = e.clientX - rect.left;
            const y       = e.clientY - rect.top;
            const centerX = rect.width  / 2;
            const centerY = rect.height / 2;
            const rotateX = ((y - centerY) / centerY) * -6;
            const rotateY = ((x - centerX) / centerX) * 6;
            tiltCard.style.transform =
                `perspective(1000px) rotateX(${rotateX}deg) rotateY(${rotateY}deg)`;
        });

        tiltCard.addEventListener("mouseleave", () => {
            tiltCard.style.transform =
                "perspective(1000px) rotateX(0deg) rotateY(0deg)";
        });
    }


    // ── 4. Canvas Drawing Setup ───────────────────────────────
    const canvas    = document.getElementById("digitCanvas");
    const ctx       = canvas.getContext("2d");
    let isDrawing   = false;
    let brushSize   = 18;
    let lastX       = 0;
    let lastY       = 0;
    let strokeCount = 0;

    // Initialize black background (MNIST format!)
    ctx.fillStyle = "#000000";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.lineWidth   = brushSize;
    ctx.lineCap     = "round";
    ctx.lineJoin    = "round";
    ctx.strokeStyle = "#ffffff";


    // ── 5. Brush Size Buttons ─────────────────────────────────
    document.querySelectorAll(".brush-btn").forEach(btn => {
        btn.addEventListener("click", () => {
            document.querySelectorAll(".brush-btn")
                .forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            brushSize      = parseInt(btn.dataset.size);
            ctx.lineWidth  = brushSize;
        });
    });


    // ── 6. Drawing Functions ──────────────────────────────────
    function getPos(e) {
        const rect = canvas.getBoundingClientRect();
        if (e.touches) {
            return {
                x: e.touches[0].clientX - rect.left,
                y: e.touches[0].clientY - rect.top
            };
        }
        return {
            x: e.clientX - rect.left,
            y: e.clientY - rect.top
        };
    }

    function startDraw(e) {
        e.preventDefault();
        isDrawing = true;
        const pos = getPos(e);
        lastX     = pos.x;
        lastY     = pos.y;
        ctx.beginPath();
        ctx.moveTo(lastX, lastY);
    }

    function draw(e) {
        e.preventDefault();
        if (!isDrawing) return;
        const pos = getPos(e);
        ctx.lineTo(pos.x, pos.y);
        ctx.stroke();
        lastX = pos.x;
        lastY = pos.y;
    }

    function stopDraw(e) {
        if (!isDrawing) return;
        isDrawing = false;
        strokeCount++;
        ctx.beginPath();

        // Auto predict after every stroke
        if (strokeCount > 0) {
            autoPredictDebounced();
        }
    }

    // Mouse events
    canvas.addEventListener("mousedown", startDraw);
    canvas.addEventListener("mousemove", draw);
    canvas.addEventListener("mouseup",   stopDraw);
    canvas.addEventListener("mouseout",  stopDraw);

    // Touch events for mobile
    canvas.addEventListener("touchstart", startDraw,
        { passive: false });
    canvas.addEventListener("touchmove",  draw,
        { passive: false });
    canvas.addEventListener("touchend",   stopDraw);


    // ── 7. Auto Predict (debounced) ───────────────────────────
    // Predicts automatically 800ms after user stops drawing
    let debounceTimer;
    function autoPredictDebounced() {
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(() => {
            predictDigit(false);
        }, 800);
    }


    // ── 8. Clear Canvas ───────────────────────────────────────
    document.getElementById("clearBtn")
        .addEventListener("click", () => {
        ctx.fillStyle = "#000000";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        strokeCount = 0;
        clearTimeout(debounceTimer);
        hideResults();
        showToast("Canvas cleared!", "success");
    });


    // ── 9. Undo Last Stroke ───────────────────────────────────
    // Simple undo - just clear (proper undo needs history array)
    document.getElementById("undoBtn")
        .addEventListener("click", () => {
        ctx.fillStyle = "#000000";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        strokeCount = Math.max(0, strokeCount - 1);
        showToast("Cleared! Redraw your digit.", "warning");
    });


    // ── 10. Predict Button ────────────────────────────────────
    document.getElementById("predictBtn")
        .addEventListener("click", () => {
        predictDigit(true);
    });


    // ── 11. Main Predict Function ─────────────────────────────
    async function predictDigit(showLoader = true) {
        // Check if canvas is empty
        const imageData = ctx.getImageData(
            0, 0, canvas.width, canvas.height
        );
        const pixels = imageData.data;
        const hasDrawing = Array.from(pixels).some(
            (val, i) => i % 4 !== 3 && val > 10
        );

        if (!hasDrawing) {
            showToast(
                "Please draw a digit first!",
                "warning"
            );
            return;
        }

        if (showLoader) {
            showLoading();
        }

        // Convert canvas to base64
        const base64 = canvas.toDataURL("image/png");

        try {
            const response = await fetch("/predict", {
                method:  "POST",
                headers: { "Content-Type": "application/json" },
                body:    JSON.stringify({ image: base64 })
            });

            const data = await response.json();

            if (data.success) {
                showResults(data);
                if (showLoader) {
                    showToast(
                        "Digit identified: " + data.digit + "!",
                        "success"
                    );
                }
            } else {
                hideLoading();
                showToast(
                    data.detail || "Prediction failed!",
                    "error"
                );
            }

        } catch (error) {
            hideLoading();
            showToast(
                "Could not connect to server!",
                "error"
            );
        }
    }


    // ── 12. Show Results ──────────────────────────────────────
    function showResults(data) {
        hideLoading();

        const placeholder    = document.getElementById(
            "result-placeholder");
        const predDisplay    = document.getElementById(
            "prediction-display");
        const predictedDigit = document.getElementById(
            "predicted-digit");
        const confidenceVal  = document.getElementById(
            "confidence-value");
        const allProbsTitle  = document.getElementById(
            "all-probs-title");

        placeholder.style.display  = "none";
        predDisplay.style.display  = "block";
        predictedDigit.textContent = data.digit;
        confidenceVal.textContent  = data.confidence + "%";

        // Show all probability bars
        allProbsTitle.style.display = "block";

        // Sort probabilities highest first
        const sorted = Object.entries(data.all_probs)
            .sort((a, b) => b[1] - a[1]);

        sorted.forEach(([digit, prob], i) => {
            const bar      = document.getElementById(
                "prob-bar-" + digit);
            const fill     = document.getElementById(
                "fill-" + digit);
            const confSpan = document.getElementById(
                "conf-" + digit);

            if (bar && fill && confSpan) {
                bar.style.display = "block";

                // Color code
                let confClass, barClass;
                if (prob >= 85) {
                    confClass = "conf-high";
                    barClass  = "bar-high";
                } else if (prob >= 50) {
                    confClass = "conf-mid";
                    barClass  = "bar-mid";
                } else {
                    confClass = "conf-low";
                    barClass  = "bar-low";
                }

                confSpan.className = "prob-value " + confClass;
                confSpan.textContent = prob + "%";
                fill.className = "bar-fill " + barClass;

                // Staggered animation
                setTimeout(() => {
                    fill.style.width = prob + "%";
                }, 100 * (i + 1));
            }
        });
    }


    // ── 13. Hide Results ──────────────────────────────────────
    function hideResults() {
        const placeholder = document.getElementById(
            "result-placeholder");
        const predDisplay = document.getElementById(
            "prediction-display");
        const allProbsTitle = document.getElementById(
            "all-probs-title");

        placeholder.style.display   = "block";
        predDisplay.style.display   = "none";
        allProbsTitle.style.display = "none";

        // Reset all bars
        for (let i = 0; i <= 9; i++) {
            const bar  = document.getElementById(
                "prob-bar-" + i);
            const fill = document.getElementById(
                "fill-" + i);
            if (bar)  bar.style.display = "none";
            if (fill) fill.style.width  = "0%";
        }
    }


    // ── 14. Loading State ─────────────────────────────────────
    function showLoading() {
        document.getElementById("loading-overlay")
            .classList.add("active");
        document.getElementById("result-placeholder")
            .style.display = "none";
        document.getElementById("prediction-display")
            .style.display = "none";
        document.getElementById("predictBtn").disabled = true;
        document.getElementById("predictBtn").textContent =
            "Analyzing...";
    }

    function hideLoading() {
        document.getElementById("loading-overlay")
            .classList.remove("active");
        document.getElementById("predictBtn").disabled = false;
        document.getElementById("predictBtn").textContent =
            "Predict";
    }


    // ── 15. Toast Notifications ───────────────────────────────
    function showToast(message, type = "success") {
        const existing = document.querySelector(".toast");
        if (existing) existing.remove();

        const icons = {
            success: "✅",
            error:   "❌",
            warning: "⚠️"
        };

        const toast = document.createElement("div");
        toast.className = `toast ${type}`;
        toast.innerHTML =
            `<span>${icons[type]}</span> ${message}`;
        document.body.appendChild(toast);

        setTimeout(() => toast.classList.add("show"), 10);
        setTimeout(() => {
            toast.classList.remove("show");
            setTimeout(() => toast.remove(), 400);
        }, 3000);
    }

});