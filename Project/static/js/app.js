document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const canvas = document.getElementById('drawing-canvas');
    const ctx = canvas.getContext('2d');
    const placeholder = document.getElementById('canvas-placeholder');
    const clearBtn = document.getElementById('clear-btn');
    const submitBtn = document.getElementById('submit-btn');
    const regenBtn = document.getElementById('regenerate-btn');
    const targetDigitEl = document.getElementById('captcha-target');
    const modelSelect = document.getElementById('model-select');
    
    // Result UI Elements
    const resultBox = document.getElementById('result-box');
    const resultIcon = document.getElementById('result-icon');
    const resultMessage = document.getElementById('result-message');
    const predDigit = document.getElementById('pred-digit');
    const predConf = document.getElementById('pred-conf');
    const btnText = document.querySelector('.btn-text');
    const loader = document.querySelector('.loader');

    // State
    let isDrawing = false;
    let hasDrawn = false;
    let currentTarget = null;

    // Initialize Canvas
    // Use an internal resolution for better quality, scale with CSS
    const initCanvas = () => {
        // Clear canvas
        ctx.fillStyle = 'black';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        // Setup brush
        ctx.strokeStyle = 'white';
        ctx.lineWidth = 18;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        
        hasDrawn = false;
        placeholder.style.opacity = '1';
    };

    // Draw Mechanics
    const startDrawing = (e) => {
        isDrawing = true;
        hasDrawn = true;
        placeholder.style.opacity = '0';
        resultBox.classList.add('hidden'); // Hide previous results
        draw(e);
    };

    const stopDrawing = () => {
        isDrawing = false;
        ctx.beginPath();
    };

    const draw = (e) => {
        if (!isDrawing) return;
        
        e.preventDefault(); // Prevent scrolling on touch

        // Calculate exact coordinates based on canvas display size vs internal size
        const rect = canvas.getBoundingClientRect();
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;

        let clientX = e.clientX;
        let clientY = e.clientY;

        // Support touch events
        if (e.touches && e.touches.length > 0) {
            clientX = e.touches[0].clientX;
            clientY = e.touches[0].clientY;
        }

        const x = (clientX - rect.left) * scaleX;
        const y = (clientY - rect.top) * scaleY;

        ctx.lineTo(x, y);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(x, y);
    };

    // Event Listeners for Drawing
    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('mouseout', stopDrawing);

    canvas.addEventListener('touchstart', startDrawing, {passive: false});
    canvas.addEventListener('touchmove', draw, {passive: false});
    canvas.addEventListener('touchend', stopDrawing);

    // API Interactions
    const fetchCaptcha = async () => {
        try {
            targetDigitEl.classList.remove('glow-digit');
            targetDigitEl.style.opacity = '0.5';
            
            const response = await fetch('/api/captcha');
            const data = await response.json();
            
            if (data.status === 'success') {
                currentTarget = data.captcha_digit;
                targetDigitEl.textContent = currentTarget;
                targetDigitEl.classList.add('glow-digit');
                targetDigitEl.style.opacity = '1';
                initCanvas();
                resultBox.classList.add('hidden');
            }
        } catch (err) {
            console.error("Failed to fetch CAPTCHA", err);
            targetDigitEl.textContent = "Err";
        }
    };

    const submitCaptcha = async () => {
        if (!hasDrawn) {
            alert("Please draw the digit before submitting.");
            return;
        }

        // UI Loading State
        btnText.classList.add('hidden');
        loader.classList.remove('hidden');
        submitBtn.disabled = true;

        // Get Base64 image
        const base64Image = canvas.toDataURL('image/png');
        const selectedModel = modelSelect.value;

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    image: base64Image,
                    target_digit: currentTarget,
                    model_name: selectedModel
                })
            });

            const data = await response.json();

            // Handle Response UI
            resultBox.classList.remove('hidden');
            resultBox.className = 'result-box'; // Reset classes
            
            if (data.status === 'success') {
                resultMessage.textContent = data.message;
                predDigit.textContent = data.predicted_digit;
                predConf.textContent = (data.confidence * 100).toFixed(1) + '%';

                if (data.is_verified) {
                    resultBox.classList.add('success');
                    resultIcon.innerHTML = '✓';
                } else {
                    resultBox.classList.add('error');
                    resultIcon.innerHTML = '✗';
                }
            } else {
                resultBox.classList.add('error');
                resultMessage.textContent = data.message || "Server Error";
                resultIcon.innerHTML = '!';
                predDigit.textContent = '-';
                predConf.textContent = '-';
            }

        } catch (err) {
            console.error("Prediction Error:", err);
            resultBox.classList.remove('hidden');
            resultBox.className = 'result-box error';
            resultMessage.textContent = "Connection failed.";
            resultIcon.innerHTML = '!';
        } finally {
            // Restore UI State
            btnText.classList.remove('hidden');
            loader.classList.add('hidden');
            submitBtn.disabled = false;
        }
    };

    // Setup Actions
    clearBtn.addEventListener('click', initCanvas);
    submitBtn.addEventListener('click', submitCaptcha);
    regenBtn.addEventListener('click', fetchCaptcha);

    // Initial load
    initCanvas();
    fetchCaptcha();
});
