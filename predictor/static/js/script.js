// File Upload Handler
document.getElementById('fileInput').addEventListener('change', function(e) {
    const file = e.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function(event) {
            const preview = document.getElementById('imagePreview');
            preview.src = event.target.result;
            preview.style.display = 'block';

            // ✅ Enable analyze button
            document.getElementById('predictBtn').disabled = false;
            document.getElementById('captureBtn').disabled = true;
        };
        reader.readAsDataURL(file);
    }
});


// Camera Capture Handler (simplified)
document.getElementById('startCameraBtn').addEventListener('click', async function() {
    try {
        // Try to access the camera
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        
        // Create a video element and append it to the preview area
        const video = document.createElement('video');
        video.srcObject = stream; // Set the video source
        video.play(); // Play the video

        const previewArea = document.getElementById('previewArea');
        previewArea.innerHTML = ''; // Clear any previous content
        previewArea.appendChild(video); // Add the video element to the DOM

        // Clear any previous capture button if it exists
        let existingCaptureBtn = document.getElementById('captureBtn');
        if (existingCaptureBtn) {
            existingCaptureBtn.remove();
        }

        // Create and append a new capture button
        const captureBtn = document.createElement('button');
        captureBtn.id = 'captureBtn';
        captureBtn.className = 'btn';
        captureBtn.innerHTML = '<i class="fas fa-camera"></i> Capture';
        captureBtn.onclick = function() {
            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

            const img = document.createElement('img');
            img.src = canvas.toDataURL('image/jpeg'); // Capture image as JPEG
            img.style.maxWidth = '100%';
            img.style.maxHeight = '400px';
            previewArea.innerHTML = '';  // Clear the preview area
            previewArea.appendChild(img);  // Append captured image

            document.getElementById('predictBtn').disabled = false;  // Enable predict button
            stream.getTracks().forEach(track => track.stop());  // Stop the camera
        };

        previewArea.appendChild(captureBtn);  // Append the capture button
    } catch (err) {
        console.error('Error accessing camera:', err.name); // Log specific error name
        console.error('Error message:', err.message); // Log specific error message
        alert('Could not access the camera. Please check permissions.');
    }
});




// document.addEventListener('DOMContentLoaded', function() {
//     // DOM Elements
//     const fileInput = document.getElementById('fileInput');
//     const video = document.getElementById('video');
//     const canvas = document.getElementById('canvas');
//     const preview = document.getElementById('imagePreview');
//     const startCameraBtn = document.getElementById('startCameraBtn');
//     const captureBtn = document.getElementById('captureBtn');
//     const predictBtn = document.getElementById('predictBtn');
//     const resultDiv = document.getElementById('result');
//     const spinner = document.getElementById('spinner');
    
//     let stream = null;

//     // Camera Functions
//     startCameraBtn.addEventListener('click', async () => {
//         try {
//             stream = await navigator.mediaDevices.getUserMedia({ video: true });
//             video.srcObject = stream;
//             showElement(video);
//             hideElement(canvas);
//             hideElement(preview);
//             captureBtn.disabled = false;
//         } catch (err) {
//             console.error('Error accessing camera:', err);
//             alert('Could not access the camera. Please check permissions.');
//         }
//     });

//     captureBtn.addEventListener('click', () => {
//         canvas.width = video.videoWidth;
//         canvas.height = video.videoHeight;
//         const ctx = canvas.getContext('2d');
//         ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        
//         preview.src = canvas.toDataURL('image/jpeg');
//         showElement(preview);
//         hideElement(video);
//         hideElement(canvas);
        
//         predictBtn.disabled = false;
//         stopCamera();
//     });
let currentPage = 1;
const remediesperpage = 3; // show 3 remedies per page (adjust as needed)
document.addEventListener('DOMContentLoaded', function() {
    const video = document.getElementById('video'); // will work only if DOM is fully loaded
    const canvas = document.getElementById('canvas');
    const preview = document.getElementById('imagePreview');
    const startCameraBtn = document.getElementById('startCameraBtn');
    const captureBtn = document.getElementById('captureBtn');
    const predictBtn = document.getElementById('predictBtn');
    const resultDiv = document.getElementById('result');
    const spinner = document.getElementById('spinner');
    const fileInput = document.getElementById('fileInput'); // ← Add this line


    let stream = null;

    startCameraBtn.addEventListener('click', async () => {
        try {
            stream = await navigator.mediaDevices.getUserMedia({ video: true });

            video.srcObject = stream;
            video.play();
            video.style.display = 'block';

            captureBtn.disabled = false;
            preview.style.display = 'none';
        } catch (err) {
            console.error('Error accessing camera:', err);
            alert('Could not access the camera. Please check permissions.');
        }
    });

    captureBtn.addEventListener('click', () => {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        preview.src = canvas.toDataURL('image/jpeg');
        preview.style.display = 'block';
        video.style.display = 'none';

        predictBtn.disabled = false;

        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            stream = null;
        }
    });

    // File Input Handling
    fileInput.addEventListener('change', () => {
        const file = fileInput.files[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = (event) => {
            preview.src = event.target.result;
            showElement(preview);
            hideElement(video);
            hideElement(canvas);
            predictBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    });

    // Prediction Function
    predictBtn.addEventListener('click', async () => {
        resultDiv.innerHTML = '';
        showSpinner();
        predictBtn.disabled = true;

        try {
            const formData = new FormData();
            
            if (preview.src.startsWith('data:image')) {
                const blob = await fetch(preview.src).then(res => res.blob());
                formData.append('file', blob, 'skin_image.jpg');
            } else {
                formData.append('file', fileInput.files[0]);
            }

            const response = await fetch('/api/predict/', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error('Server responded with error');
            }

            const data = await response.json();
            window.latestData = data; // ✅ Store data globally for remedy navigation

            currentPage = 1; // Reset page on new prediction

            displayResults(data);
            
        } catch (error) {
            console.error('Error:', error);
            resultDiv.innerHTML = `
                <div class="error-message">
                    Error: ${error.message || 'Failed to process image'}
                </div>
            `;
        } finally {
            hideSpinner();
            predictBtn.disabled = false;
        }
    });

    // Helper Functions
    function showElement(el) {
        el.style.display = 'block';
        el.classList.add('active');
    }

    function hideElement(el) {
        el.style.display = 'none';
        el.classList.remove('active');
    }

    function showSpinner() {
        spinner.style.display = 'block';
    }

    function hideSpinner() {
        spinner.style.display = 'none';
    }

    function stopCamera() {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            stream = null;
        }
    }

    const remedyState = {}; // to track pagination index per issue
let latestData = null;  // save the latest data globally for pagination


const ctx = canvas.getContext('2d');

function formatConditionName(name) {
    return name.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
}

function resizeCanvas() {
    canvas.width = preview.clientWidth;
    canvas.height = preview.clientHeight;
    canvas.style.width = preview.clientWidth + 'px';
    canvas.style.height = preview.clientHeight + 'px';
}

function drawBoundingBoxes(boxes) {
    if (!boxes || boxes.length === 0) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        return;
    }
    resizeCanvas();
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = 'red';
    ctx.lineWidth = 3;

    const scaleX = preview.clientWidth / preview.naturalWidth;
    const scaleY = preview.clientHeight / preview.naturalHeight;

    boxes.forEach(box => {
        ctx.strokeRect(
            box.x * scaleX,
            box.y * scaleY,
            box.width * scaleX,
            box.height * scaleY
        );
    });
}

function renderRemedy(issue, remedy) {
    const directions = Array.isArray(remedy.directions) ? remedy.directions : [remedy.directions];

    return `
        <div class="remedy-card">
            <div class="remedy-content">
                <h4 style="color: rgb(209, 111, 0);">${remedy.title}</h4>
                <p><strong>Amount:</strong> ${remedy.amount}</p>
                <div><strong>Directions:</strong></div>
                <ul class="remedy-steps">
                    ${directions.map(step => `<li>${step}</li>`).join('')}
                </ul>
                <div class="remedy-navigation">
                    <button class="remedy-btn prev-btn" onclick="changeRemedy('${issue}', -1)">&#8592; Previous</button>
                    <button class="remedy-btn next-btn" onclick="changeRemedy('${issue}', 1)">Next &#8594;</button>
                </div>
            </div>
            ${remedy.image_url ? 
                `<img src="${remedy.image_url}" alt="${remedy.title}" class="remedy-image">` 
                : '<div class="remedy-image-placeholder"></div>'}
        </div>
         
    `;
}

window.changeRemedy = function(issue, direction) {
    const remedies = latestData.remedies_data[issue].remedies;
    let index = remedyState[issue] + direction;

    if (index < 0) index = 0;
    if (index >= remedies.length) index = remedies.length - 1;

    remedyState[issue] = index;

    const card = document.getElementById(`card-${issue}`);
    const header = `<h3>${formatConditionName(issue)}</h3>`;
    card.innerHTML = header + renderRemedy(issue, remedies[index]);
};

function displayResults(data) {
    const resultDiv = document.getElementById('result');
    resultDiv.innerHTML = ''; // clear previous results
    latestData = data; // store latest data globally for pagination

    if (data.error) {
        resultDiv.innerHTML = `<div class="error-message">${data.error}</div>`;
        preview.style.display = 'none';
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        return;
    }

    if (data.annotated_image) {
        preview.src = 'data:image/jpeg;base64,' + data.annotated_image;
        preview.style.display = 'block';
        preview.onload = () => {
            resizeCanvas();
            // Draw bounding boxes if available
            if (data.bounding_boxes) {
                drawBoundingBoxes(data.bounding_boxes);
            } else {
                ctx.clearRect(0, 0, canvas.width, canvas.height);
            }
        };
    } else {
        preview.style.display = 'none';
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    }

    if (!data.detected_issues || data.detected_issues.length === 0) {
        resultDiv.innerHTML = `<div class="no-results">No skin conditions detected. Try a clearer image.</div>`;
        return;
    }

    // Show remedies with pagination
    for (const issue of data.detected_issues) {
        const issueData = data.remedies_data[issue];
        if (!issueData || !issueData.remedies || issueData.remedies.length === 0) continue;

        remedyState[issue] = 0; // initialize index

        const card = document.createElement('div');
        card.className = 'condition-result';
        card.id = `card-${issue}`;
        const header = `<h3>${formatConditionName(issue)}</h3>`;
        card.innerHTML = header + renderRemedy(issue, issueData.remedies[0]);

        resultDiv.appendChild(card);
    }
}


    function formatConditionName(name) {
        return name.split('_').map(word => 
            word.charAt(0).toUpperCase() + word.slice(1)
        ).join(' ');
    }
});
