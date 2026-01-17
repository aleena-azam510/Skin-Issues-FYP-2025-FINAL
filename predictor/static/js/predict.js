        document.addEventListener('DOMContentLoaded', function() {
            // DOM Elements with herbal- prefix
            const cameraCard = document.getElementById('herbal-camera-card');
            const uploadCard = document.getElementById('herbal-upload-card');
            const previewContainer = document.getElementById('herbal-preview-container');
            const cameraFeed = document.getElementById('herbal-camera-feed');
            const imagePreview = document.getElementById('herbal-image-preview');
            const placeholder = document.getElementById('herbal-placeholder');
            const fileInput = document.getElementById('herbal-file-input');
            const resetBtn = document.getElementById('herbal-reset-btn');
            const analyzeBtn = document.getElementById('herbal-analyze-btn');
            
            // Variables
            let currentStream = null;
            let selectedImage = null;
            
            // Input method selection
            cameraCard.addEventListener('click', activateCamera);
            uploadCard.addEventListener('click', activateUpload);
            
            // Drag and drop functionality
            previewContainer.addEventListener('dragover', (e) => {
                e.preventDefault();
                previewContainer.classList.add('drag-over');
            });
            
            previewContainer.addEventListener('dragleave', () => {
                previewContainer.classList.remove('drag-over');
            });
            
            previewContainer.addEventListener('drop', (e) => {
                e.preventDefault();
                previewContainer.classList.remove('drag-over');
                
                if (e.dataTransfer.files && e.dataTransfer.files[0]) {
                    const file = e.dataTransfer.files[0];
                    if (file.type.match('image.*')) {
                        handleImageFile(file);
                    }
                }
            });
            
            // File input handling
            fileInput.addEventListener('change', (e) => {
                if (e.target.files && e.target.files[0]) {
                    handleImageFile(e.target.files[0]);
                }
            });
            
            // Button events
            resetBtn.addEventListener('click', resetAll);
            analyzeBtn.addEventListener('click', analyzeImage);
            
            // Activate camera
            function activateCamera() {
                // Update UI
                cameraCard.classList.add('active');
                uploadCard.classList.remove('active');
                previewContainer.classList.add('active');
                placeholder.classList.add('herbal-hidden');
                
                // Stop any existing stream
                if (currentStream) {
                    currentStream.getTracks().forEach(track => track.stop());
                }
                
                // Access camera
                navigator.mediaDevices.getUserMedia({ video: true })
                    .then(stream => {
                        currentStream = stream;
                        cameraFeed.srcObject = stream;
                        cameraFeed.style.display = 'block';
                        imagePreview.style.display = 'none';
                    })
                    .catch(err => {
                        console.error('Error accessing camera:', err);
                        placeholder.innerHTML = `<i class="fas fa-video-slash"></i><h3>Camera Access Required</h3><p>Please enable camera permissions to use this feature</p>`;
                        placeholder.classList.remove('herbal-hidden');
                    });
            }
            
            // Activate upload
            function activateUpload() {
                // Update UI
                uploadCard.classList.add('active');
                cameraCard.classList.remove('active');
                previewContainer.classList.add('active');
                
                // Stop camera if active
                if (currentStream) {
                    currentStream.getTracks().forEach(track => track.stop());
                    currentStream = null;
                }
                
                // Hide camera feed
                cameraFeed.style.display = 'none';
                
                // Show placeholder if no image is selected
                if (!selectedImage) {
                    placeholder.classList.remove('herbal-hidden');
                    imagePreview.style.display = 'none';
                } else {
                    placeholder.classList.add('herbal-hidden');
                    imagePreview.style.display = 'block';
                }
                
                // Trigger file input
                fileInput.click();
            }
            
            // Handle image file
            function handleImageFile(file) {
                const reader = new FileReader();
                
                reader.onload = (e) => {
                    selectedImage = e.target.result;
                    imagePreview.src = selectedImage;
                    imagePreview.style.display = 'block';
                    placeholder.classList.add('herbal-hidden');
                    cameraFeed.style.display = 'none';
                    
                    // Ensure upload card is active
                    uploadCard.classList.add('active');
                    cameraCard.classList.remove('active');
                };
                
                reader.readAsDataURL(file);
            }
            
            // Reset everything
            function resetAll() {
                // Stop camera
                if (currentStream) {
                    currentStream.getTracks().forEach(track => track.stop());
                    currentStream = null;
                }
                
                // Reset UI
                cameraFeed.style.display = 'none';
                imagePreview.style.display = 'none';
                imagePreview.src = '';
                selectedImage = null;
                previewContainer.classList.remove('active');
                placeholder.classList.remove('herbal-hidden');
                placeholder.innerHTML = `<i class="fas fa-cloud-upload-alt"></i><h3>Drag & Drop Image Here</h3><p>or select an input method above</p>`;
                cameraCard.classList.remove('active');
                uploadCard.classList.remove('active');
            }
            
            // Analyze image
            function analyzeImage() {
                if (!selectedImage && !currentStream) {
                    alert('Please capture or upload an image first');
                    return;
                }
                
                // Simulate analysis process
                analyzeBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';
                analyzeBtn.disabled = true;
                
                // Simulate API call delay
                setTimeout(() => {
                    analyzeBtn.innerHTML = '<i class="fas fa-search"></i> Analyze Skin';
                    analyzeBtn.disabled = false;
                    
                    // Results would normally come from API
                    // For this demo, we're showing static results
                    
                    // Show success message
                    const statusIndicator = document.querySelector('.herbal-status-indicator');
                    statusIndicator.innerHTML = '<span class="herbal-status-dot"></span> <span>Analysis Complete</span>';
                    
                    // Add visual feedback
                    analyzeBtn.classList.add('herbal-btn-success');
                    setTimeout(() => {
                        analyzeBtn.classList.remove('herbal-btn-success');
                    }, 2000);
                    
                }, 2000);
            }
            
            // Initialize with camera disabled for safety
            resetAll();
        });