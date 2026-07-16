document.addEventListener('DOMContentLoaded', () => {
    const dropzoneA = document.getElementById('dropzone-a');
    const dropzoneB = document.getElementById('dropzone-b');
    const inputA = document.getElementById('input-a');
    const inputB = document.getElementById('input-b');
    const previewA = document.getElementById('preview-a');
    const previewB = document.getElementById('preview-b');
    const btnCompare = document.getElementById('compare-btn');
    const errorBanner = document.getElementById('error-banner');
    
    let fileA = null;
    let fileB = null;

    // Slider sync
    const slider = document.getElementById('sensitivity-slider');
    const sliderVal = document.getElementById('sensitivity-value');
    slider.addEventListener('input', (e) => {
        sliderVal.textContent = e.target.value;
    });

    // Helper: Setup Dropzone
    function setupDropzone(dropzone, input, preview, setFileCallback) {
        dropzone.addEventListener('click', () => input.click());
        
        dropzone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropzone.classList.add('dragover');
        });
        
        dropzone.addEventListener('dragleave', () => {
            dropzone.classList.remove('dragover');
        });
        
        dropzone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropzone.classList.remove('dragover');
            if (e.dataTransfer.files.length) {
                handleFile(e.dataTransfer.files[0], dropzone, preview, setFileCallback);
            }
        });

        input.addEventListener('change', (e) => {
            if (e.target.files.length) {
                handleFile(e.target.files[0], dropzone, preview, setFileCallback);
            }
        });
    }

    function handleFile(file, dropzone, preview, setFileCallback) {
        if (!file.type.startsWith('image/')) {
            showError("Please upload a valid image file.");
            return;
        }
        
        // Hide error on new valid selection
        errorBanner.classList.add('hidden');
        
        setFileCallback(file);
        
        const reader = new FileReader();
        reader.onload = (e) => {
            preview.src = e.target.result;
            preview.classList.remove('hidden');
            dropzone.querySelector('.placeholder').classList.add('hidden');
            checkReady();
        };
        reader.readAsDataURL(file);
    }

    function checkReady() {
        if (fileA && fileB) {
            btnCompare.disabled = false;
        }
    }

    function showError(msg) {
        errorBanner.textContent = msg;
        errorBanner.classList.remove('hidden');
    }

    setupDropzone(dropzoneA, inputA, previewA, (f) => fileA = f);
    setupDropzone(dropzoneB, inputB, previewB, (f) => fileB = f);

    // Compare Button
    btnCompare.addEventListener('click', async () => {
        errorBanner.classList.add('hidden');
        document.getElementById('results-section').classList.add('hidden');
        document.getElementById('loading').classList.remove('hidden');
        btnCompare.disabled = true;

        const formData = new FormData();
        formData.append('image_a', fileA);
        formData.append('image_b', fileB);
        formData.append('sensitivity', slider.value);

        try {
            const response = await fetch('/api/compare', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (!response.ok) {
                throw new Error(data.error || "Server returned an error.");
            }
            
            // Populate Results
            document.getElementById('result-composite').src = `data:image/jpeg;base64,${data.composite_b64}`;
            document.getElementById('result-heatmap').src = `data:image/jpeg;base64,${data.diff_thumb_b64}`;
            
            document.getElementById('stat-total').textContent = data.total_differences;
            document.getElementById('stat-area').textContent = `${data.flagged_area_pct.toFixed(2)}%`;
            
            const bdContainer = document.getElementById('stat-breakdown');
            bdContainer.innerHTML = '';
            if (data.classifications.length === 0) {
                bdContainer.innerHTML = '<div class="breakdown-item">0 Differences</div>';
            } else {
                data.classifications.forEach(c => {
                    const div = document.createElement('div');
                    div.className = 'breakdown-item';
                    div.innerHTML = `<span>${c.label}</span><span>${c.count}</span>`;
                    bdContainer.appendChild(div);
                });
            }

            const confBanner = document.getElementById('confidence-banner');
            if (data.confidence === 'low') {
                confBanner.classList.remove('hidden');
            } else {
                confBanner.classList.add('hidden');
            }

            document.getElementById('results-section').classList.remove('hidden');
            
        } catch (err) {
            showError(err.message);
        } finally {
            document.getElementById('loading').classList.add('hidden');
            btnCompare.disabled = false;
        }
    });
});
