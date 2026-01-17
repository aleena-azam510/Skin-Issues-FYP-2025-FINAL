document.addEventListener('DOMContentLoaded', function () {
    // ----------------------------------------------------------------------
    // 1. --- GLOBAL STATE & UI ELEMENT REFERENCES (Encapsulated) ---
    // ----------------------------------------------------------------------
    let currentPage = 0; // Not actively used but kept
    const conditionsPerPage = 2;
    let latestData = null; // Stores the entire API response
    let globalFileToSend = null; // Stores the image file for initial and follow-up calls
    let currentQuestionIndex = 0; // Manages the step-by-step question display.
    const remedyState = {}; // Stores the current index for remedies for each detected issue: { 'acne': 0, 'rosacea': 1 }
    let selectedRemedyType = 'home'; // 'home' or 'medical'
    let medicalDisclaimerShown = false;
    let onAcknowledgeCallback = null;
    const homeDisclaimerKey = 'homeDisclaimerShown';
    const medicalDisclaimerKey = 'medicalDisclaimerShown';



    const remediesByIssueKey = {}; // 🔴 REQUIRED for remedy navigation

    let followUpQuestions = []; // Stores the questions list from the initial API response
    const HOME_REMEDY_DISCLAIMER = `
        Home remedies provided here are generally safe and suitable for mild skin concerns.

        However, skin conditions can vary from person to person. 
        If you experience irritation, worsening symptoms, or no improvement,
        please discontinue use and consult a qualified dermatologist or healthcare professional.
        `;

    const MEDICAL_TREATMENT_DISCLAIMER = `
        This system provides AI-based skin condition prediction and treatment recommendations
        based on detected visual patterns and user-reported symptoms.

        Medical treatments suggested here may include prescription medications or clinical procedures
        and should be followed under the supervision of a qualified dermatologist or healthcare professional.

        This system is designed to support clinical decision-making and does not replace
        professional medical judgment. Always seek expert consultation before initiating medical treatment.
        `;

    // DOM Elements
    const video = document.getElementById('herbal-camera-feed');
    const canvas = document.getElementById('canvas');
    const preview = document.getElementById('imagePreview');
    const startCameraBtn = document.getElementById('startCameraBtn');
    const captureBtn = document.getElementById('captureBtn');
    const predictBtn = document.getElementById('herbal-analyze-btn');
    const resultsSection = document.getElementById('results');
    const diagnosisConditionNameEl = document.getElementById('diagnosisConditionName');
    const diagnosisConfidenceLevelEl = document.getElementById('diagnosisConfidenceLevel');
    const diagnosisConfidenceTextEl = document.getElementById('diagnosisConfidenceText');
    const herbalDetailedRemediesGrid = document.getElementById('herbalDetailedRemediesGrid');
    const spinner = document.getElementById('spinner');
    const fileInput = document.getElementById('fileInput');

    let stream = null;
    const ctx = canvas.getContext('2d');

    // ✅ CREATE MESSAGE BOX FIRST
const messageBoxHTML = `
<div id="customMessageBox"
     style="display:none; position:fixed; inset:0; background:rgba(0,0,0,0.5); z-index:99999;">

    <div style="
        display:flex;
        justify-content:center;
        align-items:center;
        width:100%;
        height:100%;
    ">
        <div style="
            background:white;
            padding:24px;
            border-radius:14px;
            max-width:420px;
            width:90%;
            box-shadow:0 10px 40px rgba(0,0,0,0.3);
        ">
            <h4 style="font-size:20px;font-weight:700;color:#166534;margin-bottom:12px;">
                Important Health Notice
            </h4>

            <p id="messageBoxText" style="color:#374151;margin-bottom:20px;"></p>

            <button onclick="window.hideMessageBox()"
                style="
                    width:100%;
                    background:#198526;
                    color:white;
                    padding:10px;
                    border:none;
                    border-radius:10px;
                    font-weight:600;
                    cursor:pointer;
                ">
                Acknowledge
            </button>
        </div>
    </div>
</div>
`;
document.body.insertAdjacentHTML('beforeend', messageBoxHTML);


    // ----------------------------------------------------------------------
    // 2. --- UTILITY FUNCTIONS ---
    // ----------------------------------------------------------------------

    // --- Message Box Functions ---
    function showMessageBox(message, onAcknowledge) {
    const box = document.getElementById('customMessageBox');
    const textEl = document.getElementById('messageBoxText');

    if (!box || !textEl) return;

    textEl.textContent = message;
    onAcknowledgeCallback = onAcknowledge || null;

    box.style.display = 'block';
    box.classList.remove('opacity-0', 'pointer-events-none');
    box.classList.add('opacity-100');
}


    function hideMessageBox() {
    const box = document.getElementById('customMessageBox');
    if (!box) return;

    box.classList.remove('opacity-100');
    box.classList.add('opacity-0', 'pointer-events-none');

    setTimeout(() => {
        box.style.display = 'none';

        // ✅ PROCEED ONLY AFTER ACKNOWLEDGE
        if (typeof onAcknowledgeCallback === 'function') {
            const cb = onAcknowledgeCallback;
            onAcknowledgeCallback = null;
            cb();
        }
    }, 300);
}
window.hideMessageBox = hideMessageBox;



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

    function formatConditionName(name) {
        if (!name || name.toLowerCase() === 'none' || name === '0' || /^\d+$/.test(name.trim())) {
            return 'Unclassified Issue';
        }
        return name.split(/[\s_-]+/).map(word =>
            word.charAt(0).toUpperCase() + word.slice(1)
        ).join(' ');
    }
    function makeSafeKey(str) {
    return str.toLowerCase().replace(/[^a-z0-9]/g, '_');
}


    function formatListContent(content) {
    if (!content || (Array.isArray(content) && content.length === 0) || content === 'N/A') {
        return '<p class="italic text-gray-500 text-sm">Detailed information is being updated.</p>';
    }
    
    if (Array.isArray(content)) {
        const items = content.filter(item => item.trim().length > 0);
        if (items.length === 0) return '<p class="italic text-gray-500 text-sm">Information not available.</p>';
        
        // Removed list-disc and ml-5 to remove bullets and indentation
        return `<ul class="space-y-2 list-none p-0 m-0">
            ${items.map(item => `<li class="text-sm" style="list-style: none;">${item.strip ? item.strip() : item}</li>`).join('')}
        </ul>`;
    }
    
    return `<p class="text-sm">${content.toString().replace(/\n/g, '<br>')}</p>`;
}

    function getConfidenceScore(issue) {
        if (!latestData || !latestData.confidence_scores) return 0;
        
        // Prioritize confidence_scores from the API response
        const score = latestData.confidence_scores[issue];

        if (typeof score === 'number' && !isNaN(score) && isFinite(score)) {
            return Math.max(0, Math.min(100, Math.round(score)));
        }
        return 0; // Default if not found or invalid
    }

    // ----------------------------------------------------------------------
    // 3. --- QUESTION FLOW LOGIC ---
    // ----------------------------------------------------------------------

    // Helper function to save the current question's answer to sessionStorage
    function saveCurrentAnswer(question, index) {
        const uniqueId = `q_${index}_${question.symptom_key}`;
        const questionContainer = document.getElementById('question-flow-container');
        if (!questionContainer) return;

        // Use the querySelector on the current container to find the checked radio button
        const answer = questionContainer.querySelector(`input[name="${uniqueId}"]:checked`)?.value;
        if (answer) {
            sessionStorage.setItem(uniqueId, answer);
        }
    }

    // Helper function to check if all questions are answered
    function checkFormValidity(questions, submitBtn) {
        let answeredCount = 0;
        const requiredQuestions = questions.length;
        
        questions.forEach((q, index) => {
            const uniqueId = `q_${index}_${q.symptom_key}`;
            const answer = sessionStorage.getItem(uniqueId); 
            if (answer === 'yes' || answer === 'no') { // Check for explicit 'yes' or 'no'
                answeredCount++;
            }
        });
        
        const isComplete = answeredCount === requiredQuestions;
        if(submitBtn) submitBtn.disabled = !isComplete;
        
        return isComplete;
    }

    

    // ----------------------------------------------------------------------
    // 4. --- REMEDY & UI RENDERING LOGIC ---
    // ----------------------------------------------------------------------
function renderRemedyCardHTML(issueKey, remedy, remedyIndex, totalRemedies, data) {
    const directions = Array.isArray(remedy.directions)
        ? remedy.directions
        : (remedy.directions ? [remedy.directions] : []);

    const isPrevDisabled = remedyIndex === 0 ? 'disabled' : '';
    const isNextDisabled = remedyIndex === totalRemedies - 1 ? 'disabled' : '';

    const styleId = 'remedy-card-final-alignment-css';
    if (!document.getElementById(styleId)) {
        const style = document.createElement('style');
        style.id = styleId;
        style.innerHTML = `
            .viewport-safety-wrapper { width: 100%; padding: 0; box-sizing: border-box; display: flex; justify-content: center; }
            .herbal-remedy-card { width: 100%; max-width: 650px; /* Wider card for larger text */ background: #FFF8F0; border: 1px solid #FFE0B2; border-radius: 24px; overflow: hidden; box-shadow: 0 12px 30px rgba(0,0,0,0.08); }
            .remedy-body { padding: clamp(20px, 5vw, 40px); }
            
            /* Fluid Typography: Scales from 1rem (mobile) to 1.5rem (large desktop) */
            .direction-text { 
                font-size: clamp(1rem, 1.2vw + 0.8rem, 1.5rem); 
                line-height: 1.7;
                color: #4e342e;
            }

            .direction-row { display: flex; gap: 15px; margin-bottom: 15px; align-items: flex-start; }
            .direction-num { color: #068901; font-weight: 900; min-width: 28px; font-size: 1.1em; }
            
            .remedy-title-text {
                font-size: clamp(1.5rem, 3vw, 2.2rem) !important;
                margin-bottom: 20px !important;
            }

            @media (max-width: 500px) {
                .remedy-body { padding: 20px; }
                .viewport-safety-wrapper { padding: 0 10px; }
                .nav-text { display: none; }
            }
        `;
        document.head.appendChild(style);
    }

    return `
        <div class="viewport-safety-wrapper">
            <div class="herbal-remedy-card">
                <div class="remedy-body">
                    <div style="color: #068901; font-weight: 800; font-size: 1.2rem; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px;">Recommended Remedy 🌿</div>
                    <div class="remedy-title-text" style="color: #FF7043; font-weight: 800; line-height: 1.2;">${remedy.title}</div>
                    
                    <div class="direction-text">
                        ${remedy.amount ? `<p style="margin-bottom: 15px; background: #fff3e0; padding: 10px 15px; border-radius: 10px; border-left: 4px solid #FF9800;"><strong>Amount:</strong> ${remedy.amount}</p>` : ''}
                        <div style="font-weight: bold; color: #068901; margin-bottom: 12px; font-size: 1.2em;">How to apply:</div>
                        <div style="margin-bottom: 20px;">
                            ${directions.map((step, i) => `
                                <div class="direction-row">
                                    
                                    <span style="flex: 1;">${step}</span>
                                </div>
                            `).join('')}
                        </div>
                    </div>

                    <div style="width: 100%; height: 250px; border-radius: 15px; overflow: hidden; border: 2px solid #FFE0B2; background: #eee; margin-top: 10px;">
                        ${remedy.image_url ? 
                            `<img src="${remedy.image_url}" style="width: 100%; height: 100%; object-fit: cover;">` : 
                            `<div style="height:100%; display:flex; align-items:center; justify-content:center; color:#999;">No Image Available</div>`}
                    </div>
                </div>

                <div style="display: flex; justify-content: space-between; align-items: center; padding: 20px 30px; background: #FFE0B2; border-top: 1px solid #FFD180;">
                    <button onclick="changeRemedy('${issueKey}', -1)" ${isPrevDisabled} style="background: #FF9800; color: white; border: none; padding: 12px 20px; border-radius: 12px; font-weight: bold; cursor: pointer; font-size: 1rem;">
                        ← <span class="nav-text">Previous</span>
                    </button>
                    <span style="font-weight: 800; color: #5d4037; font-size: 1.1rem;">${remedyIndex + 1} / ${totalRemedies}</span>
                    <button onclick="changeRemedy('${issueKey}', 1)" ${isNextDisabled} style="background: #FF9800; color: white; border: none; padding: 12px 20px; border-radius: 12px; font-weight: bold; cursor: pointer; font-size: 1rem;">
                        <span class="nav-text">Next Step</span> →
                    </button>
                </div>
            </div>
        </div>
    `;
}    // Expose changeRemedy globally so it can be called from the dynamically generated HTML
        window.changeRemedy = function (issueKey, direction) {

    const remedies = remediesByIssueKey[issueKey];
    if (!remedies || remedies.length === 0) return;

    let currentIndex = remedyState[issueKey] ?? 0;
    let newIndex = currentIndex + direction;

    newIndex = Math.max(0, Math.min(newIndex, remedies.length - 1));
    remedyState[issueKey] = newIndex;

    const wrapper = document.getElementById(`remedy-card-wrapper-${issueKey}`);
    if (!wrapper) return;

    wrapper.innerHTML = renderRemedyCardHTML(
        issueKey,
        remedies[newIndex],
        newIndex,
        remedies.length
    );
};


function renderIssueCardHTML(index, rawName, issueData, confidence) { 
    const formattedIssueName = formatConditionName(rawName); 
    const confidenceDisplay = confidence > 0 ? `${confidence}%` : 'Detected'; 

    const causesId = `causes-section-${index}`;
    const symptomsId = `symptoms-section-${index}`;

    const causesHtml = formatListContent(issueData.causes);
    const symptomsHtml = formatListContent(issueData.symptoms);

    return ` 
        <div class="card-viewport-container" style="width: 100%; padding: 0 10px; box-sizing: border-box; display: flex; justify-content: center;">
            <div class="card herbal-issue-card shadow-lg mb-4" style="width: 100%; max-width: 500px; border-radius: 20px; border: 1px solid #e0e0e0; background: #fff; overflow: hidden; box-sizing: border-box;">
                <div class="herbal-card-body" style="padding: 20px; box-sizing: border-box;">  
                    <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 10px; margin-bottom: 15px;">
                        <h3 style="margin: 0; color: #4A8E1D; font-size: 2rem; font-weight: 800; display: flex; align-items: center; gap: 8px; flex: 1; min-width: 150px;">  
                            <i class="fas fa-exclamation-triangle" style="color: #064e3b;"></i> 
                            <span>${formattedIssueName}</span>
                        </h3>  
                        
                        <span style="background-color: #4CAF50; color: white; padding: 5px 12px; border-radius: 20px; font-size: 1.2rem; font-weight: 700; white-space: nowrap;">  
                            ${confidenceDisplay} Match
                        </span>  
                    </div>

                    <div class="herbal-issue-buttons" style="display: flex; flex-wrap: wrap; gap: 10px; margin-top: 15px;">  
                        <button class="toggle-causes-btn" 
                                style="flex: 1; min-width: 130px; padding: 10px; font-size: 0.9rem; border-radius: 10px; cursor: pointer; 
                                       background-color: #FFF3E0; border: 1.5px solid #FF7043; color: #BF360C; 
                                       font-weight: 700; display: flex; align-items: center; justify-content: center; gap: 8px;"
                                onclick="document.getElementById('${causesId}').classList.toggle('hidden')">
                            <i class="fas fa-magnifying-glass"></i> Causes
                        </button>  
                        
                        <button class="toggle-symptoms-btn" 
                                style="flex: 1; min-width: 130px; padding: 10px; font-size: 0.9rem; border-radius: 10px; cursor: pointer; 
                                       background-color: #F0FDF4; border: 1.5px solid #4CAF50; color: #166534; 
                                       font-weight: 700; display: flex; align-items: center; justify-content: center; gap: 8px;"
                                onclick="document.getElementById('${symptomsId}').classList.toggle('hidden')">
                            <i class="fas fa-stethoscope"></i> Symptoms
                        </button>  
                    </div>

                    <div class="mt-3">
                        <div id="${causesId}" class="hidden rounded-lg mb-2" style="background-color: #FFF3E0; border-left: 5px solid #FF7043; padding: 12px; box-sizing: border-box;">
                            <p style="font-size: 0.75rem; font-weight: 900; color: #BF360C; text-transform: uppercase; margin-bottom: 5px;">Possible Causes</p>
                            <div style="font-size: 0.85rem; color: #BF360C; line-height: 1.4;">${causesHtml}</div>
                        </div>
                        
                        <div id="${symptomsId}" class="hidden rounded-lg" style="background-color: #F0FDF4; border-left: 5px solid #4CAF50; padding: 12px; box-sizing: border-box;">
                            <p style="font-size: 0.75rem; font-weight: 900; color: #166534; text-transform: uppercase; margin-bottom: 5px;">Common Symptoms</p>
                            <div style="font-size: 0.85rem; color: #166534; line-height: 1.4;">${symptomsHtml}</div>
                        </div>
                    </div>
                </div>  
            </div>
        </div>
    `; 
}
function attachToggleButtonListeners() {
        // Detach old listeners to prevent memory leaks/duplicate calls
        document.querySelectorAll('.toggle-causes-btn, .toggle-symptoms-btn').forEach(button => {
             button.removeEventListener('click', handleToggleButtonClick);
        });

        function handleToggleButtonClick(event) {
            const button = event.currentTarget;
            const targetId = button.getAttribute('data-target');
            const content = document.getElementById(targetId);

            if (content) {
                // Toggle the 'hidden' class for visibility
                content.classList.toggle('hidden');
                button.classList.toggle('active');

                // Update button text
                const isCauses = button.classList.contains('toggle-causes-btn');
                const isViewing = !content.classList.contains('hidden'); // Check the final state

                if (isCauses) {
                    button.textContent = isViewing ? 'Hide Causes' : 'View Causes';
                } else {
                    button.textContent = isViewing ? 'Hide Symptoms' : 'View Symptoms';
                }
            }
        }

        document.querySelectorAll('.toggle-causes-btn, .toggle-symptoms-btn').forEach(button => {
            button.addEventListener('click', handleToggleButtonClick);
        });
    }
    function findRemedyData(issue, remediesData = {}) {
    if (!remediesData) return {};

    // Exact match
    if (remediesData[issue]) return remediesData[issue];

    // Case-insensitive match
    const matchKey = Object.keys(remediesData).find(
        key => key.toLowerCase() === issue.toLowerCase()
    );

    return remediesData[matchKey] || {};
}
// ------------------------------
// 1. Normalize strings for matching
// ------------------------------
function normalizeStr(str) {
    if (!str) return '';
    return str.toLowerCase().replace(/[\s_-]+/g, '');
}

function getFilteredIssues(data) {
    const answeredYesConditions = new Set();

    followUpQuestions.forEach((q, i) => {
        const answer = sessionStorage.getItem(`q_${i}_${q.symptom_key}`);
        if (answer === 'yes') {
            answeredYesConditions.add(normalizeStr(q.condition));
        }
    });

    // Debug: log everything
    console.log("Answered Yes conditions:", Array.from(answeredYesConditions));
    console.log("Detected issues from API:", data.detected_issues);

    return (data.detected_issues || []).filter(issue => {
        const normIssue = normalizeStr(issue);
        const match = answeredYesConditions.has(normIssue);
        console.log(`Issue "${issue}" normalized="${normIssue}" match=${match}`);
        return match;
    });
}
// ------------------------------
// BACK BUTTON LOGIC
// ------------------------------

// Global reference to the remedies grid container
const grid = document.getElementById('herbalDetailedRemediesGrid');

// Function to create back button
function createBackButton(onClick) {
    const backBtn = document.createElement('button');
    backBtn.textContent = '← Back';
    backBtn.style.marginBottom = '20px';
    backBtn.style.padding = '10px 18px';
    backBtn.style.borderRadius = '12px';
    backBtn.style.border = '1px solid #e2e8f0';
    backBtn.style.background = '#ffffff';
    backBtn.style.cursor = 'pointer';
    backBtn.style.fontWeight = '600';
    backBtn.style.boxShadow = '0 3px 8px rgba(0,0,0,0.08)';
    backBtn.style.fontSize = '14px';
    backBtn.style.transition = 'all 0.2s ease';

    backBtn.onmouseover = () => backBtn.style.background = '#f7fafc';
    backBtn.onmouseout = () => backBtn.style.background = '#ffffff';

    backBtn.addEventListener('click', onClick);
    return backBtn;
}

// Function to show content with back button without overwriting your cards
function showWithBack(renderFn, data) {
    // Clear grid
    grid.innerHTML = '';

    // Create a container to hold Back button and content
    const container = document.createElement('div');
    container.style.display = 'flex';
    container.style.flexDirection = 'column';
    container.style.gap = '20px';

    // Add Back button on top
    const backBtn = createBackButton(() => showRemediesStep(data));
    container.appendChild(backBtn);

    // Create content wrapper for your existing render function
    const contentWrapper = document.createElement('div');
    container.appendChild(contentWrapper);

    // Call your existing render function, passing wrapper as container
    renderFn(contentWrapper);

    // Append everything to grid
    grid.appendChild(container);
    showElement(grid);
}


// 1. Mapping of Condition Names (from your Model) to URL Slugs (for Django)
const conditionSlugMap = {
    "Acne": "acne",
    "Rosacea": "rosacea",
    "Dark Circles": "dark-circles",
    "Pigmentation": "pigmentation",
    "Wrinkles": "wrinkles",
    "Blackheads": "blackheads",
    "Sun Spots": "sun-spots",
    "Eye Bags": "eye-bags",
    "Freckles": "freckles",
    "Skin Cancer": "skin-cancer",
    "Psoriasis": "psoriasis",
    "Eczema": "eczema",
    "Shingles": "shingles",
    "Warts": "warts",
    "Hives": "hives",
    "Chicken Pox": "chicken-pox"
};
/**
 * Renders the Treatment Path selection step.
 * Features: 
 * - Side-by-side row on Desktop
 * - Vertical stack on Mobile
 * - Perfect horizontal centering
 */
function showRemediesStep(data) {
    const herbalDetailedRemediesGrid = document.getElementById('herbalDetailedRemediesGrid');
    herbalDetailedRemediesGrid.innerHTML = '';
    herbalDetailedRemediesGrid.style.display = 'block';
    herbalDetailedRemediesGrid.style.width = '100%';

    const validIssues = getFilteredIssues(data);
    if (!validIssues || !validIssues.length) return;

    const rawIssue = validIssues[0];
    const conditionName = typeof rawIssue === 'string' ? rawIssue : (rawIssue.condition || rawIssue.name);

    // 1. Inject Premium CSS
    const styleId = 'treatment-selection-premium-v4';
    if (document.getElementById(styleId)) document.getElementById(styleId).remove();

    const style = document.createElement('style');
    style.id = styleId;
    style.innerHTML = `
        .treatment-selection-wrapper {
            width: 100%;
            margin: 40px auto;
            padding: 20px;
            box-sizing: border-box;
            display: flex;
            flex-direction: column;
            align-items: center; /* Centers everything horizontally */
        }

        .treatment-header {
            text-align: center;
            margin-bottom: 40px;
            width: 100%; /* Ensures header spans full width to allow centering */
        }

        .treatment-header h2 {
            font-size: 2.5rem;
            color: #1a365d;
            font-weight: 800;
            margin: 0 auto 15px auto;
            display: block;
        }

        .treatment-header p {
            font-size: 2rem; 
            color: #718096;
            margin: 0 auto;
            max-width: 600px;
        }

        /* Desktop Row: Balanced spacing */
        .treatment-cards-container {
            display: flex !important;
            flex-direction: row !important;
            justify-content: center !important; 
            gap: 25px !important;
            width: 100% !important;
            flex-wrap: nowrap !important;
        }

        .treatment-card {
            background: #ffffff !important;
            border: 1px solid #e2e8f0 !important;
            border-radius: 24px !important;
            padding: 40px 25px !important;
            /* WIDTH ADJUSTMENT: 280px provides better text balance than 240px */
            width: 260px !important;
            min-width: 240px !important; 
            display: flex !important;
            flex-direction: column !important;
            align-items: center !important;
            text-align: center !important;
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
            cursor: pointer !important;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05) !important;
            box-sizing: border-box !important;
        }

        .treatment-card:hover {
            transform: translateY(-10px) scale(1.02) !important;
            border-color: #e1c142 !important; /* Your specific gold-ish color */
            box-shadow: 0 20px 30px rgba(225, 193, 66, 0.2) !important;
        }

        .card-icon {
            font-size: 3.5rem !important;
            margin-bottom: 25px !important;
        }

        .treatment-card h3 {
            font-size: 2rem !important;
            color: #2d3748 !important;
            font-weight: 700 !important;
            margin-bottom: 12px !important;
        }

        .treatment-card p {
            font-size: 1.6rem !important;
            color: #4a5568 !important;
            line-height: 1.6 !important;
            margin: 0 !important;
        }

        /* Mobile Adjustments */
        @media (max-width: 1000px) {
            .treatment-cards-container {
                flex-direction: column !important;
                align-items: center !important;
                gap: 20px !important;
            }
            .treatment-card {
                width: 90% !important;
                max-width: 400px !important;
                min-width: unset !important;
            }
            .treatment-header h2 { font-size: 2rem; }
        }
    `;
    document.head.appendChild(style);

    // 2. Build UI
    const mainWrapper = document.createElement('div');
    mainWrapper.className = 'treatment-selection-wrapper';

    mainWrapper.innerHTML = `
        <div class="treatment-header">
            <h2>Choose Your Treatment Path</h2>
            <p>Select an option to explore personalized solutions for ${conditionName}</p>
        </div>
        <div class="treatment-cards-container" id="cardsWrapper"></div>
    `;

    const cardsWrapper = mainWrapper.querySelector('#cardsWrapper');

    // 3. Card Creation Helper with Restored Pop-up Logic
    const createCard = (emoji, title, desc, type) => {
        const card = document.createElement('div');
        card.className = 'treatment-card';
        card.innerHTML = `
            <div class="card-icon">${emoji}</div>
            <h3>${title}</h3>
            <p>${desc}</p>
        `;
        
        card.onclick = () => {
            if (type === 'home') {
                if (!sessionStorage.getItem('homeDisclaimer')) {
                    sessionStorage.setItem('homeDisclaimer', 'true');
                    showMessageBox(HOME_REMEDY_DISCLAIMER, () => showWithBack(() => renderRemediesCards(data, 'home'), data));
                } else {
                    showWithBack(() => renderRemediesCards(data, 'home'), data);
                }
            } 
            else if (type === 'medical') {
                if (!sessionStorage.getItem('medicalDisclaimer')) {
                    sessionStorage.setItem('medicalDisclaimer', 'true');
                    showMessageBox(MEDICAL_TREATMENT_DISCLAIMER, () => showWithBack(() => renderRemediesCards(data, 'medical'), data));
                } else {
                    showWithBack(() => renderRemediesCards(data, 'medical'), data);
                }
            }
            else if (type === 'lifestyle') {
                showWithBack(wrapper => renderLifestylePlan(wrapper, [conditionName]), data);
            }
        };
        return card;
    };

    cardsWrapper.appendChild(createCard('🏠', 'Home Remedies', 'Natural remedies using everyday ingredients to soothe and support your skin safely.', 'home'));
    cardsWrapper.appendChild(createCard('💊', 'Medical Treatments', 'Evidence-based medical options for managing your condition under supervision.', 'medical'));
    cardsWrapper.appendChild(createCard('🧘‍♀️', 'Lifestyle Plan', 'A personalized daily routine covering diet, skincare, and healthy habits.', 'lifestyle'));

    herbalDetailedRemediesGrid.appendChild(mainWrapper);
}

function renderLifestylePlan(wrapper, conditionNames) {
    const planContainer = document.createElement('div');
    planContainer.style.cssText = "padding: 40px; background: white; border-radius: 12px; text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.05);";

    planContainer.innerHTML = `
        <h2 style="color:#2d3748; margin-bottom: 15px;">Personalized Lifestyle Plan</h2>
        <p style="font-size: 1.5rem; color: #4a5568;">Ready to download your custom guide for: <br><strong>${conditionNames.join(', ')}</strong></p>
        <button id="downloadPlanBtn" style="background-color: #198526ff; color: white; padding: 14px 28px; border-radius: 10px; border: none; font-weight: bold; cursor: pointer; margin-top: 25px; font-size: 1rem;">
            Download PDF Report
        </button>
    `

    wrapper.appendChild(planContainer);

    const downloadBtn = planContainer.querySelector('#downloadPlanBtn');
    downloadBtn.addEventListener('click', () => {
    // 1. Create a lowercase version of your map for safety
    const safeMap = {};
    Object.keys(conditionSlugMap).forEach(key => {
        safeMap[key.toLowerCase().trim()] = conditionSlugMap[key];
    });

    // 2. Map the names safely
    const slugs = conditionNames
        .map(name => {
            if (!name) return null;
            const cleanName = name.toString().toLowerCase().trim();
            return safeMap[cleanName];
        })
        .filter(Boolean); // Removes nulls or undefined

    // Debugging: Check the console to see what is failing
    console.log("Input Names:", conditionNames);
    console.log("Mapped Slugs:", slugs);

    if (slugs.length === 0) {
        alert(`Error: Could not find a plan for "${conditionNames.join(', ')}". Check if the name matches the slug map.`);
        return;
    }

    const query = slugs.join(',');
    const url = `/download-personalized-plan/?conditions=${encodeURIComponent(query)}`;
    window.open(url, '_blank');
});
}
// ------------------------------
// 4. Render remedies cards (no change to most code)
// ------------------------------
function renderRemediesCards(data, type) {
    herbalDetailedRemediesGrid.innerHTML = ''; // Clear option screen

    const validIssues = getFilteredIssues(data);
    if (validIssues.length === 0) return;

    validIssues.forEach((issue, index) => {
        const issueKey = makeSafeKey(issue);
        const issueData = findRemedyData(issue, data.remedies_data);

        const remedies =
            type === 'home' ? (issueData.home_remedies || []) :
            type === 'medical' ? (issueData.medical_remedies || []) :
            (issueData.remedies || []);

        // Store index and remedies globally
        remedyState[`${issueKey}_${type}`] = 0;
        remediesByIssueKey[`${issueKey}_${type}`] = remedies;

        const container = document.createElement('div');
        container.className = 'issue-remedy-pair grid md:grid-cols-2 gap-6 mb-8';

        // Issue card
        container.innerHTML = renderIssueCardHTML(index, issue, issueData, getConfidenceScore(issue));

        // Remedies card
        const remedyWrapper = document.createElement('div');
        remedyWrapper.id = `remedy-card-wrapper-${issueKey}_${type}`;
        remedyWrapper.style.display = 'flex';
        remedyWrapper.style.flexDirection = 'column';

        if (remedies.length > 0) {
            remedyWrapper.innerHTML = renderRemedyCardHTML(`${issueKey}_${type}`, remedies[0], 0, remedies.length);
        } else {
            remedyWrapper.innerHTML = `
                <div class="p-6 text-center border-2 rounded-xl shadow-md bg-gray-50 text-gray-500 italic">
                    No ${type === 'home' ? 'home' : 'medical'} remedies found for this condition.
                </div>
            `;
        }

        container.appendChild(remedyWrapper);
        herbalDetailedRemediesGrid.appendChild(container);
    });

    setTimeout(attachToggleButtonListeners, 100);
    showElement(herbalDetailedRemediesGrid);
}


    // ----------------------------------------------------------------------
    // 5. --- FLOW CONTROLLERS ---
    // ----------------------------------------------------------------------
    // Function to render a single question card or the review step


// --- RENDER SINGLE QUESTION OR REVIEW ---
function renderQuestion(index) {
    const questions = followUpQuestions;
    const container = document.getElementById('question-flow-container');
    const submitBtn = document.getElementById('submitFollowupBtn');
    if (!container) return;

    container.innerHTML = '';

    if (index >= 0 && index < questions.length) {
        currentQuestionIndex = index;
        const q = questions[index];
        const uniqueId = `q_${index}_${q.symptom_key}`;

        const questionHtml = `
            <div class="question-item active-question" data-question-index="${index}">
                <p class="question-title"><strong>${q.condition} (${index + 1} of ${questions.length}):</strong> ${q.question}</p>
                <div class="options-group" data-symptom-key="${q.symptom_key}">
                    <input type="radio" id="${uniqueId}_yes" name="${uniqueId}" value="yes" required>
                    <label for="${uniqueId}_yes" class="radio-label yes-option"><span class="radio-icon">✓</span> Yes</label>

                    <input type="radio" id="${uniqueId}_no" name="${uniqueId}" value="no" required>
                    <label for="${uniqueId}_no" class="radio-label no-option"><span class="radio-icon">✗</span> No</label>
                </div>
            </div>
        `;

        container.insertAdjacentHTML('beforeend', questionHtml);

        // Restore previous answer
        const prev = sessionStorage.getItem(uniqueId);
        if (prev) {
            const input = container.querySelector(`input[name="${uniqueId}"][value="${prev}"]`);
            if (input) input.checked = true;
        }

        // Auto-advance on change
        const inputs = container.querySelectorAll('input[type="radio"]');
        inputs.forEach(input => {
            input.addEventListener('change', () => {
                sessionStorage.setItem(uniqueId, input.value);

                setTimeout(() => {
                    if (currentQuestionIndex < questions.length - 1) {
                        renderQuestion(currentQuestionIndex + 1);
                    } else {
                        renderQuestion(questions.length); // Go to review
                    }
                }, 200);
            });
        });

        if (submitBtn) submitBtn.style.display = 'none';
    }

    else if (index === questions.length) {
        // Review / submit step
        currentQuestionIndex = index;
        const reviewHtml = questions.map((q, i) => {
            const uniqueId = `q_${i}_${q.symptom_key}`;
            const answer = sessionStorage.getItem(uniqueId) || 'Unanswered';
            const icon = answer === 'yes' ? '✅' : (answer === 'no' ? '❌' : '❓');
            return `
                <div class="review-item">
                    <p><strong>${q.condition}:</strong> ${q.question}</p>
                    <p>${icon} Your Answer: ${answer.toUpperCase()}</p>
                </div>
            `;
        }).join('');

        container.innerHTML = `
            <div class="final-step-message card p-6">
                <h4>✅ Review Your Answers</h4>
                <div class="review-summary-container">${reviewHtml}</div>
            </div>
        `;

        if (submitBtn) submitBtn.style.display = 'block';
    }
}

// --- GENERATE FOLLOW-UP QUESTIONS ---
async function generateFollowupQuestions(data) {
    currentQuestionIndex = 0;
    sessionStorage.clear();

    // Normalize follow-up questions
    followUpQuestions = (data.follow_up_questions || []).map((q, i) => {
        let questionText = '';

        if (typeof q.question === 'string' && q.question.trim()) questionText = q.question;
        else if (Array.isArray(q.question)) questionText = q.question.filter(s => s.trim()).join(', ');
        else if (typeof q.question === 'object' && q.question !== null) {
            questionText = Object.values(q.question)
                .filter(val => typeof val === 'string' && val.trim())
                .join(', ');
        }

        // Friendly default if question is missing
        if (!questionText) questionText = `Do you have ${q.condition || 'this symptom'}?`;

        let symptomKey = '';
        if (typeof q.symptom_key === 'string') symptomKey = q.symptom_key;
        else if (Array.isArray(q.symptom_key)) symptomKey = q.symptom_key.join(',');
        else if (typeof q.symptom_key === 'object' && q.symptom_key !== null) symptomKey = Object.keys(q.symptom_key).join(',');

        if (!symptomKey) symptomKey = `q${i}`;

        return {
            index: i,
            condition: String(q.condition || 'Unknown'),
            question: questionText,
            symptom_key: symptomKey
        };
    });

    if (followUpQuestions.length === 0) {
        showRemediesStep(data);
        return;
    }

    herbalDetailedRemediesGrid.innerHTML = `
        <div class="followup-section-header p-4 mb-4 border-b">
            <h3 class="text-2xl font-bold text-green-800">Refine Your Diagnosis 🌿</h3>
            <p class="text-gray-700">Please answer the following questions to suggest the most relevant remedies:</p>
        </div>
        <form id="followup-questions-form">
            <div id="question-flow-container" class="space-y-4"></div>
        </form>
        <button id="submitFollowupBtn" class="herbal-btn herbal-btn-primary mt-6 w-full" disabled>
            Submit Answers & Get Remedies
        </button>
    `;

    renderQuestion(currentQuestionIndex);

    const submitBtn = document.getElementById('submitFollowupBtn');

    document.getElementById('followup-questions-form').addEventListener('change', () => {
        checkFormValidity(followUpQuestions, submitBtn);
    });

    submitBtn.addEventListener('click', async () => {
    if (!checkFormValidity(followUpQuestions, submitBtn)) return showMessageBox('Please answer all questions.');
    if (!globalFileToSend) return showMessageBox('Image missing. Please upload again.');

    const answeredSymptoms = followUpQuestions
        .filter((q, i) => sessionStorage.getItem(`q_${i}_${q.symptom_key}`) === 'yes')
        .map(q => q.symptom_key);

    const formData = new FormData();
    formData.append('file', globalFileToSend);
    formData.append('data', JSON.stringify({ symptoms: answeredSymptoms, answers_submitted: true }));

    try {
        showSpinner();
        const response = await fetch('/api/predict/', { method: 'POST', body: formData });
        const result = await response.json();
        hideSpinner();

        console.log('Follow-up API result:', result); // 🔹 Add this line to debug

        if (result.status === 'success') showRemediesStep(result);
        else showMessageBox(result.error || 'Prediction failed.');
    } catch (err) {
        hideSpinner();
        console.error(err);
        showMessageBox('Error submitting answers.');
    }
});


    showElement(herbalDetailedRemediesGrid);
}


    function renderAnalysisSummary(data) {
        console.log("API Response Data:", JSON.stringify(data, null, 2));

        latestData = data;
        // ✅ SHOW "SEND REPORT TO DOCTOR" CARD AFTER PREDICTION
        const sendReportCard = document.getElementById("sendReportCard");
        if (sendReportCard) {
            sendReportCard.style.display = "block";
        }


        // --- A. Image Preview and Bounding Boxes ---
        if (data.annotated_image) {
            preview.src = 'data:image/jpeg;base64,' + data.annotated_image;
            showElement(preview);
            
            // Wait for image to load to correctly size the canvas
            preview.onload = () => {
                // Resize canvas to match the preview image's rendered size
                canvas.width = preview.clientWidth;
                canvas.height = preview.clientHeight;
                canvas.style.width = preview.clientWidth + 'px';
                canvas.style.height = preview.clientHeight + 'px';
                
                // Draw boxes if present
                if (data.all_boxes && data.all_boxes.length > 0) { // Using all_boxes from the server response
                    drawBoundingBoxes(data.all_boxes);
                } else {
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                    hideElement(canvas);
                }
            };
            // Handle case where image is already cached/loaded
            if (preview.complete) {
                 preview.onload();
            }
        } else {
            hideElement(preview);
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            hideElement(canvas);
        }

        // --- B. Analysis Summary & Probability Distribution ---
        const existingDistribution = diagnosisConfidenceTextEl.parentNode.querySelector('.probability-distribution');
        if (existingDistribution) {
            existingDistribution.remove();
        }

        if (data.detected_issues?.length > 0) {
            const issuesList = data.detected_issues
                .map(issue => ({
                    name: issue,
                    confidence: getConfidenceScore(issue),
                    formattedName: formatConditionName(issue)
                }))
                .filter(issue => {
                    const issueName = issue.name ? issue.name.trim() : null;
                    return issueName && !/^\d+$/.test(issueName) && issue.formattedName !== 'Unclassified Issue';
                })
                .sort((a, b) => b.confidence - a.confidence); // Sort by confidence descending

            const primaryIssue = issuesList[0] || {};
            diagnosisConditionNameEl.textContent = primaryIssue.formattedName || 'Condition Detected';
            
            // Main Confidence Bar Update
            const mainConfidence = primaryIssue.confidence !== null ? primaryIssue.confidence : 0;
            diagnosisConfidenceLevelEl.style.width = `${mainConfidence}%`;
            diagnosisConfidenceTextEl.textContent = `Confidence: ${mainConfidence}%`;

            const validScoredIssues = issuesList.filter(issue => issue.confidence > 0);

            if (validScoredIssues.length > 0) {
                const distributionHtml = `
                    <div class="probability-distribution mt-4 p-4 border rounded-lg bg-gray-50">
                        <div class="probability-header mb-3">
                            <h4 class="text-lg font-semibold text-gray-800">Model Analysis</h4>
                            <p class="text-sm text-gray-600">Detection confidence levels:</p>
                        </div>
                        <div class="probability-bars space-y-2">
                            ${validScoredIssues.map(issue => `
                                <div class="probability-item">
                                    <div class="probability-label text-sm font-medium">${issue.formattedName}</div>
                                    <div class="probability-bar-container bg-gray-200 h-2.5 rounded-full overflow-hidden">
                                        <div class="probability-bar bg-green-500 h-2.5" style="width: ${issue.confidence}%">
                                        </div>
                                        <span class="probability-value text-xs ml-2">${issue.confidence}%</span>
                                    </div>
                                </div>
                            `).join('')}
                        </div>
                    </div>
                `;
                diagnosisConfidenceTextEl.insertAdjacentHTML('afterend', distributionHtml);
            }

            // --- C. Decide Next Step (Questions or Remedies) ---
            if (data.follow_up_questions && data.follow_up_questions.length > 0) {
                generateFollowupQuestions(data); 
            } else {
                showRemediesStep(data);
            }

        } else {
            // No issues detected
            diagnosisConditionNameEl.textContent = 'No Specific Condition Detected';
            diagnosisConfidenceLevelEl.style.width = '0%';
            diagnosisConfidenceTextEl.textContent = 'Confidence: 0%';

            herbalDetailedRemediesGrid.innerHTML = `
                <div class="no-results-message text-center p-6 bg-green-50 rounded-lg border-l-4 border-green-500">
                    <p class="text-gray-700">No specific skin conditions were detected. You have healthy skin! Continue to maintain it.</p>
                </div>
            `;
        }
        
        // --- D. Progress Summary Display (New addition/improvement) ---
        const progressEl = document.getElementById('progressSummaryContainer');
        if (progressEl && data.progress_summary) {
            let summaryHtml = `<div class="p-4 bg-blue-100 rounded-lg shadow-sm">
                                   <h4 class="text-lg font-semibold text-blue-800 mb-2">Progress Check 📈</h4>
                                   <p class="text-sm text-blue-700">${data.progress_summary.status}</p>`;

            if (data.progress_summary.improvements) {
                summaryHtml += `<h5 class="font-medium mt-2 text-green-700">Improvements:</h5><ul class="list-disc ml-5 text-sm text-green-800">`;
                for (const [issue, message] of Object.entries(data.progress_summary.improvements)) {
                    summaryHtml += `<li>${message}</li>`;
                }
                summaryHtml += `</ul>`;
            }
            if (data.progress_summary.regressions) {
                summaryHtml += `<h5 class="font-medium mt-2 text-red-700">Regressions:</h5><ul class="list-disc ml-5 text-sm text-red-800">`;
                for (const [issue, message] of Object.entries(data.progress_summary.regressions)) {
                    summaryHtml += `<li>${message}</li>`;
                }
                summaryHtml += `</ul>`;
            }
            summaryHtml += `</div>`;
            progressEl.innerHTML = summaryHtml;
        }

        showElement(resultsSection);
    }
    
    // --- Bounding Box Drawing (Uses all_boxes from API instead of generating client-side) ---
    function drawBoundingBoxes(all_boxes) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        if (!all_boxes || all_boxes.length === 0) {
            hideElement(canvas);
            return;
        }

        showElement(canvas);

        ctx.strokeStyle = 'red';
        ctx.lineWidth = 3;
        ctx.font = '24px Arial'; // Increased font size for better visibility

        // Scale factors relative to the natural size of the image vs the displayed size
        const scaleX = preview.clientWidth / preview.naturalWidth;
        const scaleY = preview.clientHeight / preview.naturalHeight;

        // The API response all_boxes should contain: [x_min, y_min, x_max, y_max, label_name, score]
        all_boxes.forEach(box => {
            const [x_min, y_min, x_max, y_max, label_name, score] = box;
            
            // Calculate scaled coordinates and dimensions
            const x = x_min * scaleX;
            const y = y_min * scaleY;
            const width = (x_max - x_min) * scaleX;
            const height = (y_max - y_min) * scaleY;

            // Draw the box
            ctx.strokeRect(x, y, width, height);

            // Draw the label
            const text = `${formatConditionName(label_name)} (${Math.round(score)}%)`;
            const textY = y > 30 ? y - 10 : y + 20; // Position text above or below box
            
            ctx.fillStyle = 'red';
            ctx.fillRect(x, textY - 20, ctx.measureText(text).width + 10, 30); // Background for text
            ctx.fillStyle = 'white';
            ctx.fillText(text, x + 5, textY);
        });
    }

    // ----------------------------------------------------------------------
    // 6. --- MAIN PREDICTION EXECUTION ---
    // ----------------------------------------------------------------------

    async function runPrediction(fileToSend, symptoms = []) {
        showSpinner();
        predictBtn.disabled = true;
        
        // If this is the *initial* call (no symptoms), clear questions and set file
        if (symptoms.length === 0) {
            globalFileToSend = fileToSend;
            sessionStorage.clear(); // Clear session storage for a new primary analysis
        }

        try {
            const formData = new FormData();
            formData.append('file', fileToSend);

            if (symptoms.length > 0) {
                // The 'data' payload for LR refinement
                formData.append('data', JSON.stringify({ symptoms: symptoms }));
                // Hide the question container during the refinement call
                hideElement(herbalDetailedRemediesGrid);
            }

            const response = await fetch('/api/predict/', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ message: response.statusText }));
                throw new Error(errorData.error || errorData.message || 'Server responded with an unknown error');
            }

            const data = await response.json();
            renderAnalysisSummary(data);

        } catch (error) {
            console.error('Error during prediction:', error);
            showMessageBox(`Analysis Error: ${error.message || 'Failed to process image'}`);
            
            // Reset UI state on error
            diagnosisConditionNameEl.textContent = 'Analysis Error';
            diagnosisConfidenceTextEl.textContent = `Error: ${error.message || 'Failed to process image'}`;
            hideElement(preview);
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            predictBtn.disabled = false; 
        } finally {
            hideSpinner();
            // Re-enable the main predict button only if we are not in the question flow
            if (!document.getElementById('submitFollowupBtn')) {
                predictBtn.disabled = false;
            }
        }
    }


    // ----------------------------------------------------------------------
    // 7. --- INITIALIZATION & EVENT LISTENERS ---
    // ----------------------------------------------------------------------
    
    // Initialize the custom message box HTML (better placement for the final version)
    
    
    // Add a container for progress summary, assuming it exists in HTML
    if (!document.getElementById('progressSummaryContainer')) {
        const resultsHeader = document.querySelector('#results h2') || resultsSection;
        const progressDiv = document.createElement('div');
        progressDiv.id = 'progressSummaryContainer';
        resultsHeader.insertAdjacentElement('afterend', progressDiv);
    }
    
    // Attach Listeners
    startCameraBtn.addEventListener('click', async () => {
        // Camera startup logic (unchanged)
        try {
            stream = await navigator.mediaDevices.getUserMedia({ video: true });
            video.srcObject = stream;
            video.play();
            showElement(video);
            captureBtn.disabled = false;
            hideElement(preview);
            hideElement(canvas);
            globalFileToSend = null;
        } catch (err) {
            console.error('Error accessing camera:', err);
            showMessageBox('Could not access the camera. Please check permissions and try again.');
        }
    });

    captureBtn.addEventListener('click', () => {
        // Image capture logic (unchanged)
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        preview.src = canvas.toDataURL('image/jpeg');
        showElement(preview);
        hideElement(video);
        hideElement(canvas);

        predictBtn.disabled = false;

        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            stream = null;
        }

        canvas.toBlob((blob) => {
            if (blob) {
                globalFileToSend = new File([blob], 'captured_image.jpeg', { type: 'image/jpeg' });
            } else {
                console.error("Failed to create blob from canvas.");
                globalFileToSend = null;
            }
        }, 'image/jpeg', 1.0);
    });

    fileInput.addEventListener('change', () => {
        // File input logic (unchanged)
        const file = fileInput.files[0];
        if (!file) return;

        globalFileToSend = file;

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

    predictBtn.addEventListener('click', async () => {
        // Main prediction trigger
        if (!globalFileToSend) {
            showMessageBox('No image selected or captured to analyze. Please capture an image or choose a file.');
            return;
        }
        
        diagnosisConditionNameEl.textContent = 'Analyzing...';
        diagnosisConfidenceLevelEl.style.width = '0%';
        diagnosisConfidenceTextEl.textContent = 'Confidence: --%';
        herbalDetailedRemediesGrid.innerHTML = '';
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        hideElement(canvas);
        
        await runPrediction(globalFileToSend, []);
    });
    
    
});