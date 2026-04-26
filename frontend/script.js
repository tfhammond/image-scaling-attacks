const METHOD_DEFAULTS = {
    bicubic: {
        lam: "0.25",
        eps: "0.0",
        gamma: "1.0",
        dark_frac: "0.3"
    },
    bilinear: {
        lam: "1.0",
        eps: "0.0",
        gamma: "0.9",
        dark_frac: "0.3",
        anti_alias: false
    },
    nearest: {
        lam: "0.25",
        eps: "0.0",
        gamma: "1.0",
        offset: "2"
    }
};

const state = {
    loading: false,
    method: "bicubic",
    decoyImageData: null,
    decoyName: "",
    decoyDimensions: null,
    previewResult: null,
    generationResult: null
};

const elements = {
    loading: document.getElementById("loading"),
    error: document.getElementById("error"),
    errorMessage: document.getElementById("errorMessage"),
    textInput: document.getElementById("textInput"),
    fontSizeInput: document.getElementById("fontSizeInput"),
    alignmentInput: document.getElementById("alignmentInput"),
    previewButton: document.getElementById("previewButton"),
    previewPanel: document.getElementById("previewPanel"),
    previewImage: document.getElementById("previewImage"),
    previewSize: document.getElementById("previewSize"),
    previewStatus: document.getElementById("previewStatus"),
    previewWarning: document.getElementById("previewWarning"),
    decoyInput: document.getElementById("decoyInput"),
    uploadStatus: document.getElementById("uploadStatus"),
    methodInput: document.getElementById("methodInput"),
    lamInput: document.getElementById("lamInput"),
    epsInput: document.getElementById("epsInput"),
    gammaInput: document.getElementById("gammaInput"),
    offsetField: document.getElementById("offsetField"),
    offsetInput: document.getElementById("offsetInput"),
    darkFracField: document.getElementById("darkFracField"),
    darkFracInput: document.getElementById("darkFracInput"),
    antiAliasField: document.getElementById("antiAliasField"),
    antiAliasInput: document.getElementById("antiAliasInput"),
    generateButton: document.getElementById("generateButton"),
    resultsPanel: document.getElementById("resultsPanel"),
    resultSummary: document.getElementById("resultSummary"),
    resultParameters: document.getElementById("resultParameters"),
    targetImage: document.getElementById("targetImage"),
    adversarialImage: document.getElementById("adversarialImage"),
    downsampledImage: document.getElementById("downsampledImage"),
    targetStage: document.getElementById("targetStage"),
    adversarialStage: document.getElementById("adversarialStage"),
    downsampledStage: document.getElementById("downsampledStage"),
    targetSize: document.getElementById("targetSize"),
    adversarialSize: document.getElementById("adversarialSize"),
    downsampledSize: document.getElementById("downsampledSize")
};

function setLoading(nextValue, message = "Working...") {
    state.loading = nextValue;
    elements.loading.hidden = !nextValue;
    elements.loading.textContent = message;
    elements.previewButton.disabled = nextValue;
    updateGenerateButtonState();
}

function showError(message) {
    elements.error.hidden = false;
    elements.errorMessage.textContent = message;
}

function clearError() {
    elements.error.hidden = true;
    elements.errorMessage.textContent = "";
}

function clearPreview() {
    state.previewResult = null;
    elements.previewPanel.hidden = true;
    elements.previewImage.removeAttribute("src");
    elements.previewWarning.hidden = true;
    elements.previewWarning.textContent = "";
    elements.previewStatus.textContent = "The backend will render the preview at the default size.";
    elements.previewSize.textContent = "1092x1092";
}

function clearResults() {
    state.generationResult = null;
    elements.resultsPanel.hidden = true;
    elements.targetImage.removeAttribute("src");
    elements.adversarialImage.removeAttribute("src");
    elements.downsampledImage.removeAttribute("src");
}

function resetDecoyState(statusText = "No image selected.") {
    state.decoyImageData = null;
    state.decoyName = "";
    state.decoyDimensions = null;
    elements.uploadStatus.textContent = statusText;
    updateGenerateButtonState();
}

function updateGenerateButtonState() {
    const hasText = elements.textInput.value.trim().length > 0;
    const hasDecoy = Boolean(state.decoyImageData);
    elements.generateButton.disabled = state.loading || !hasText || !hasDecoy;
}

function applyMethodDefaults(method) {
    const defaults = METHOD_DEFAULTS[method];
    elements.lamInput.value = defaults.lam;
    elements.epsInput.value = defaults.eps;
    elements.gammaInput.value = defaults.gamma;
    if (method === "nearest") {
        elements.offsetInput.value = defaults.offset;
    } else {
        elements.darkFracInput.value = defaults.dark_frac;
        elements.antiAliasInput.checked = Boolean(defaults.anti_alias);
    }
}

function syncMethodFields() {
    const method = elements.methodInput.value;
    state.method = method;
    elements.offsetField.hidden = method !== "nearest";
    elements.darkFracField.hidden = method !== "bicubic" && method !== "bilinear";
    elements.antiAliasField.hidden = method !== "bilinear";
}

function parseNumberInput(element, label) {
    const value = Number(element.value);
    if (Number.isNaN(value)) {
        throw new Error(`${label} must be a number.`);
    }
    return value;
}

function readFileAsDataURL(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = () => reject(new Error("Failed to read the selected file."));
        reader.readAsDataURL(file);
    });
}

function loadImage(dataUrl) {
    return new Promise((resolve, reject) => {
        const image = new Image();
        image.onload = () => resolve(image);
        image.onerror = () => reject(new Error("Failed to decode the selected image."));
        image.src = dataUrl;
    });
}

async function requestJson(path, payload) {
    const response = await fetch(path, {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify(payload)
    });

    const data = await response.json();
    if (!response.ok) {
        throw new Error(data.error || "Request failed.");
    }
    return data;
}

async function handlePreview() {
    clearError();
    clearResults();

    const text = elements.textInput.value.trim();
    if (!text) {
        showError("Text is required before generating a preview.");
        return;
    }

    try {
        setLoading(true, "Generating text preview...");
        const payload = {
            text,
            font_size: parseNumberInput(elements.fontSizeInput, "Font size"),
            alignment: elements.alignmentInput.value
        };
        const result = await requestJson("/preview-text", payload);
        state.previewResult = result;

        elements.previewImage.src = result.image;
        elements.previewSize.textContent = result.size;
        elements.previewStatus.textContent = `Preview ready at ${result.size}.`;
        if (result.text_overflowed) {
            elements.previewWarning.hidden = false;
            elements.previewWarning.textContent = "Text overflowed the preview bounds and may be cut off.";
        } else {
            elements.previewWarning.hidden = true;
            elements.previewWarning.textContent = "";
        }
        elements.previewPanel.hidden = false;
    } catch (error) {
        showError(error.message);
    } finally {
        setLoading(false);
    }
}

async function handleDecoyUpload(event) {
    clearError();
    clearResults();

    const file = event.target.files[0];
    if (!file) {
        resetDecoyState();
        return;
    }

    if (file.type !== "image/png" && !file.name.toLowerCase().endsWith(".png")) {
        resetDecoyState("No image selected.");
        showError("Please choose a PNG image file.");
        event.target.value = "";
        return;
    }

    try {
        const dataUrl = await readFileAsDataURL(file);
        const image = await loadImage(dataUrl);

        if (image.width !== image.height) {
            throw new Error(`Decoy image must be square. Got ${image.width}x${image.height}.`);
        }
        if (image.width % 4 !== 0) {
            throw new Error(
                `Decoy image size must be divisible by 4. Got ${image.width}x${image.height}.`
            );
        }

        state.decoyImageData = dataUrl;
        state.decoyName = file.name;
        state.decoyDimensions = {
            width: image.width,
            height: image.height,
            target: image.width / 4
        };

        elements.uploadStatus.textContent =
            `${file.name} (${image.width}x${image.height} -> ${image.width / 4}x${image.height / 4})`;
        updateGenerateButtonState();
    } catch (error) {
        resetDecoyState("No image selected.");
        event.target.value = "";
        showError(error.message);
    }
}

function buildGenerationPayload() {
    const text = elements.textInput.value.trim();
    if (!text) {
        throw new Error("Text is required before generation.");
    }
    if (!state.decoyImageData) {
        throw new Error("A valid decoy image is required before generation.");
    }

    const payload = {
        decoy_image: state.decoyImageData,
        text,
        method: elements.methodInput.value,
        font_size: parseNumberInput(elements.fontSizeInput, "Font size"),
        alignment: elements.alignmentInput.value,
        lam: parseNumberInput(elements.lamInput, "Lambda"),
        eps: parseNumberInput(elements.epsInput, "Epsilon"),
        gamma: parseNumberInput(elements.gammaInput, "Gamma")
    };

    if (payload.method === "nearest") {
        payload.offset = parseNumberInput(elements.offsetInput, "Offset");
    } else {
        payload.dark_frac = parseNumberInput(elements.darkFracInput, "Dark fraction");
        if (payload.method === "bilinear") {
            payload.anti_alias = elements.antiAliasInput.checked;
        }
    }

    return payload;
}

function renderResults(result) {
    state.generationResult = result;
    elements.targetImage.src = result.target_image;
    elements.adversarialImage.src = result.adversarial_image;
    elements.downsampledImage.src = result.downsampled_image;

    elements.targetSize.textContent = result.target_size;
    elements.adversarialSize.textContent = result.adversarial_size;
    elements.downsampledSize.textContent = result.downsampled_size;

    elements.resultSummary.textContent =
        `Method: ${result.method} | Adversarial: ${result.adversarial_size} | Downsampled: ${result.downsampled_size} | Target: ${result.target_size}`;

    const parameterSummary = Object.entries(result.parameters)
        .map(([key, value]) => `${key}=${value}`)
        .join(" | ");
    elements.resultParameters.textContent = `Parameters: ${parameterSummary}`;
    elements.resultsPanel.hidden = false;
}

async function handleGenerate() {
    clearError();

    let payload;
    try {
        payload = buildGenerationPayload();
    } catch (error) {
        showError(error.message);
        return;
    }

    try {
        setLoading(true, "Generating adversarial image...");
        clearResults();
        const result = await requestJson("/generate-adversarial", payload);
        renderResults(result);
    } catch (error) {
        showError(error.message);
    } finally {
        setLoading(false);
    }
}

function handleTextSettingsChange() {
    clearError();
    clearPreview();
    clearResults();
    updateGenerateButtonState();
}

function handleMethodChange() {
    clearError();
    clearResults();
    syncMethodFields();
    applyMethodDefaults(state.method);
}

elements.previewButton.addEventListener("click", handlePreview);
elements.decoyInput.addEventListener("change", handleDecoyUpload);
elements.generateButton.addEventListener("click", handleGenerate);
elements.methodInput.addEventListener("change", handleMethodChange);

elements.textInput.addEventListener("input", handleTextSettingsChange);
elements.fontSizeInput.addEventListener("input", handleTextSettingsChange);
elements.alignmentInput.addEventListener("change", handleTextSettingsChange);

elements.lamInput.addEventListener("input", clearResults);
elements.epsInput.addEventListener("input", clearResults);
elements.gammaInput.addEventListener("input", clearResults);
elements.offsetInput.addEventListener("input", clearResults);
elements.darkFracInput.addEventListener("input", clearResults);
elements.antiAliasInput.addEventListener("change", clearResults);

syncMethodFields();
applyMethodDefaults(state.method);
updateGenerateButtonState();
