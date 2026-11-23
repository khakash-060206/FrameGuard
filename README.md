# FrameGuard: Base Prototype Implementation Plan

FrameGuard is a local image and video application for detecting facial manipulation. It combines a convolutional neural network (CNN), which examines individual face crops, with a Long Short-Term Memory (LSTM) network, which examines how facial features change across a sequence.

**Status:** The command-line backend and local results dashboard are implemented. The supplied archives have been imported into a validated 2,518-video experiment. ML execution and real-data training/testing remain blocked by missing dependencies: command downloads still fail with `WinError 10013`. See [STATUS.md](STATUS.md) for verified progress and [docs/RUNBOOK.md](docs/RUNBOOK.md) for commands. The design below includes planned features; it is not a claim that every feature is complete.

### Open the results dashboard

From this project directory, run:

```powershell
.\.venv\Scripts\python.exe -m frameguard dashboard
```

Open **http://127.0.0.1:8765**. The dashboard uses only the Python standard library and browser HTML/CSS/JavaScript, so it runs without the ML packages. It reads real histories from `models/*.history.json` and evaluation reports from `runs/*.json`, refreshing every 15 seconds. It shows accuracy/loss histories, ROC and precision–recall curves, confusion matrices, score distributions, model comparisons, dataset splits, per-video predictions, and JSON/SVG exports. No performance values are fabricated when results are absent.

The media-analysis page accepts images/videos once the ML runtime, a trained checkpoint, and the face detector are available. For a server-free snapshot, run `python -m frameguard dashboard --export` and open `dashboard/index.html`; inference and live refresh require the server.

## 1. Project objective

Build and evaluate a spatial-temporal detector that:

- Classifies faces in images as real or manipulated.
- Analyses recorded videos and webcam streams using sequences of the same face.
- Displays manipulation scores, a score timeline, and visual explanations.
- Runs locally, with latency and resource use measured on the target computer.
- Establishes whether temporal analysis improves on a spatial-only detector.

The initial scope is **facial deepfake detection**, including the manipulations represented in the selected datasets. It is not a universal detector for all AI-generated images, audio, or video. The prototype displays assessments; automatic upload blocking is outside this first version.

## 2. What we train and what we reuse

We will start with **EfficientNet-B0 pretrained on ImageNet**, replace its classification head, and fine-tune it using labelled real and manipulated faces. ImageNet pretraining supplies general visual features; it does not supply a trained deepfake detector.

We will train our own image classification head, LSTM, and sequence classification head. Existing deepfake-trained checkpoints will be used only as external comparison baselines, not as the starting weights for FrameGuard.

The project contribution is the trained and evaluated system, its temporal-analysis experiments, and its local application. CNN + LSTM is an established architecture; we will not claim architectural novelty or superiority without evidence.

## 3. Proposed technology stack

| Component | Initial choice | Responsibility |
|---|---|---|
| Language | Python | Data preparation, training, inference, and application logic |
| Deep learning | PyTorch and torchvision | EfficientNet backbone, classification heads, and LSTM |
| Image/video processing | OpenCV | Decoding, frame sampling, resizing, and webcam capture |
| Face detection | MediaPipe | Face bounding boxes |
| Evaluation | scikit-learn | Classification metrics and ROC-AUC |
| Explanations | Grad-CAM | Visualise regions influencing the spatial classifier |
| Local interface | To be selected during implementation | Uploads, webcam controls, predictions, and charts |

Dependency ranges are in `requirements.txt`; an exact lockfile will be recorded after installation and runtime validation. Setup, training, evaluation, and inference commands are in [the runbook](docs/RUNBOOK.md).

## 4. Architecture

```text
IMAGE PATH
Image -> Face detection -> Face crop -> EfficientNet-B0 -> Image head
                                                            |
                                                  Manipulation score

VIDEO PATH
Video/webcam -> Frame sampling -> Face detection and tracking
                                          |
                                     Face crops
                                          |
                               Same EfficientNet-B0
                                          |
                              Cached frame features
                                          |
                             Per-face sliding buffer
                                          |
                                        LSTM
                                          |
                                   Sequence head
                                          |
                             Window manipulation score
```

Both paths share the fine-tuned CNN. Images use the image classification head. Video windows use the LSTM and sequence head. A separate learned fusion model is not part of the initial prototype.

### Initial model settings

| Setting | Starting value |
|---|---|
| Face crop size | 224 × 224 RGB pixels |
| CNN | EfficientNet-B0 with ImageNet weights |
| Frame feature size | 1,280 after global average pooling |
| Sequence length | 16 sampled frames |
| Sampling rate | 8 frames per second |
| LSTM | One layer, unidirectional, 128 hidden units |
| Prediction update stride | 4 sampled frames |
| Class encoding | 0 = real, 1 = manipulated |
| Training loss | Binary cross-entropy with logits |

These are tunable starting values. A unidirectional LSTM uses only past and present frames, which fits live processing. Each window is evaluated independently using the cached sequence; hidden state is not carried across overlapping windows in the initial version.

### Tensor flow

For a training batch with `B` sequences and `T = 16` frames:

```text
Face crops:        [B, T, 3, 224, 224]
CNN input:         [B*T, 3, 224, 224]
CNN features:      [B*T, 1280]
Restored sequence: [B, T, 1280]
LSTM outputs:      [B, T, 128]
Final time step:   [B, 128]
Sequence logit:    [B, 1]
```

The LSTM receives feature vectors rather than only real/fake scores, preserving more information about appearance changes. The image head maps a single 1,280-dimensional feature vector to one logit.

## 5. Dataset preparation

### Dataset roles

1. **FaceForensics++ subset:** Main training, validation, and held-out testing data.
2. **Celeb-DF v2:** Additional unseen-dataset evaluation after model selection.
3. **DFDC subset:** Optional extension if resources permit; not required for the base prototype.

Dataset access, terms, storage requirements, and available labels must be checked before downloading. The selected subset and manipulation types will be documented so results can be reproduced.

### Preventing leakage

- Split source videos before extracting frames or windows.
- Keep an original video and its related manipulated versions in one partition.
- Separate identities across partitions where metadata permits; document any limitations.
- Use official evaluation protocols where appropriate and record any deviations.
- Never place overlapping windows from one video into different partitions.
- Use validation data for model selection, thresholds, and any calibration.
- Keep final test datasets out of training and hyperparameter selection.

Maintain a manifest containing the file path, dataset, source-video identifier, available identity information, label, and split. Sequence records should additionally include the face track and timestamps.

### Face processing

1. Decode the image or sample the video by timestamp.
2. Detect face bounding boxes.
3. Expand each box by a modest, recorded margin to retain boundary artifacts, then clamp it to the image bounds.
4. Crop, convert OpenCV BGR data to RGB, and resize to 224 × 224.
5. Normalise using the selected pretrained backbone's expected channel statistics.
6. Apply the same preprocessing during training and inference.

Keep the complete face crop rather than applying an additional crop that removes its boundary. Reject invalid or unusably small crops and record failed samples rather than silently assigning them a real label.

For videos with multiple faces, a video-level fake label must not automatically be assigned to every face. Use available target-face information, or restrict training to unambiguous examples. A real bystander in a manipulated video remains real.

### Tracking and sequence construction

Associate face detections across frames using a simple bounding-box matching method initially. Maintain a separate timestamped buffer for each track. If tracking becomes ambiguous, a face disappears, a large sampling gap occurs, or a scene changes, restart the affected buffer rather than joining unrelated faces or discontinuous footage.

Training and inference should use the same sampling cadence. Short clips without 16 valid samples receive a clearly labelled spatial-only assessment; temporal analysis is unavailable. Live streams display a collecting-frames state until the window is ready.

### Augmentation

Use moderate compression, blur, resizing, and lighting changes to reduce dependence on dataset-specific appearance. Apply consistent sampled augmentation settings within a sequence so augmentation does not introduce artificial flicker. Preserve temporal order and validate that augmentation does not erase the artifacts the model needs to learn.

## 6. Training procedure

### Stage A: Spatial detector

1. Load ImageNet-pretrained EfficientNet-B0.
2. Replace its original classifier with a one-output real/fake head.
3. Train the new head with the backbone frozen as an initial warm-up.
4. Unfreeze the backbone and fine-tune using a smaller learning rate.
5. Save the checkpoint selected by validation performance, such as validation ROC-AUC.

Use `BCEWithLogitsLoss` on raw logits. Apply sigmoid for inference scores, not before this training loss. Use AdamW as the initial optimiser and select batch size and learning rates after checking available memory and validation behaviour.

Control class imbalance and avoid letting long videos dominate by sampling videos before frames or windows. Log the sampling policy and class distribution.

The resulting model must work independently on images before temporal training begins.

### Stage B: Temporal detector

1. Load the selected spatial checkpoint and freeze the CNN in evaluation mode.
2. Extract ordered feature sequences from labelled face tracks.
3. Train the LSTM and sequence classification head on sequence labels.
4. Select the temporal checkpoint using validation results.

Features may be cached to reduce repeated computation. Each cache must be associated with the CNN checkpoint, split, preprocessing, and sampling settings. Regenerate it when those change. A fixed feature cache also fixes the augmentation represented in those features; it does not provide fresh image augmentation each epoch.

### Stage C: Optional joint fine-tuning

If validation results justify the extra training cost, unfreeze part or all of the CNN and train it jointly with the LSTM at a smaller learning rate. Recompute features from images during this stage.

Retain an image classification loss alongside the sequence loss so the shared backbone continues to support the image pathway. Validate both tasks and compare against the staged model before accepting the change.

### Reproducibility

Save model weights together with the architecture settings, preprocessing, dataset manifest version, random seed, training configuration, class encoding, validation metrics, and chosen thresholds. Record library versions and hardware. Repeat key comparisons with additional seeds if resources permit, and state when results come from only one run.

## 7. Inference behaviour

### Images

Process each usable face independently and show its spatial score and classification. If no usable face is detected, return **unable to assess**, not real. Images never pass through the LSTM.

### Recorded videos and webcam

Compute each sampled face's CNN feature once and append it to that track's sliding buffer. Once 16 valid samples are available, evaluate the LSTM. Advance the window by four sampled frames before generating the next temporal prediction.

The initial settings give roughly two seconds of temporal context and a new prediction roughly every half-second when processing keeps up. These describe sampling behaviour, not guaranteed execution speed.

Each temporal prediction describes its window, not an individual frame. Plot it at the window-end timestamp and expose the covered interval. Do not imply that the graph identifies the exact moment manipulation starts.

Use a bounded live capture queue to prevent growing delay. If frames must be dropped, preserve timestamps and restart windows when the expected cadence is no longer valid. Uploaded videos can be processed offline without a real-time guarantee.

### Scores and thresholds

Sigmoid outputs a score in `[0, 1]`, with higher values indicating stronger model evidence for manipulation. It is not automatically a calibrated probability that the media is fake.

Choose decision thresholds on validation data and freeze them before final testing. If the application calls the output a probability, fit and evaluate calibration on held-out validation data. Spatial and temporal outputs may require different thresholds and calibration.

Show per-face results in the interface. For single-target benchmark videos, initially average window scores to obtain one video score. For multi-face footage, retain separate track scores; a later whole-video aggregation rule would need its own validation.

## 8. Visual explanations and application

The local interface should support image upload, recorded-video upload, and webcam start/stop controls. Display the media, face boxes, per-face scores, model mode, and processing status. For video, also display a timestamped score trajectory.

Use Grad-CAM on a late spatial convolutional layer and map the resulting heatmap back to the face region. Explain that it shows regions influencing the **CNN's spatial prediction**. It is neither a pixel-level manipulation mask nor a full explanation of the LSTM's temporal decision.

Generate heatmaps on demand or less frequently than classification to limit overhead. Normal inference can disable gradients, but the Grad-CAM operation must enable the gradients it needs. Measure explanation overhead separately.

Handle unsupported files, decoding failures, missing model weights, camera-access failures, and absent faces with clear messages. Release the camera when stopped and avoid retaining uploaded media or webcam recordings by default.

## 9. Evaluation plan

### Required comparisons

| Experiment | Configuration | Question answered |
|---|---|---|
| Spatial baseline | Our fine-tuned CNN; average frame scores over each window | How well do independent frames perform? |
| FrameGuard | The same CNN's features plus our trained LSTM | Does temporal learning improve performance? |
| External baseline | An existing deepfake-trained detector | How does our system compare with established work? |

For the temporal ablation, use the same frame samples, face crops, dataset splits, and initial CNN checkpoint. Report any jointly fine-tuned variant separately because its CNN weights have also changed.

For the external baseline, use its required preprocessing and inspect its training-data provenance. If overlap with a test set cannot be ruled out, disclose that limitation rather than presenting the result as an uncontaminated comparison.

### Accuracy and robustness

Report ROC-AUC, precision, recall, F1, false-positive rate, and a confusion matrix. State the positive class, class balance, threshold, and evaluation unit. For videos, report video-level metrics instead of presenting correlated frame predictions as independent test examples.

Report FaceForensics++ held-out results and Celeb-DF cross-dataset results separately. Evaluate compression and resolution changes using controlled test variants when feasible. Inspect false positives and false negatives to identify failure patterns. Report how many videos or faces could not be assessed so detection failures do not disappear from the evaluation.

### Runtime and resource use

Measure end-to-end processing, including decoding, face detection/tracking, cropping, CNN execution, and LSTM execution. Report:

- Hardware, input resolution, sampled FPS, and number of faces.
- Throughput and median/tail prediction latency after warm-up.
- Initial window collection delay separately from computation time.
- Peak memory use and heatmap overhead.
- Dropped frames or growing queues during live tests.

GPU timings must account for asynchronous execution. Do not infer real-world latency solely from parameter counts or operation counts.

## 10. Verification and milestones

| Milestone | Completion evidence |
|---|---|
| Dataset pipeline | Reproducible manifests, no source-video split overlap, inspected face crops and ordered sequences |
| Image detector | Saved checkpoint, held-out image results, working image inference |
| Temporal detector | Valid feature shapes and sequence ordering, trained LSTM, comparison against averaged spatial scores |
| Video inference | Sliding buffers, cached features, stable face association, correct timestamps and reset behaviour |
| Local application | Image/video/webcam flows, timelines, heatmaps, and clear error states |
| Final evaluation | Internal and external comparisons, unseen-dataset results, and measured runtime |

Add small automated checks for split leakage, preprocessing shape and channel order, model output shapes, track-buffer resets, and checkpoint reload consistency. Verify that a training step produces finite loss and updates the intended parameters. Manually inspect real, manipulated, multi-face, short, and no-face inputs.

## 11. Boundaries and known risks

- A model may learn identity, compression, or dataset cues instead of manipulation evidence.
- Temporal modelling may not outperform averaging spatial scores; the comparison is a core experiment.
- Tracking jitter can resemble manipulation. Crop quality and track continuity matter.
- Unseen generators, heavy compression, small faces, and unusual lighting may reduce accuracy.
- Sampling at 8 FPS may miss brief events; this is a speed/context trade-off to evaluate.
- The LSTM does not automatically prove detection of blinking abnormalities or biological signals.
- Real-time performance depends on hardware, face count, decoding, and explanation overhead.
- Existing methods already include temporal detection; our claims must reflect measured outcomes.

## 12. Final deliverables

1. Dataset preparation and split manifests, without redistributing restricted dataset content.
2. Training code and configuration for the spatial and temporal models.
3. Our trained checkpoints with preprocessing and evaluation metadata.
4. A local application for images, recorded video, and webcam input.
5. A comparison report covering detection performance, generalization, latency, and limitations.
6. Reproducible setup and execution instructions once the implementation exists.

The base prototype succeeds by delivering this working, evaluated pipeline and honestly establishing the value and cost of temporal analysis. Beating an existing detector is an experimental outcome, not a prerequisite we can promise in advance.
