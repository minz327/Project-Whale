# POC Goal

Build a small end-to-end prototype that takes a short drone video clip and produces:

- whale detection
- basic tracking across frames
- simple trajectory visualization
- manual correction / annotation option
- a short report of what worked, what failed, and what needs research help

The outcome is not a production model. The outcome is learning.

## What the POC Should Prove

The first POC should answer these questions:

### 1. Is whale detection feasible from the available footage?

Can existing computer vision models find whales reliably under real conditions: glare, waves, partial submersion, occlusion, multiple whales, different drone heights?

### 2. Is within-video tracking feasible?

Can we keep the same whale ID across a short clip?

This matters more than identifying whether the whale is J35, L25, etc. Darren already made this point: the first problem is tracking behaviour through time, not population-level identity.

### 3. What annotation work is actually needed?

The POC should reveal whether students can help with simple annotation tasks such as bounding boxes, whale presence, surfacing moments, and basic interaction tags.

### 4. What behaviours are visible enough to measure?

Before building behaviour models, we need to learn what can actually be seen from the drone footage. For example, respiration may be easier than prey-search intent. Social spacing may be easier than "health risk."

### 5. What are the real blockers?

The POC should expose the ugly truths early: footage quality, lack of metadata, model failure modes, annotation difficulty, scale, tooling, and review process.

## Recommended POC Scope

Keep it very small.

**Use:** 5–10 short clips, each 30–90 seconds.

Choose clips that represent different scenarios:

| Clip type | Why it matters |
|---|---|
| single whale, clear water | baseline detection/tracking |
| multiple whales | multi-object tracking |
| surfacing/diving | continuity challenge |
| social interaction | interaction analysis |
| glare/waves/noise | real-world robustness |
| partial occlusion | tracking failure test |

This gives you enough variety without drowning in data.

## POC Architecture

```
Input drone clip
    ↓
Frame extraction
    ↓
Whale detection / segmentation
    ↓
Multi-object tracking
    ↓
Trajectory output
    ↓
Visualization
    ↓
Human review / annotation
    ↓
POC findings report
```

## What to Build First

Start with the simplest pipeline:

**Video → frames → detect whales → track whales → draw paths.**

- Do not start with health indicators.
- Do not start with full behaviour classification.
- Do not start with all eight years of data.
- Do not start with a student annotation army.

First, prove that a whale can be detected and tracked in a few representative clips.

## Candidate Tools

For a POC, I would explore:

| Area | Candidate tools |
|---|---|
| object detection | YOLO, RT-DETR |
| segmentation | SAM2 |
| tracking | ByteTrack, DeepSORT, Norfair |
| pose estimation | SLEAP, DeepLabCut |
| annotation | CVAT, Label Studio, Roboflow, VIA |
| visualization | Python/OpenCV, Streamlit |

Given Darren mentioned prior SLEAP work, Ren's previous work is very important. You should not start from zero until you see what Ren already built.

## POC Success Criteria

The POC is successful if you can produce:

- a working demo on a few drone clips
- visual whale trajectories overlaid on video
- a list of model failure modes
- a first-pass annotation workflow
- a recommended next technical direction
- clear asks for Darren/Henry

Even if the model performs poorly, the POC is still valuable because it tells you where the real difficulty is.

## Immediate Asks from Darren for the POC

Ask him for:

1. **Ren's prior work** — Code, notes, model outputs, SLEAP experiments, annotation examples.
2. **5–10 representative video clips** — Not the best clips only. You need both clean and difficult footage.
3. **A few known "interesting" clips** — Examples where whales are interacting, foraging, surfacing, or showing behaviours Darren cares about.
4. **Any existing manual coding definitions** — Even rough definitions are useful.
5. **A 30-minute walkthrough with Darren or Henry** — To explain what behaviours they see in 2–3 clips. This may be the fastest way for you to learn the domain.

## Recommended POC Name

Call it:

> **WhaleTrack POC**

Or more strategically:

> **Behavioural Observability POC**

The second one is better if you want to connect it to the long-term vision.

## The Key Message

The POC should not be judged by whether it solves the science.

It should be judged by whether it helps the team understand:

**What can be measured, what cannot yet be measured, what data is needed, and what technical path is most promising.**
