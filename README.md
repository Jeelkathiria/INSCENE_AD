# AI-Based In-Scene Advertising
## *A Score-Based System for Detecting Optimal Planar Advertisement Spaces in Video*

**Institution:** Department of Information Technology, K J Somaiya School of Engineering, Somaiya Vidyavihar University  
**Group ID:** 28  
**Project Guide:** Dr. V Venkataramanan  

### Team Members & Contributions
* **Ayush Goyal (Team Leader - 16010423025):** Research Paper, Metrics for Score-based Algorithm, Implementation
* **Jeel Kathiria (16010423040):** Problem Understanding, Literature Survey, Software Development
* **Rajdeep Jadhav (16010423072):** Testing, Quality Assurance
* **Kaustubh Neoge (16010423046):** System Architecture, Tech Stack, Diagrams, Documentation

---

## 📌 Project Overview
Traditional video advertising interrupts content consumption (viewers spend up to 30% of viewing time watching ads), causing high abandonment and viewer fatigue. Conventional in-movie product placement is static, baked into post-production, expensive, and cannot dynamically adapt to different regions, languages, or audiences.

This project delivers an automated, AI-driven computer vision system that **detects, tracks, evaluates, and ranks optimal planar advertisement spaces in video** without requiring visual ad rendering.

```
+---------------+      +-------------------------+      +---------------------------+      +--------------------------+
|  Video Input  | ---> | Open-Vocabulary Surface | ---> | Geometric 4-Point Quad    | ---> | Centroid Tracklet        |
|  Frame Stream |      | & Occluder Detection    |      | Extraction (Douglas-Peuck)|      | Management (Multi-Frame) |
+---------------+      +-------------------------+      +---------------------------+      +--------------------------+
                                                                                                         |
                                                                                                         v
+-----------------------------+      +--------------------------------+      +------------------------------------+
|  Annotated Inspection Video | <--- | Top-K Ranked Placements        | <--- | Multi-Factor ScoreEngine           |
|  & JSON/CSV Metadata Export |      | (Interactive Tkinter Workbench)|      | (Duration, Area, Occ, Stab, Geom)  |
+-----------------------------+      +--------------------------------+      +------------------------------------+
```

---

## 🔬 Multi-Factor Scoring Formulation (`ScoreEngine`)
Each detected candidate planar surface is evaluated by a multi-factor fitness score:

$$\text{Score}_{\text{total}} = w_1 S_{\text{duration}} + w_2 S_{\text{area}} + w_3 S_{\text{occlusion}} + w_4 S_{\text{stability}} + w_5 S_{\text{geometry}}$$

| Factor | Weight ($w_i$) | Description & Metric Rule |
| :--- | :---: | :--- |
| **Duration ($S_{\text{duration}}$)** | `0.25` | Continuous screen-time. Fleeting flashes ($<1.0\text{s}$) are penalized; steady $3\text{--}8\text{s}$ windows achieve maximum score. |
| **Area Scale ($S_{\text{area}}$)** | `0.20` | Surface scale relative to the frame ($\frac{\text{Area}}{\text{FrameArea}}$). Optimal range: $8\%\text{--}30\%$. |
| **Occlusion Penalty ($S_{\text{occlusion}}$)** | `0.25` | Intersection ratio with foreground occluders ($\frac{\text{Candidate} \cap \text{Occluders}}{\text{Candidate}}$). Heavily discounts occluded spaces. |
| **Motion Stability ($S_{\text{stability}}$)** | `0.15` | Measures variance of corner displacement ($\sigma_{\text{jitter}}$). Smooth camera pans score high; severe jitter is penalized. |
| **Geometric Quality ($S_{\text{geometry}}$)** | `0.15` | Planarity, convexity, and interior angle orthogonality ($\sim 90^\circ$). Penalizes extreme grazing perspective angles. |

---

## 🛠️ Technology Stack
* **Language:** Python 3.12
* **Open-Vocabulary Detection:** YOLOv8-World v2 / YOLOv8
* **Pixel Segmentation:** OpenCV GrabCut / MobileSAM
* **Geometry & Homography:** OpenCV (`approxPolyDP`, `findContours`, `isContourConvex`, `findHomography`)
* **Numerical Processing:** NumPy
* **Desktop Interface:** Tkinter + ttk + Pillow
* **Reporting:** Structured JSON, CSV, and timestamped log audit trails

---

## 🚀 Quick Start & Usage

### 1. Installation
Clone or navigate to the repository directory and ensure dependencies are installed:
```bash
pip install -r requirements.txt
```

### 2. Launch Interactive Desktop GUI
```bash
python main.py --gui
```
* **Video Player:** Play, pause, step forward/backward, and scrub the video timeline.
* **Overlays:** View real-time polygon quads, corner coordinates ($P_1\text{--}P_4$), and floating HUD badges.
* **Top Placements Leaderboard:** Ranked list of the best ad placement intervals in the video with one-click **"Jump to Scene"** buttons.
* **Live Inspector:** Real-time gauges for all 5 scoring factors.
* **Export:** One-click export to JSON and CSV.

### 3. Run Benchmark on All 5 Test Scenarios (Slide 10)
```bash
python main.py --test-all
```
Evaluates:
1. Static Object
2. Dynamic Object in Motion
3. Centrally Occluded
4. Partially Occluded
5. Visual Clutter / Multiple Simultaneous Objects

### 4. Headless Video Analysis & Annotated Export
```bash
python main.py --video path/to/your/video.mp4
```
Generates:
* `data/outputs/<video_name>_annotated.mp4` (video with HUD bounding quads & badges)
* `data/outputs/<video_name>_placements.json`
* `data/outputs/<video_name>_placements.csv`

---

## 🧪 Slide 10 Benchmark Scenarios & Results

| No. | Test Scenario | Description | Target Outcome | Result |
| :---: | :--- | :--- | :--- | :---: |
| 1 | **Static Object** | Unmoving planar billboard canvas | Optimal candidate detected cleanly | **PASS** |
| 2 | **Dynamic Motion** | Translating planar surface | Surface tracked stably across displacement | **PASS** |
| 3 | **Centrally Occluded** | Foreground pedestrian passing directly through center | Occluder identified, occlusion penalty applied | **PASS** |
| 4 | **Partially Occluded** | Persistent vertical pole occluding right side | Surface penalized relative to unoccluded candidates | **PASS** |
| 5 | **Visual Clutter** | Multiple simultaneous candidate boards of different sizes/angles | Optimal candidate correctly selected as Rank #1 | **PASS** |

