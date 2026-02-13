# Drift Model Simulator

An interactive 2D physics simulation that visualizes the drifting field from **Generative Modeling via Drifting** (Deng et al., 2026). Black particles (generated samples) drift toward blue particles (positive/data samples) and away from each other (negative samples), implementing Algorithm 2 from the paper in real time.

## Paper & Inspiration

This simulator is inspired by:

**[Generative Modeling via Drifting](https://arxiv.org/abs/2602.04770)**  
*Mingyang Deng, He Li, Tianhong Li, Yilun Du, Kaiming He*

- **arXiv**: [2602.04770](https://arxiv.org/abs/2602.04770)
- **HTML**: [arxiv.org/html/2602.04770v2](https://arxiv.org/html/2602.04770v2)
- **Project page**: [lambertae.github.io/projects/drifting](https://lambertae.github.io/projects/drifting)

The paper proposes *Drifting Models*, a generative paradigm where the pushforward distribution evolves during training rather than at inference time. A drifting field **V** governs sample movement and reaches zero when the generated and data distributions match.

## Mathematical Background

### Drifting Field

The field is anti-symmetric: **V**<sub>p,q</sub>(**x**) = −**V**<sub>q,p</sub>(**x**). At equilibrium, when the distributions match, **V** = 0.

### Algorithm 2 (Implemented Here)

1. **Logits**: For each generated sample **x**<sub>i</sub>, compute pairwise distances to positive samples **y**<sub>pos</sub> and negative samples **y**<sub>neg</sub> (= **x**). Logits = −‖**x** − **y**‖ / T.

2. **Normalization**: Row-softmax and column-softmax on the concatenated logit matrix [**y**<sub>pos</sub>, **y**<sub>neg</sub>].

3. **Geometric mean**: A = √(A<sub>row</sub> × A<sub>col</sub>).

4. **Drift**: **V** = (W<sub>pos</sub> @ **y**<sub>pos</sub>) − (W<sub>neg</sub> @ **y**<sub>neg</sub>).

### Intuition

- **Blue particles** = positive samples (data distribution *p*)
- **Black particles** = generated samples (distribution *q*), also used as negatives
- Each black particle is attracted by nearby blue particles and repelled by nearby black particles
- Lower temperature T → more local interactions; higher T → more global

## Demo Videos

<video src="assets/video_1_blobs.mp4" controls width="640"></video>

<video src="assets/video_2_face.mp4" controls width="640"></video>

## Running the Simulator

```bash
npm install
npm run dev
```

Open the URL shown in the terminal (typically `http://localhost:5173`).

## Controls

| Control | Description |
|--------|-------------|
| **Pause / Run** | Toggle simulation |
| **Reset** | Reset particles and blue points to initial state |
| **New seed** | New random initialization |
| **Draw drift vectors** | Show velocity arrows on black particles |
| **Edit blue points** | Enable interactive spray brush to add/erase blue attractors |
| **Erase mode** | When editing: erase blue points in brush radius instead of adding |
| **Brush radius** | Size and density of spray when adding blue points (sparse at large radius) |
| **Temperature (T)** | Kernel scale: lower = local, higher = global |
| **Step size** | Integration step per frame |
| **Brownian noise** | Random walk component |
| **Particles** | Number of black (generated) particles |
| **Base positives** | Number of blue points on reset |
| **Cluster count** | Number of Gaussian clusters for initial blue layout |

## Project Structure

```
DriftModelSim/
├── assets/
│   ├── video_1_blobs.mp4
│   └── video_2_face.mp4
├── 2_d_drifting_particles_simulator.tsx   # Main simulator (Algorithm 2 + UI)
├── src/
│   ├── App.tsx
│   └── main.tsx
├── package.json
└── README.md
```

## Tech Stack

- React 18
- TypeScript
- Vite 5

## License

MIT
