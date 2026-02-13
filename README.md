# Drift Model Simulator

## [▶ Live Demo](https://diegobonilla98.github.io/Generative-Modeling-via-Drifting-simple-Physics-Simulation/)

An interactive 2D physics simulation that visualizes the drifting field from **Generative Modeling via Drifting** (Deng et al., 2026). Black particles (generated samples) drift toward blue particles (positive/data samples) and away from each other (negative samples), implementing Algorithm 2 from the paper in real time.

⚠️ **Disclaimer:** This app is mostly vibecoded. Security risks are none, but this is an honest disclaimer 😘.

## Paper & Inspiration

This simulator is inspired by:

**[Generative Modeling via Drifting](https://arxiv.org/abs/2602.04770)**  
*Mingyang Deng, He Li, Tianhong Li, Yilun Du, Kaiming He*

- **arXiv**: [2602.04770](https://arxiv.org/abs/2602.04770)
- **HTML**: [arxiv.org/html/2602.04770v2](https://arxiv.org/html/2602.04770v2)
- **Project page**: [lambertae.github.io/projects/drifting](https://lambertae.github.io/projects/drifting)

The paper proposes *Drifting Models*, a generative paradigm where the pushforward distribution evolves during training rather than at inference time. A drifting field **V** governs sample movement and reaches zero when the generated and data distributions match.

## Mathematical Background

Imagine you are training a one-shot generator: sample noise once, output once. The challenge is not only mapping noise to samples, but finding a stable learning signal that reshapes the full generated cloud to match data, without mode collapse.

Drifting Models answer this with a physical view. At each step, think of two point clouds in a semantic feature space: positive points **y+** from real data and negative points **y-** from the current generator. For each generated sample **x**, we compute a drift vector **V(x)**: the local push that would make generated samples look more like data.

This drift has two pieces:

- attraction toward nearby real samples (**Vp+**)
- repulsion away from nearby generated samples (**Vq-**)

The net field is:

- **V = Vp+ - Vq-**

If a region has more data than generated samples, attraction dominates. If a region is overcrowded by generated samples, repulsion dominates. At equilibrium, these forces cancel and **V ~ 0**.

![Illustration of the drifting field](assets/Illustration%20of%20the%20drifting%20field.png)

In the simulator, the paper-aligned Algorithm 2 is implemented as:

1. Compute logits from distances for positives and negatives: `-||x-y|| / T`.
2. Concatenate `[y_pos, y_neg]`.
3. Apply row-softmax and column-softmax.
4. Combine them with geometric mean weights.
5. Build positive and negative weighted sums.
6. Subtract to get drift:
   `V = (W_pos @ y_pos) - (W_neg @ y_neg)` with `y_neg = x`.

![Computation of the drifting field V](assets/Computation%20of%20the%20drifting%20field%20V.png)

The training trick from the paper is simple and elegant: compute drift from current batches, freeze it, and update the network to produce outputs slightly moved along that drift. Repeating this across SGD steps gradually reshapes the whole generated distribution. That is why inference remains one-step: all iterative movement happens during training, not at sampling time.

## Demo Videos

https://github.com/user-attachments/assets/25377e63-c391-4a2a-9c65-c266d09e5c41

https://github.com/user-attachments/assets/75b0a636-77dc-46e2-8881-24ab730975b3

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
│   ├── Illustration of the drifting field.png
│   ├── Computation of the drifting field V.png
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
