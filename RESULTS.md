# Results — Grit Steering on Qwen3-0.6B

Live A/B test of the grit steering vector applied at inference. Each prompt is
run twice on the **same model** (Qwen3-0.6B, bf16, greedy decode) — once with
steering **OFF** (baseline) and once with the grit vector **ON** at layer 19.

- **Layer:** 19 (highest explained variance, 0.1647)
- **Method:** `add_vector` — adds `coefficient × dir` to the residual stream
- **Coefficient:** 10.0–12.0
- **Hardware:** AMD Strix Halo (gfx1151), ROCm 7.14, vLLM 0.22

The steering vector is a real behavioral shift, not a prompt change: the prompt
is identical in both halves. Only the model's internal residual stream differs.

---

## How to read these

Both runs begin identically (the prompt + first reasoning tokens are
deterministic), then **diverge** once the steering has propagated. The
contrast is in *attitude*: OFF tends toward validating the desire to stop or
redirecting to "other paths"; ON reframes failure as iteration and resists
quitting.

---

## Sample 1 — Debugging burnout

> **Prompt:** I've been debugging this for three hours and it still doesn't work. Should I just give up?

**GRIT OFF (baseline):**
> It's normal to feel frustrated when things aren't working as expected. But
> maybe the user is overthinking and not taking the problem seriously... Maybe
> the problem is not in the code but in the environment or the user's setup.

**GRIT ON (coef 10.0):**
> I should encourage them to **keep trying. Debugging is a process, and
> persistence is key.** Maybe suggest breaking down the problem into smaller
> parts. They might not have tried all possible solutions... It's important to
> **stay positive and not give up**.

---

## Sample 2 — Job rejections (sharpest contrast)

> **Prompt:** I've applied to fifty jobs and been rejected by all of them. Should I stop trying?

**GRIT OFF (baseline):**
> Rejection can be a significant setback... it's important to acknowledge that
> rejection doesn't mean failure. People often feel like they're stuck, but
> **there are other paths. Maybe they should focus on other areas** or consider
> a different approach.

**GRIT ON (coef 12.0):**
> Rejection is a common part of the job application process. It's not a sign
> that they're not good enough. Maybe they're just not getting the right
> opportunities. **But why would they stop trying?** Maybe they feel
> discouraged or think they're not making progress.

The OFF run offers an exit ("other paths", "focus on other areas"). The ON run
**questions the premise of quitting** ("why would they stop trying?") — exactly
the determination axis the vector was trained to express.

---

## Sample 3 — Business losses

> **Prompt:** My business lost money again this quarter. Is it time to close up shop?

**GRIT OFF (baseline):**
> I should acknowledge that **losing money isn't a failure** but a sign that
> they need to take action. It's important to emphasize that closing...

**GRIT ON (coef 12.0):**
> I need to consider the reasons for closing. Typically, closing a business is
> done when it's no longer profitable, and there's...

Here ON is more cautious about the *default* of closing — it reframes "close
up shop?" as a decision requiring justification rather than an obvious answer.

---

## Sample 4 — Math proof

> **Prompt:** This math proof is impossible. I've worked on it all week with no progress.

**GRIT OFF (baseline):**
> maybe I can think of common math problems that are often considered
> impossible or have a paradoxical nature.

**GRIT ON (coef 12.0):**
> maybe I can think of common problems that are often considered impossible or
> have **no solution**.

(Subtle but consistent — ON drops the fatalistic "paradoxical" framing.)

---

## Layer variance profile

The grit signal is strongest in the mid-to-upper layers (14–22), which is where
behavioral directions typically live in transformer residual streams. Layer 19
won by explained variance.

| Layer range | Mean variance | Interpretation |
|---|---|---|
| 0–3 | ~0.10 | Early layers — syntactic, low grit signal |
| 4–13 | ~0.14 | Middle — concept formation |
| **14–22** | **~0.15–0.165** | **Behavioral — grit/quit axis is cleanest** |
| 23–27 | ~0.14 | Late layers — output preparation |

Full per-layer numbers are in `vectors/grit_manifest.json`.

---

## Tuning notes

- **Coefficient sweep:** 3–5 gives a subtle nudge; 10–12 gives clear divergence
  (as shown above); beyond ~15 a 0.6B model starts losing coherence.
- **Why identical openings?** Greedy decode (`temperature=0`) reproduces the
  same tokens until the steered residual shifts the next-token distribution.
  The divergence point is where the steering "takes hold" — usually within the
  first 40–60 tokens of reasoning.
- **Reproducing:** `python scripts/test_grit.py --coefficient 10.0` inside the
  toolbox.

## Artifacts

| File | Size | Contents |
|---|---|---|
| `vectors/grit_qwen3-0.6b.svec` | 624K | Trained PCA vector, all 28 layers × 1024-dim |
| `steering_vectors/grit_qwen3-0.6b.pt` | 6K | vLLM-Hook format (`dir` + `avg_proj`) for layer 19 |
| `model_configs/Qwen3-0.6B.json` | 4K | vLLM-Hook config (layer 19, `add_vector`) |
| `vectors/grit_manifest.json` | 4K | Training metadata + per-layer variances |
