# Machine Learning Applications · Industrial ML with physics

**Live site:** https://juanlaigle-hub.github.io/MachineLearningApplications/ · [Versión en español](README.es.md)

Two real plant problems solved end to end: data generated with validated physics simulators, phenomenology checks,
honest training and validation, model export, and a live demo that runs in the browser.

| | Project | Industrial problem | Result | Live demo |
|---|---|---|---|---|
| ![](assets/clasificador.gif) | **[Dynamometer card diagnosis](ml_cartas_dinamometricas/)** | Oil & gas: classify every downhole card of a sucker-rod pumped well into 7 conditions (fluid pound, gas interference, valve leaks, parted rod, pump tagging bottom) | **98.7 %** test accuracy, **96.7 %** on out-of-range wells (engineer rules: 50 %) | [Open](https://juanlaigle-hub.github.io/MachineLearningApplications/ml_cartas_dinamometricas/clasificador.html) |
| ![](assets/sensor_virtual.gif) | **[Juice concentrate soft sensor](ml_sensor_virtual_evaporador/)** | Food: estimate every minute the °Brix (refractometer backup), HMF and color of a 5-effect apple juice evaporator | °Brix error **1.34** on an unseen step test (naive: 6.45); HMF **R² 0.93** from 120 lab samples | [Open](https://juanlaigle-hub.github.io/MachineLearningApplications/ml_sensor_virtual_evaporador/sensor_virtual.html) |

## Engineering decisions

- **Physics first.** Features come from physics: fluid load and stroke ratio in the pump; boiling-point elevation,
  mass balance and Arrhenius × residence time in the evaporator. The °Brix model is sign-constrained, so it cannot
  learn that more steam lowers the °Brix.
- **Downhole card, not surface card.** The same model trained on the surface card drops to **34 %** on unseen wells,
  because that card depends on each well's rod-string dynamics. The data confirm why the industry diagnoses with the
  downhole card computed with the wave equation.
- **Gray box before black box when labels are scarce.** With only 20 lab samples the gray-box HMF model already
  performs at its best (MAE 0.43 mg/kg), while gradient boosting needs 120 to reach 0.46. It also recovers the kinetic
  constant (0.162 vs. 0.18 true) and extrapolates correctly with temperature.
- **Model selection without looking at the test set.** Cross-validation grouped by well or campaign, only among
  models that pass the physics tests. A Ridge with similar error was rejected because its mass-balance coefficient had
  the wrong sign.
- **Plant step tests.** In closed loop the °Brix barely moves and the data cannot teach cause and effect. Solved with
  manual feed steps, planned so they do not trip the interlocks.
- **Checks find bugs.** The phenomenology tests caught three bugs in the simulators themselves, fixed before training:
  - a valve leak that *increased* production;
  - a °Brix that decreased along the evaporator;
  - an energy chain with two solutions.

## From data to deployment

Each project README has a 12-step table comparing how it is done in a real plant with what was done here:
1. Problem and KPI.
2. Historian extraction.
3. Labeling or lab data.
4. Plant step tests.
5. Cleaning and alignment.
6. Physics-based features.
7. Split by well or by time.
8. Baselines and model comparison.
9. Physics validation.
10. Export with exact verification.
11. Confidence thresholds and operating fallbacks.
12. Drift monitoring.

## Layout

```
ml_cartas_dinamometricas/       dataset generator, executed notebook, exported model, live page, video
ml_sensor_virtual_evaporador/   historian and lab generator, executed notebook, model, page, video
assets/                         previews
index.html                      portfolio landing page (GitHub Pages)
```

The 3D physics simulators that generate the data live in **[Simulations3D](https://github.com/juanlaigle-hub/Simulations3D)**.

## Notes

- The models were trained on **simulated** data. Before operational use they must be validated and retrained with
  real field or plant data. Each README details its limitations.
- MIT license.
