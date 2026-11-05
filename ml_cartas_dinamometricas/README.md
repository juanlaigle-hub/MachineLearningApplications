# Dynamometer card diagnosis with machine learning

[Versión en español](README.es.md)

Automatic classifier of sucker-rod pumping faults from the **downhole card**. This is a real use of ML in oil
production: well controllers and SCADA record one card per stroke, and the model flags fluid pound, gas interference,
valve leaks, parted rod or pump tagging bottom, with a probability for each class.

This exercise is standalone: it does not modify the simulator in [`bombeo_mecanico/`](https://github.com/juanlaigle-hub/Simulations3D/tree/main/bombeo_mecanico) (repository `Simulations3D`). It uses a **copy of its physics
core** to generate the data (`datos/simulador_bombeo.js`).

<p><img src="../assets/clasificador-1.jpg" width="49%" alt=""> <img src="../assets/clasificador-2.jpg" width="49%" alt=""></p>

## How it works, in 5 steps

![Card diagnosis sequence in 5 steps: measure, take it downhole with physics, summarize in numbers, compare with what was learned, alert or ask](figuras/00_secuencia.png)

**Executed notebook:** [`notebook/entrenamiento_clasificador.ipynb`](notebook/entrenamiento_clasificador.ipynb), viewable on GitHub with all its plots.

## Results

| Model | Test accuracy | Macro F1 | Accuracy on out-of-range wells |
|---|---|---|---|
| Engineer's rules (fillage, max./min. load) | 49.9 % | 0.38 | 46.7 % |
| **Gradient boosting** (deployed) | **98.7 %** | **0.987** | **96.7 %** |
| MLP neural network 64-32 | 99.0 % | 0.990 | 96.2 % |

Out-of-range wells are deeper (2250–2600 m), faster (11–12.5 SPM) and have other plungers and fluids. The model never
saw them and still gets 96.7 % right.

**Downhole or surface card?** The same model trained on the **surface** card reaches 89.9 % on the test set, but only
**33.8 %** out of range. The surface card depends on each well's rod-string dynamics. That is why the industry
diagnoses with the downhole card, computed with the wave equation, and this analysis confirms it with data.

![Confusion matrix](figuras/02_matriz_confusion.png)

## Phenomenology: how it was verified that everything respects the physics

1. **When generating the data.** Each sample is discarded if its physics does not match its label (`fenomenologiaOK` in `datos/generar_dataset.js`):
   - fluid pound: fillage < 0.88;
   - gas interference: fillage < 0.92;
   - leaks: at least 8 % less production than the same healthy well;
   - parted rod: production ≈ 0;
   - tagging bottom: compressive load < −3 kN.

   476 "tagging bottom" cases where the plunger never reached the bottom were discarded.
2. **A simulator bug caught by these checks.** The travelling-valve leak *increased* production, because the leaked
   liquid was counted again as produced. It was fixed in the core and also in
   [`bombeo_mecanico/`](https://github.com/juanlaigle-hub/Simulations3D/tree/main/bombeo_mecanico), in a separate commit. Production now drops with severity: 251 → 192 / 144 / 94 bbl/d.
3. **Before training.** The notebook checks the per-class averages with `assert`: fillage, relative production and loads.
4. **After training, with continuous sweeps.** A well the model never saw is used:
   - inflow decreases gradually: the model switches from "normal" to "fluid pound" when fillage drops below 0.88,
     exactly where the physics says;
   - leak severity goes from 0 to 5: the probability of the correct leak grows to 1.

   At severities lower than those in training the model hesitates, and that is shown.
5. **Explainability.** The most important features are the plunger/surface stroke ratio (parted rod), the relative
   minimum load (tagging bottom), the maximum load and the card area (fillage), all physically meaningful.

![Phenomenology sweeps](figuras/05_fenomenologia_barridos.png)

## Step by step: how this is done in a real field

| Step | In a real project | In this exercise |
|---|---|---|
| **1. Define the problem and the KPI** | Which faults matter, how much missing them costs and how much a false alarm costs. Goal: high recall on parted rod and leaks | 7 classes; recall on critical faults is prioritized |
| **2. Get the data** | Export surface cards from months of operation from SCADA or the POCs, with each well's conditions (depth, rods, plunger, SPM, fluid level) | `datos/generar_dataset.js` simulates 2800 different wells with dynamometer noise |
| **3. Compute the downhole card** | With the Gibbs wave equation from the surface card. It is standard software in diagnostic systems | The simulator delivers the downhole card directly |
| **4. Label** | Production engineers label a representative sample, and labels are cross-checked with pulling jobs (what was found when the pump came out) | Labels by construction, with phenomenology checks that discard ambiguous cases |
| **5. Clean and normalize** | Discard incomplete cards, resample to fixed points, normalize by the well's fluid load | 32 + 32 point resampling and scale features relative to $F_o$ |
| **6. Split the data** | **By well**, never randomly across cards of the same well, to avoid leakage. Hold out a whole field or period for the final test | Each sample is a different well, plus a set of out-of-range wells |
| **7. Baseline** | Compare against the rules the POC already has | The rules get 50 % right |
| **8. Train and compare models** | Gradient boosting, neural networks and, with abundant real data, convolutional networks on the card image | Gradient boosting and MLP |
| **9. Validate the physics** | Review confusion matrices, important features and edge cases with specialists | Continuous sweeps, `assert` and explainability |
| **10. Deploy** | Export the model (ONNX, PMML or the SCADA format) and check it gives the same result as in Python. Run it at the end of each stroke and send alerts to the control room | Exported to JSON (`modelo/clasificador_cartas.json`), verified with zero difference against scikit-learn, and run in `clasificador.html` |
| **11. Confidence threshold** | If the maximum probability is low, the card goes to human review | 70 % threshold |
| **12. Monitor and retrain** | Watch feature drift month by month, log engineers' corrections and retrain | Discussed in the notebook |

## How to use it

```bash
cd ml_cartas_dinamometricas/datos
node generar_dataset.js 400 7 base      # 2800 cards (≈ 8 min)
node generar_dataset.js 60 99 fuera     # 420 cards from out-of-range wells
node generar_dataset.js 1 5 barrido     # sweeps to verify the phenomenology
cd ../notebook
pip install -r ../requirements.txt
python construir_notebook.py entrenamiento_clasificador.py   # trains, evaluates, exports and saves the executed .ipynb
```

Then open `clasificador.html` in the browser: it is the pumping simulator with the "Automatic diagnosis · machine
learning" panel. At every stroke it shows the predicted class, the confidence, the 7 probabilities and the fault that is
active in the simulation, for comparison. The "Faults" tab lets you trigger any fault, and the "▶ Guided demo" button
goes through them one by one.

## Files

| File | Contents |
|---|---|
| `datos/simulador_bombeo.js` | Copy of the pumping physics core (wave equation, pump, valves) |
| `datos/generar_dataset.js` | Dataset generator with phenomenology checks |
| `datos/dataset.csv`, `dataset_fuera_de_rango.csv`, `dataset_barrido.csv` | Generated data (2800 + 420 + 73 cards) |
| `notebook/entrenamiento_clasificador.py` / `.ipynb` | Training, evaluation, phenomenology and export |
| `modelo/clasificador_cartas.json` | Deployable model (1750 trees) |
| `modelo/comparacion_modelos.csv` | Results table |
| `figuras/` | Notebook plots |
| `clasificador.html` | Simulator with live diagnosis |

## Limitations

- The data is synthetic. The model is a **starting point**: before using it in operation, its accuracy must be measured
  on real labelled cards and it must be retrained with them.
- Some real faults are not modelled by the simulator: worn barrel, stuck plunger, friction in deviated wells,
  combined faults.
- With low intake pressure, free gas produces cards similar to fluid pound. This is a real confusion in the industry,
  and the diagnosis should be cross-checked with the measured fluid level (echometer).
