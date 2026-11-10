# GenImage AI Detector

Progetto di deep learning per il riconoscimento di immagini generate da intelligenza artificiale.

L'obiettivo è sviluppare e confrontare modelli di reti neurali in grado di distinguere immagini reali da immagini generate artificialmente, utilizzando il dataset GenImage.

## Obiettivi

- Preparare e analizzare il dataset GenImage.
- Addestrare un classificatore real/fake.
- Confrontare modelli basati su CNN e, opzionalmente, Transformer.
- Valutare la capacità del modello di generalizzare su immagini generate da modelli diversi.

## Tecnologie

- Python 3.11
- PyTorch / torchvision / timm
- scikit-learn, pandas
- matplotlib, seaborn
- Optuna (ottimizzazione iperparametri)
- TensorBoard (logging training)

## Struttura

```
configs/      # YAML per esperimenti (uno per modello)
data/         # (gitignored) subset GenImage + splits/ — solo sulla VM
scripts/      # download_subset, build_splits, make_fixtures, slurm/
src/          # datasets, transforms, degradations, models, train, evaluate, metrics, utils
tests/        # smoke test + fixtures finte per CPU
notebooks/    # EDA e visualizzazione risultati
experiments/  # (gitignored) output per-run: config, log, checkpoint
results/      # (gitignored) tabelle e grafici finali
```

## Workflow di sviluppo (locale ↔ VM)

Sviluppo **ibrido**: codice scritto/testato in locale (CPU), training eseguito sulla VM (GPU T4 via SLURM).
Il dataset e la GPU stanno **solo sulla VM**. Il ponte tra i due ambienti è **git**.

### Setup locale (CPU, per sviluppo e smoke test)

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows;  source .venv/bin/activate su Linux/Mac
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### Setup VM (GPU, CUDA)

```bash
conda env create -f environment.yml
conda activate genimage
```

### Smoke test (locale, senza GenImage)

```bash
python scripts/make_fixtures.py   # genera un dataset finto minuscolo in tests/fixtures/
pytest tests/test_smoke.py -v     # verifica utility di base su CPU
```

I componenti dati-dipendenti accettano `--data-root`, così lo stesso codice punta a
`tests/fixtures/` in locale e a `data/` sulla VM.

## Valutazione con image degradation

La valutazione supporta degradazioni deterministiche applicate al test set dopo
Resize e CenterCrop, prima di ToTensor/normalizzazione, cosi' le intensita' sono
confrontabili tra classi e generatori. La singola eval storica resta invariata:

```bash
python -m src.evaluate --config experiments/cnn_custom_baseline/config.yaml \
    --data-root data/genimage_subset
```

Per valutare il test in-distribution con la suite base:

```bash
python -m src.evaluate \
    --config configs/swin_tiny_crossgen_7gen_test_sdv5.yaml \
    --data-root data/genimage_subset \
    --split splits/crossgen_7gen_test_sdv5/test_in_distribution.csv \
    --out-dir experiments/swin_tiny_crossgen_7gen_test_sdv5/eval_in_distribution \
    --degradations basic
```

Per valutare l'heldout con la suite base:

```bash
python -m src.evaluate \
    --config configs/swin_tiny_crossgen_7gen_test_sdv5.yaml \
    --data-root data/genimage_subset \
    --split splits/crossgen_7gen_test_sdv5/test_sdv5.csv \
    --out-dir experiments/swin_tiny_crossgen_7gen_test_sdv5/eval_heldout \
    --degradations basic
```

Output per split valutato:

```text
experiments/<run>/eval_in_distribution/<degradation>/metrics.json
experiments/<run>/eval_in_distribution/<degradation>/metadata.json
experiments/<run>/eval_in_distribution/<degradation>/confusion_matrix.png
experiments/<run>/eval_heldout/<degradation>/metrics.json
experiments/<run>/eval_heldout/<degradation>/metadata.json
experiments/<run>/eval_heldout/<degradation>/confusion_matrix.png
```

## Preparazione dati GenImage (sulla VM)

GenImage si distribuisce su Google Drive come **ZIP multi-volume per generatore**
(`imagenet_ai_<data>_<gen>.z01 … .z12 + .zip`, ~36 GB l'uno). Flusso, **un generatore alla volta**:

```bash
# 1. ricomporre i volumi ed estrarre (NON unzippare i singoli .z0x)
zip -s 0 imagenet_ai_0508_adm.zip --out adm_full.zip
unzip adm_full.zip                       # -> imagenet_ai_0508_adm/{train,val}/{ai,nature}/

# 2. campionare un subset bilanciato nel dataset incrementale persistente
python scripts/sample_generator.py --src imagenet_ai_0508_adm \
    --dst data/genimage_subset --train-per-class 5000 --val-per-class 2000

# 3. liberare disco: cancellare archivi grezzi + estrazione, tenere solo il subset
rm -rf imagenet_ai_0508_adm adm_full.zip imagenet_ai_0508_adm.z*
```

Ripetendo per ogni generatore, `data/genimage_subset/` cresce con `imagenet_<gen>/{train,val}/{ai,nature}/`.
Poi si generano i CSV di split per gli esperimenti (non copia immagini, solo percorsi):

```bash
# baseline real/fake misto su tutti i generatori scaricati
python scripts/build_splits.py --data-root data/genimage_subset --name baseline_misto --mode mixed
# cross-generator: train sul pool, test su ogni held-out
python scripts/build_splits.py --data-root data/genimage_subset --name crossgen_sdv4 \
    --mode cross-gen --pool sdv4 --heldout adm biggan glide
```

`sample_generator` copia solo `train/val`; `build_splits` ricava il **test** dividendo la val
(GenImage non ha test ufficiale) e compone gli split. In locale lo stesso `build_splits` gira su `tests/fixtures/`.

## Stato

- [x] **M0** — Setup ambiente, utility, fixtures, smoke test
- [ ] **M1** — Pipeline dati (subset GenImage, Dataset/DataLoader)
- [ ] **M2** — Baseline CNN + training/eval loop
- [ ] **M3** — Modelli pretrained (ResNet-50, EfficientNet-B0)
- [ ] **M4** — Transformer (Swin-Tiny)
- [ ] **M5** — Cross-generator · **M6** — Degradation · **M7** — Optuna · **M8** — Report
