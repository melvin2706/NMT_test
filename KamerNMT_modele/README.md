# KamerNMT

Modèle neuronal de traduction multilingue (français, anglais, langues camerounaises),
léger et optimisé pour l'inférence CPU. Développement indépendant : aucun fichier,
donnée, tokenizer, poids ni script de Cameroon-int8 n'est utilisé (voir §4 de la spécification).

> **État actuel :** la chaîne complète (données → tokenizer → entraînement →
> évaluation → export INT8 → benchmark) est implémentée et testée de bout en bout.
> Données : dictionnaire duala→français et lexique français→duala (`data/dictionaries/`,
> ~35 600 paires dont ~9 650 phrases d'exemple) et ~92 000 paires de mots français↔anglais.
> La transcription duala n'est pas encore relue par un locuteur : l'utiliser avec
> `--allow-unvalidated`. Aucune direction n'est encore déclarée « prise en charge »
> (il faut d'abord une évaluation humaine).

## Démarrage rapide

```bash
setup.bat                      # Windows — ou ./setup.sh sous Linux/macOS
.venv\Scripts\activate         # Linux/macOS : source .venv/bin/activate
python main.py --demo          # vérifier que tout marche (quelques minutes)
python main.py                 # entraîner le modèle -> dossier prêt à l'emploi dans models/
kamernmt                       # traduire en mode interactif
```

`setup.bat` / `setup.sh` crée l'environnement `.venv`, installe PyTorch (CPU), les bibliothèques
de `requirements.txt` et la commande `kamernmt`. `setup.bat gpu` installe PyTorch pour carte
NVIDIA (entraînement plus rapide). Installation manuelle :

```bash
python -m venv .venv
.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m pip install -e .
```

## Entraîner : `main.py`

`main.py` enchaîne : vérification des sources → corpus → tokenizer → entraînement →
évaluation → export INT8 → évaluation INT8 (+ comparaison) → benchmark CPU.
Options utiles : `--config configs/tiny.yaml` (modèle plus petit, plus rapide), `--steps train,evaluate`,
`--force` (refaire les étapes déjà faites), `--allow-unvalidated` (utiliser des paires non relues).

Résultats :

- **`models/kamernmt-<expérience>-int8/`** : le modèle prêt à l'emploi, autonome. Il contient
  le modèle compressé, le tokenizer et `kamernmt.json` (langues connues, directions entraînées,
  pivot, version). On peut copier ce dossier sur une autre machine et le charger par son chemin.
- `runs/<expérience>/` : points de reprise de l'entraînement ; `reports/<expérience>/` : métriques, benchmark.

## Traduire : commande `kamernmt`, `translate.py` ou Python

La commande `kamernmt` (installée par `setup.bat` ou `pip install -e .`) et `python translate.py`
(depuis le dossier du projet, sans installation) font exactement la même chose :

```bash
kamernmt                                          # mode interactif : tapez une phrase, Entrée
kamernmt --tgt dua "Je vais au marché."           # une ou plusieurs phrases
kamernmt --src fr --tgt dua --input doc.json --output doc.dua.json   # document JSON
kamernmt --model chemin/vers/un-modele ...        # utiliser un dossier de modèle précis
```

Le modèle est trouvé automatiquement : variable `KAMERNMT_MODEL`, sinon le dossier le plus récent
de `models/` (dossier courant, puis dossier du projet), sinon `runs/*/best.pt`.

- **Français = langue pivot** (`pivot: fr` dans `configs/languages.yaml`, recopié dans
  `kamernmt.json`) : quand ni la source ni la cible n'est le français, le texte passe par le
  français (`en → dua` = `en → fr → dua`). `--no-pivot` pour traduire directement.
- **Le résultat garde le format de l'entrée** : une phrase donne une phrase ; un document JSON
  donne le même document (mêmes clés, même imbrication, même ordre), seules les valeurs texte
  sont traduites. Nombres, booléens et `null` sont inchangés.
- **Ce qui n'est jamais traduit** : `{name}`, `{{count}}`, `%s`, `%(user)s`, `${var}`, balises
  `<b>…</b>`, URL, e-mails, retours à la ligne, espaces de début et de fin, chaînes sans lettres.
- `--skip-keys id,url,code` : clés dont les valeurs ne sont jamais traduites ;
  `--only-keys title,description` : traduire uniquement ces clés.
- Les textes longs sont découpés en phrases, traduits puis recollés avec leurs séparateurs.
- Mode interactif : `:src en`, `:tgt dua`, `:swap`, `:langs` (indique les langues entraînées),
  `:help`, `:q`.

Exemple : `{"titre": "Bienvenue {name} !", "prix": 2500, "lien": "https://exemple.cm", "menu": ["Accueil", "Mon compte"]}`
devient le même objet avec `titre` et `menu` traduits, et `{name}`, `prix`, `lien` intacts.

Depuis Python (intégration dans une application) :

```python
from kamernmt import Translator

tr = Translator()                                   # trouve le modèle tout seul
tr.translate("Je vais au marché.", src="fr", tgt="dua")                 # -> une phrase
tr.translate(["Bonjour", "Merci"], src="fr", tgt="dua")                 # -> une liste
tr.translate(document_json, src="fr", tgt="dua", skip_keys={"id"})      # -> le même document

import kamernmt
kamernmt.translate("Je vais au marché.", "fr", "dua")                   # encore plus court
```

## GPU et CPU

KamerNMT utilise la carte graphique NVIDIA quand elle est disponible, et le processeur sinon.

- **Entraînement** (`device: auto` dans `configs/small.yaml` / `tiny.yaml`) : GPU si PyTorch le voit,
  avec précision mixte (bf16/fp16) et TF32. Le CPU prépare les lots pendant ce temps. Si la mémoire
  du GPU est pleine, la taille des lots est divisée par deux et l'entraînement continue au lieu de s'arrêter.
- **Traduction** (`--device auto`, par défaut, pour `kamernmt`, `kamernmt-server` et `Translator(device=...)`) :
  le modèle est chargé sur le GPU ; un lot qui ne tient pas en mémoire GPU est découpé, et ce qui
  échoue encore est traduit par le CPU. Si le GPU échoue à répétition, le CPU prend le relais pour de bon.
  `--device cpu` force le processeur, `--device cuda` exige le GPU.
- `GET /health` du service indique le matériel utilisé (`"device": "cuda (+cpu fallback)"` ou `"cpu"`).

Pour utiliser un GPU : pilote NVIDIA à jour, puis `setup.bat gpu` (ou `./setup.sh gpu`), qui installe
PyTorch avec CUDA. CTranslate2 utilise le GPU s'il trouve les bibliothèques CUDA 12 ; pour un modèle
ONNX sur GPU, installer `onnxruntime-gpu` à la place de `onnxruntime`.

## Service pour applications externes (API JSON)

```bash
kamernmt-server                     # http://127.0.0.1:8000  (ou : python server.py)
```

Une application envoie `POST /translate` avec `source_language`, `target_language` et `content`
(texte, liste ou objet JSON) et reçoit `content` traduit, avec la même structure : **seules les
valeurs sont traduites, jamais les clés**. `?download=1` renvoie le résultat comme fichier JSON.
Sans serveur : `kamernmt --request requete.json --output traduction.json`.
Détails, codes d'erreur et exemples (curl, Python, JavaScript) : **[docs/api.md](docs/api.md)**.

## Vérifier l'installation (démonstration)

```bash
python main.py --demo
```

Génère des phrases françaises/anglaises par gabarits et une **langue factice** `xxa`
(règle réversible, pas une vraie langue), puis exécute toute la chaîne dans `work/smoke/`.

## Chaîne de production

Toutes les commandes : `python -m kamernmt.cli <commande> --help`.

| Étape (spec) | Commande |
|---|---|
| 1. Déclarer chaque source | éditer `configs/sources.yaml`, puis `check-sources` |
| 2. Corpus : import, normalisation, validation, dédoublonnage, découpage | `build-corpus --out data/corpus/v1` |
| 3. Phrases à faire traduire par des locuteurs | `worksheet-export` → traduction humaine → `worksheet-import` |
| 4. Tokenizer : BPE vs Unigram, 8k/16k/32k | `tokenizer-sweep --corpus data/corpus/v1` |
| 5. Taille exacte du modèle | `params --config configs/small.yaml --vocab 16000` |
| 6. Entraînement (reprise : `--resume`) | `train --config configs/small.yaml` |
| 7. Évaluation par direction (BLEU, chrF++, TER) | `evaluate --model runs/small-v1/best.pt --corpus data/corpus/v1 --out reports/small-v1-fp32` |
| 8. Export CPU INT8 | `export-ct2 --checkpoint runs/small-v1/best.pt --out models/kamernmt-small-v1-int8` / `export-onnx ...` |
| 9. Qualité après quantification | `evaluate --model models/kamernmt-small-v1-int8 ...` puis `compare old/metrics.json new/metrics.json` |
| 10. Benchmark CPU (latence P95, RAM, débit, taille) | `benchmark --model models/kamernmt-small-v1-int8 --input phrases.txt --src fr --tgt en` |
| 11. Évaluation humaine | `human-eval-sheet --eval-dir reports/... --out human_eval.tsv` |
| 12. Rétro-traduction (données synthétiques) | `backtranslate ...` puis déclarer la source avec `synthetic: true` |

Traduction locale : voir « Traduire (moteur de traduction) » plus haut.

## Choix techniques

- **Normalisation** (`kamernmt/data/normalize.py`) : NFC uniquement ; accents, tons,
  ɛ ɔ ŋ ə ʉ conservés ; les apostrophes sont unifiées (pas supprimées) car elles sont
  des lettres dans plusieurs orthographes (Ghomálá', Lamnso'). Caractère cible réglable par langue.
- **Données** : chaque paire garde l'identifiant de sa source. Toute source déclarée dans
  `configs/sources.yaml` est utilisée (sauf `enabled: false`) ; celles qui correspondent à la
  liste noire (`cameroon-int8`, `flagship`) sont refusées. Les paires non relues
  (`validated: false`) vont en quarantaine, sauf avec `--allow-unvalidated`. Le découpage train/valid/test
  se fait par groupe (une phrase et sa traduction inverse restent dans la même partition) ;
  valid/test ne contiennent que des phrases validées, jamais des mots isolés ni du synthétique.
- **Données synthétiques** : les traductions vers les langues camerounaises viennent de
  traducteurs humains (feuilles de travail). La rétro-traduction est marquée `synthetic`,
  jamais inversée (le texte machine n'est jamais une cible), exclue par défaut (`include_synthetic: false`).
- **Tokenizer** : SentencePiece, normalisation `identity`, `character_coverage=1.0`,
  `byte_fallback`, balises `<src:xx>` / `<tgt:xx>` pour toutes les langues du registre
  (ajouter une langue ne décale pas les identifiants existants). Échantillonnage par température
  pour que le français et l'anglais n'écrasent pas le vocabulaire.
- **Modèle** (`kamernmt/model.py`) : Transformer encodeur-décodeur pre-norm écrit de zéro,
  embeddings partagés, encodage positionnel sinusoïdal, recherche en faisceau avec cache KV.

  | Config | Paramètres (vocab 8k / 16k / 32k) | INT8 approx. |
  |---|---|---|
  | Tiny (4+4 couches, d=256) | 9,4 M / 11,5 M / 15,6 M | 9–16 Mo |
  | Small (6+6 couches, d=512) | 48,2 M / 52,3 M / 60,5 M | 48–61 Mo |

- **Entraînement** : échantillonnage multilingue par température, lissage des étiquettes,
  warmup + inverse-sqrt, perte de validation par direction, arrêt anticipé, points de reprise.
  `run_info.json` enregistre configuration, hachages du corpus et du tokenizer, graine et versions.
- **Export** : conversion directe vers CTranslate2 (correspondance des poids écrite à la main)
  et ONNX (+ quantification dynamique INT8 d'onnxruntime). En float32, les deux exports
  donnent des sorties identiques à PyTorch (vérifié).

## Ajouter un dictionnaire ou une langue (spec §12.1)

Guide complet, avec les formats acceptés et des exemples : **[docs/ajouter_un_dictionnaire.md](docs/ajouter_un_dictionnaire.md)**.

En bref :

1. Mettre le fichier dans `data/dictionaries/` : tableau CSV/TSV, JSON au format de
   `duala_fr.json` / `fr_duala.json`, ou phrases parallèles TSV.
2. Le déclarer dans `configs/sources.yaml` (`id`, `path`, `format`, `kind`, `language`, `validated`) ;
   ajouter la langue dans `configs/languages.yaml` si elle est nouvelle. Vérifier avec
   `python -m kamernmt.cli check-sources`.
3. Relancer `python main.py --force` (`--force` est indispensable : nouveau tokenizer et
   nouvel entraînement), puis comparer avec la version précédente (`compare`).
4. Ne déclarer une direction dans `supported_directions` qu'après évaluation humaine.

## Arborescence

```
configs/        registre des langues, sources, configurations Tiny/Small/smoke
data/dictionaries/  duala_fr.json, fr_duala.json, dictionary-fr.json
docs/           ajout d'un dictionnaire, description des données
kamernmt/
  data/         schéma, normalisation, import, validation, corpus, feuilles de travail
  tokenizer.py  entraînement et évaluation SentencePiece
  model.py      Transformer + génération
  train.py      boucle d'entraînement
  export_ct2.py / export_onnx.py
  engine.py     moteur de traduction : pivot français, phrases, documents JSON
  inference.py  chargement des modèles (PyTorch, CTranslate2, ONNX)
  evaluate.py   BLEU / chrF++ / TER, comparaison de versions, fiche d'évaluation humaine
  benchmark.py  latence, débit, RAM, CPU, taille
  translate_cli.py  commande `kamernmt` (interactif, phrases, documents JSON, fichier de requête)
  server.py     service HTTP JSON (`kamernmt-server`)
  cli.py        commandes avancées (python -m kamernmt.cli …)
main.py         entraînement complet
translate.py    traduction sans installation (= commande kamernmt)
server.py       service HTTP sans installation (= kamernmt-server)
setup.bat / setup.sh  installation en une commande
models/         modèles prêts à l'emploi (créé par main.py)
scripts/        smoke_test.py (démo), dictionary_review.py (relecture d'un dictionnaire)
```
