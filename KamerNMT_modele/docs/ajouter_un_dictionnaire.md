# Ajouter un dictionnaire ou une langue

Trois étapes, sans modifier le code :

1. mettre le fichier dans `data/dictionaries/` au bon format ;
2. le déclarer dans `configs/sources.yaml` (et la langue dans `configs/languages.yaml` si elle est nouvelle) ;
3. relancer `python main.py --force`.

## 1. Formats acceptés

### A. Tableau CSV / TSV (`format: csv` ou `tsv`)

Le plus simple, pour un dictionnaire saisi à la main ou exporté d'Excel. Une ligne par mot ;
plusieurs valeurs dans une case séparées par `|`. Encodage UTF-8.

```text
headword,fr,en,pos,variants,example_src,example_fr,example_en
mbɔ́ŋ,manioc,cassava,n.,,,,
ndáp,maison|case,house,n.,ndab,ndáp ya mə́,ma maison,my house
```

| Colonne | Contenu | Obligatoire |
|---|---|---|
| `headword` | le mot dans la langue locale | oui |
| `fr`, `en` | traductions (séparées par `\|`) | au moins une |
| `pos` | catégorie grammaticale | non |
| `variants` | autres orthographes | non |
| `synonyms` | synonymes | non |
| `example_src`, `example_fr`, `example_en` | une phrase d'exemple et ses traductions | non |
| `morphology` | informations morphologiques | non |

Si les colonnes du fichier portent d'autres noms, on les fait correspondre avec `columns:`
dans la déclaration (voir étape 2).

### B. JSON structuré (`format: entries_jsonl`)

Même format que `data/dictionaries/duala_fr.json` — adapté aux dictionnaires riches
(plusieurs sens et plusieurs exemples par mot) :

```json
{
 "direction": "ewo-fr",
 "entries": [
  {"headword": "ndáp", "pos": "n.", "senses": ["maison", "case"],
   "examples": [{"dua": "ndáp ya mə́", "fr": "ma maison"}],
   "see": [], "uncertain": []}
 ]
}
```

- `direction` doit commencer par le code de la langue locale (`ewo-fr`).
- Dans `examples`, la clé de la phrase locale s'appelle `dua` quelle que soit la langue.
- `~` dans un exemple est remplacé automatiquement par le mot-vedette.
- Texte descriptif qui n'est pas une traduction → `definition` (non utilisé pour l'entraînement).

Pour un lexique français → langue locale, copier le format de `fr_duala.json` :
`"direction": "fr-ewo"` et des entrées `{"fr": "maison", "dua": ["ndáp"]}`.

### C. Phrases parallèles (`format: parallel_tsv`, `kind: parallel`)

Pour des traductions de phrases (feuilles de traduction, corpus). TSV avec en-tête :

```text
src_lang	tgt_lang	src_text	tgt_text	group	validated
fr	ewo	Je vais au marché.	<traduction validée>	g1	true
```

`group` et `validated` sont facultatifs. Le sens inverse est ajouté automatiquement.

### D. Export Wiktionary compact (`format: wiktionary_jsonl`)

Format de `data/dictionaries/dictionary-fr.json` : une ligne JSON par mot,
`{"": "manger", "p": ["verb"], "d": ["(transitive) to eat", "food, foodstuff"], ...}`.
Déclarer `language` (langue des mots) et `gloss_language` (langue des définitions `d`).
Seules les gloses utilisables comme traductions sont gardées : les formes fléchies
(« plural of… », « first-person singular… of… »), noms propres, abréviations et définitions
longues sont écartés (`kamernmt/data/wiktionary.py`). Si `gloss_language` = `language`
(dictionnaire monolingue), aucune paire n'est produite.

```yaml
  - id: wiktionary-fr-en
    path: data/dictionaries/dictionary-fr.json
    format: wiktionary_jsonl
    kind: dictionary
    language: fr
    gloss_language: en
    origin: "English Wiktionary, compact export"
    validated: true
```

## 2. Déclarer la source

Dans `configs/sources.yaml`, sous `sources:` :

```yaml
  - id: ewo-dictionnaire-2026            # nom unique
    path: data/dictionaries/ewondo.csv
    format: csv                          # csv | tsv | entries_jsonl | parallel_tsv | wiktionary_jsonl
    kind: dictionary                     # dictionary | parallel
    language: ewo                        # code de la langue locale
    origin: "dictionnaire constitué pour KamerNMT"   # description libre (facultatif)
    validated: true                      # true si relu par un locuteur compétent
    # enabled: false                     # pour mettre une source de côté sans la supprimer
    columns:                             # seulement si les colonnes ont d'autres noms
      headword: mot
      fr: francais
```

Règles appliquées automatiquement :

- champ obligatoire manquant (`id`, `path`, `format`, `kind`, `language` pour un dictionnaire)
  ou fichier introuvable → **arrêt** du pipeline avec un message clair ;
- nom contenant `cameroon-int8` ou `flagship` → **refusée** (spécification §4.2) ;
- `enabled: false` → source ignorée ;
- `validated: false` → paires mises en quarantaine, sauf avec `--allow-unvalidated`.

**Nouvelle langue :** ajouter son code dans `configs/languages.yaml`
(ewondo, duala, bassa, fulfulde, ghomálá', medumba, yemba, lamnso', kom y sont déjà).

Vérifier la déclaration :

```bash
python -m kamernmt.cli check-sources
```

## 3. Relancer

```bash
python main.py --force
python main.py --force --allow-unvalidated     # si la source n'est pas encore relue
```

**`--force` est indispensable** quand on ajoute une langue ou des caractères nouveaux :
sans lui, `main.py` réutilise l'ancien tokenizer (qui ne connaît pas les nouveaux caractères)
et tente de reprendre l'ancien entraînement, incompatible avec le nouveau vocabulaire.

Comparer ensuite avec la version précédente :

```bash
python -m kamernmt.cli compare reports/<ancienne>/fp32/metrics.json reports/<nouvelle>/fp32/metrics.json
```

Ne déclarer une direction dans `supported_directions` (`configs/languages.yaml`) qu'après
évaluation humaine (spécification §5.2).

## Vérifier avant d'entraîner

Voir exactement ce que le modèle apprendra d'une source :

```bash
python scripts/dictionary_review.py --source-id ewo-dictionnaire-2026 --out work/review/ewondo
```

Produit `pairs.tsv` (paires d'entraînement), `review.tsv` (à faire relire, colonnes
« correct » / « correction ») et `review.html`.

## Cas d'un livre scanné

Un PDF sans texte doit d'abord être transcrit vers le format B. Les outils utilisés pour le
dictionnaire Helmlinger (scan, fichiers page par page, script de transcription) ont été supprimés
le 2026-11-08 ; seul le résultat est conservé (`duala_fr.json`, `fr_duala.json`). Pour un autre
livre scanné, il faudra recréer un outil de transcription.

## Conseils

- Les **phrases d'exemple** apprennent beaucoup plus au modèle que les listes de mots.
- Garder les caractères propres à la langue (ɛ ɔ ŋ ə, tons, apostrophes) : la normalisation
  les conserve (NFC uniquement).
- Une langue doit avoir des données relues par un locuteur avant d'être déclarée prise en charge.
