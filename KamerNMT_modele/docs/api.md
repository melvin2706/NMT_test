# Service de traduction (API JSON)

KamerNMT peut servir de moteur de traduction à une application externe : l'application envoie
les données à traduire en JSON et reçoit le même contenu traduit, avec exactement la même structure.
**Seules les valeurs sont traduites, jamais les clés.**

## Démarrer le service

```bash
kamernmt-server                                   # http://127.0.0.1:8000 (cet ordinateur seulement)
kamernmt-server --host 0.0.0.0 --port 8080        # accessible depuis le réseau
kamernmt-server --model models/kamernmt-small-v1-int8 --cors https://mon-app.cm
python server.py ...                              # identique, depuis le dossier du projet
```

| Option | Rôle |
|---|---|
| `--host` | `127.0.0.1` (défaut) : appels depuis cet ordinateur uniquement ; `0.0.0.0` : depuis le réseau |
| `--port` | port d'écoute (défaut 8000) |
| `--model` | dossier de modèle ; par défaut le plus récent de `models/` |
| `--cors ORIGINE` | autoriser une application web (navigateur) de cette origine, ou `*` |
| `--beam N` | qualité / vitesse (défaut 4 ; 1 = le plus rapide) |
| `--no-pivot` | traduire directement sans passer par le français |
| `--device` | `auto` (défaut : GPU si disponible, le CPU prend le relais), `cpu`, `cuda` |
| `--quiet` | ne pas afficher chaque requête |

Le service fonctionne hors ligne ; il utilise le GPU s'il y en a un, sinon le CPU. Le modèle est chargé une seule fois au démarrage.

## Requête : `POST /translate`

En-tête `Content-Type: application/json`. Corps :

| Champ | Obligatoire | Description |
|---|---|---|
| `source_language` | **oui** | code de la langue source : `fr`, `dua`, `en`… |
| `target_language` | **oui** | code de la langue cible |
| `content` | **oui** | ce qu'il faut traduire : un texte, une liste ou un objet JSON (toute profondeur) |
| `skip_keys` | non | liste de clés dont les valeurs ne sont jamais traduites, ex. `["id", "url", "code"]` |
| `only_keys` | non | liste de clés : seules leurs valeurs sont traduites |

Exemple :

```json
{
  "source_language": "fr",
  "target_language": "dua",
  "content": {
    "id": "a-12",
    "titre": "Bonjour {name}, bienvenue !",
    "prix": 2500,
    "menu": ["Accueil", "Mon compte"],
    "detail": {"texte": "<b>Attention</b> : le marché ferme à 18 h."}
  },
  "skip_keys": ["id"]
}
```

## Réponse : `200`

```json
{
  "source_language": "fr",
  "target_language": "dua",
  "content": { "...": "même structure que la requête, valeurs traduites" },
  "route": ["fr", "dua"],
  "model": "kamernmt-small-v1-int8"
}
```

- `content` a **exactement la structure** du `content` envoyé : mêmes clés, même ordre, même
  imbrication. Un texte donne un texte, une liste une liste, un objet un objet.
- Nombres, booléens et `null` sont renvoyés tels quels.
- Conservés à l'identique dans les textes : `{name}`, `{{count}}`, `%s`, `%(user)s`, `${var}`,
  balises HTML, URL, e-mails, retours à la ligne, espaces de début et de fin.
- `route` indique le chemin suivi : `["en", "fr", "dua"]` quand le français sert de pivot.

### Recevoir un fichier : `POST /translate?download=1`

Même requête ; la réponse est **uniquement le contenu traduit**, envoyé comme fichier JSON
(`Content-Disposition: attachment; filename="translation.<langue cible>.json"`).

## Erreurs

Toujours en JSON : `{"error": "message"}`.

| Code | Cas |
|---|---|
| 400 | champ obligatoire manquant, langue inconnue, JSON invalide, `skip_keys`/`only_keys` mal formés |
| 404 | mauvaise adresse (utiliser `POST /translate`) |
| 413 | requête trop grosse (plus de 20 Mo) |
| 500 | erreur pendant la traduction (le service continue de fonctionner) |

## Autres adresses

- `GET /health` : `{"status": "ok", "model": …, "pivot": "fr", "languages": [...]}` — vérifier que le service répond.
- `GET /languages` : langues connues, directions réellement entraînées, langue pivot.

## Exemples d'appel

**curl**

```bash
curl -X POST http://127.0.0.1:8000/translate -H "Content-Type: application/json" \
     -d "{\"source_language\": \"fr\", \"target_language\": \"dua\", \"content\": {\"titre\": \"Je vais au marché.\"}}"

curl -X POST "http://127.0.0.1:8000/translate?download=1" -H "Content-Type: application/json" \
     --data-binary @requete.json -o traduction.dua.json
```

**Python**

```python
import requests
r = requests.post("http://127.0.0.1:8000/translate", json={
    "source_language": "fr", "target_language": "dua",
    "content": {"titre": "Je vais au marché.", "prix": 2500}})
r.raise_for_status()
traduit = r.json()["content"]
```

**JavaScript**

```javascript
const r = await fetch("http://127.0.0.1:8000/translate", {
  method: "POST",
  headers: {"Content-Type": "application/json"},
  body: JSON.stringify({source_language: "fr", target_language: "dua",
                        content: {titre: "Je vais au marché."}}),
});
const {content} = await r.json();
```

## Sans serveur : échange par fichiers

Même format de requête, dans un fichier ; le résultat est un fichier contenant le contenu traduit :

```bash
kamernmt --request requete.json --output traduction.dua.json
```

## À savoir

- Les traductions sont faites une par une (le modèle est partagé) ; les textes identiques d'un
  même document ne sont traduits qu'une fois, et le service garde en mémoire les traductions déjà faites.
- Le service n'a pas d'authentification : avec `--host 0.0.0.0`, le placer derrière un pare-feu ou
  un proxy (nginx, etc.) si le réseau n'est pas de confiance.
