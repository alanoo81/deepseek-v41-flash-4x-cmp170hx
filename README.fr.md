# DeepSeek-V4.1-Flash-1017 sur 4x CMP 170HX, la fidélité d'abord

[English version](README.md) (référence complète ; ce résumé renvoie aux mêmes fichiers)

Ce dépôt décrit comment nous servons [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) sur 4 cartes CMP 170HX de 64 Go (PCIe Gen2 x16), en partant de la [recette GNmac-lab](https://github.com/GNmac-lab/deepseek-v41-flash-1017-cmp170hx) @ `2ad5741ab1`. Nos changements :

- couches 18-22 en 3 bits au lieu de 2 ;
- pas de drafter DSpark ;
- tables Engram réellement chargées ;
- `gpu_memory_utilization` à 0,98.

Résultats mesurés :

- test de copie **300/300 exact**, sorties identiques octet pour octet ;
- perplexité **-1,80 %** au global et **-4,45 % en français** par rapport au pack 2.9 bpw publié ;
- cache KV de **2 717 420 jetons** à 262 144 de contexte ;
- environ 105 tok/s pour un utilisateur, 449 à 560 tok/s en agrégé pour 8 utilisateurs ;
- préremplissage à froid à 2 514 tok/s.

Le dépôt ne contient que des patchs, des scripts, des tests et des données brutes : pas de poids.

## Configuration en service

| Élément | Valeur |
|---|---|
| Matériel | ASRock Rack ROMED8-2T, EPYC 7F52, 512 Go DDR4-3200 (8 canaux), 4x CMP 170HX, 200 W, 170tune +200 MHz @ 1500 |
| Experts routés | LibertAIDAI 2.9 bpw @ `0be63c33`, couches 18-22 greffées depuis le pack 3.5 bpw @ `04a492fc` (toutes les couches en 3 bits) |
| Reste | FP8 officiel @ `1bae0e10` |
| Engram | `ENGRAM_STORAGE=ssd` lu depuis un tmpfs en RAM (`recipe/engram-ram.sh`) |
| Cache KV | `fp8_ds_mla`, 0,98 : 2 717 420 jetons (marge d'environ 0,23 Gio seulement si vous utilisez `prompt_logprobs` sur des fenêtres de 1 024 jetons ; sinon passer à 0,97) |
| Divers | TP4, 8 séquences, vision et appels d'outils actifs, API sur `127.0.0.1` sans clé par défaut |

## Principaux résultats

- **Le mode Engram « host » échoue sans bruit, et c'était notre erreur de configuration.** Avec `DSV41_ENGRAM_STORAGE` vide, les tables sont allouées mais jamais remplies. Symptômes : 15/20 copies exactes, des « _ » perdus dans les identifiants, et des sorties qui varient à température 0. Correctif : mode `ssd` lu depuis un tmpfs. Après correctif, 100/100 sur 3 passes ([`data/logs/03-copytest-100-and-determinism.txt`](data/logs/03-copytest-100-and-determinism.txt)).
- **La greffe des couches 18-22 en 3 bits apporte un gain net**, mesuré en NLL appariée. Données : [`data/logs/05-ab-2.9orig-vs-graft.txt`](data/logs/05-ab-2.9orig-vs-graft.txt).

  | Corpus | Perplexité | IC 95 % | Fenêtres meilleures |
  |---|---|---|---|
  | Français | **-4,45 %** | [-4,97 ; -3,93] | 64 sur 64 |
  | Global | **-1,80 %** | [-2,00 ; -1,60] | |
- **Les couches 4 bits n'apportent rien de mesurable.** Couches 0, 1, 3 et 6 : +0,08 %, IC [-0,01 ; +0,18], pour environ 1,58 Gio de VRAM en plus par carte (cache KV de 2 417 392 à 1 670 368 jetons à 0,97). Nous sommes donc revenus au tout 3 bits.
- **DSpark :** x1,9 à x3,5 pour un utilisateur, à peu près nul pour 8 utilisateurs, et le cache KV tombe à 1 113 781 jetons. Taux d'acceptation : 99,4 % sur le prompt structuré (un comptage), environ 70 % sur le code et 43-53 % sur la prose. Nous l'avons écarté, pour la place en VRAM et non pour la qualité (il est sans perte).
- **Les résultats sont reproductibles bit pour bit d'un redémarrage à l'autre** (logprobs de préremplissage, schéma A-B-A, configuration tout 3 bits). Pour le décodage, nous n'avons vérifié l'identité des sorties qu'au sein d'un même processus.
- **Le pack 3.5 bpw complet ne tient pas en VRAM.** Nous gardons LibertAI plutôt que Mia-AiLab : l'auteur amont a mesuré +1,55 % de NLL avec Mia dans cet hybride.

## Méthode

- [`docs/quality-research.md`](docs/quality-research.md) : la recherche sur la qualité.
- [`docs/agentic-methodology.md`](docs/agentic-methodology.md) : comment la recherche a été menée, avec des workflows d'agents IA (Claude Code), une relecture critique et les décisions laissées à l'humain, ainsi que nos règles de mesure.
- [`docs/results.md`](docs/results.md) : toutes les mesures.
- [`docs/comparison-morrowmake.md`](docs/comparison-morrowmake.md) : la comparaison avec la recette Morrowmake, **dont une mesure A-B-A sur la même machine (30/10/2026)**. Sur des fenêtres jusqu'à 1 024 jetons, les deux moteurs font jeu égal à poids identiques (moins de 0,3 % de perplexité d'écart) ; notre configuration est meilleure d'environ 2 % (5 % en français) grâce à ses couches 18-22 en 3 bits. HumanEval et GSM8K ne départagent pas les deux piles. Le contrôle P2P de Morrowmake passe sur notre pilote sans son patch TRAP31. Vitesse et contextes longs non comparés sur une même machine.

## Organisation du dépôt

- `recipe/` : patchs contre l'amont, téléchargement à révisions épinglées, script de greffe, script Engram en RAM, `env.example`.
- `tests/` : copie, déterminisme, stress, NLL.
- `data/` : mesures brutes (`INVALID-*` = l'exécution sans tables Engram).
- `docs/` : documentation (en anglais).

## Démarrage rapide

Voir [`docs/setup.md`](docs/setup.md). En résumé :

1. Cloner l'amont et se placer sur `2ad5741ab1`.
2. Appliquer les patchs avec `patch -p1`.
3. Copier `recipe/tools/*` dans `tools/`, `recipe/engram-ram.sh` à la racine, et `recipe/env.example` en `.env`.
4. Lancer `tools/download-pinned.sh` (environ 428 Gio).
5. Lancer `tools/make_hybrid_graft.py <MODELS_DIR> DeepSeek-V4.1-Flash-1017-hybrid k3`.
6. Lancer `./start.sh`.

Prévoir environ 190 Gio de RAM pour les tables Engram.

## Remerciements

- GNmac-lab : la recette de base.
- LibertAIDAI : les packs EXL3.
- deepseek-ai : le modèle.
- Curtis White : `quantization_config.json` et la recette hybride.
- Morrowmake : la recette comparée.
- MiaAI-Lab : le protocole de banc.
- sudoingX : les données de sensibilité par couche.
- turboderp : exllamav3.
- vcruz305 : vllm-exl3.

Les liens exacts sont dans la [version anglaise](README.md#credits).

## Licences

- Ce dépôt est sous licence MIT.
- La recette amont est sous licence MIT, sauf `patches/vllm-exl3/` (AGPL-3.0), que nous ne redistribuons pas. L'image moteur construite depuis la recette amont contient donc du code AGPL-3.0 : si vous la servez à d'autres personnes sur un réseau, l'article 13 de l'AGPL-3.0 s'applique (voir [`NOTICE.md`](NOTICE.md)).
- Les poids ne sont pas inclus ; ils sont sous licence MIT de leurs propriétaires.
- Le script de banc MiaAI-Lab (AGPL-3.0) n'est pas inclus.

## Avertissement

Projet de homelab : une seule machine, des mesures faites sur notre matériel, sans aucune garantie. Nous ne sommes affiliés à aucun des projets cités. Les écarts de perplexité sont relatifs entre nos configurations : la distance au modèle non quantifié n'a pas été mesurée par nous.
