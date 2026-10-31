# DeepSeek-V4.1-Flash-1017 sur 4x CMP 170HX, la fidélité d'abord

[English version](README.md)

Nous servons [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) sur
quatre cartes NVIDIA CMP 170HX (sm_80, 64 Go chacune, PCIe Gen2 x16), et nous privilégions la fidélité au modèle
d'origine plutôt que la vitesse brute. Chaque choix ci-dessous a été mesuré sur cette machine contre son alternative,
par perplexité appariée sur des textes figés, avec les données brutes publiées.

Ce que nous servons aujourd'hui (31/10/2026) :

- **Le moteur de [Morrowmake](https://github.com/Morrowmake/dsv41-flash-cmp170hx-recipe), version 1.0.0** : il
  reproduit l'arrondi des caches d'attention du modèle de référence, et cela se voit à partir de 32k jetons.
- **Les experts routés des couches 18 à 22 greffés de 2 à 3 bits** : le plus gros gain de qualité que nous ayons
  trouvé.
- **Un cache KV compact sans perte, de notre cru** : les mêmes bits que le cache BF16 de Morrowmake, dans la moitié
  de la mémoire.
- **Le décodage spéculatif DSpark activé** : il génère le même texte, 2 à 3,7 fois plus vite pour un utilisateur.

Le dépôt contient des correctifs, des outils, des tests, les données brutes des mesures et la documentation. Il ne
contient aucun poids ; son seul code de moteur est le cache KV compact, sous forme de correctif du moteur de
Morrowmake.

## En bref

| | Mesuré |
|---|---|
| Qualité contre notre production précédente (moteur GNmac, même greffe) | **-1,64 %** de perplexité sur des fenêtres de code de 32k jetons (16 sur 16 meilleures), **-0,58 %** sur de la prose de 32k ; +0,14 % sur des fenêtres de 1 024 jetons |
| Gain de la greffe 3 bits (couches 18-22) | **-1,81 %** de perplexité au global, **-4,45 %** en français, -2,31 % sur la prose de 32k |
| Cache KV compact | log-probabilités **identiques au bit près** sur 1,39 M jetons ; réserve KV **x1,99** (1,60 M → 3,19 M jetons sans DSpark, 1,36 M avec) ; décodage -2 à -5 % |
| Vitesse, un utilisateur (protocole MiaAI, structuré / code / prose) | **267 / 212 / 153 jetons/s** avec DSpark, 72 sans |
| Vitesse, 8 utilisateurs, cumulée | **570 / 291 / 349 jetons/s** avec DSpark |
| Longs prompts | premier jeton en **93 s à 259k jetons** |
| Contexte de 1M jetons (profil séparé, sans DSpark) | rappel exact de 8 codes cachés : **8 / 8 à 997k jetons** (premier jeton après 12 min) ; réserve KV de 4,2 M jetons ([résultats](docs/results.md#5-contexts-beyond-262144-tokens)) |

## Comparatifs de qualité

[`tests/nll_eval.py`](tests/nll_eval.py) note les mêmes fenêtres figées sur chaque configuration (livres du domaine
public en français et en anglais, code Python) et les compare fenêtre par fenêtre. Ces notations se reproduisent au
bit près d'un redémarrage du serveur à l'autre : un écart ci-dessous n'est donc pas du bruit de redémarrage, et
l'intervalle de confiance bootstrap à 95 % dit à quel point il est sûr. Négatif signifie que B prédit mieux le texte.
La « prose 32k » compte 8 fenêtres en français, 8 en anglais et 3 de code.

| A → B | Fenêtres de 1 024 | Fenêtres de 40 | Prose 32k | Code 32k |
|---|---|---|---|---|
| **Notre production précédente → ce que nous servons** | +0,14 % [+0,03 ; +0,26] | -0,21 % [-0,52 ; +0,11] | **-0,58 % [-0,74 ; -0,43]** | **-1,64 % [-1,92 ; -1,37]** |
| Moteur seul, mêmes poids (GNmac → Morrowmake) | +0,15 % [+0,02 ; +0,29] | -0,29 % [-0,59 ; +0,02] | **-0,57 % [-0,73 ; -0,42]** | non mesuré |
| Greffe des couches 18-22, 2 → 3 bits (moteur Morrowmake) | **-1,81 % [-2,01 ; -1,61]** | -0,49 % [-0,84 ; -0,13] | **-2,31 % [-2,75 ; -1,79]** | -0,06 % [-0,12 ; -0,00] |
| Cache KV compact (lignes BF16 → lignes compactes) | identique | identique | identique | identique |
| Couches 36-39, 3 → 4 bits | -0,04 % [-0,08 ; +0,01] | -0,08 % [-0,22 ; +0,07] | -0,11 % [-0,15 ; -0,07] | -0,11 % [-0,14 ; -0,09] |
| Couches 0, 1, 3, 6, 3 → 4 bits (moteur GNmac) | +0,08 % [-0,01 ; +0,18] | -0,02 % [-0,30 ; +0,25] | non mesuré | non mesuré |
| Raccourcis de la release désactivés (Morrowmake) | +0,04 % [-0,05 ; +0,13] | non mesuré | non mesuré | +0,00 % [-0,04 ; +0,05] |

Ce que dit le tableau :

- **Les poids comptent le plus, et c'est la greffe qui paie.** Les couches 18-22 sont les seules en 2 bits du pack
  publié. Les passer en 3 bits rapporte 1,8 % au global et 4,5 % en français, dans les deux moteurs. Passer d'autres
  couches de 3 à 4 bits rapporte au mieux 0,1 % : les experts sont au plateau.
- **Le moteur compte sur les contextes longs.** À poids égaux, le moteur de Morrowmake fait 0,15 % moins bien que
  celui de GNmac sur des fenêtres de 1 024 jetons, et 0,6 % mieux à 32k (2 % mieux sur les trois fenêtres de code
  de cet ensemble). *Interprétation :*
  il reproduit l'arrondi FP4 de référence du cache compressé et des clés de l'indexeur, là où GNmac stocke des
  enregistrements FP8 ; l'indexeur choisit quelles positions passées chaque jeton relit, et ce choix devient plus
  difficile quand il y a plus de positions.
- **Nous avons choisi les contextes longs.** Contre notre production précédente, la configuration actuelle fait
  1,6 % mieux sur le code long et 0,6 % mieux sur la prose longue, et 0,14 % moins bien sur les fenêtres courtes.
  Le travail agentique et le code se passent dans des contextes longs.
- **Le cache KV compact ne coûte rien en qualité** : pas un petit écart, aucun écart, sur chaque jeton mesuré.

Détails, lignes par langue et fichiers de données : [docs/results.md](docs/results.md#2-quality).

## Nos optimisations

1. **Greffe 3 bits des couches 18-22.** Les experts routés de ces cinq couches viennent du pack 3,5 bpw de LibertAI
   au lieu du pack 2,9 bpw (mêmes noms de tenseurs, 3 bits au lieu de 2). Coût : 1,98 Gio par carte. Outil :
   [`morrowmake/tools/graft_layers.py`](morrowmake/tools/graft_layers.py), qui vérifie les ensembles de tenseurs et
   la largeur en bits (et refuse toute autre largeur que celle attendue) avant de réécrire l'index.
2. **Cache KV compact sans perte** ([docs/compact-kv-cache.md](docs/compact-kv-cache.md)). La référence arrondit
   déjà le latent compressé et les clés de l'indexeur en flottants 4 bits avec une échelle par bloc : nous stockons
   les codes 4 bits et les octets d'échelle, puis nous les redécodons en exactement les mêmes bits. Le latent et la
   clé d'indexeur d'une même position partagent une ligne de 512 octets (1 280 octets par jeton et par carte au lieu
   de 3 200). Trois versions :
   - v1 : exacte, mais -5 à -9 % en décodage et -10 à -35 % en lecture de prompt ;
   - v2 : décodage vectorisé, et attention à travers une copie BF16 des lignes choisies par chaque requête, si bien
     que les noyaux de Morrowmake tournent sans modification et que l'identité tient par construction (décodage -2
     à -5 %, lecture de prompt -6 à -10 %) ;
   - v1.5 : les clés de l'indexeur passent dans les lignes du latent (+16 % de réserve en plus).
3. **Le moteur, choisi sur les contextes longs.** Voir plus haut : la première comparaison, à 1 024 jetons, aurait
   désigné l'autre moteur ([docs/comparison-morrowmake.md](docs/comparison-morrowmake.md)).
4. **DSpark gardé.** Mêmes 768 jetons générés à l'auto-contrôle, log-probabilités de décodage à moins de 2e-5 ;
   vitesse pour un utilisateur x2,1 à x3,7. Le profil sans DSpark reste disponible pour une réserve de 3,19 M jetons.
5. **Contextes jusqu'à 1M jetons, en profil séparé.** Sans DSpark, `--max-model-len 1048576` : 8 codes cachés sur 8
   restitués exactement à 131k, 522k et 997k jetons. Un maximum de 1M ralentit toutes les requêtes de 21 à 25 %, d'où
   un profil à part plutôt que le réglage par défaut.
6. **Mesuré et écarté :** experts en 4 bits (deux essais, 0,1 % au plus, -12 à -17 % de vitesse) ; raccourcis
   numériques de la release désactivés (aucun écart mesurable, 2,3 fois plus lent) ; réglage mémoire à 0,98 (mêmes
   bits, manque de mémoire) ([docs/results.md, section 6](docs/results.md#6-settings-we-tried-and-rejected)).

## Vitesse

Protocole MiaAI-Lab (température 0, réflexion coupée, 400 jetons au plus), jetons/s, structuré / code / prose :

| | 1 utilisateur | 8 utilisateurs, cumulé |
|---|---|---|
| **Ce que nous servons** (Morrowmake + greffe + cache compact + DSpark) | **267 / 212 / 153** | **570 / 291 / 349** |
| Idem, sans DSpark | 72 / 73 / 72 | 397 / 245 / 341 |
| Notre production précédente (GNmac + greffe, sans DSpark) | 105 / 105 / 105 | 560 / 482 / 449 |
| GNmac + greffe + DSpark | 366 / 268 / 202 | 884 / 563 / 476 |

**Le moteur de GNmac est plus rapide**, d'environ 1,45 fois pour un flux sans DSpark et 1,3 à 1,4 fois avec.
Comparez les colonnes structuré et prose : la réponse au prompt de code fait 48 jetons sur Morrowmake et 95 sur
GNmac, cette colonne ne se compare donc pas d'un moteur à l'autre. Nous avons échangé cette vitesse contre la qualité
en contexte long. Morrowmake lit les longs prompts 1,4 à 1,7 fois plus vite :
le premier jeton d'un prompt de 259k jetons arrive après 93 s au lieu de 159 s. Décodage par flux en contexte long et
réserves KV : [docs/results.md](docs/results.md#4-speed).

## Configuration en production

| Élément | Valeur |
|---|---|
| Matériel | ASRock Rack ROMED8-2T, AMD EPYC 7F52, 512 Go DDR4-3200 (8 canaux), 4x CMP 170HX 64 Go en PCIe Gen2 x16, 200 W, 170tune +200 MHz @ 1500 MHz |
| Pilote | pilote ouvert NVIDIA 610.43.02 avec un fork de cmpunlocker et un correctif P2P ; le contrôle P2P de Morrowmake passe sans son correctif TRAP31 |
| Moteur | Morrowmake 1.0.0 (`7f9023cd`), image `ghcr.io/morrowmake/vllm-cmp170hx@sha256:f67d9be4…` (moteur `a0c57b52`), inchangée sauf les sept fichiers du cache compact montés par-dessus |
| Poids | tenseurs officiels FP8 hors experts et tables Engram (`deepseek-ai/DeepSeek-V4.1-Flash-1017` @ `1bae0e10`) ; experts routés du pack 2,9 bpw de LibertAI (révision épinglée par Morrowmake), couches 18-22 du pack 3,5 bpw de LibertAI @ `04a492fc` |
| Cache KV | compact v1.5 (`VLLM_SM80_COMPACT_KV=1`), cache de préfixes activé |
| Service | TP 4, contexte de 262 144 jetons, 8 séquences, DSpark (7 jetons proposés), réglage mémoire 0,95, tables Engram en RAM |
| Réserve KV | 1 360 109 jetons (5,2 requêtes de longueur maximale) ; 3 189 000 sans DSpark |
| API | analyseurs d'appels d'outils et de raisonnement `deepseek_v41`, clé d'API obligatoire, `/health` ouvert |

## Organisation du dépôt

```
morrowmake/                       la configuration actuelle, au-dessus de Morrowmake 1.0.0
  compact-kv/                     cache KV compact sans perte (fichiers de moteur sous Apache-2.0)
    compact-kv.patch              modifications de six fichiers du moteur
    vllm/.../ops/sm80_kv_pack.py  compactage, décompactage, chargements et écritures Triton
    build-overlay.sh              construit kvc-overlay/ depuis l'image épinglée + le correctif
    tests/test_gpu.py             test unitaire d'identité au bit près sur GPU ; tests/bench_gpu.py : temps des noyaux
  tools/graft_layers.py           greffe des couches d'un autre pack EXL3 dans le checkpoint exporté
  start.sh.patch                  monte kvc-overlay/ par-dessus les fichiers de l'image
  prepare.sh.patch                empreinte de l'index hybride de la greffe 18-22
  env.example, start-dspark.sh, start-nospec.sh
recipe/                           la configuration précédente, au-dessus de GNmac-lab 1.2.0
tests/
  nll_eval.py                     NLL / perplexité appariée (build, run, compare, selfcheck)
  identical.py                    identité au bit près de deux mesures
  speed.py                        décodage et lecture de prompt par flux, de 3,5k à 259k jetons
  longctx.py                      rappel exact de 8 codes cachés dans un long document (jusqu'à 1M jetons)
  copytest-*.py, stress-dsv41.py  tests de copie exacte, de déterminisme et de charge
data/
  nll/                            toutes les mesures notées (long/, codex/ = prose et code 32k ; .gz = JSON compressé)
  bench/                          mesures de vitesse (speed/), sorties MiaAI (mia-json*/), anciens journaux
  logs/                           journaux d'expériences du premier tour
docs/
  results.md                      toutes les mesures de la configuration actuelle
  compact-kv-cache.md             conception et validation du cache compact
  setup.md                        reconstruction pas à pas
  comparison-morrowmake.md        les deux moteurs, mesurés côte à côte
  quality-research.md, agentic-methodology.md, results-gnmac.md, setup-gnmac.md
```

La documentation détaillée est en anglais.

## Démarrage rapide

Étapes complètes avec leurs vérifications : [docs/setup.md](docs/setup.md).

1. Installer Morrowmake 1.0.0 (`7f9023cd`) et le lancer une fois tel quel.
2. Télécharger les cinq shards donneurs du pack 3,5 bpw de LibertAI, lancer `morrowmake/tools/graft_layers.py`,
   appliquer `prepare.sh.patch`, reconstruire l'hybride.
3. Lancer `morrowmake/compact-kv/build-overlay.sh`, appliquer `start.sh.patch`, régler `VLLM_SM80_COMPACT_KV=1`,
   passer le test unitaire GPU.
4. `./start-dspark.sh`. Le journal doit annoncer une réserve KV de 1 360 109 jetons.

## La configuration précédente

Jusqu'au 31/10/2026 nous servions la [recette GNmac-lab](https://github.com/GNmac-lab/deepseek-v41-flash-1017-cmp170hx)
1.2.0 avec la même greffe, sans DSpark, à 0,98. Elle reste documentée dans [`recipe/`](recipe/),
[docs/setup-gnmac.md](docs/setup-gnmac.md) et [docs/results-gnmac.md](docs/results-gnmac.md). Ses constats restent
valables, en particulier :

- **Les tables Engram peuvent ne pas être chargées sans que rien ne le signale.** Dans le mode « host » de GNmac,
  les tables étaient réservées mais jamais remplies ; le modèle continuait de répondre, avec des identifiants
  corrompus et une sortie non déterministe. La vitesse de décodage ne le montre pas ; un test de copie exacte, si
  ([docs/agentic-methodology.md, section 7](docs/agentic-methodology.md#7-failure-analysis-the-engram-tables-were-empty)).
- **Un plan A-B-A d'un redémarrage à l'autre donne un bruit de fond nul** sur ces piles, ce qui rend mesurables de
  petits écarts.

## Méthode

Un agent de programmation IA (Claude Code) a mené la recherche, les expériences et les mesures ; un opérateur humain
a pris chaque décision. Des passes critiques ont relu les rapports de recherche, et chaque affirmation qui a guidé une
décision a été mesurée. [docs/agentic-methodology.md](docs/agentic-methodology.md) décrit la méthode, ce que les
relectures critiques ont attrapé et les pièges de mesure dont nous nous gardons.
[docs/quality-research.md](docs/quality-research.md) couvre le premier tour de recherche sur la fidélité (options de
quantification, budgets de VRAM, diagnostic Engram).

## Remerciements

- [Morrowmake/dsv41-flash-cmp170hx-recipe](https://github.com/Morrowmake/dsv41-flash-cmp170hx-recipe) 1.0.0 et
  [Morrowmake/vllm-cmp170hx](https://github.com/Morrowmake/vllm-cmp170hx) : le moteur et la recette que nous servons.
- [GNmac-lab/deepseek-v41-flash-1017-cmp170hx](https://github.com/GNmac-lab/deepseek-v41-flash-1017-cmp170hx) 1.2.0 :
  la recette de notre première configuration.
- [LibertAIDAI](https://huggingface.co/LibertAIDAI) : les packs EXL3,
  [2,9 bpw](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw) et
  [3,5 bpw](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw) @ `04a492fc`.
- [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) @ `1bae0e10` :
  le modèle.
- Curtis White ([CurtisAccelerate/DeepSeekv41-Flash-1017-EXL3-CMP170HX](https://github.com/CurtisAccelerate/DeepSeekv41-Flash-1017-EXL3-CMP170HX)) :
  la disposition du checkpoint hybride.
- [MiaAI-Lab](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks) : le protocole de mesure du décodage
  (non inclus ici).
- [sudoingX/dgx-spark-deepseek-v41](https://github.com/sudoingX/dgx-spark-deepseek-v41) : données de sensibilité par
  couche.
- [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3) et [vLLM](https://github.com/vllm-project/vllm).

## Licences

- **Ce dépôt** est sous licence MIT ([LICENSE](LICENSE)), **sauf les fichiers de moteur de `morrowmake/compact-kv/`**
  (`compact-kv.patch` et `sm80_kv_pack.py`), qui modifient le fork vLLM de Morrowmake, sous Apache-2.0, et sont donc
  sous Apache-2.0 ([leur LICENSE](morrowmake/compact-kv/LICENSE)).
  [NOTICE.md](NOTICE.md) liste chaque fichier dérivé d'un autre projet.
- **Les poids** ne sont pas inclus : ceux de DeepSeek et les packs de LibertAI sont sous MIT, selon les conditions de
  leurs propriétaires.
- L'image du moteur de la configuration précédente (GNmac) contient du code AGPL-3.0 ; voir [NOTICE.md](NOTICE.md) si
  vous la construisez.

## Avertissement

Projet de homelab : une machine, une série de mesures. Nous ne sommes affiliés ni à DeepSeek, ni à NVIDIA, ni à
Morrowmake, GNmac-lab, LibertAI ou tout autre projet cité. Les CMP 170HX demandent un pilote modifié pour fonctionner
comme décrit ; vérifiez les conditions qui s'appliquent à vous. Chaque chiffre est ce que nous avons mesuré sur notre
matériel, avec son fichier de données. Les écarts de perplexité sont relatifs entre nos configurations : nous n'avons
pas mesuré la distance au modèle non quantifié.
