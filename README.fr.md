# DeepSeek-V4.1-Flash-1017 sur 4x CMP 170HX, la fidélité d'abord

[English version](README.md)

Nous servons [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) sur
quatre cartes NVIDIA CMP 170HX (sm_80, 64 Go chacune, PCIe Gen2 x16), et nous privilégions la fidélité au modèle
d'origine plutôt que la vitesse brute. Chaque choix ci-dessous a été mesuré sur cette machine contre son alternative,
par perplexité appariée sur des textes figés, avec les données brutes publiées.

Ce que nous servons aujourd'hui (01/11/2026) :

- **Le moteur de [Morrowmake](https://github.com/Morrowmake/dsv41-flash-cmp170hx-recipe), version 1.0.1** : il
  reproduit l'arrondi des caches d'attention du modèle de référence, et cela se voit à partir de 32k jetons.
- **Les experts routés des couches 18 à 22 greffés de 2 à 3 bits** : le plus gros gain de qualité que nous ayons
  trouvé.
- **Un cache KV compact sans perte, de notre cru** : les mêmes bits que le cache BF16 de Morrowmake, dans la moitié
  de la mémoire.
- **Le décodage spéculatif DSpark activé** : il génère le même texte, 2 à 3,6 fois plus vite pour un utilisateur.

Le dépôt contient des correctifs, des outils, des tests, les données brutes des mesures et la documentation. Il ne
contient aucun poids ; son seul code de moteur est le cache KV compact, sous forme de correctif du moteur de
Morrowmake.

## En bref

| | Mesuré sur cette machine, mêmes outils pour chaque pile |
|---|---|
| Qualité contre **Morrowmake 1.0.1** tel que publié | **-1,76 %** de perplexité (français **-4,39 %**), **-2,28 %** sur de la prose de 32k jetons |
| Qualité contre **GNmac-lab 1.2.0** tel que publié | **-1,63 %** de perplexité (français **-4,09 %**), **-2,83 %** sur de la prose de 32k jetons |
| Autres langues, contre Morrowmake | allemand **-3,06 %** ; italien -0,98 %, porté par ses deux livres les plus connus ([détails](#comparatifs-de-qualité)) |
| Cache KV compact | log-probabilités **identiques au bit près** sur 1,39 M jetons ; réserve KV **x1,99** contre les lignes BF16 de Morrowmake à poids égaux (1,56 M → 3,09 M jetons sans DSpark, 1,47 M avec) |
| Réserve KV contre Morrowmake tel que publié | **+13 %** avec DSpark (1 466 159 contre 1 296 145 jetons), **+47 %** sans (3 090 900 contre 2 105 722) |
| Vitesse, un utilisateur (protocole MiaAI, structuré / code / prose) | **288 / 251 / 155 jetons/s** avec DSpark, 79 sans |
| Vitesse, 8 utilisateurs, cumulée | **639 / 307 / 362 jetons/s** avec DSpark |
| Longs prompts | premier jeton en **93 s à 259k jetons** |
| Contexte de 1M jetons (profil séparé, sans DSpark) | rappel exact de 8 codes cachés : **8 / 8 à 997k jetons** (premier jeton après 12 min) ; réserve KV de 4,06 M jetons ([résultats](docs/results.md#5-contexts-beyond-262144-tokens)) |

## Comparatifs de qualité

[`tests/nll_eval.py`](tests/nll_eval.py) note les mêmes fenêtres figées sur chaque configuration (livres du domaine
public en français et en anglais, code Python) et les compare fenêtre par fenêtre. Ces notations se reproduisent au
bit près d'un redémarrage du serveur à l'autre : un écart ci-dessous n'est donc pas du bruit de redémarrage, et
l'intervalle de confiance bootstrap à 95 % dit à quel point il est sûr. Négatif signifie que notre configuration
prédit mieux le texte. La « prose 32k » compte 8 fenêtres en français, 8 en anglais et 3 de code.

**Contre les deux recettes publiques, telles que publiées** (leurs poids : le pack 2,9 bpw de LibertAI) :

| Recette → nous | Fenêtres de 1 024 | Fenêtres de 40 | Prose 32k | Code 32k |
|---|---|---|---|---|
| **Morrowmake 1.0.1** | **-1,76 % [-1,97 ; -1,55]** | -0,23 % [-0,57 ; +0,12] | **-2,28 % [-2,70 ; -1,75]** | -0,06 % [-0,11 ; -0,02] |
| **GNmac-lab 1.2.0** | **-1,63 % [-1,85 ; -1,40]** | -0,34 % [-0,77 ; +0,09] | **-2,83 % [-3,36 ; -2,21]** | non mesuré |

**Autres langues** (Morrowmake tel que publié → nous, 64 fenêtres de 1 024 jetons tirées de 8 livres Gutenberg par
langue) :

| Langue | Écart | Fenêtres où nous faisons mieux |
|---|---|---|
| Français | **-4,39 % [-4,94 ; -3,83]** | 63 sur 64 |
| Allemand | **-3,06 % [-3,78 ; -2,35]** | 54 sur 64, chaque livre mieux |
| Anglais | -0,68 % [-0,94 ; -0,45] | 47 sur 64 |
| Italien | -0,98 % [-1,63 ; -0,38] | 29 sur 64 |

Le gain en italien vient de ses deux livres les plus connus (*I promessi sposi* -5,2 %, *Pinocchio* -4,5 %, 8 fenêtres
sur 8 chacun) ; les six romans moins connus bougent de -0,2 à +0,63 %, un peu moins bien dans l'ensemble.
*Interprétation :* une partie de ce que la greffe rétablit pourrait être le rappel de textes vus à l'entraînement ;
ces mesures ne permettent pas de dire quelle part.

**D'où vient le gain**, un changement à la fois :

| A → B | Fenêtres de 1 024 | Fenêtres de 40 | Prose 32k | Code 32k |
|---|---|---|---|---|
| Greffe des couches 18-22, 2 → 3 bits (moteur Morrowmake) | **-1,76 % [-1,97 ; -1,55]** | -0,23 % [-0,57 ; +0,12] | **-2,28 % [-2,70 ; -1,75]** | -0,06 % [-0,11 ; -0,02] |
| Moteur GNmac → moteur Morrowmake, mêmes poids publiés | +0,14 % [+0,02 ; +0,25] | -0,12 % [-0,44 ; +0,20] | **-0,57 % [-0,73 ; -0,42]** | non mesuré |
| Moteur GNmac → moteur Morrowmake avec notre cache compact, mêmes poids (greffés des deux côtés) | +0,17 % [+0,06 ; +0,30] | +0,23 % [-0,09 ; +0,55] | **-0,54 % [-0,70 ; -0,38]** | **-1,62 % [-1,89 ; -1,35]** |
| Cache KV compact (lignes BF16 de Morrowmake → lignes compactes) | identique | identique | identique | identique |
| Couches 36-39, 3 → 4 bits (sur 1.0.0) | -0,04 % [-0,08 ; +0,01] | -0,08 % [-0,22 ; +0,07] | -0,11 % [-0,15 ; -0,07] | -0,11 % [-0,14 ; -0,09] |
| Couches 0, 1, 3, 6, 3 → 4 bits (moteur GNmac) | +0,08 % [-0,01 ; +0,18] | -0,02 % [-0,30 ; +0,25] | non mesuré | non mesuré |
| Raccourcis de la release Morrowmake désactivés (sur 1.0.0) | +0,04 % [-0,05 ; +0,13] | non mesuré | non mesuré | +0,00 % [-0,04 ; +0,05] |

Ce que disent les tableaux :

- **Contre Morrowmake, tout le gain vient de la greffe.** Même moteur, même image ; le cache compact ne change
  aucun bit. Morrowmake livre les couches 18-22 en 2 bits, les seules du pack. En 3 bits elles rapportent 1,8 % au
  global, 4,4 à 4,7 % en français à partir de 1 024 jetons, et peu sur le code.
- **Les prompts très courts gagnent peu.** Sur les fenêtres de 40 jetons, le gain reste dans le bruit sur la 1.0.1
  (-0,23 % ; anglais -0,99 % [-1,57 ; -0,40], français -0,02 %, code +0,33 %). Il était de -0,49 % sur la 1.0.0 ; nous
  n'avons pas trouvé ce qui l'a fait bouger ([results.md, 2.5](docs/results.md#25-from-morrowmake-100-to-101)).
- **Contre GNmac, le moteur s'ajoute à la greffe.** À poids égaux, le moteur de Morrowmake a 0,14 % de retard sur
  celui de GNmac en fenêtres de 1 024 jetons, et prend l'avantage à partir de 32k : 0,6 % sur la prose, 1,6 % sur
  le code (16 fenêtres sur 16). *Interprétation :* Morrowmake reproduit l'arrondi FP4 de référence du cache
  compressé et des clés de l'indexeur, là où GNmac stocke des enregistrements FP8 ; l'indexeur choisit quelles
  positions passées chaque jeton relit, et ce choix devient plus difficile quand il y a plus de positions. Le
  travail agentique et le code se passent en contexte long : nous avons choisi ce moteur.
- **Les experts sont au plateau.** Passer d'autres couches de 3 à 4 bits rapporte au mieux 0,1 %.
- **Le cache KV compact ne coûte rien en qualité** : pas un petit écart, aucun écart, sur chaque jeton mesuré.

Détails, lignes par langue et fichiers de données : [docs/results.md](docs/results.md#2-quality) et
[docs/comparison-morrowmake.md](docs/comparison-morrowmake.md).

## Contre Morrowmake 1.0.1, en bref

Nous servons l'image même de Morrowmake. Ce qui change, mesuré sur la même machine
([docs/comparison-morrowmake.md](docs/comparison-morrowmake.md)) :

| | Morrowmake 1.0.1 tel que publié | Nous |
|---|---|---|
| Couches 18-22 | experts en 2 bits | experts en 3 bits (+1,98 Gio par carte) |
| Caches d'attention compressée | lignes BF16 (1 280 octets par position) | lignes compactes, mêmes bits (512 octets) |
| Perplexité, fenêtres de 1 024 / 32k jetons | référence | **-1,76 % / -2,28 %** (français -4,39 % / -4,72 %) |
| Réserve KV, avec / sans DSpark | 1 296 145 / 2 105 722 | **1 466 159 / 3 090 900** |
| Vitesse avec DSpark, 1 utilisateur, structuré / prose | 297 / 158 jetons/s | 288 / 155 jetons/s (-3 % / -2 %) |
| Vitesse avec DSpark, 8 utilisateurs, structuré / prose | 624 / 377 jetons/s | 639 / 362 jetons/s (+2 % / -4 %) |
| Premier jeton d'un prompt de 259k jetons | 88 s | 93 s |
| Contexte de 1M jetons | non validé | rappel 8 / 8 à 997k jetons (profil séparé) |
| Installation | `install.sh`, `download.sh` | plus 24 Gio de shards donneurs, l'outil de greffe, une empreinte, l'overlay |

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

   Le passage à Morrowmake 1.0.1 n'a changé que des chemins d'import, et les deux contrôles de validation sont
   repassés.
3. **Le moteur, choisi sur les contextes longs.** À 1 024 jetons le moteur de GNmac est légèrement devant ; à partir
   de 32k c'est celui de Morrowmake. Une comparaison arrêtée aux fenêtres courtes aurait désigné l'autre.
4. **DSpark gardé.** Mêmes 768 jetons générés à l'auto-contrôle ; vitesse pour un utilisateur x2,0 à x3,6. Le profil
   sans DSpark reste disponible pour une réserve de 3,09 M jetons.
5. **Contextes jusqu'à 1M jetons, en profil séparé.** Sans DSpark, `--max-model-len 1048576` : 8 codes cachés sur 8
   restitués exactement à 131k, 522k et 997k jetons. Un maximum de 1M ralentit toutes les requêtes de 19 à 31 %, d'où
   un profil à part plutôt que le réglage par défaut.
6. **Mesuré et écarté :** experts en 4 bits (deux essais, 0,1 % au plus, -12 à -17 % de vitesse) ; raccourcis
   numériques de Morrowmake désactivés (aucun écart mesurable, 2,3 fois plus lent) ; réglage mémoire à 0,98 (mêmes
   bits, manque de mémoire), le tout sur la 1.0.0 ([docs/results.md, section 6](docs/results.md#6-settings-we-tried-and-rejected)).

## Comment fonctionne le cache KV compact

**Pourquoi il peut être sans perte.** La référence de DeepSeek-V4.1 arrondit deux des caches avant de les stocker,
et Morrowmake reproduit cet arrondi exactement :

```
latent compressé, par bloc de 16 valeurs :  s = E4M3(max|x| / 6)    q = E2M1(x / s)    stocké = bf16(q * s)
clés de l'indexeur, par bloc de 32 valeurs : s = 2^ceil(log2(max|x| / 6))  (E8M0, une puissance de deux)
```

`q` vaut 0, ±0,5, ±1, ±1,5, ±2, ±3, ±4 ou ±6 : un code de 4 bits. Les lignes BF16 de Morrowmake contiennent donc des
nombres de 16 bits qui portent chacun 4 bits d'information, plus une échelle par bloc : 3,6 fois plus de mémoire que
d'information. Nous stockons les codes et les octets d'échelle à la place. Le décodage recalcule `q * s`, qui a au
plus 6 bits significatifs (2 pour `q`, 4 pour une échelle E4M3) : un résultat exact en FP32 et exactement
représentable en BF16. La valeur décodée est donc la valeur BF16 stockée, au bit près, zéros signés compris. Nos
fonctions de compactage font les mêmes choix que la référence (même plancher d'échelle, division correctement
arrondie, mêmes arrondis), donc elles produisent les mêmes codes.

**Ce qui pouvait encore mal tourner, et comment nous l'avons écarté.** Des valeurs identiques ne suffisent pas : il
faut aussi que les noyaux calculent la même chose. Décoder une tuile compacte dans les registres juste avant le
produit matriciel de l'attention décalait un élément de sortie sur 40 000 environ d'un ulp BF16, avec des opérandes
identiques, parce que le compilateur emprunte un autre chemin pour un opérande construit en registres. Les noyaux
d'attention ne sont donc pas modifiés du tout : les 512 lignes choisies par chaque requête sont décodées dans un
petit tampon BF16, et **les noyaux de Morrowmake eux-mêmes** le lisent avec leurs propres chargements. L'identité
tient par construction. L'indexeur, dont les scores restaient exacts avec un décodage en registres, décode sur place.

**Disposition.** Une ligne de 512 octets par position compressée contient le latent (256 octets de codes, 32 octets
d'échelles) et la clé d'indexeur (64 + 4 octets), là où Morrowmake utilise 1 024 + 256 octets répartis dans deux
caches. L'indexeur lit ses clés à travers une vue des lignes du latent : son propre cache disparaît.

**Comment nous l'avons vérifié.** Un test unitaire sur GPU compare chaque bit de sortie de chaque noyau qui lit ou
écrit ces caches avec le chemin BF16, aux tailles réelles de service ; puis les log-probabilités du modèle servi sont
comparées jeton par jeton avec le cache BF16 de Morrowmake : 1,39 M jetons, 0 différent, sur la 1.0.0 puis de nouveau
sur la 1.0.1. Trois versions (exacte mais lente, puis rapide, puis une ligne par position) ont chacune passé les deux
contrôles. Conception complète, exemple
chiffré et historique : [docs/compact-kv-cache.md](docs/compact-kv-cache.md).

## Vitesse

Protocole MiaAI-Lab (température 0, réflexion coupée, 400 jetons au plus), tout sur cette machine à 200 W par carte,
jetons/s, structuré / code / prose :

| | 1 utilisateur | 8 utilisateurs, cumulé |
|---|---|---|
| **Nous** (Morrowmake + greffe + cache compact + DSpark) | **288 / 251 / 155** | **639 / 307 / 362** |
| Nous, sans DSpark | 79 / 80 / 79 | 421 / 253 / 345 |
| Morrowmake 1.0.1 tel que publié, DSpark | 297 / 235 / 158 | 624 / 378 / 377 |
| Moteur GNmac-lab 1.2.0, DSpark, mêmes poids que nous | 366 / 268 / 202 | 884 / 563 / 476 |
| Moteur GNmac-lab 1.2.0, sans DSpark, mêmes poids que nous | 105 / 105 / 105 | 560 / 482 / 449 |

- **Contre Morrowmake tel que publié**, le décodage va de -4 % à +2 % sur le structuré et la prose, et la lecture de
  prompt est 3 à 6 % plus lente. *Interprétation :* c'est le coût du cache compact ; sur la 1.0.0, où nous les avons
  mesurés séparément, la greffe ne coûtait rien de mesurable.
- **Le moteur de GNmac est plus rapide**, d'environ 1,3 fois pour un flux avec ou sans DSpark, et de 1,3 à 1,4 fois à
  8 utilisateurs avec DSpark. Nous avons échangé cette vitesse contre la qualité en contexte long.
- **Comparez sur les colonnes structuré et prose.** Seuls, Morrowmake 1.0.1 et nous répondons au prompt de code en 48
  jetons, GNmac en 95. En lots, les réponses gloutonnes diffèrent encore (à 8 utilisateurs, 760 jetons au total pour
  Morrowmake, 478 pour nous) : cette colonne ne mesure pas le même travail.
- **Nous ne retrouvons pas les chiffres à 8 utilisateurs publiés par Morrowmake** (868 / 795 / 464 sur sa machine,
  624 / 378 / 377 ici ; à un utilisateur 297 / 297 / 170 contre 297 / 235 / 158). L'écart existait déjà en 1.0.0 (760
  contre 585 en structuré). Nous ne savons pas quelle part vient de la machine et quelle part de la méthode.
- Morrowmake lit les longs prompts 1,4 à 1,7 fois plus vite que GNmac : le premier jeton d'un prompt de 259k jetons
  arrive après 93 s au lieu de 159 s.
- Décodage en contexte long et réserves KV : [docs/results.md](docs/results.md#4-speed).

## Configuration en production

| Élément | Valeur |
|---|---|
| Matériel | ASRock Rack ROMED8-2T, AMD EPYC 7F52, 512 Go DDR4-3200 (8 canaux), 4x CMP 170HX 64 Go en PCIe Gen2 x16, 200 W, 170tune +200 MHz @ 1500 MHz |
| Pilote | pilote ouvert NVIDIA 610.43.02 avec un fork de cmpunlocker et un correctif P2P ; le contrôle P2P de Morrowmake passe sans son correctif TRAP31 |
| Moteur | Morrowmake 1.0.1 (tag `v1.0.1`, `f6643be`), image `ghcr.io/morrowmake/vllm-cmp170hx@sha256:32ae8b0e…` (vLLM `330c5bff`, queue `e05403fd`, greffon `vllm_sm80` `0b34823c`), inchangée sauf les sept fichiers du cache compact montés par-dessus |
| Poids | tenseurs officiels FP8 hors experts et tables Engram (`deepseek-ai/DeepSeek-V4.1-Flash-1017` @ `1bae0e10`) ; experts routés du pack 2,9 bpw de LibertAI (révision épinglée par Morrowmake), couches 18-22 du pack 3,5 bpw de LibertAI @ `04a492fc` |
| Cache KV | compact v1.5 (`VLLM_SM80_COMPACT_KV=1`), cache de préfixes activé |
| Service | TP 4, contexte de 262 144 jetons, 8 séquences, DSpark (7 jetons proposés), réglage mémoire 0,95, tables Engram en RAM |
| Réserve KV | 1 466 159 jetons (5,6 requêtes de longueur maximale) ; 3 090 900 sans DSpark |
| API | analyseurs d'appels d'outils et de raisonnement `deepseek_v41`, clé d'API obligatoire, `/health` ouvert |

## Organisation du dépôt

```
morrowmake/                       la configuration actuelle, au-dessus de Morrowmake 1.0.1
  compact-kv/                     cache KV compact sans perte (fichiers de moteur sous Apache-2.0)
    compact-kv.patch              modifications de six fichiers du moteur
    vllm_sm80/.../sm80_kv_pack.py compactage, décompactage, chargements et écritures Triton
    build-overlay.sh              construit kvc-overlay/ depuis l'image épinglée + le correctif
    tests/test_gpu.py             test unitaire d'identité au bit près sur GPU ; tests/bench_gpu.py : temps des noyaux
  tools/graft_layers.py           greffe des couches d'un autre pack EXL3 dans le checkpoint exporté
  start.sh.patch                  monte kvc-overlay/ par-dessus les fichiers de l'image
  prepare.sh.patch                empreinte de l'index hybride de la greffe 18-22
  env.example, start-dspark.sh, start-nospec.sh
recipe/                           correctifs pour GNmac-lab 1.2.0, pour mesurer son moteur avec nos poids
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
  quality-research.md, agentic-methodology.md
  results-gnmac.md, setup-gnmac.md  GNmac-lab 1.2.0 mesuré avec nos correctifs (Engram, tests de copie et de charge)
```

La documentation détaillée est en anglais.

## Démarrage rapide

Étapes complètes avec leurs vérifications : [docs/setup.md](docs/setup.md).

1. Installer Morrowmake 1.0.1 (tag `v1.0.1`) et le lancer une fois tel quel.
2. Télécharger les cinq shards donneurs du pack 3,5 bpw de LibertAI, lancer `morrowmake/tools/graft_layers.py`,
   appliquer `prepare.sh.patch`, reconstruire l'hybride.
3. Lancer `morrowmake/compact-kv/build-overlay.sh`, appliquer `start.sh.patch`, régler `VLLM_SM80_COMPACT_KV=1`,
   passer le test unitaire GPU.
4. `./start-dspark.sh`. Le journal doit annoncer une réserve KV de 1 466 159 jetons.

## Méthode

Un agent de programmation IA (Claude Code) a mené la recherche, les expériences et les mesures ; un opérateur humain
a pris chaque décision. Des passes critiques ont relu les rapports de recherche, et chaque affirmation qui a guidé une
décision a été mesurée. [docs/agentic-methodology.md](docs/agentic-methodology.md) décrit la méthode, ce que les
relectures critiques ont attrapé et les pièges de mesure dont nous nous gardons.
[docs/quality-research.md](docs/quality-research.md) couvre le premier tour de recherche sur la fidélité (options de
quantification, budgets de VRAM, diagnostic Engram).

## Remerciements

- [Morrowmake/dsv41-flash-cmp170hx-recipe](https://github.com/Morrowmake/dsv41-flash-cmp170hx-recipe) 1.0.1 et
  [Morrowmake/vllm-cmp170hx](https://github.com/Morrowmake/vllm-cmp170hx) : le moteur et la recette que nous servons.
- [GNmac-lab/deepseek-v41-flash-1017-cmp170hx](https://github.com/GNmac-lab/deepseek-v41-flash-1017-cmp170hx) 1.2.0 :
  l'autre recette que nous avons mesurée, et la base de nos premières mesures.
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
  (`compact-kv.patch` et `sm80_kv_pack.py`), qui modifient le greffon du fork vLLM de Morrowmake, sous Apache-2.0, et sont donc
  sous Apache-2.0 ([leur LICENSE](morrowmake/compact-kv/LICENSE)).
  [NOTICE.md](NOTICE.md) liste chaque fichier dérivé d'un autre projet.
- **Les poids** ne sont pas inclus : ceux de DeepSeek et les packs de LibertAI sont sous MIT, selon les conditions de
  leurs propriétaires.
- L'image du moteur de GNmac-lab, utilisée pour les comparaisons, contient du code AGPL-3.0 ; voir
  [NOTICE.md](NOTICE.md) si vous la construisez.

## Avertissement

Projet de homelab : une machine, une série de mesures. Nous ne sommes affiliés ni à DeepSeek, ni à NVIDIA, ni à
Morrowmake, GNmac-lab, LibertAI ou tout autre projet cité. Les CMP 170HX demandent un pilote modifié pour fonctionner
comme décrit ; vérifiez les conditions qui s'appliquent à vous. Chaque chiffre est ce que nous avons mesuré sur notre
matériel, avec son fichier de données. Les écarts de perplexité sont relatifs entre nos configurations : nous n'avons
pas mesuré la distance au modèle non quantifié.
