# Submission Plan — V6 Paper

> **Strategy:** arXiv preprint first (no deadline, establishes timestamp + DOI). Then target one workshop submission and one full-venue submission in parallel.

---

## Tier 1 — arXiv (do this first, then iterate before the next deadline)

| Field | Value |
|---|---|
| Venue | arXiv.org cs.LG (Machine Learning) |
| Cross-list to | cs.AI (Artificial Intelligence), cs.MA (Multi-Agent Systems) |
| Deadline | None (rolling) |
| Format | Any LaTeX template; **use NeurIPS style for consistency** with downstream submissions |
| Length | No limit |
| Anonymous | No |
| Review | None (preprint) |
| Time to publish | 1–3 business days after submission |

**Recommendation:** Submit to arXiv first regardless of which conference we target. Dr. Zheng explicitly said this in his May 18 email. The arXiv preprint:
- Establishes priority (important for "first Pokemon Q-ensemble paper" claim)
- Lets you cite it in grad school applications
- Provides a stable URL to share
- Does NOT preclude conference submission (most conferences accept arXiv preprints)

---

## Tier 2 — Workshop venues (target one)

These are reasonable targets for undergraduate research with strong empirical results but modest theoretical novelty. Acceptance rates 30–50%. Less prestigious than the main conference, but appropriate for V6's scope.

### 2.1 NeurIPS 2026 — Workshop on Reinforcement Learning for Video Games (RLVG)
| Field | Value |
|---|---|
| URL | https://sites.google.com/view/rlvg-workshop-2025/home (2025 version; check 2026 site near deadline) |
| Pokemon precedent | Yes — Dr. Zheng explicitly mentioned this workshop in his March 5 email |
| Likely deadline | September 2026 (NeurIPS workshops typically due ~6 weeks before conference) |
| Format | Short paper, 4–8 pages NeurIPS |
| Fit | **Excellent.** Workshop is explicitly for RL in video games. |
| Risk | Workshop may not return; check 2026 listing |

**Recommended primary target.** Best fit, Dr. Zheng-endorsed.

### 2.2 NeurIPS 2026 — Workshop on Deep Reinforcement Learning
| Field | Value |
|---|---|
| Likely deadline | September 2026 |
| Format | 4-page short / 8-page long, NeurIPS |
| Fit | Moderate — V6 is tabular, not deep |
| Risk | "Deep RL" in title may exclude tabular work |

Backup if RLVG doesn't run in 2026.

### 2.3 ICML 2027 — Workshop on RL Foundations
| Field | Value |
|---|---|
| Likely deadline | April 2027 |
| Fit | Good for the foundational claim (ensemble Q-learning) |
| Risk | Foundations workshops often theory-heavy; our paper is empirical |

### 2.4 ICLR 2027 — Workshop on Self-Play and Games
| Field | Value |
|---|---|
| Likely deadline | February 2027 |
| Fit | Good — Pokemon is a self-play game |
| Risk | Often focused on perfect-information games; Pokemon is POMDP |

---

## Tier 3 — Full conference venues (more competitive; stretch goal)

### 3.1 NeurIPS 2026
| Field | Value |
|---|---|
| Main deadline | Passed for 2026 cycle (deadlines were May 2026) |
| Page limit | 9 + unlimited refs/appendix |
| Acceptance rate | ~26% |
| Realistic? | No for 2026; **could target 2027** if we extend the work substantially |

### 3.2 RLC 2026 — Reinforcement Learning Conference
| Field | Value |
|---|---|
| Status | Passed for 2026 (was January 2026 deadline) |
| Page limit | 9 |
| Acceptance rate | ~30% (newer venue) |
| Notable | **Metamon was published here in 2025** — direct precedent |
| Realistic? | Target **RLC 2027** (January 2027 deadline) — strong fit |

### 3.3 AAMAS 2027 — Autonomous Agents and Multi-Agent Systems
| Field | Value |
|---|---|
| Likely deadline | October 2026 |
| Page limit | 7 + 2 refs |
| Acceptance rate | ~25% |
| Fit | Good — Pokemon is multi-agent and stochastic |
| Realistic? | Yes |

### 3.4 IEEE Conference on Games (CoG) 2027
| Field | Value |
|---|---|
| Likely deadline | February 2027 |
| Page limit | 8 |
| Acceptance rate | ~40% |
| Fit | Excellent — game-playing is the venue's focus |
| Notable | Pokemon Showdown AI Competition was at IEEE CIG (CoG predecessor) in 2017 |
| Realistic? | Yes |

### 3.5 AAAI 2027
| Field | Value |
|---|---|
| Likely deadline | August 2026 |
| Page limit | 7 + refs |
| Acceptance rate | ~20% |
| Realistic? | Tighter; tighten the theoretical contribution first |

---

## Recommended sequence (after Friday's meeting)

| Date | Action |
|---|---|
| Week of June 1, 2026 | Draft Method section + Results section in LaTeX |
| Week of June 15 | Draft Intro + Related Work; complete first internal pass |
| End of June | Internal review with Dr. Zheng; revise |
| Week of July 14 | Submit v1 to arXiv |
| Aug 1 | If AAMAS deadline confirmed for Oct: prep submission |
| Sep 1 | Submit to NeurIPS 2026 RLVG workshop (if running) |
| Oct (cycle TBD) | Submit to AAMAS 2027 |
| Jan 2027 | Submit to RLC 2027 |
| Feb 2027 | Submit to IEEE CoG 2027 |

Most papers submit to multiple workshops over time after the arXiv release. The conference targets above are not mutually exclusive in the way reviewers care about — they care about timely peer review, not whether the work was previously on arXiv.

---

## Pre-submission checklist (do before *any* submission)

- [ ] All authors listed in correct order (Arzaan first author, Dr. Zheng senior author)
- [ ] Affiliations correct (Tulane University, Department of Computer Science)
- [ ] Acknowledgment of NTC URAF CAIDS: Focus on Data grant (small mention is appropriate)
- [ ] Reproducibility statement included (code link, hyperparameters, seeds)
- [ ] References complete; every cited work has bibliographic info
- [ ] All figure CIs computed via Wilson method
- [ ] No em dashes anywhere
- [ ] All numbers cross-checked against eval JSONs

---

## Note on co-authorship

Standard for undergraduate research projects under faculty supervision:
- **First author:** Arzaan Singh (you do the work, you go first)
- **Second/Senior author:** Dr. Zizhan Zheng (PI, sets direction)

Confirm this with Dr. Zheng on Friday. Some advisors prefer to be co-authors, some prefer to be acknowledged but not on the byline. The norm in CS is the former.
