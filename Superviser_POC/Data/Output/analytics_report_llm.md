Ironbridge Logistics, LLC — Loss & Risk Analytics
=================================================

Submission snapshot
- Insured: Ironbridge Logistics, LLC
- Policy years covered: 4
- Source files: 04_Pinnacle_Casualty_LossRun_Scanned.pdf.md; 02_Continental_Assurance_LossRun_Detail.pdf.md; 03_Summit_Underwriters_LossRun_Excel.pdf.md; 01_Meridian_Mutual_LossRun_CleanTable.pdf.md

1) Overall risk picture — at a glance
- Total premium (all years): 169000.0
- Total incurred (all years): 1055153.27
- Overall loss ratio: 624.4%
- Total claim count (all years): 24
- Average claims per year: 6.0

Interpretation: The account is producing very high incurred relative to premium: overall loss ratio 624.4% and total incurred 1055153.27 versus premium 169000.0.

2) Year-by-year breakdown (each prior policy year)
| Policy year | Carrier | Premium | Claim count | Total incurred | Loss ratio (%) | Open claims | % claims open | Reserve share of incurred (%) | Prior-carrier profit result |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 06/01/2020 - 06/01/2021 | PINNACLE CASUALTY & SURETY CO. | 36500.0 | 5 | 127669.4 | 349.8 | 1 | 20.0 | 17.9 | Unprofitable (loss ratio 349.8% > 100%) |
| 06/01/2021 - 06/01/2022 | SUMMIT UNDERWRITERS EXCHANGE | 39800.0 | 8 | 326277.04 | 819.8 | 5 | 62.5 | 21.9 | Unprofitable (loss ratio 819.8% > 100%) |
| 06/01/2022 - 06/01/2023 | CONTINENTAL ASSURANCE GROUP | 44200.0 | 5 | 382355.74 | 865.1 | 2 | 40.0 | 5.9 | Unprofitable (loss ratio 865.1% > 100%) |
| 06/01/2023 - 06/01/2024 | MERIDIAN MUTUAL INSURANCE COMPANY | 48500.0 | 6 | 218851.09 | 451.2 | 2 | 33.3 | 8.5 | Unprofitable (loss ratio 451.2% > 100%) |

3) Trend (direction and context)
- Oldest year (06/01/2020 - 06/01/2021) loss ratio: 349.8%
- Newest year (06/01/2023 - 06/01/2024) loss ratio: 451.2%
- Trend direction provided in data: worsening

Interpretation: The computed trend is "worsening" (newest-year loss ratio 451.2% is higher than the oldest-year loss ratio 349.8%). Note that several intervening years have extremely high loss ratios (819.8% and 865.1%), which creates substantial volatility and sustained poor performance across the portfolio.

4) Claim types / cause-of-loss patterns
Top causes of loss (as provided)
| Cause of loss | Count |
|---|---:|
| (blank / unspecified) | 8 |
| Premises LIABILITY - p | 2 |
| Water damage - burst p | 2 |
| Cyber incident - data breach | 2 |
| Water damage - burst pipe | 2 |

Interpretation:
- A large share of claims (8) are recorded with a blank/unspecified cause — this limits precise cause-pattern analysis.
- Water-damage variants appear multiple times (entries for "Water damage - burst p" and "Water damage - burst pipe"), indicating repeated water-related incidents (count 2 and 2).
- Premises liability and cyber/data-breach incidents are present (each count 2).
- Because many claims lack a specified cause, the apparent patterns should be treated with caution; see Data Quality notes below.

5) Large / notable individual claims (items that stand out in the claims list)
(Note: no pre-flagged items were included in the provided large_losses array — large_losses is empty — but the claims listing contains several high-incurred files.)
Notable claim entries (claim number — policy year — total_incurred):
- CL-2024-00102 — 06/01/2022 - 06/01/2023 — total_incurred 163812.66 (Premises liability - parking lot)
- 0006 — 06/01/2021 - 06/01/2022 — total_incurred 118009.86
- CL-2023-00105 — 06/01/2022 - 06/01/2023 — total_incurred 107705.12 (Employee injury - forklift accident)
- 0005 — 06/01/2021 - 06/01/2022 — total_incurred 96211.25
- CL-2022-00004 — 06/01/2023 - 06/01/2024 — total_incurred 90359.71 (Water damage - burst pipe; status Open)
- CL-2024-00103 — 06/01/2022 - 06/01/2023 — total_incurred 32636.97 (Water damage - burst pipe)
- CL-2024-00006 — 06/01/2023 - 06/01/2024 — total_incurred 37772.57 (Rear-end collision - company vehicle)

Observations:
- Several very large individual incurred amounts are present across multiple policy years, including a 163812.66 premises-liability parking-lot loss and multiple six-figure incurred items in 2021–2023 years.
- Some of the largest items are closed, while others are open/reopened (see open exposure table below).

6) Open exposure (currently reserved potential future payments)
- Open claim count (aggregate): 10
- Total outstanding reserve (aggregate): 135399.6

Open claims (from the claims list; status not "Closed")
| Claim # | Loss date | Policy year | Status | Cause | Total incurred | Reserve |
|---|---:|---|---:|---|---:|---:|
| CL-2025-0038 | 12/07/20 | 06/01/2020 - 06/01/2021 | Reopened | Water damage - burst p | 67174.0 | 22897.0 |
| CL-2023-00101 | 04/25/2022 | 06/01/2022 - 06/01/2023 | Open | Product liability - defective part | 52080.67 | 6983.14 |
| CL-2023-00105 | 06/30/2023 | 06/01/2022 - 06/01/2023 | Reopened | Employee injury - forklift accident | 107705.12 | 15506.26 |
| 0001 | 03/11/2021 | 06/01/2021 - 06/01/2022 | Open | (blank) | 46610.08 | 7366.19 |
| 0002 | 03/12/2021 | 06/01/2021 - 06/01/2022 | Open | (blank) | 20418.67 | 4747.14 |
| 0004 | 06/11/2021 | 06/01/2021 - 06/01/2022 | Reopened | (blank) | 16544.79 | 3356.89 |
| 0005 | 08/29/2021 | 06/01/2021 - 06/01/2022 | Open | (blank) | 96211.25 | 24095.51 |
| 0006 | 05/30/2022 | 06/01/2021 - 06/01/2022 | Open | (blank) | 118009.86 | 31800.38 |
| CL-2022-00004 | 02/27/2023 | 06/01/2023 - 06/01/2024 | Open | Water damage - burst pipe | 90359.71 | 9840.16 |
| CL-2020-00003 | 12/02/2024 | 06/01/2023 - 06/01/2024 | Open | Rear-end collision - company vehicle | 28108.95 | 8806.93 |

Notes:
- The dataset records 10 open claims and total outstanding reserve 135399.6 (per risk_metrics.open_exposure).
- Open claims include several large incurred items (see table). These reserves represent risk of additional future paid loss/expense.

7) Data maturity & important caveats (must-read)
Maturity caveats included verbatim:
- 06/01/2022 - 06/01/2023: 40.0% of claims still open (5.9% of incurred is still reserve, not paid) - this year's loss ratio (865.1%) is likely to rise further before this policy year is fully developed.
- 06/01/2021 - 06/01/2022: 62.5% of claims still open (21.9% of incurred is still reserve, not paid) - this year's loss ratio (819.8%) is likely to rise further before this policy year is fully developed.
- 06/01/2023 - 06/01/2024: 33.3% of claims still open (8.5% of incurred is still reserve, not paid) - this year's loss ratio (451.2%) is likely to rise further before this policy year is fully developed.

Data quality / anomalies observed in the provided records (do not change numbers)
- The provided top_causes_of_loss list includes a cause entry that is blank with count 8; many claims in the claims table show blank cause_of_loss values. This reduces confidence in detailed cause-pattern conclusions.
- The large_losses array in the risk_metrics is empty (large_losses: []), yet the claims list contains multiple large incurred items; no separate large-loss summary was provided.
- There are duplicate claim numbers in the claims list: CL-2024-0031 appears twice with differing totals (one record total_incurred 48403.0, another 965.4). This appears to be a data inconsistency and should be validated with source loss runs.
- Some claim loss_date values (e.g., 10/17/2024, 12/02/2024) fall after the policy_end "06/01/2024" reported for that policy year, indicating possible date-entry or mapping issues that should be reconciled against source documents.

Summary — key underwriting takeaways (facts only)
- Aggregate performance is very poor: overall loss ratio 624.4% on total premium 169000.0 with total incurred 1055153.27 and 24 claims.
- Every prior carrier experienced an unprofitable year (loss ratios 349.8%, 819.8%, 865.1%, 451.2%).
- Trend labeled "worsening"; recent years show large, high-severity losses and high percentages of open claims in multiple years.
- Repeated water damage and premises liability items appear in the claim list, but cause-of-loss analysis is hampered by 8 claims with blank cause entries.
- Open exposure: 10 open claims with total outstanding reserve 135399.6 — several open claims have sizable incurred and reserves.
- Data issues identified (blank causes, duplicate claim numbers, date/policy mismatches) should be resolved with the source loss run documents prior to final underwriting conclusions.

If you want, I can:
- Produce a one-page PDF summary with the tables above.
- Extract and format individual open-claim text summaries for follow-up.
- Flag the duplicate claim-number and date mismatches for a data-correction request to the broker.