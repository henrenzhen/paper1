# CTID campaign influence audit v1

Post-result sensitivity analysis; not a new confirmatory test.

## Leave-one-campaign-out NDCG@5

| Comparison | Full delta | LOCO min | Omitted at min | LOCO max | Positive / zero / negative campaigns | Stability |
|---|---:|---:|---|---:|---:|---|
| B0-A0 | +0.0188 | -0.0017 | fin6::unknown | +0.0371 | 6 / 0 / 4 | single-campaign fragile |
| B0-K | +0.0492 | +0.0347 | fin6::unknown | +0.0566 | 8 / 0 / 2 | single-campaign sign-stable |
| B0-T | +0.0383 | +0.0271 | fin6::unknown | +0.0424 | 10 / 0 / 0 | single-campaign sign-stable |
| B0-HM | +0.0741 | +0.0665 | wizard_spider::unknown | +0.0858 | 9 / 0 / 1 | single-campaign sign-stable |
| B0-HM+S | +0.0741 | +0.0665 | wizard_spider::unknown | +0.0858 | 9 / 0 / 1 | single-campaign sign-stable |
| T+B0-B0 | +0.0001 | -0.0001 | turla_carbon::unknown | +0.0002 | 1 / 8 / 1 | single-campaign fragile |

## Per-campaign NDCG@5 differences

| Campaign | B0-A0 | B0-K | B0-T | B0-HM | B0-HM+S | T+B0-B0 |
|---|---:|---:|---:|---:|---:|---:|
| apt29::unknown | -0.0453 | -0.0007 | +0.0045 | -0.0318 | -0.0318 | +0.0000 |
| carbanak::unknown | +0.0028 | +0.0222 | +0.0021 | +0.0816 | +0.0816 | +0.0000 |
| fin6::unknown | +0.2037 | +0.1804 | +0.1394 | +0.0683 | +0.0683 | +0.0000 |
| fin7::unknown | +0.0227 | +0.0814 | +0.0561 | +0.0587 | +0.0587 | +0.0000 |
| menu_pass::unknown | +0.1173 | +0.0408 | +0.0206 | +0.0420 | +0.0420 | +0.0000 |
| oilrig::unknown | +0.0994 | +0.1083 | +0.0568 | +0.1415 | +0.1415 | +0.0000 |
| sandworm::unknown | +0.0087 | -0.0172 | +0.0259 | +0.1122 | +0.1122 | -0.0007 |
| turla_carbon::unknown | -0.0611 | +0.0209 | +0.0079 | +0.0871 | +0.0871 | +0.0018 |
| turla_snake::unknown | -0.1461 | +0.0541 | +0.0251 | +0.0387 | +0.0387 | +0.0000 |
| wizard_spider::unknown | -0.0139 | +0.0023 | +0.0451 | +0.1424 | +0.1424 | +0.0000 |

A sign-stable label means that no single CTID campaign can reverse the positive point estimate. It does not imply statistical significance or generalization to a new source.
