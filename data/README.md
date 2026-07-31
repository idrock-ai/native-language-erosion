# DTM benchmark data

The DTM benchmark (1,000 Uzbek university-entrance multiple-choice questions, subjects
`ona_tili` / `tarix` / `matematika` / `fizika`) is released on IEEE Dataport:

**DOI: [10.21227/e4h4-kp42](https://dx.doi.org/10.21227/e4h4-kp42)**

Place the benchmark JSON here as `DTM_benchmark.json`. Each entry has the fields
`question`, `option_A`..`option_D`, `answer`, `subject`. The loader in `src/data.py`
reads this file, shuffles options under a fixed seed, and produces the stratified
train/dev/test split used in the paper.

## Public replication set

`DTM2019_public.csv` is the public 2,066-question complement of the 1,000-question
benchmark (same DTM 2019 corpus and extraction pipeline). Columns: `question_id`,
`subject`, `topic`, `question`, `option_A`..`option_D`, `correct_answer`; subjects
`math` / `physics` / `history` / `ona_tili`.

It is gitignored. Place it at `data/DTM2019_public.csv` to enable the E4 replication
arm and `tests/test_public.py`. Obtain it from the IDROCK team or the IEEE Dataport
supplementary materials.
