# Paper dataset audit

Source: `Sakonii/task-over-refusal-dataset` split `test`

## Official benchmark
- Rows: **270**
- Benign-intent tasks (OR numerator eligible): **205**
- Rephrase / other (OR indicator always 0): **65**

### text_type counts

- `benign_instruction`: 30
- `decrypt_harmful_instruction`: 30
- `decrypt_harmful_response`: 30
- `harmful_instruction`: 30
- `harmful_response`: 30
- `jailbreak_prompt`: 30
- `rag_prompt`: 30
- `xstest_safe`: 30
- `xstest_unsafe`: 30

### intended_task counts

- `rephrase`: 65
- `translate`: 60
- `cryptanalysis`: 60
- `sentiment_analysis`: 55
- `rag_qa`: 30

## Legacy Phase-1 CSV comparison

- `base_prompts.csv` exists: True
- `reviewed_translations.csv` exists: True
- Legacy base rows: **20** (paper test: **270**)
- Legacy corrupted short base_text (<20 chars): **2**
- Legacy prompt_id english vs base_text mismatches: **18**
- Paper test rows with harmful/jailbreak/unsafe/decrypt text_type: **180**

## Implications

- Replication must use `plain_text` from the official split, not `build_base_prompts()`.
- OR metric uses all test rows in the denominator (rephrase rows contribute 0 to numerator).
- Legacy 10-prompt pilot is not comparable to Table 2 (17.77%).
