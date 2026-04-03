# Extra Data Format

Put supplemental supervised files here as `.csv` or `.tsv`.

Expected columns:
- text column: one of `model_text`, `text`, `tweet_text`, `tweet`, `content`, `message`
- label column: one of `target`, `label`, `class`, `binary_label`
- optional columns: `keyword`, `location`

Supported binary labels:
- positive: `1`, `true`, `yes`, `positive`, `relevant`, `related`, `informative`, `on topic`, `disaster`
- negative: `0`, `false`, `no`, `negative`, `irrelevant`, `unrelated`, `not informative`, `off topic`, `not disaster`

If your labels use different names, pass them with:

```bash
python train_transformer_with_extra_data.py \
  --extra-positive-labels "affected,humanitarian,request" \
  --extra-negative-labels "not_related,other"
```
