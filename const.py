CUE_TABLE = dict(unconfident=1, confident=2, neutral=3)
CUE_TABLE_INV = {value: key for key, value in CUE_TABLE.items()}
DECISION_TABLE = dict(AI=1, SELF=2)
DECISON_TABLE_INV = {value: key for key, value in DECISION_TABLE.items()}
MASK_TOKEN = 0