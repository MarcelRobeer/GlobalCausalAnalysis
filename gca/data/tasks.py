FAIRNESS_FEATURES = ['male_words', 'female_words', 'non-binary_words', 'has_name', 'has_religion']
ROBUSTNESS_FEATURES = ['len_chr', 'len_tok', 'len_snt', 'all_lower', 'flesch_grade', 'is_active']
TASK_FEATURES = ['subreddit', 'has_emoji',
                 'NRC_valence', 'NRC_arousal', 'NRC_dominance',
                 'NRC_anger', 'NRC_anticipation', 'NRC_disgust', 'NRC_fear',
                 'NRC_joy', 'NRC_sadness', 'NRC_surprise', 'NRC_trust']

COLORS = {feature: '#ffffff' for feature in TASK_FEATURES}
COLORS |= {feature: '#ffdca9' for feature in FAIRNESS_FEATURES}
COLORS |= {feature: '#e8f3d6' for feature in ROBUSTNESS_FEATURES}
