import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest

def run_earnings_analysis(file_path):
    # --- Load Data ---
    print("Loading dataset...")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print("Error: File not found.")
        return

    # --- Step 3 & Preprocessing: Filter & Clean ---
    # Filter for Post-Earnings Window (0 to 168 hours / 1 week)
    # Drop rows where essential columns are missing
    df_clean = df.dropna(subset=['vader_compound', 'beat', 'sentiment_label', 'hours_after_earnings', 'event_id'])
    df_window = df_clean[(df_clean['hours_after_earnings'] >= 0) & (df_clean['hours_after_earnings'] <= 168)].copy()
    
    print(f"Data filtered to {len(df_window)} comments within the 168-hour post-earnings window.")
    
    # Split into Groups
    beats = df_window[df_window['beat'] == 1]
    misses = df_window[df_window['beat'] == 0]
    
    print(f"Observations: {len(beats)} Beats, {len(misses)} Misses")
    
    results = {}
    alpha = 0.05

    # --- Step 4: Methodology ---

    # 1. Sentiment Analysis: Two-sample (Welch) T-test
    # Hypothesis: Is there a difference in mean sentiment between beats and misses?
    t_stat_sent, p_val_sent = stats.ttest_ind(
        beats['vader_compound'], 
        misses['vader_compound'], 
        equal_var=False,
        nan_policy='omit'
    )
    
    results['Sentiment Analysis'] = {
        'Metric': 'Mean VADER Compound Score',
        'Beat Mean': beats['vader_compound'].mean(),
        'Miss Mean': misses['vader_compound'].mean(),
        'P-Value': p_val_sent,
        'Significant': p_val_sent < alpha
    }

    # 2. Negativity Analysis: Difference in Proportions
    # Hypothesis: Is the proportion of "negative" comments different?
    beat_neg = (beats['sentiment_label'] == 'negative').sum()
    beat_total = len(beats)
    miss_neg = (misses['sentiment_label'] == 'negative').sum()
    miss_total = len(misses)
    
    if beat_total > 0 and miss_total > 0:
        counts = np.array([beat_neg, miss_neg])
        nobs = np.array([beat_total, miss_total])
        z_stat_neg, p_val_neg = proportions_ztest(counts, nobs)
    else:
        p_val_neg = np.nan

    # Storing as 'Mean' for consistency with other metrics (proportion is a mean of 0/1)
    results['Negativity Analysis'] = {
        'Metric': 'Proportion of Negative Comments',
        'Beat Mean': beat_neg / beat_total if beat_total else 0,
        'Miss Mean': miss_neg / miss_total if miss_total else 0,
        'P-Value': p_val_neg,
        'Significant': p_val_neg < alpha if pd.notnull(p_val_neg) else False
    }

    # 3. Engagement Analysis (Volume): Comments per Event
    # Hypothesis: Do beats generate more comments per event than misses?
    beat_counts = beats.groupby('event_id').size()
    miss_counts = misses.groupby('event_id').size()
    
    # We align indices to ensure we are comparing distributions of event counts
    if len(beat_counts) > 1 and len(miss_counts) > 1:
        t_stat_eng, p_val_eng = stats.ttest_ind(
            beat_counts, 
            miss_counts, 
            equal_var=False
        )
    else:
        p_val_eng = np.nan

    results['Engagement Volume'] = {
        'Metric': 'Mean Comments per Earnings Event',
        'Beat Mean': beat_counts.mean(),
        'Miss Mean': miss_counts.mean(),
        'P-Value': p_val_eng,
        'Significant': p_val_eng < alpha if pd.notnull(p_val_eng) else False
    }

    # 4. Engagement Analysis (Timing): Average Hours to Comment
    # Hypothesis: Do investors react faster (lower hours_after_earnings) to beats vs misses?
    t_stat_time, p_val_time = stats.ttest_ind(
        beats['hours_after_earnings'],
        misses['hours_after_earnings'],
        equal_var=False
    )

    results['Reaction Timing'] = {
        'Metric': 'Mean Hours After Earnings',
        'Beat Mean': beats['hours_after_earnings'].mean(),
        'Miss Mean': misses['hours_after_earnings'].mean(),
        'P-Value': p_val_time,
        'Significant': p_val_time < alpha
    }

    # --- Step 5: Evaluation & Output ---
    print("\n" + "="*50)
    print("STATISTICAL ANALYSIS RESULTS")
    print("="*50)
    
    for category, res in results.items():
        print(f"\n[{category}]")
        print(f"  Metric: {res['Metric']}")
        # Standardized printing loop now works for all metrics
        print(f"  Beat Group: {res['Beat Mean']:.4f}")
        print(f"  Miss Group: {res['Miss Mean']:.4f}")
        print(f"  Difference: {res['Beat Mean'] - res['Miss Mean']:.4f}")
        
        if pd.notnull(res['P-Value']):
            print(f"  P-Value:    {res['P-Value']:.4f}")
        else:
            print(f"  P-Value:    N/A (Insufficient Data)")
            
        if res['Significant']:
            print("  -> RESULT: Statistically Significant Difference (p < 0.05)")
        else:
            print("  -> RESULT: No Significant Difference")

# Execute
run_earnings_analysis('reddit_with_earnings_linked.csv')