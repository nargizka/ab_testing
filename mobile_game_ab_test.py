import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
# import warnings
# warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class CookieCatsABTest:
    """Class for A/B testing analysis on Cookie Cats dataset."""
    
    def __init__(self, filepath):
        """Initialize with data filepath"""
        self.df = None
        self.filepath = filepath
        self.bootstrap_results_1d = None
        self.bootstrap_results_7d = None
        
    def load_data(self):
        """Load data and list basic information about the dataset"""
        print("___LOADING DATA___")
        
        self.df = pd.read_csv(self.filepath)
        
        print(f"\nDataset shape: {self.df.shape}")
        print(f"\nView first rows and data types.")
        print(self.df.head())
        
        print(self.df.dtypes)
        
        print(f"\nBasic Info:")
        print(self.df.describe())
        
        print(f"\nMissing values:")
        print(self.df.isnull().sum())

        duplicates = self.df['userid'].duplicated().sum()
        print(f"Duplicate user IDs: {duplicates}")

        print(f"Group distribution:")
        print(self.df['version'].value_counts())
        print(f"Group proportions:")
        print(self.df['version'].value_counts(normalize=True))
        
        return self
    

    
    def visualize_distributions(self):
        """Create Plots"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # game rounds distribution by group
        ax1 = axes[0, 0]
        for version in ['gate_30', 'gate_40']:
            data = self.df[self.df['version'] == version]['sum_gamerounds']
            ax1.hist(data, bins=50, alpha=0.5, label=version, edgecolor='black')
        ax1.set_xlabel('Game Rounds')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Game Rounds by Group')
        ax1.legend()
        ax1.set_xlim(0, 200)
        
        # box plot of game rounds
        ax2 = axes[0, 1]
        self.df.boxplot(column='sum_gamerounds', by='version', ax=ax2)
        ax2.set_xlabel('Version')
        ax2.set_ylabel('Game Rounds')
        ax2.set_title('Game Rounds Distribution by Group')
        plt.sca(ax2)
        plt.xticks(rotation=0)
        
        # retention rates comparison
        ax3 = axes[1, 0]
        retention_data = self.df.groupby('version')[['retention_1', 'retention_7']].mean()
        retention_data.plot(kind='bar', ax=ax3, color=['#3498db', '#e74c3c'])
        ax3.set_xlabel('Version')
        ax3.set_ylabel('Retention Rate')
        ax3.set_title('Retention Rates by Group')
        ax3.legend(['1-Day Retention', '7-Day Retention'])
        ax3.set_ylim(0, 0.5)
        plt.sca(ax3)
        plt.xticks(rotation=0)
        
        for container in ax3.containers:
            ax3.bar_label(container, fmt='%.3f')
        
        # sample sizes
        ax4 = axes[1, 1]
        sample_sizes = self.df['version'].value_counts()
        ax4.bar(sample_sizes.index, sample_sizes.values, color=['#2ecc71', '#f39c12'])
        ax4.set_xlabel('Version')
        ax4.set_ylabel('Number of Users')
        ax4.set_title('Sample Size by Group')
        
        for i, v in enumerate(sample_sizes.values):
            ax4.text(i, v + 500, str(v), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
        
        return self
    
    def calculate_retention_metrics(self):
        """Calculates retention metrics for each group."""
        
        metrics = self.df.groupby('version').agg({
            'retention_1': ['sum', 'mean', 'count'],
            'retention_7': ['sum', 'mean']
        }).round(4)
        
        print("=== Retention Metrics ===")
        print(metrics)
        
        gate_30_ret1 = self.df[self.df['version'] == 'gate_30']['retention_1'].mean()
        gate_40_ret1 = self.df[self.df['version'] == 'gate_40']['retention_1'].mean()
        
        gate_30_ret7 = self.df[self.df['version'] == 'gate_30']['retention_7'].mean()
        gate_40_ret7 = self.df[self.df['version'] == 'gate_40']['retention_7'].mean()
        
        print("=== 1-DAY RETENTION ===")
        print(60*"=")
        print(f"gate_30 (Control):   {gate_30_ret1:.4f} ({gate_30_ret1*100:.2f}%)")
        print(f"gate_40 (Treatment): {gate_40_ret1:.4f} ({gate_40_ret1*100:.2f}%)")
        print(f"Difference: {(gate_40_ret1 -gate_30_ret1):.4f} ({(gate_40_ret1 -gate_30_ret1)*100:.2f}%)")
        
        print("\n")
        print("=== 7-DAY RETENTION ===")
        print(f"gate_30 (Control):   {gate_30_ret7:.4f} ({gate_30_ret7*100:.2f}%)")
        print(f"gate_40 (Treatment): {gate_40_ret7:.4f} ({gate_40_ret7*100:.2f}%)")
        print(f"Difference: {(gate_40_ret7 -gate_30_ret7):.4f} ({(gate_40_ret7 -gate_30_ret7)*100:.2f}%)")
        
        return self
    
    
    def z_test_proportions(self, metric='retention_1', alpha=0.05):
        """
        Perform z-test for proportions (two-proportion z-test)
        H0: p1 = p2 (retention rates are equal)
        H1: p1 != p2 (retention rates are different)
        """
        print(f"=== Z-TEST FOR PROPORTIONS - {metric.upper()} ===\n")
        
        gate_30_data = self.df[self.df['version'] == 'gate_30'][metric]
        gate_40_data = self.df[self.df['version'] == 'gate_40'][metric]
        
        # calculate proportions
        n1 = len(gate_30_data)
        n2 = len(gate_40_data)
        p1 = gate_30_data.mean()
        p2 = gate_40_data.mean()
        
        # pooled proportion
        p_pool = (gate_30_data.sum() + gate_40_data.sum()) / (n1 + n2)
        
        #standard error
        se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
        
        # z-statistic
        z_stat = (p2 - p1) / se
        
        # p-value (two-tailed)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        
        # confidence interval for difference
        se_diff = np.sqrt(p1*(1-p1)/n1 + p2*(1-p2)/n2)
        ci_lower = (p2 - p1) - 1.96 * se_diff
        ci_upper = (p2 - p1) + 1.96 * se_diff
        
        print(f"\nSample sizes:")
        print(f"  gate_30: n = {n1}")
        print(f"  gate_40: n = {n2}")
        
        print(f"\nProportions:")
        print(f"  gate_30: p1 = {p1:.6f}")
        print(f"  gate_40: p2 = {p2:.6f}")
        print(f"  Pooled:  p  = {p_pool:.6f}")
        
        print(f"\nTest statistics:")
        print(f"  z-statistic: {z_stat:.4f}")
        print(f"  p-value: {p_value:.6f}")
        print(f"  std error: {se:.6f}")
        
        print(f"\n95% CI (confidence interval) for difference ((gate_40 - gate_30)):")
        print(f"  [{ci_lower:.6f}, {ci_upper:.6f}]")
        
        # Interpretation
        print("=== INTERPRETATION ===\n")
        if p_value < alpha:
            print(f"REJECT NULL HYPOTHESIS (p = {p_value:.6f} < {alpha})")
            print(f"The retention rates are statistically significantly different.")
            if p2 > p1:
                print(f"gate_40 has significantly HIGHER {metric} than gate_30.")
            else:
                print(f"gate_30 has significantly HIGHER {metric} than gate_40.")
        else:
            print(f"FAIL TO REJECT NULL HYPOTHESIS (p = {p_value:.6f} >= {alpha})")
            print(f"There is no statistically significant difference in retention rates.")
        
        return self
    
    def bootstrap_analysis(self, metric='retention_1', n_bootstrap=1000, alpha=0.05):
        print(f"=== BOOTSTRAP ANALYSIS: {metric.upper()} ===\n")
        
        print(f"Generating {n_bootstrap} bootstrap samples...")
        
        gate_30 = self.df[self.df['version'] == 'gate_30'][metric].values
        gate_40 = self.df[self.df['version'] == 'gate_40'][metric].values

        boot_diffs = []
        boot_30_means = []
        boot_40_means = []

        for _ in range(n_bootstrap):
            sample_30 = np.random.choice(gate_30, size=len(gate_30), replace=True)
            sample_40 = np.random.choice(gate_40, size=len(gate_40), replace=True)
            m30 = sample_30.mean()
            m40 = sample_40.mean()
            boot_30_means.append(m30)
            boot_40_means.append(m40)
            boot_diffs.append(m40 - m30)

        boot_df = pd.DataFrame({
            'gate_30': boot_30_means,
            'gate_40': boot_40_means,
            'diff': boot_diffs,
        })

        
        if metric == 'retention_1':
            self.bootstrap_results_1d = boot_df
        else:
            self.bootstrap_results_7d = boot_df
        
        # calculate statistics
        diff_mean = boot_df['diff'].mean()
        diff_std = boot_df['diff'].std()
        diff_ci_lower = boot_df['diff'].quantile(alpha/2)
        diff_ci_upper = boot_df['diff'].quantile(1 - alpha/2)
        
        prob_positive = (boot_df['diff'] > 0).sum() / len(boot_df)
        
        print(20*"=" + " Bootstrap Results "+ 20*"=")
        print(f"* mean difference: {diff_mean:.6f} ({diff_mean*100:.3f}%)")
        print(f"* std deviation: {diff_std:.6f}")
        print(f"* {int((1-alpha)*100)}% CI: [{diff_ci_lower:.6f}, {diff_ci_upper:.6f}]")
        print(f"* Prob(gate_40 > gate_30): {prob_positive:.2%}")
        
        # Plots
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Distribution of differences
        ax1 = axes[0]
        boot_df['diff'].hist(bins=50, ax=ax1, edgecolor='black', alpha=0.7)
        ax1.axvline(0, color='red', linestyle='--', linewidth=2, label='No difference')
        ax1.axvline(diff_mean, color='green', linestyle='-', linewidth=2, label=f'Mean: {diff_mean:.4f}')
        ax1.set_xlabel('Difference in Retention Rates (gate_40 - gate_30)')
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'Bootstrap Distribution of Difference in {metric.upper()}')
        ax1.legend()
        
        # KDE plot
        ax2 = axes[1]
        boot_df[['gate_30', 'gate_40']].plot(kind='kde', ax=ax2, linewidth=2)
        ax2.set_xlabel('Retention Rate')
        ax2.set_ylabel('Density')
        ax2.set_title(f'Bootstrap Distribution of {metric.upper()} by Group')
        ax2.legend(['gate_30 (Control)', 'gate_40 (Treatment)'])
        
        plt.tight_layout()
        plt.show()
        
        print("=== INTERPRETATION ===\n")
        if diff_ci_lower > 0:
            print(f"We can be {int((1-alpha)*100)}% confident that gate_40 has HIGHER {metric} than gate_30.")
        elif diff_ci_upper < 0:
            print(f"We can be {int((1-alpha)*100)}% confident that gate_40 has LOWER {metric} than gate_30.")
        else:
            print(f"X We cannot be {int((1-alpha)*100)}% confident that there is a true difference in {metric}.")
        print(f"\nThere is a {prob_positive:.1%} probability that gate_40 has higher {metric} than gate_30.")
        
        return self
    
    def effect_size_analysis(self):
        """Calculate effect sizes (Cohen's h for proportions)"""
        print("=== EFFECT SIZE ANALYSIS ===\n")
        
        for metric in ['retention_1', 'retention_7']:
            p1 = self.df[self.df['version'] == 'gate_30'][metric].mean()
            p2 = self.df[self.df['version'] == 'gate_40'][metric].mean()
            
            # Cohen's h for proportions
            # Positive h means treatment > control
            h = 2 * (np.arcsin(np.sqrt(p2)) - np.arcsin(np.sqrt(p1)))
            
            if abs(h) < 0.2:
                interpretation = "small effect"
            elif abs(h) < 0.5:
                interpretation = "medium effect"
            else:
                interpretation = "large effect"
            
            print(f"\n{metric.upper()}:")
            print(f"* Cohen's h: {h:.4f}")
            print(f"* Interpretation: {interpretation}")
            print(f"* Absolute difference: {abs(p1-p2):.4f} ({abs(p1-p2)*100:.2f}%)")
        
        return self


def main_test():
    filepath = "./ab_test/data/cookie_cats.csv" 
    print(20*"*" + "MOBILE GAME (COOKIE CATS) A/B TESTING ANALYSIS" + 20*"*")
    
    abtest_obj = CookieCatsABTest(filepath)
    
    try:
        abtest_obj.load_data()
        abtest_obj.visualize_distributions()
        abtest_obj.calculate_retention_metrics()
        abtest_obj.z_test_proportions(metric='retention_1')
        abtest_obj.z_test_proportions(metric='retention_7')
        abtest_obj.bootstrap_analysis(metric='retention_1', n_bootstrap=1000)
        abtest_obj.bootstrap_analysis(metric='retention_7', n_bootstrap=1000)
        abtest_obj.effect_size_analysis()  
    except Exception as e:
        print(f"\nError during analysis: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    
    # Cookie Cats A/B Testing Analysis
    # This script performs a comprehensive A/B testing analysis on 
    # the Cookie Cats mobile game dataset from Kaggle.
    main_test()
