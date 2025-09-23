import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

warnings.filterwarnings('ignore')

# Set visualization style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Create output directory
output_dir = './fraud_rst'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Global variable to store all analysis results
analysis_results = []


def add_result(section, content):
    """Add analysis result to global results list"""
    analysis_results.append(f"\n{'=' * 60}")
    analysis_results.append(f"{section}")
    analysis_results.append(f"{'=' * 60}")
    analysis_results.append(content)


def load_and_prepare_data(file_path):
    """Load and prepare transaction data for analysis"""
    df = pd.read_csv(file_path)

    # Convert amount column to numeric
    df['Amount'] = df['Amount'].replace('[\$,]', '', regex=True).astype(float)

    # Convert dates
    df['Post Date'] = pd.to_datetime(df['Post Date'], errors='coerce')
    df['Account Open Date'] = pd.to_datetime(df['Account Open Date'], errors='coerce')

    # Create fraud indicator
    df['Is_Fraud'] = df['Fraud Adjustment Indicator'].notna() & (df['Fraud Adjustment Indicator'].str.strip() != '')

    # Clean IDs
    df['Member ID'] = df['Member ID'].astype(str).str.lstrip('0')
    df['Account ID'] = df['Account ID'].astype(str).str.lstrip('0')

    # Basic statistics
    total_transactions = len(df)
    fraud_cases = df['Is_Fraud'].sum()
    fraud_rate = fraud_cases / total_transactions * 100

    result = f"""Dataset loaded successfully:
- Total transactions: {total_transactions:,}
- Fraud cases: {fraud_cases:,}
- Fraud rate: {fraud_rate:.4f}%
- Date range: {df['Post Date'].min().strftime('%Y-%m-%d')} to {df['Post Date'].max().strftime('%Y-%m-%d')}
- Unique accounts: {df['Account ID'].nunique():,}
- Unique members: {df['Member ID'].nunique():,}"""

    add_result("DATASET OVERVIEW", result)
    return df


def analyze_fraud_amounts(df):
    """Analyze fraud transaction amount patterns"""
    fraud_df = df[df['Is_Fraud']].copy()
    normal_df = df[~df['Is_Fraud']].copy()

    # Basic fraud amount statistics
    total_fraud = len(fraud_df)
    min_amount = fraud_df['Amount'].min()
    max_amount = fraud_df['Amount'].max()
    mean_amount = fraud_df['Amount'].mean()
    median_amount = fraud_df['Amount'].median()
    std_amount = fraud_df['Amount'].std()

    # Round number analysis
    fraud_df['Is_Round'] = fraud_df['Amount'].apply(lambda x: x % 1 == 0 and x % 10 == 0)
    normal_df['Is_Round'] = normal_df['Amount'].apply(lambda x: x % 1 == 0 and x % 10 == 0)

    round_fraud_pct = fraud_df['Is_Round'].mean() * 100
    round_normal_pct = normal_df['Is_Round'].mean() * 100

    # Amount ranges analysis
    bins = [0, 25, 50, 100, 200, 500, 1000, 5000, float('inf')]
    labels = ['$0-25', '$25-50', '$50-100', '$100-200', '$200-500', '$500-1K', '$1K-5K', '>$5K']

    fraud_df['Amount_Range'] = pd.cut(fraud_df['Amount'], bins=bins, labels=labels, right=False)
    df['Amount_Range'] = pd.cut(df['Amount'], bins=bins, labels=labels, right=False)

    fraud_by_range = fraud_df['Amount_Range'].value_counts().sort_index()
    total_by_range = df['Amount_Range'].value_counts().sort_index()
    fraud_rate_by_range = (fraud_by_range / total_by_range * 100).fillna(0)

    # Action Type analysis
    action_analysis = []
    for action in df['Action Type'].unique():
        if pd.notna(action):
            action_data = df[df['Action Type'] == action]
            action_fraud = action_data[action_data['Is_Fraud']]
            if len(action_data) > 0:
                action_analysis.append(
                    f"  {action}: {len(action_fraud)} fraud / {len(action_data)} total ({len(action_fraud) / len(action_data) * 100:.2f}%)")

    # Source Type analysis
    source_analysis = []
    for source in df['Source Type'].unique():
        if pd.notna(source):
            source_data = df[df['Source Type'] == source]
            source_fraud = source_data[source_data['Is_Fraud']]
            if len(source_data) > 10:
                source_analysis.append(
                    f"  {source}: {len(source_fraud)} fraud / {len(source_data)} total ({len(source_fraud) / len(source_data) * 100:.2f}%)")

    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Fraud Amount Pattern Analysis', fontsize=16, fontweight='bold')

    # 1. Amount distribution comparison
    axes[0, 0].hist(normal_df['Amount'], bins=50, alpha=0.7, label='Normal', density=True)
    axes[0, 0].hist(fraud_df['Amount'], bins=50, alpha=0.7, label='Fraud', density=True)
    axes[0, 0].set_xlabel('Amount ($)')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('Amount Distribution Comparison')
    axes[0, 0].legend()
    axes[0, 0].set_xlim(0, 2000)

    # 2. Fraud rate by amount range
    bars = axes[0, 1].bar(range(len(fraud_rate_by_range)), fraud_rate_by_range.values)
    axes[0, 1].set_xticks(range(len(labels)))
    axes[0, 1].set_xticklabels(labels, rotation=45)
    axes[0, 1].set_ylabel('Fraud Rate (%)')
    axes[0, 1].set_title('Fraud Rate by Amount Range')
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                        f'{height:.2f}%', ha='center', va='bottom', fontsize=8)

    # 3. Round vs Non-round amounts
    round_data = ['Fraud', 'Normal']
    round_percentages = [round_fraud_pct, round_normal_pct]
    bars = axes[0, 2].bar(round_data, round_percentages)
    axes[0, 2].set_ylabel('Round Amount Percentage (%)')
    axes[0, 2].set_title('Round Amount Preference')
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        axes[0, 2].text(bar.get_x() + bar.get_width() / 2., height + 1,
                        f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

    # 4. Top fraud amounts
    top_fraud_amounts = fraud_df['Amount'].value_counts().head(10)
    if len(top_fraud_amounts) > 0:
        axes[1, 0].barh(range(len(top_fraud_amounts)), top_fraud_amounts.values)
        axes[1, 0].set_yticks(range(len(top_fraud_amounts)))
        axes[1, 0].set_yticklabels([f'${x:.2f}' for x in top_fraud_amounts.index])
        axes[1, 0].set_xlabel('Frequency')
        axes[1, 0].set_title('Top 10 Most Common Fraud Amounts')

    # 5. Action Type fraud analysis
    action_types = []
    action_fraud_counts = []
    for action in df['Action Type'].unique():
        if pd.notna(action):
            action_fraud = df[(df['Action Type'] == action) & df['Is_Fraud']]
            if len(action_fraud) > 0:
                action_types.append(action)
                action_fraud_counts.append(len(action_fraud))

    if action_types:
        axes[1, 1].bar(action_types, action_fraud_counts)
        axes[1, 1].set_ylabel('Fraud Count')
        axes[1, 1].set_title('Fraud Count by Action Type')
        axes[1, 1].tick_params(axis='x', rotation=45)

    # 6. Source Type fraud analysis
    source_types = []
    source_fraud_rates = []
    for source in df['Source Type'].unique():
        if pd.notna(source):
            source_data = df[df['Source Type'] == source]
            source_fraud = source_data[source_data['Is_Fraud']]
            if len(source_data) > 10:
                source_types.append(source[:15])
                source_fraud_rates.append(len(source_fraud) / len(source_data) * 100)

    if source_types:
        axes[1, 2].bar(source_types, source_fraud_rates)
        axes[1, 2].set_ylabel('Fraud Rate (%)')
        axes[1, 2].set_title('Fraud Rate by Source Type')
        axes[1, 2].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fraud_amount_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Compile results
    result = f"""FRAUD AMOUNT STATISTICS:
- Total fraud cases: {total_fraud}
- Amount range: ${min_amount:.2f} - ${max_amount:.2f}
- Mean amount: ${mean_amount:.2f}
- Median amount: ${median_amount:.2f}
- Standard deviation: ${std_amount:.2f}
- Round amount preference: {round_fraud_pct:.1f}% (vs {round_normal_pct:.1f}% normal)

FRAUD RATE BY AMOUNT RANGE:
{chr(10).join([f"  {label}: {fraud_by_range.get(label, 0)} fraud / {total_by_range.get(label, 0)} total ({fraud_rate_by_range.get(label, 0):.2f}%)" for label in labels])}

FRAUD BY ACTION TYPE:
{chr(10).join(action_analysis)}

FRAUD BY SOURCE TYPE:
{chr(10).join(source_analysis)}"""

    add_result("FRAUD AMOUNT ANALYSIS", result)
    return fraud_df


def analyze_fraud_demographics(df):
    """Analyze fraud patterns by demographics"""
    fraud_df = df[df['Is_Fraud']].copy()

    # Age analysis
    age_bins = [0, 25, 35, 45, 55, 65, 100]
    age_labels = ['18-25', '26-35', '36-45', '46-55', '56-65', '65+']

    df['Age_Group'] = pd.cut(df['Member Age'], bins=age_bins, labels=age_labels, right=False)
    fraud_df['Age_Group'] = pd.cut(fraud_df['Member Age'], bins=age_bins, labels=age_labels, right=False)

    # Account type analysis
    account_fraud_stats = []
    for acc_type in df['Account Type'].unique():
        if pd.notna(acc_type):
            acc_data = df[df['Account Type'] == acc_type]
            acc_fraud = acc_data[acc_data['Is_Fraud']]
            account_fraud_stats.append(
                f"  {acc_type}: {len(acc_fraud)} fraud / {len(acc_data)} total ({len(acc_fraud) / len(acc_data) * 100:.2f}%)")

    # Age group analysis
    age_fraud_stats = []
    for age_group in age_labels:
        age_data = df[df['Age_Group'] == age_group]
        age_fraud = age_data[age_data['Is_Fraud']]
        if len(age_data) > 0:
            age_fraud_stats.append(
                f"  {age_group}: {len(age_fraud)} fraud / {len(age_data)} total ({len(age_fraud) / len(age_data) * 100:.2f}%)")

    # Create visualizations
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Fraud Demographic Analysis', fontsize=16, fontweight='bold')

    # 1. Age distribution comparison
    axes[0, 0].hist(df[~df['Is_Fraud']]['Member Age'], bins=30, alpha=0.7, label='Normal', density=True)
    axes[0, 0].hist(fraud_df['Member Age'], bins=30, alpha=0.7, label='Fraud', density=True)
    axes[0, 0].set_xlabel('Age')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('Age Distribution Comparison')
    axes[0, 0].legend()

    # 2. Fraud rate by age group
    age_groups = []
    age_rates = []
    for age_group in age_labels:
        age_data = df[df['Age_Group'] == age_group]
        age_fraud = age_data[age_data['Is_Fraud']]
        if len(age_data) > 0:
            age_groups.append(age_group)
            age_rates.append(len(age_fraud) / len(age_data) * 100)

    if age_groups:
        axes[0, 1].bar(age_groups, age_rates)
        axes[0, 1].set_ylabel('Fraud Rate (%)')
        axes[0, 1].set_title('Fraud Rate by Age Group')
        axes[0, 1].tick_params(axis='x', rotation=45)

    # 3. Account type analysis
    acc_types = []
    acc_rates = []
    for acc_type in df['Account Type'].unique():
        if pd.notna(acc_type):
            acc_data = df[df['Account Type'] == acc_type]
            acc_fraud = acc_data[acc_data['Is_Fraud']]
            acc_types.append(acc_type[:15])
            acc_rates.append(len(acc_fraud) / len(acc_data) * 100 if len(acc_data) > 0 else 0)

    if acc_types:
        axes[0, 2].bar(acc_types, acc_rates)
        axes[0, 2].set_ylabel('Fraud Rate (%)')
        axes[0, 2].set_title('Fraud Rate by Account Type')
        axes[0, 2].tick_params(axis='x', rotation=45)

    # 4. Age group fraud counts
    if age_groups:
        age_counts = [len(df[(df['Age_Group'] == group) & df['Is_Fraud']]) for group in age_groups]
        axes[1, 0].bar(age_groups, age_counts)
        axes[1, 0].set_ylabel('Fraud Count')
        axes[1, 0].set_title('Fraud Count by Age Group')
        axes[1, 0].tick_params(axis='x', rotation=45)

    # 5. Average fraud amount by age group
    if age_groups:
        age_avg_amounts = []
        for group in age_groups:
            group_fraud = df[(df['Age_Group'] == group) & df['Is_Fraud']]
            age_avg_amounts.append(group_fraud['Amount'].mean() if len(group_fraud) > 0 else 0)

        axes[1, 1].bar(age_groups, age_avg_amounts)
        axes[1, 1].set_ylabel('Average Fraud Amount ($)')
        axes[1, 1].set_title('Average Fraud Amount by Age Group')
        axes[1, 1].tick_params(axis='x', rotation=45)

    # 6. Age vs Amount scatter plot for fraud
    if len(fraud_df) > 0:
        axes[1, 2].scatter(fraud_df['Member Age'], fraud_df['Amount'], alpha=0.6)
        axes[1, 2].set_xlabel('Member Age')
        axes[1, 2].set_ylabel('Fraud Amount ($)')
        axes[1, 2].set_title('Fraud Cases: Age vs Amount')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fraud_demographic_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Compile results
    result = f"""DEMOGRAPHIC ANALYSIS:
- Age range in fraud: {fraud_df['Member Age'].min():.0f} - {fraud_df['Member Age'].max():.0f}
- Average fraud victim age: {fraud_df['Member Age'].mean():.1f}
- Median fraud victim age: {fraud_df['Member Age'].median():.1f}
- Standard deviation: {fraud_df['Member Age'].std():.1f}

FRAUD BY ACCOUNT TYPE:
{chr(10).join(account_fraud_stats)}

FRAUD BY AGE GROUP:
{chr(10).join(age_fraud_stats)}"""

    add_result("FRAUD DEMOGRAPHIC ANALYSIS", result)
    return fraud_df


def analyze_fraud_timing(df):
    """Analyze fraud timing patterns"""
    fraud_df = df[df['Is_Fraud']].copy()

    # Time analysis - Parse HHMMSS format (6 digits: hour, minute, second)
    def parse_time_to_hour(time_val):
        if pd.notna(time_val):
            time_str = str(int(time_val)).zfill(6)  # Ensure 6 digits with leading zeros
            hour = int(time_str[:2])  # First 2 digits are hour
            return hour
        return np.nan

    fraud_df['Hour'] = fraud_df['Post Time'].apply(parse_time_to_hour)
    df['Hour'] = df['Post Time'].apply(parse_time_to_hour)

    # Date analysis
    fraud_df['Day'] = fraud_df['Post Date'].dt.day
    fraud_df['Weekday'] = fraud_df['Post Date'].dt.day_name()

    df['Day'] = df['Post Date'].dt.day
    df['Weekday'] = df['Post Date'].dt.day_name()

    # Hourly analysis
    hourly_fraud = fraud_df['Hour'].value_counts().sort_index()
    hourly_total = df['Hour'].value_counts().sort_index()
    hourly_fraud_rate = (hourly_fraud / hourly_total * 100).fillna(0)

    # Weekday analysis
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekday_fraud = fraud_df['Weekday'].value_counts().reindex(weekday_order).fillna(0)
    weekday_total = df['Weekday'].value_counts().reindex(weekday_order).fillna(0)
    weekday_rate = (weekday_fraud / weekday_total * 100).fillna(0)

    # Daily analysis
    daily_fraud = fraud_df['Day'].value_counts().sort_index()

    # Create visualizations
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Fraud Timing Pattern Analysis', fontsize=16, fontweight='bold')

    # 1. Hourly fraud distribution
    axes[0, 0].bar(hourly_fraud.index, hourly_fraud.values)
    axes[0, 0].set_xlabel('Hour')
    axes[0, 0].set_ylabel('Fraud Count')
    axes[0, 0].set_title('Fraud Distribution by Hour')
    axes[0, 0].set_xlim(-0.5, 23.5)

    # 2. Hourly fraud rate
    axes[0, 1].plot(hourly_fraud_rate.index, hourly_fraud_rate.values, marker='o')
    axes[0, 1].set_xlabel('Hour')
    axes[0, 1].set_ylabel('Fraud Rate (%)')
    axes[0, 1].set_title('Fraud Rate by Hour')
    axes[0, 1].set_xlim(-0.5, 23.5)
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Daily distribution
    axes[0, 2].bar(daily_fraud.index, daily_fraud.values)
    axes[0, 2].set_xlabel('Day of Month')
    axes[0, 2].set_ylabel('Fraud Count')
    axes[0, 2].set_title('Fraud Distribution by Day of Month')

    # 4. Weekday fraud count
    axes[1, 0].bar(range(len(weekday_fraud)), weekday_fraud.values)
    axes[1, 0].set_xticks(range(len(weekday_fraud)))
    axes[1, 0].set_xticklabels([day[:3] for day in weekday_order], rotation=45)
    axes[1, 0].set_ylabel('Fraud Count')
    axes[1, 0].set_title('Fraud Count by Weekday')

    # 5. Weekday fraud rate
    axes[1, 1].bar(range(len(weekday_rate)), weekday_rate.values)
    axes[1, 1].set_xticks(range(len(weekday_rate)))
    axes[1, 1].set_xticklabels([day[:3] for day in weekday_order], rotation=45)
    axes[1, 1].set_ylabel('Fraud Rate (%)')
    axes[1, 1].set_title('Fraud Rate by Weekday')

    # 6. Time distribution heatmap
    if len(fraud_df) > 20:
        fraud_time_data = fraud_df.groupby(['Hour', 'Day']).size().reset_index(name='Count')
        if len(fraud_time_data) > 0:
            pivot_data = fraud_time_data.pivot(index='Hour', columns='Day', values='Count').fillna(0)
            sns.heatmap(pivot_data, ax=axes[1, 2], cmap='YlOrRd', cbar_kws={'label': 'Fraud Count'})
            axes[1, 2].set_title('Fraud Heatmap: Hour vs Day')
            axes[1, 2].set_xlabel('Day of Month')
            axes[1, 2].set_ylabel('Hour')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fraud_timing_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Compile results
    hourly_analysis = '\n'.join([f"  Hour {hour:2.0f}: {count} cases" for hour, count in hourly_fraud.items()])
    weekday_analysis = '\n'.join(
        [f"  {day}: {weekday_fraud[day]:.0f} cases ({weekday_rate[day]:.2f}%)" for day in weekday_order])

    result = f"""TIMING ANALYSIS:
- Peak fraud hour: {hourly_fraud.idxmax()}:00 ({hourly_fraud.max()} cases)
- Fraud time range: {fraud_df['Hour'].min()}:00 - {fraud_df['Hour'].max()}:00
- Most fraud-prone weekday: {weekday_fraud.idxmax()} ({weekday_fraud.max():.0f} cases)
- Least fraud-prone weekday: {weekday_fraud.idxmin()} ({weekday_fraud.min():.0f} cases)

HOURLY FRAUD DISTRIBUTION:
{hourly_analysis}

WEEKDAY FRAUD ANALYSIS:
{weekday_analysis}

DAILY FRAUD DISTRIBUTION (TOP 10):
{chr(10).join([f"  Day {day}: {count} cases" for day, count in daily_fraud.head(10).items()])}"""

    add_result("FRAUD TIMING ANALYSIS", result)
    return fraud_df


def create_comprehensive_summary(df):
    """Create comprehensive fraud summary"""
    fraud_df = df[df['Is_Fraud']].copy()

    # Key statistics
    total_fraud_amount = fraud_df['Amount'].sum()
    avg_fraud_amount = fraud_df['Amount'].mean()
    fraud_rate = len(fraud_df) / len(df) * 100
    unique_fraud_accounts = fraud_df['Account ID'].nunique()
    unique_fraud_members = fraud_df['Member ID'].nunique()

    # Compile comprehensive summary
    result = f"""COMPREHENSIVE FRAUD SUMMARY:
- Total fraud cases: {len(fraud_df)}
- Total fraud amount: ${total_fraud_amount:,.2f}
- Average fraud amount: ${avg_fraud_amount:.2f}
- Median fraud amount: ${fraud_df['Amount'].median():.2f}
- Fraud rate: {fraud_rate:.4f}%
- Unique fraud accounts: {unique_fraud_accounts}
- Unique fraud members: {unique_fraud_members}
- Fraud amount standard deviation: ${fraud_df['Amount'].std():.2f}
- Min fraud amount: ${fraud_df['Amount'].min():.2f}
- Max fraud amount: ${fraud_df['Amount'].max():.2f}
- Date range: {fraud_df['Post Date'].min().strftime('%Y-%m-%d')} to {fraud_df['Post Date'].max().strftime('%Y-%m-%d')}"""

    add_result("COMPREHENSIVE FRAUD SUMMARY", result)
    return fraud_df


def save_unified_results():
    """Save all analysis results to a unified text file"""
    with open(os.path.join(output_dir, 'fraud_analysis_simplified_report.txt'), 'w', encoding='utf-8') as f:
        f.write("FRAUD TRANSACTION PATTERN ANALYSIS - SIMPLIFIED REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n")

        for result in analysis_results:
            f.write(result + "\n")

        f.write("\n" + "=" * 80)
        f.write("\nANALYSIS COMPLETE")
        f.write("\n" + "=" * 80)
        f.write(f"\nTotal sections analyzed: {len([r for r in analysis_results if r.startswith('=')]) // 2}")
        f.write(f"\nCharts generated: 3 PNG files")
        f.write(f"\nAll results saved in: {output_dir}/")


def main():
    """Main execution function"""
    print("FRAUD TRANSACTION PATTERN ANALYSIS - SIMPLIFIED")
    print("=" * 60)

    file_path = '../transaction_data.csv'

    try:
        # Load data
        df = load_and_prepare_data(file_path)

        if df['Is_Fraud'].sum() == 0:
            print("No fraud cases found in the dataset. Please check the Fraud Adjustment Indicator column.")
            return None

        print(f"\nGenerating analysis and saving to '{output_dir}' directory...")

        # Run all analyses
        fraud_df_amount = analyze_fraud_amounts(df)
        fraud_df_demo = analyze_fraud_demographics(df)
        fraud_df_timing = analyze_fraud_timing(df)
        fraud_df_summary = create_comprehensive_summary(df)

        # Save unified results
        save_unified_results()

        print(f"\nAnalysis complete!")
        print(f"All results saved to '{output_dir}' directory:")
        print(f"- 3 PNG chart files")
        print(f"- 1 unified text report: fraud_analysis_simplified_report.txt")
        print(f"\nCheck '{output_dir}/fraud_analysis_simplified_report.txt' for complete analysis results.")

        return df

    except FileNotFoundError:
        print(f"Error: Could not find file at {file_path}")
        return None
    except Exception as e:
        print(f"Error: {str(e)}")
        return None


if __name__ == "__main__":
    result_df = main()