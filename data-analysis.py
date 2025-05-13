# 📌 Step 1: Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configure styles
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# 📌 Step 2: Load Dataset
df = pd.read_csv("global_unemployment_data.csv")
print("Dataset shape:", df.shape)
print(df.head())

# 📌 Step 3: Basic Data Info
print("\nMissing Values:\n", df.isnull().sum())
print("\nData Types:\n", df.dtypes)

# 📌 Step 4: Reshape data from wide to long format
# First, clean column names
df.columns = df.columns.str.strip().str.lower()

# Create a copy of the original data for reference
df_original = df.copy()

# Identify categorical columns (non-year columns)
id_cols = ['country_name', 'indicator_name', 'sex', 'age_group', 'age_categories']
year_cols = [str(year) for year in range(2014, 2025)]

# Reshape data from wide format (years as columns) to long format (year as a column)
df_long = pd.melt(
    df, 
    id_vars=id_cols,
    value_vars=year_cols,
    var_name='year',
    value_name='unemployment_rate'
)

# Convert year to integer for proper sorting
df_long['year'] = df_long['year'].astype(int)

# 📌 Step 5: Analysis Questions & Code

# 1️⃣ Which countries had the highest unemployment in 2024?
top_10_2024 = df_long[df_long["year"] == 2024].sort_values(by="unemployment_rate", ascending=False).head(10)
print("\nTop 10 Countries in 2024:\n", top_10_2024[["country_name", "sex", "age_group", "unemployment_rate"]])

# Plot
plt.figure(figsize=(12, 8))
sns.barplot(data=top_10_2024, x="unemployment_rate", y="country_name", hue="sex", palette="Reds_r")
plt.title("Top 10 Highest Unemployment Rates (2024)")
plt.xlabel("Unemployment Rate (%)")
plt.ylabel("Country")
plt.tight_layout()
plt.savefig("top_10_highest_unemployment_2024.png")  # Save the plot as PNG
plt.show()

# 2️⃣ Trend: Global Average Unemployment (2014–2024)
global_trend = df_long.groupby("year")["unemployment_rate"].mean().reset_index()

plt.figure(figsize=(12, 6))
sns.lineplot(data=global_trend, x="year", y="unemployment_rate", marker="o")
plt.title("Global Average Unemployment Rate (2014–2024)")
plt.ylabel("Unemployment Rate (%)")
plt.xlabel("Year")
plt.xticks(global_trend["year"])
plt.grid(True)
plt.savefig("global_average_unemployment_trend.png")  # Save the plot as PNG
plt.show()

# 3️⃣ Gender-Based Unemployment in 2024
gender_2024 = df_long[df_long["year"] == 2024].groupby("sex")["unemployment_rate"].mean().reset_index()

plt.figure(figsize=(10, 6))
sns.barplot(data=gender_2024, x="sex", y="unemployment_rate", hue="sex", palette="Blues", legend=False)
plt.title("Average Unemployment by Gender (2024)")
plt.ylabel("Unemployment Rate (%)")
plt.savefig("gender_unemployment_2024.png")  # Save the plot as PNG
plt.show()

# 4️⃣ Youth vs Adult (Age Group) in 2024
age_2024 = df_long[df_long["year"] == 2024].groupby("age_group")["unemployment_rate"].mean().reset_index()

plt.figure(figsize=(10, 6))
sns.barplot(data=age_2024, x="age_group", y="unemployment_rate", hue="age_group", palette="Purples", legend=False)
plt.title("Unemployment by Age Group (2024)")
plt.ylabel("Unemployment Rate (%)")
plt.savefig("age_group_unemployment_2024.png")  # Save the plot as PNG
plt.show()

# 5️⃣ Countries with largest Increase/Decrease in Unemployment (2014–2024)
# First, let's calculate average unemployment rate per country per year
country_year_avg = df_long.groupby(['country_name', 'year'])['unemployment_rate'].mean().reset_index()

# Now calculate changes between 2014 and 2024
country_changes = []

for country in country_year_avg['country_name'].unique():
    country_data = country_year_avg[country_year_avg['country_name'] == country]
    
    # Check if we have data for both 2014 and 2024
    if 2014 in country_data['year'].values and 2024 in country_data['year'].values:
        rate_2014 = country_data[country_data['year'] == 2014]['unemployment_rate'].values[0]
        rate_2024 = country_data[country_data['year'] == 2024]['unemployment_rate'].values[0]
        change = rate_2024 - rate_2014
        
        country_changes.append({
            'country_name': country,
            'rate_2014': rate_2014,
            'rate_2024': rate_2024,
            'change': change
        })

# Convert to DataFrame
changes_df = pd.DataFrame(country_changes)

# Top 5 increase
top_increase = changes_df.sort_values('change', ascending=False).head(5)
print("\nTop 5 Countries with Increased Unemployment (2014–2024):\n", top_increase)

# Top 5 decrease
top_decrease = changes_df.sort_values('change').head(5)
print("\nTop 5 Countries with Decreased Unemployment (2014–2024):\n", top_decrease)

# Plot both
top_changes = pd.concat([top_increase, top_decrease])

plt.figure(figsize=(12, 8))
sns.barplot(data=top_changes, x="change", y="country_name", palette="coolwarm")
plt.title("Countries with Largest Unemployment Change (2014–2024)")
plt.xlabel("Change in Unemployment Rate (%)")
plt.axvline(0, color="black", linestyle="--")
plt.tight_layout()
plt.savefig("countries_largest_unemployment_change.png")  # Save the plot as PNG
plt.show()

# 6️⃣ Countries with consistently low unemployment (average < 4% over 10 years)
avg_unemp = df_long.groupby("country_name")["unemployment_rate"].mean().sort_values()
low_unemp = avg_unemp[avg_unemp < 4].head(10)
print("\nCountries with Consistently Low Unemployment:\n", low_unemp)

# 7️⃣ Female unemployment higher than male (avg over all years)
# First, calculate average unemployment by country and sex
gender_avg = df_long.groupby(['country_name', 'sex'])['unemployment_rate'].mean().reset_index()

# Convert to wide format to compare Female vs Male
gender_wide = gender_avg.pivot(index='country_name', columns='sex', values='unemployment_rate')

# Calculate gender gap (only for countries with both Female and Male data)
if 'Female' in gender_wide.columns and 'Male' in gender_wide.columns:
    gender_wide['gap'] = gender_wide['Female'] - gender_wide['Male']
    female_dominant = gender_wide[gender_wide['gap'] > 2].sort_values('gap', ascending=False)
    print("\nCountries with Significantly Higher Female Unemployment:\n", female_dominant)
else:
    print("\nCannot calculate gender gap - missing data for one or both genders")

# 8️⃣ Gender-Unemployment Correlation Over Time
gender_yearly = df_long.groupby(["year", "sex"])["unemployment_rate"].mean().reset_index()
plt.figure(figsize=(12, 6))
sns.lineplot(data=gender_yearly, x="year", y="unemployment_rate", hue="sex", marker="o")
plt.title("Unemployment Rate by Gender Over Years")
plt.ylabel("Unemployment Rate (%)")
plt.savefig("gender_unemployment_correlation.png")  # Save the plot as PNG
plt.show()

# 9️⃣ Youth vs Adult Gap (for 2024)
# Group by country and age category for 2024
age_2024_by_country = df_long[(df_long["year"] == 2024) & 
                             (df_long["age_group"].isin(["15-24", "25+"]))].copy()

# Calculate average by country and age group
age_avg = age_2024_by_country.groupby(['country_name', 'age_group'])['unemployment_rate'].mean().reset_index()

# Reshape to wide format to calculate gap
age_wide = age_avg.pivot(index='country_name', columns='age_group', values='unemployment_rate')

# Calculate youth-adult gap (if both categories exist)
if '15-24' in age_wide.columns and '25+' in age_wide.columns:
    age_wide['gap'] = age_wide['15-24'] - age_wide['25+']
    significant_gap = age_wide[age_wide['gap'] > 5].sort_values('gap', ascending=False)
    print("\nCountries where Youth Unemployment >> Adult (2024):\n", significant_gap.head(10))

# 🔟 Countries with highest volatility in unemployment
volatility = df_long.groupby('country_name')['unemployment_rate'].std().sort_values(ascending=False)
high_volatility = volatility.head(10)
print("\nCountries with Highest Volatility in Unemployment:\n", high_volatility)
