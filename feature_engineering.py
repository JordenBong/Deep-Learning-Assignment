import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load the datasets
bid_set = pd.read_csv("bids.csv")
test_set = pd.read_csv("test.csv")
train_set = pd.read_csv("train.csv")

# Check the info
print("Bid Set Info:")
bid_set.info()
print("\nTrain Set Info:")
train_set.info()

# View the first few rows of the datasets
print(bid_set.head(), "\n")  
print(train_set.head(), "\n")

# Check the shape of the datasets
print("\nbid shape:")
print(bid_set.shape)  # Show the shape of the DataFrame
print("\ntrain shape:")
print(train_set.shape)  # Show the shape of the DataFrame

# # Statistics of the datasets
# print("\nBid Set Statistics:\n", bid_set.describe())
# print("\nTrain Set Statistics:\n", train_set.describe())

# Check for missing values
print("\nMissing Values in Bid Set:\n", bid_set.isnull().sum())
print("\nMissing Values in Train Set:\n", train_set.isnull().sum())

# categorical feature Analysis
print("\nUnique values in categorical columns (Bid Set):\n", bid_set.nunique())
print("\nUnique values in categorical columns (Train Set):\n", train_set.nunique())

missing_percent = bid_set['country'].isnull().mean()
print(f"\nPercentage of missing data in country column: {missing_percent*100: .2f}%")

# unique_countries = bid_set['country'].value_counts()
# filling missing values with the mode
bid_set['country'] = bid_set['country'].fillna(bid_set['country'].mode()[0])

merged_df = pd.merge(train_set, bid_set, on='bidder_id', how='left')
missing_values_count = merged_df.isnull().sum()
print("\nMissing Value count after merged: \n" , missing_values_count )

# Check the percentage of missing values in each column
missing_values_ratio = merged_df.isnull().sum() / len(merged_df) * 100
print("\nMissing Values Ratio:\n", missing_values_ratio)


# for all the missing value observation, drop it
merged_df = merged_df.dropna()
missing_values_count_dropped = merged_df.isnull().sum()
print("\nMissing Values Count after dropping rows with missing values:\n", missing_values_count_dropped )


# EDA
# Distribution of auction outcomes
outcome_counts = merged_df['outcome'].value_counts()

# Visualizing the distribution of auction outcomes
plt.figure(figsize=(8, 6))
outcome_counts.plot(kind='bar', color=['blue', 'orange'])
plt.title('Distribution of Outcomes')
plt.xlabel('Outcome')
plt.ylabel('Number of Bidders')
plt.show()

# Visualizing the top 10 merchandise types with the highest number of bids per label (outcome)
plt.figure(figsize=(14, 8))
top_merchandise_by_label = merged_df.groupby(['outcome', 'merchandise']).size().nlargest(30).reset_index(name='count')

sns.barplot(x='merchandise', y='count', hue='outcome', data=top_merchandise_by_label)
plt.title('Top 30 Merchandise Types with the Highest Number of Bids by Label')
plt.xlabel('Merchandise Type')
plt.ylabel('Number of Bids')
plt.legend(title='Label (Outcome)')
plt.show()

# Visualizing the top 10 countries with the highest number of bids
plt.figure(figsize=(14, 8))
top_countries = merged_df['country'].value_counts().nlargest(10)
sns.barplot(x=top_countries.index, y=top_countries.values)
plt.title('Top 10 Countries with the Highest Number of Bids')
plt.xlabel('Country')
plt.ylabel('Number of Bids')
plt.show()

# Visualizing the top 40 countries with the highest number of bids by label (outcome)
plt.figure(figsize=(14, 8))
top_countries_by_label = merged_df.groupby(['outcome', 'country']).size().nlargest(40).reset_index(name='count')

sns.barplot(x='country', y='count', hue='outcome', data=top_countries_by_label)
plt.title('Top 40 Countries with the Highest Number of Bids by Label')
plt.xlabel('Country')
plt.ylabel('Number of Bids')
plt.legend(title='Label (Outcome)')
plt.show()


# df = pd.DataFrame(data = merged_df)
# df.to_csv("merged_df.csv", index=False)

# Feature Engineering: Grouping the data by bidder_id
# Get the number of bids per bidder
bids_per_bidder = merged_df.groupby('bidder_id').size().reset_index(name='num_bids')

# Get the number of unique countries per bidder
unique_countries_per_bidder = merged_df.groupby('bidder_id')['country'].nunique().reset_index(name='num_countries')

# Get the number of unique merchandise types per bidder
unique_merchandise_per_bidder = merged_df.groupby('bidder_id')['merchandise'].nunique().reset_index(name='num_merchandise')

# Get the number of unique URLs per bidder
unique_urls_per_bidder = merged_df.groupby('bidder_id')['url'].nunique().reset_index(name='num_urls')

# Get the number of unique IPs per bidder
unique_ips_per_bidder = merged_df.groupby('bidder_id')['ip'].nunique().reset_index(name='num_ips')

# Get the number of unique devices per bidder
unique_devices_per_bidder = merged_df.groupby('bidder_id')['device'].nunique().reset_index(name='num_devices')

# Get the number of auctions joined per bidder
auctions_joined_per_bidder = merged_df.groupby('bidder_id')['auction'].nunique().reset_index(name='num_auctions')

# Get the total duration between the first and last bid for each bidder
total_duration = merged_df.groupby('bidder_id')['time'].agg(lambda x: x.max() - x.min()).reset_index(name='total_duration')
total_duration['total_duration'] = total_duration['total_duration'] / 1e9  # Convert to smaller values

# Get the majority outcome per bidder
majority_outcome_per_bidder = merged_df.groupby('bidder_id')['outcome'].agg(lambda x: x.mode()[0]).reset_index(name='outcome')

# Merge all the features into a single DataFrame
featured_df = bids_per_bidder
featured_df = pd.merge(featured_df, unique_countries_per_bidder, on='bidder_id', how='left')
featured_df = pd.merge(featured_df, unique_merchandise_per_bidder, on='bidder_id', how='left')
featured_df = pd.merge(featured_df, unique_urls_per_bidder, on='bidder_id', how='left')
featured_df = pd.merge(featured_df, unique_ips_per_bidder, on='bidder_id', how='left')
featured_df = pd.merge(featured_df, unique_devices_per_bidder, on='bidder_id', how='left')
featured_df = pd.merge(featured_df, auctions_joined_per_bidder, on='bidder_id', how='left')
featured_df = pd.merge(featured_df, total_duration, on='bidder_id', how='left')
featured_df = pd.merge(featured_df, majority_outcome_per_bidder, on='bidder_id', how='left')

# Visualizing the table by printing the data for 10 bidders 
print(featured_df.head(10))

# Normalizing the features
scaler = StandardScaler()
features_to_scale = ['num_bids', 'num_countries', 'num_merchandise', 'num_urls', 'num_ips', 'num_devices', 'num_auctions', 'total_duration']
featured_df[features_to_scale] = scaler.fit_transform(featured_df[features_to_scale])

# Splitting the dataset into training and testing sets
X = featured_df.drop(['bidder_id','outcome'], axis=1)  # Features
y = featured_df['outcome']  # Target variable (1 - Robot, 0 - Human)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)