import pandas as pd
import matplotlib.pyplot as plt

#Method for visualising relationships between a data column and survival rate using a scatter plot
def scatterColumnAgainstSurvived(dataframe, columnName):
    x = dataframe[columnName].to_numpy()
    y = dataframe["Survived"].to_numpy()
    plt.scatter(x, y)
    plt.title(f"{columnName} against Survived")
    plt.xlabel(columnName)
    plt.ylabel("Survived")
    plt.show()

#Method for visualising relationships between a data column and survival rate using a bar plot
def barChartAgainstSurvived(xlabel, survivalRate, labels=None):
    plt.bar(survivalRate.index, survivalRate.values)
    plt.xlabel(xlabel)
    plt.ylabel("Survival rate (%)")
    plt.title(f"Survival Rate by {xlabel}")
    if labels:
        plt.xticks(ticks=survivalRate.index, labels=labels)
    plt.show()

#Importing data from CSV file to a dataframe
df = pd.read_csv("titanic_800.csv")

#Exploring the data
print(f"Number of columns in total: {len(df.columns)}")
print(df.describe().to_string())
print(df.count())

#Dropping column Cabin
df.drop("Cabin", axis=1, inplace=True)

#Removing row with missing Embarked value
df.dropna(subset=["Embarked"], inplace=True)

#Insert the average age in age column where values are missing
averageAge = df['Age'].mean()
df["Age"].fillna(averageAge, inplace = True)

#Replacing male with 0 and female with 1
df['Sex'] = df['Sex'].replace(['female'], 1.0)
df['Sex'] = df['Sex'].replace(['male'], 0.0)

#Dropping name column
df.drop('Name' , axis = 1 , inplace=True)

#Checking for relationship between PassengerId value and survival rate
scatterColumnAgainstSurvived(df, "PassengerId")

#Dropping column PassengerId
df.drop( 'PassengerId', axis=1, inplace=True)

#Checking for relationship between SibSp value and survival rate
sibSpAmounts = sorted(df["SibSp"].unique())
survivalRateBySibSp = df.groupby("SibSp")["Survived"].mean() * 100
barChartAgainstSurvived("A passengers amount of siblings and spouse aboard", survivalRateBySibSp, labels=sibSpAmounts)

#Checking for relationship between Parch value and survival rate
parchAmounts = sorted(df["Parch"].unique())
survivalRateByParch = df.groupby("Parch")["Survived"].mean() * 100
barChartAgainstSurvived("A passengers amount of parents and children aboard", survivalRateByParch, labels=parchAmounts)

#Checking for relationship between Ticket value and survival rate
scatterColumnAgainstSurvived(df, "Ticket")

#Dropping column Ticket
df.drop("Ticket", axis=1, inplace=True)

#Checking for relationship between Fare value and survival rate
scatterColumnAgainstSurvived(df, "Fare")

#Checking for relationship between Embarked value and survival rate
embarkedAmounts = sorted(df["Embarked"].unique())
survivalRateByEmbarked = df.groupby("Embarked")["Survived"].mean() * 100
barChartAgainstSurvived("A passengers embark", survivalRateByEmbarked, labels=embarkedAmounts)

#Checking for relationship between Age value and survival rate
scatterColumnAgainstSurvived(df, "Age")

#Checking for relationship between Sex value and survival rate
survivalRateBySex = df.groupby("Sex")["Survived"].mean() * 100
barChartAgainstSurvived("Sex", survivalRateBySex, labels=["Men", "Women"])

#Checking for relationship between Pclass value and survival rate
survivalRateByPClas = df.groupby("Pclass")["Survived"].mean()
barChartAgainstSurvived("Pclass", survivalRateByPClas, labels=["First", "Second", "Third"])

#Turning categorical variables into multiple dummy variables
df = pd.get_dummies(df, columns=["Embarked", "Pclass"], prefix=["Embarked", "Pclass"])

#Printing description of final dataset
print(df.describe(include='all').to_string())
print(df.count())



# Split into training (80%) and test (20%) sets
from sklearn.model_selection import train_test_split
X = df.drop("Survived", axis=1)
y = df["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Scaling the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Train Neural Network model
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(50, 50, 50), max_iter=1000, random_state=42)
mlp.fit(X_train, y_train)

# Make predictions
y_pred_NN = mlp.predict(X_test)

# Evaluate the model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
accuracyNN = accuracy_score(y_test, y_pred_NN)
print(f"Accuracy NN: {accuracyNN:.4f}")
print("Classification Report NN:")
print(classification_report(y_test, y_pred_NN))
print("Confusion Matrix NN:")
print(confusion_matrix(y_test, y_pred_NN))

#Train Random Forest model
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=50)  # using 20 trees
clf.fit(X_train,y_train)

# Make predictions
y_pred_random_forest = clf.predict(X_test)

# Evaluate the model
accuracyRandomForest = accuracy_score(y_test, y_pred_random_forest)
print(f"Accuracy Random Forest: {accuracyRandomForest:.4f}")
print("Classification Report Random Forest:")
print(classification_report(y_test, y_pred_random_forest))
print("Confusion Matrix Random Forest:")
print(confusion_matrix(y_test, y_pred_random_forest))

