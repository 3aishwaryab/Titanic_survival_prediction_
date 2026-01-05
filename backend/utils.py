import pandas as pd

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Family size
    df['FamilySize'] = df.get('SibSp', 0) + df.get('Parch', 0) + 1

    # Is alone
    df['IsAlone'] = 0
    df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1

    # Title from name
    if 'Name' in df.columns:
        df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
        df['Title'] = df['Title'].replace({
            'Mlle': 'Miss', 'Mme': 'Mrs', 'Ms': 'Miss'
        })
        rare_titles = ['Dr', 'Rev', 'Col', 'Major', 'Countess', 'Lady', 'Jonkheer', 'Don', 'Dona', 'Capt', 'Sir']
        df['Title'] = df['Title'].apply(lambda x: x if x in ['Mr', 'Miss', 'Mrs', 'Master'] else ('Rare' if x in rare_titles else x))
    else:
        df['Title'] = 'Rare'

    # Age group
    if 'Age' in df.columns:
        df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 100],
                           labels=['Child', 'Teenager', 'Young Adult', 'Adult', 'Senior'])
    else:
        df['AgeGroup'] = 'Adult'

    # Fare per person
    if 'Fare' in df.columns:
        df['FarePerPerson'] = df['Fare'] / (df['FamilySize'] + 1e-6)
    else:
        df['FarePerPerson'] = 0.0

    return df


def prepare_payload(payload: dict) -> pd.DataFrame:
    """Convert incoming JSON payload into a DataFrame suitable for the model pipeline.
    Expected payload keys (at minimum): 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Name'
    """
    df = pd.DataFrame([payload])
    df = create_features(df)
    return df
