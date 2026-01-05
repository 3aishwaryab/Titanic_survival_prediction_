import pandas as pd
from src.feature_engineering import build_preprocessor, TitleExtractor, FamilyFeatures, BinningTransformer


def test_title_family_bins_transformers():
    sample = pd.DataFrame([
        {"Name": "Smith, Mr. John", "SibSp": 1, "Parch": 0, "Age": 28, "Fare": 7.25, "Sex": "male", "Pclass": 3},
        {"Name": "Brown, Mrs. Anna", "SibSp": 0, "Parch": 0, "Age": 45, "Fare": 71.2833, "Sex": "female", "Pclass": 1},
    ])

    te = TitleExtractor()
    ff = FamilyFeatures()
    bt = BinningTransformer()

    tdf = te.transform(sample)
    fdf = ff.transform(sample)
    bdf = bt.transform(sample)

    assert "Title" in tdf.columns
    assert "FamilySize" in fdf.columns and "IsAlone" in fdf.columns
    assert "AgeBin" in bdf.columns and "FareBin" in bdf.columns

    # Composite feature creator should include interaction features
    fc = build_preprocessor()[0].named_steps["feature_creator"]
    fcd = fc.transform(sample)
    assert "Sex_Pclass" in fcd.columns
