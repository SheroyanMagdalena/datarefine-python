NUM_VS_CAT = """
print("Numerical vs categorical columns:")

numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
categorical_cols = df.select_dtypes(exclude=["number"]).columns.tolist()

print("\\nNumeric columns:")
print(numeric_cols)

print("\\nCategorical columns:")
print(categorical_cols)
"""
