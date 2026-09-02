import pickle

with open("Model/model.pkl", "rb") as f:
    model = pickle.load(f)

print("MODEL TYPE")
print(type(model))

print("\nPIPELINE STEPS")
for name, step in model.steps:
    print(name, "->", type(step))

print("\nFEATURES")
print(model.feature_names_in_)

print("\nCLASSES")
print(model.classes_)

print("\nPREPROCESSOR")
preprocessor = model.named_steps["preprocessor"]

print("Transformers:")
for name, transformer, columns in preprocessor.transformers_:
    print("\nName:", name)
    print("Transformer:", type(transformer))
    print("Columns:", columns)

print("\nCLASSIFIER")
classifier = model.named_steps["classifier"]

print("Type:", type(classifier))
print("Number of trees:", classifier.n_estimators)
print("Max depth:", classifier.max_depth)
print("Random state:", classifier.random_state)