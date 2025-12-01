from financial_TSF.pipeline.pipeline import Pipeline


print("Starting pipeline...")
pipeline = Pipeline()
print(pipeline.steps)
pipeline.run(data=None)

