from sklearn import decomposition
import pandas as pd

# Read data from file.
data = pd.read_csv('data.csv', header=None)

# Reduce word embeddings to one dimension.
embeddings = data.ix[:, 1:]
pca = decomposition.PCA(n_components=1)
pca.fit(embeddings)
values = pca.transform(embeddings)

# Build result columns.
labels = pd.Series(data.ix[:, 0], name='labels')
values = pd.Series(values[:, 0], name='values')

# Build results and sort.
result = pd.concat([labels, values], axis=1)
result = result.sort(['values'], ascending=[True])

# Output to console.
print(result)
