
import pandas as pd
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors

plt.ion()

# MATCH (n:Person) RETURN n.deepWalk AS `emb`, n.name AS `label`,
# MATCH (n:Movie) RETURN n.deepWalk AS `emb`, n.title AS `label`

files = ['movies.csv', 'persons.csv']

for fname in files: 

	data = pd.read_csv(fname)
	X = data['emb'].apply(lambda x: pd.Series(eval(x)))
	y = data['label']

	pca = PCA(n_components=2)
	reduced = pca.fit_transform(X)

	plt.scatter(reduced[:,0], reduced[:,1])

	for i, txt in enumerate(y):
	    plt.annotate(txt, reduced[i], fontsize=5)

	nbrs = NearestNeighbors(n_neighbors=5)

	nbrs.fit(X)

	distances, indices = nbrs.kneighbors(X)

	print("\n\n%s\n\n" % fname.upper())
	for idx, (dists, nbrs) in enumerate(zip(distances, indices)):
		closest = [(y[nbr], round(dists[i+1],1)) for i, nbr in enumerate(nbrs[1:])]
		print("%s: %s" % (y[idx], closest))


plt.legend(['Movies', 'Actors'])

