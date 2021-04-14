{"metadata":{"kernelspec":{"language":"python","display_name":"Python 3","name":"python3"},"language_info":{"pygments_lexer":"ipython3","nbconvert_exporter":"python","version":"3.6.4","file_extension":".py","codemirror_mode":{"name":"ipython","version":3},"name":"python","mimetype":"text/x-python"}},"nbformat_minor":4,"nbformat":4,"cells":[{"cell_type":"code","source":"# %% [code]\nimport pandas as pd\nimport numpy as np\n\ndef dist(x1,x2):\n    return np.sqrt(sum((x1-x2)**2)) # calculating distance\n\n# main algo \ndef knn(X,Y,queryPoint,k=5):\n    \n    vals = [] # creating list to append all distances\n    m = X.shape[0]\n    \n    for i in range(m):\n        d = dist(queryPoint,X[i])\n        vals.append((d,Y[i])) #appending all distances \n        \n    #sorting the list\n    vals = sorted(vals)\n    # choose first k distances \n    vals = vals[:k]\n    \n    vals = np.array(vals)\n\n    \n    new_vals = np.unique(vals[:,1],return_counts=True)\n    \n    index = new_vals[1].argmax()\n    pred = new_vals[0][index]\n    \n    return pred\n\n\n## For testing Purposes\n'''\n## Importing libraries\n\nimport sklearn.datasets\nimport matplotlib.pyplot as plt\n\n## creating dataset\n\nx,y = sklearn.datasets.make_classification(n_samples=1000, n_classes=2,\nn_clusters_per_class=1, n_features=2,n_informative=2, n_redundant=0, n_repeated=0)\n\n\n## Visualization\n\nquery_p = np.array([0.5,0.5])   \nplt.scatter(query_p[0],query_p[1],c = 'r') ## plot the query point\nplt.scatter(x[:,0],x[:,1],c = y)\nplt.show()\n\n\n## testing the algorithm\n\nresult = knn(x,y,query_p)    ### query point ==> x = 0.5,y = 0.5  \nprint(result)\n'''","metadata":{"_uuid":"caa1d711-a2e9-46ea-914a-707ce64372e9","_cell_guid":"f37ec52a-2a75-4352-821f-7f23f08b0501","collapsed":false,"jupyter":{"outputs_hidden":false},"trusted":true},"execution_count":null,"outputs":[]}]}