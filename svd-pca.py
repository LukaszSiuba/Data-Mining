from sklearn.datasets import load_digits
import numpy as np; import matplotlib.pyplot as plt; import pandas as pd
from sklearn.decomposition import PCA

def digits():
    #Load and standardize the data
    digits = load_digits()
    data = digits.data
    mean = np.mean(data, axis=0)
    std =  np.std(data, axis=0)
    #Check if in std array we have 0, if so replace value to 1 to prevent division by zero so Nan  avoided
    for i in range(len(std)):
        if std[i] == 0:
            std[i] = 1
    standardized = (data-mean)/std

    #Compute the SVD of the data input: (M,N) ->  K = min(M,N) ret (M,K) and (K,N) for U and Vt respectively. U Left singular Vectors, S singular values, Vt right singular vectors
    U, S, Vt = np.linalg.svd(standardized, full_matrices=False)
    #Check if singular values sorted in decreasing order
    for i in range(len(S) - 1):
        if S[i] <  S[i+1]:
            raise ValueError("Singualr values are not sorted in descending error.")
        
    """
    Retained Energy, defined by lecture 3 is;

    The summation of singular values up to k / summation of singular values to r or total sum of singular values
    The summation up to k can be simply the cummulative sum of singular values

    So retained energy = Cumulative summation of singular values up to k / total sum of singular values
    """
    retained = np.cumsum(S) / np.sum(S) #Returns retained energy array
    min_k_90 = 0
    min_k_95 = 0
    for i in range(len(retained)):
        if retained[i] >= .90 and min_k_90 == 0:
            min_k_90 = i
        if retained[i] >= .95 and min_k_95 == 0:
            min_k_95 = i
    print("\nMin singular indices to retain 90% energy:", min_k_90, "and retained 95% energy:", min_k_95)
    print("K values for 90% and 95% Retained Energy respectively is:", min_k_90+1,",", min_k_95+1, '\n') 

    #Plotting for 2c
    plt.figure("Retained Energy")
    #{1-64 for k}
    plt.plot(range(1,(len(S)+1)), retained, c = 'k')
    plt.xlabel("Number of Singular Values S")
    plt.xticks(np.arange(1, 65, step=1)) 
    plt.axhline(y=.90, label="90% Retained Energy", ls='-', c='b')
    plt.axhline(y=.95, label="95% Retained Energy", ls='-', c='m')
    plt.axvline(x=min_k_90 + 1, ls=':', c='r')
    plt.axvline(x=min_k_95 + 1, ls=':', c='r')
    plt.ylabel("Retained Energy")
    plt.legend()
    plt.title("Percentage of Retained Energy as a Function of the Number of Singular Values")

    print("Energy retained with 2 dimensions:", retained[1])

    # Reduced-dimension data matrix D^(2) with two dimensions by projection(dot) on Vt
    first_two_Vt = Vt[:2] #Shape is (2,64), standardized shape is (1797,64)
    first_two_V = np.transpose(first_two_Vt) #Shape now is (64,2) Dimensions now align for dot
    D_2 = np.dot(standardized, first_two_V) #Shape is (1797,2) expected

    #Scatter plot for 2d.
    plt.figure("Reduced Dimension Representation")
    scatter = plt.scatter(D_2[:, 0], D_2[:, 1], c=digits.target, cmap='viridis')
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.title("Digits Projected onto First Two Right Singular Vectors")
    plt.colorbar(scatter)


    #SVD Reconstruction of the data compute dot product between 
    # SVD reconstruction using k needed for 90% retained energy
    k = min_k_90 + 1 #46 singular values
    U_k = U[:, :k]
    S_k = S[:k]
    Vt_k = Vt[:k, :]
    Sigma_k = np.diag(S_k)
    D_rec = U_k @ Sigma_k @ Vt_k

    # Residual matrix
    R = standardized - D_rec
    # Norms of rows in R
    l2_norms = np.linalg.norm(R, axis=1)
    #Sort the indexes resulting in the smallest to largest values
    sorted_indices = np.argsort(l2_norms)
    largest_norm_instances = sorted_indices[-5:]
    print("5 instances of largest norm: ", largest_norm_instances)

    #Plotting for e
    plt.figure("SVD Outliers")
    #Loop over our largest_norm_instances list, for each plot the images each instance represents and the image they correspond to.
    for i, id in enumerate(largest_norm_instances[:5]):
        plt.style.use('grayscale')
        plt.subplot(1, 5, i+1)
        plt.imshow(digits.images[id])
        plt.title(f"ID: {id}")
        plt.axis('off')


#Question3 
def personality():
    #A) Load the data, fill in missing values with the median of that column, standardize the data
    data = pd.read_csv('bfi.csv')
    median_of_col = data.median()
    filled = data.fillna(median_of_col)
    mean = np.mean(filled, axis=0)
    std = np.std(filled, axis=0)
    #Avoid division by 0. Need to use iloc here due to pandas. Same functionality as digits to change values from 0 to 1 at whatever i.
    for i in range(len(std)):
        if std.iloc[i] == 0:
            std.iloc[i] = 1
    standardized = (filled-mean)/std

    #B) Compute the covariance matrix of the data as a matrix product of the centrered data So Sigma = 1/n (Z^t @ Z). Z is already centered as standardized
    n = standardized.shape[0] 
    Z = standardized
    Z_t = np.transpose(standardized)
    sigma = (1/n)*(Z_t @ Z)

    #C) Compute the eigenvectors and eigenvalues of sigma, make sure to sort eigenvalues from largest to smallest
    eigenvalues, eigenvectors = np.linalg.eig(sigma)
    #corresponding index of descending sorted order of eigen values. Then sort eigenvalues with corresponding eigenvectors descending order
    index= np.argsort(eigenvalues)[::-1]
    sort_eigvalues = eigenvalues[index]
    sort_eigvectors = eigenvectors[:,index]

    #PCA, returns the amount of variance (eigenvalues) assigned to each component
    pca = PCA()
    pca.fit(standardized) #Fit the data for pca to use our standardized data to match the above.

    #Since built-in PCA uses 1 / (n-1) for the covariance, we need to cancel it out to match our covariance matrix
    pca_eigvalues = ((n - 1) / n) * pca.explained_variance_
    print("Do Eigen Values match?",np.allclose(sort_eigvalues, pca_eigvalues))

    #3d percentage of retained variance, same structure as retained energy
    retained_variance = np.cumsum(sort_eigvalues) / np.sum(sort_eigvalues)
    min_k_90 = 0
    min_k_95 = 0
    for i in range(len(retained_variance)):
        if retained_variance[i] >= .90 and min_k_90 == 0:
            min_k_90 = i
        if retained_variance[i] >= .95 and min_k_95 == 0:
            min_k_95 = i

    print("\nMin component indices to retain 90% variance:", min_k_90, "and retained 95% variance:", min_k_95)
    print("Number of components for 90% and 95% Retained variance respectively is:", min_k_90+1,",", min_k_95+1, '\n') 

    #Plotting for 3d
    plt.figure("Retained Variance")
    #{1-64 for components}
    plt.plot(range(1,len(sort_eigvalues)+1), retained_variance, c ='k')
    plt.xlabel("Number of Components")
    plt.xticks(np.arange(1, 29, step=1)) 
    plt.axhline(y=.90, label="90% Retained Variance", ls='-', c='b')
    plt.axhline(y=.95, label="95% Retained Variance", ls='-', c='m')
    plt.axvline(x=min_k_90 + 1, ls=':', c='r')
    plt.axvline(x=min_k_95 + 1, ls=':', c='r')
    plt.ylabel("Retained Variance")
    plt.legend()
    plt.title("Retained Variance as a Function of the Number of Components")

    #3e plotting, the first 3 factors as lineplots
    plt.figure("Eigen vector factors")
    x_axis = data.columns
    for i, factor in enumerate(range(3)):
        plt.subplot(1,3, i+1)
        #Since our eigen vectors are sorted, just pull all rows and the corresponding factor
        plt.plot(range(len(x_axis)), sort_eigvectors[:, factor], c ='k')
        plt.xticks(range(len(x_axis)), x_axis, rotation=90, fontsize=8)
        plt.title(f"Factor: {factor+1}")
        plt.xlabel("Variables")
        plt.ylabel("Eigenvector")
    plt.tight_layout()    

    


#Loader, do not alter plt.show() so all plots produce.
if __name__ == '__main__':
    digits()
    personality()
    plt.show()