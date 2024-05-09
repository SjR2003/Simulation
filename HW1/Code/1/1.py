# Import the necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Open the raw data file
with open('1/batch-yield-and-purity.csv', newline='') as csvfile:
    data = pd.read_csv(csvfile)
    print("Raw data :")
    print(data)

    # U = purity & Y = yield

    # Normalize the data:
    Y = np.array((data["yield"] - min(data["yield"])) / (max(data["yield"]) - min(data["yield"])))
    U = np.array((data["purity"] - min(data["purity"])) / (max(data["purity"]) - min(data["purity"])))

    # Plot Raw data:
    plt.scatter(Y, U, color='blue', label='Raw data') 
    plt.title("Raw data")
    plt.xlabel("purity")
    plt.ylabel("yield") 
    plt.grid() 
    plt.legend() 
    plt.show()

    # Calculate the least square With equations:
    X = np.vstack([U, np.ones(len(U))]).T

    U_T = X.T

    U_RESULT = np.dot(U_T, X)

    U_RESULT_INV =np.linalg.inv(U_RESULT)

    theta = np.dot(np.dot(U_RESULT_INV, U_T), Y)

    y = np.dot(theta[0], U) + theta[1]
    
    # Plot Raw data & Least square answer:
    plt.scatter(U, Y, color='blue', label='Original data') 
    plt.plot(U, y, color='red', label='Fitted line')
    plt.title("With equations")
    plt.xlabel("purity")
    plt.ylabel("yield")  
    plt.grid() 
    plt.legend() 
    plt.show()

    # Calculate the least square With library:
    m, c = np.linalg.lstsq(X, Y, rcond=None)[0]

    # Plot Raw data & Least square answer:
    plt.title("With the library")
    plt.plot(U, Y, "o", label='Original data', markersize=10)
    plt.plot(U, m*U + c, label='Fitted line')
    plt.legend()
    plt.show()

    # Comparison of equations method and library
    plt.plot(U, y, color='purple',linewidth='4', label='equations')
    plt.plot(U, m*U + c, color='yellow', label='library')
    plt.legend()
    plt.show()

    
   