# Import the necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from io import StringIO

# Open the raw data file
file_path = './pHdata.dat' 
with open(file_path, 'r') as file: 
    content = file.read() 
    print(type(content))

    
    data = StringIO(content)
    df = pd.read_csv(data, sep="   ", header=None)
   
    # T = time-steps & U1 = input u1 & U2 = input u2 & Y = yield
    T = df.iloc[:, 0].values
    U1 = df.iloc[:, 1].values
    U2 = df.iloc[:, 2].values
    Y = df.iloc[:, 3].values
    
    # Normalize the data:
    Y = np.array((Y - min(Y)) / (max(Y) - min(Y)))
    U1 = np.array((U1 - min(U1)) / (max(U1) - min(U1)))
    U2 = np.array((U2 - min(U2)) / (max(U2) - min(U2)))

    # Plot Raw data:
    plt.scatter(U1, Y, color='blue', label='Raw data') 
    plt.title("Raw data")
    plt.xlabel("In(U1)")
    plt.ylabel("OUT") 
    plt.grid() 
    plt.legend() 
    plt.show()

    plt.scatter(U2, Y, color='blue', label='Raw data') 
    plt.title("Raw data")
    plt.xlabel("In(U2)")
    plt.ylabel("OUT") 
    plt.grid() 
    plt.legend() 
    plt.show()

    ax = plt.axes(projection ="3d")
    ax.scatter3D(U1, U2, Y, color='blue')
    plt.title("Raw data")
    ax.set_xlabel('In(U1)') 
    ax.set_ylabel('In(U2)') 
    ax.set_zlabel('OUT(Y)')
    plt.grid() 
    plt.legend() 
    plt.show()

    # make X matrix contain U array
    X = np.column_stack((np.ones_like(U1), U1**4, U2**4))

    # Calculate the least square With library:
    c, m, n = np.linalg.lstsq(X, Y, rcond=None)[0]
    Y_fIT =  m * U1 + n * U2 + c

    # Plot Raw data & Least square answer:
    ax = plt.axes(projection ="3d")
    ax.scatter3D(U1, U2, Y_fIT, color='blue', label='Fitted line')
    ax.scatter3D(U1, U2, Y, color='red', label='Original data')
    plt.title("Result")
    ax.set_xlabel('In(U1)') 
    ax.set_ylabel('In(U2)') 
    ax.set_zlabel('OUT(Y)')
    plt.grid() 
    plt.legend() 
    plt.show()

    # Erorr calculation
    e = Y - Y_fIT

    # Plot Erorr:
    plt.title("Erorr")
    plt.plot(e, label='erorr')
    plt.grid() 
    plt.show()

    # Parameters
    _lambda = 0.9  # recommended: 0.7 < lambda < 0.9

    # Forgetting Factor
    theta = np.zeros((X.shape[1], 1))
    P = np.eye(X.shape[1]) / _lambda

    # Loop through all available data points (up to the length of y)
    for i in range(len(Y)):
        x_i = X[i, :].reshape(-1, 1)
        y_predicted = np.dot(x_i.T, theta)
        e = Y[i] - y_predicted
        K = np.dot(P, x_i) / (_lambda + np.dot(x_i.T, np.dot(P, x_i)))
        theta = theta + K * e
        P = (P - np.dot(K, np.dot(x_i.T, P))) / _lambda

    # Error
    # Calculate prediction values
    y_fit = np.dot(X, theta)
    E = Y - y_fit

    # Plotting Error
    plt.figure()
    plt.plot(E, 'k')
    plt.xlabel('index')
    plt.ylabel('Error')
    plt.title('E = Actual Value - Fitted Value ')
    plt.grid(True)
    plt.show()

    # Sliding Window
    Window_size = 50
    Step_size = 1

    num_points = len(Y)
    num_windows = (num_points - Window_size) // Step_size + 1
    intercepts = np.zeros(num_windows)
    slope_U1 = np.zeros(num_windows)
    slope_U2 = np.zeros(num_windows)

    errors = np.zeros((num_windows, Window_size))

    # Sliding window least square
    for i in range(num_windows):
        start_idx = (i * Step_size)
        end_idx = start_idx + Window_size

        U1_window = U1[start_idx:end_idx]
        U2_window = U2[start_idx:end_idx]
        Y_window = Y[start_idx:end_idx]

        X_window = np.column_stack((np.ones_like(U1_window), U1_window, U2_window))
        theta_window = np.dot(np.linalg.pinv(X_window), Y_window)
        intercepts[i] = theta_window[0]
        slope_U1[i] = theta_window[1]
        slope_U2[i] = theta_window[2]

        y_fit_window = np.dot(X_window, theta_window)

        errors[i, :] = Y_window - y_fit_window

    # Plot Raw data & Sliding window LS answer:
    fig = plt.figure()
    ax1 = fig.add_subplot(projection='3d')

    for i in range(num_windows):
        U1_window = U1[i * Step_size:i * Step_size + Window_size]
        U2_window = U2[i * Step_size:i * Step_size + Window_size]
        y_fit = intercepts[i] + slope_U1[i] * U1_window + slope_U2[i] * U2_window
        ax1.plot(U1_window, U2_window, y_fit)

    ax1.scatter(U1, U2, Y, color='red', marker='o')
    ax1.set_xlabel('Acid')
    ax1.set_ylabel('Base')
    ax1.set_zlabel('pH')
    ax1.set_title('Fitted Lines Sliding Window')
    plt.grid() 
    plt.show()

    # Plotting Error
    plt.plot(errors.T)
    plt.xlabel('index')
    plt.ylabel('Error')
    plt.title('Error')
    plt.grid() 
    plt.show()


    ## RLS method for Sliding window
    Window_Size = 50
    Step_Size = 1

    # Parameters
    intercepts = np.zeros(len(Y) - Window_Size + 1)
    slope_U1 = np.zeros(len(Y) - Window_Size + 1)
    slope_U2 = np.zeros(len(Y) - Window_Size + 1)

    errors = np.zeros((len(Y) - Window_Size + 1, Window_Size))


    for i in range(len(Y) - Window_Size + 1):
        U1_window = U1[i:i + Window_Size]
        U2_window = U2[i:i + Window_Size]
        Y_window = Y[i:i + Window_Size]

        P = np.eye(3)
        theta = np.zeros((3, 1))
        window_errors = np.zeros(Window_Size)

        for j in range(Window_Size):
            x = np.array([[1], [U1_window[j]], [U2_window[j]]])
            e = Y_window[j] - np.dot(x.T, theta)
            K = (np.dot(P, x)) / (1 + np.dot(x.T, np.dot(P, x)))
            theta = theta + K * e
            P = P - np.dot(K, np.dot(x.T, P))
            
            window_errors[j] = e

        errors[i, :] = window_errors
    
    # Plot Raw data & Sliding window RLS answer:
    fig = plt.figure()
    ax1 = fig.add_subplot(projection='3d')

    for i in range(len(Y) - Window_Size + 1):
        U1_window = U1[i:i + Window_Size]
        U2_window = U2[i:i + Window_Size]
        y_fit = intercepts[i] + slope_U1[i] * U1_window + slope_U2[i] * U2_window
        ax1.plot(U1_window, U2_window, y_fit)

    ax1.scatter(U1, U2, Y, color='magenta', marker='o')
    ax1.set_xlabel('Acid')
    ax1.set_ylabel('Base')
    ax1.set_zlabel('pH ')
    ax1.set_title('Fitted Lines Sliding Window RLS')
    plt.grid() 
    plt.show()

    # Plotting Error
    plt.plot(errors.T)
    plt.xlabel('index')
    plt.ylabel('Error')
    plt.title('Errors plot')
    plt.grid() 
    plt.show()

