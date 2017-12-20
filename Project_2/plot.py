import matplotlib.pyplot as plt




def error_visualization(parameter, rmse, parameter_name):
    """visualization the curves of mse_tr and mse_te."""
    #plt.semilogx(lambds, mse_tr, marker=".", color='b', label='train error')
    plt.semilogx(parameter, rmse, marker=".", color='r', label='test error')
    plt.xlabel(parameter_name)
    plt.ylabel("rmse")
    plt.title('RMSE')
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig('RMSE' +str(parameter_name))