import numpy as np
import pandas as pd
import random
import copy
from scipy.stats import moment
import matplotlib.pyplot as plt
import math
import yfinance as yf
import datetime
import scipy


def CorrolationMetrix(StartTime, EndTime, LogReturn_df):
  CorrMatrix = LogReturn_df[(LogReturn_df.index>=StartTime) & (LogReturn_df.index<=EndTime)].cov().to_numpy()
  return CorrMatrix

def Get_Magnitization(StartTime, EndTime, LogReturn_df):
    LogReturn_df = LogReturn_df[(LogReturn_df.index>=StartTime) & (LogReturn_df.index<=EndTime)]
    m = LogReturn_df.sum()/len(LogReturn_df)
    return m

import numpy as np

def energy_difference(spins, J, h, i):

    interaction_term = np.dot(J[i], spins)

    delta_E = 2 * spins[i] * (interaction_term + h[i])
    
    return delta_E

def energy_difference_higher_order(spins, Gamma, J, h, i):


    three_body_term = np.einsum('jk,j,k->', Gamma[i], spins, spins)

    interaction_term = np.dot(J[i], spins)
    

    delta_E = 2 * spins[i] * (three_body_term + interaction_term + h[i])
    
    return delta_E



def Energy(spins, J, h):
        # print(len(spins))
        return -0.5 * np.dot(spins.T, np.dot(J, spins)) - np.dot(h, spins)

def Energy_higher_order(spins, h, J, Gamma):
        result = - np.einsum('ijk,i,j,k->', Gamma, spins, spins, spins) - 0.5 * np.dot(spins.T, np.dot(J, spins)) - np.dot(h, spins)
        return result

def MCMC(SpinConfiguration, J, h, beta, NumberOfitterations, NumberOfSamples):

    
    old_spins = SpinConfiguration.copy()
    length = len(SpinConfiguration)
    # spin_evolution = np.zeros((NumberOfitterations, length))
    spin_samples = np.zeros((NumberOfSamples, length))


    for i in range(NumberOfitterations+NumberOfSamples):
      for _ in range(length):
          
          random_spin = np.random.randint(0, length)
        #   SpinConfigurationFlipped = SpinConfiguration.copy()
        #   SpinConfigurationFlipped[random_spin] *= -1
        #   dE = Energy(SpinConfigurationFlipped, J, h) - Energy(SpinConfiguration, J, h)
          dE = energy_difference(SpinConfiguration, J, h, random_spin)
          if dE <= 0:
              SpinConfiguration[random_spin] *= -1
              old_spins = copy.deepcopy(SpinConfiguration)


          elif np.random.random() < np.exp(-dE*beta):
              SpinConfiguration[random_spin] *= -1


      if i >= NumberOfitterations:
          spin_samples[i - NumberOfitterations] = SpinConfiguration



    return _, spin_samples


def MCMC_higher_order(SpinConfiguration, h, J, Gamma, beta, NumberOfitterations, NumberOfSamples):

    
    old_spins = SpinConfiguration.copy()
    length = len(SpinConfiguration)
    # spin_evolution = np.zeros((NumberOfitterations, length))
    spin_samples = np.zeros((NumberOfSamples, length))


    for i in range(NumberOfitterations+NumberOfSamples):
      for _ in range(length):
          
          random_spin = np.random.randint(0, length)
        #   SpinConfigurationFlipped = SpinConfiguration.copy()
        #   SpinConfigurationFlipped[random_spin] *= -1

        #   dE = Energy_higher_order(SpinConfigurationFlipped, h, J, Gamma) - Energy_higher_order(SpinConfiguration, h, J, Gamma)
          dE = energy_difference_higher_order(SpinConfiguration, Gamma, J, h, random_spin)
          if dE <= 0:
              SpinConfiguration[random_spin] *= -1
            #   old_spins = copy.deepcopy(SpinConfiguration)

          elif np.random.random() < np.exp(-dE*beta):
              SpinConfiguration[random_spin] *= -1


      if i >= NumberOfitterations:
          spin_samples[i - NumberOfitterations] = SpinConfiguration


    return _, spin_samples


def h(spins, J):
  length_of_series = len(spins)
  number_of_spins = len(spins[0])
  h = np.zeros([length_of_series, number_of_spins])

  for i in range(length_of_series):

    h[i] = np.matmul(J, spins[i])

  return h


def TAP(m ,ts ,J ,beta ,tol):
    for _ in range(ts):
        m_prev = m.copy()
        m = np.tanh(beta * (np.dot(J, m) - beta * np.sum(J**2 * (1 - m**2) * m)))
        # Check for convergence
        if np.max(np.abs(m - m_prev)) < tol:
            break
    return m

def calculate_covariance_matrix(samples):
    return np.cov(samples, rowvar=False)


def Generate_J(std, mean, N):
#   J = np.random.normal(size=(N,N), scale = std/N**0.5, loc = mean/N)
#   J = (J + J.T)/(2**0.5)
  J = np.triu(np.random.normal(loc = mean/N, scale = std / N**0.5, size=(N, N)), 1)
  J = J + J.T 
  np.fill_diagonal(J, 0)

  return J

def Generate_h(std, mean, N):
    h = np.random.normal(loc=mean, scale = std, size=N)

    return h

def Generate_Gamma(std, mean, N):
#   J = np.random.normal(size=(N,N), scale = std/N**0.5, loc = mean/N)
#   J = (J + J.T)/(2**0.5)
    Gamma = np.zeros([N,N,N])

    for i in range(N):
        for j in range(N):
            for k in range(N):
                if i!=j and j!=k and i!=k:

                    value = np.random.normal(loc = mean/N**2, scale = std / N)

                    Gamma[i, j, k] = value
                    Gamma[k, j, i] = value
                    Gamma[i, k, j] = value
                    Gamma[k, i, j] = value
                    Gamma[j, k, i] = value
                    Gamma[j, i, k] = value
    return Gamma


def Get_C_M_J(NumberOfItterations, NumberOfSamples, Size, mean_J, std_J, mean_h, std_h, beta):
    h_init = Generate_h(std_h, mean_h, Size)
    J_init = Generate_J(std_J, mean_J, Size)
    InitialSpins = np.ones(Size)
    spin_evolution, spin_samples = MCMC(InitialSpins, J_init, h_init, beta, NumberOfItterations, NumberOfSamples)
    C = calculate_covariance_matrix(spin_samples)
    M = np.sum(spin_samples, axis=0) / len(spin_samples)

    return J_init, h_init, C, M


def Get_C_M_J_higher_order(NumberOfIterations, NumberOfSamples, Size, mean_Gamma, std_Gamma, mean_J, std_J, mean_h, std_h, beta):
    
    h_init = Generate_h(std_h, mean_h, Size)
    J_init = Generate_J(std_J, mean_J, Size)
    Gamma_init = Generate_Gamma(std_Gamma, mean_Gamma, Size)

    InitialSpins = np.ones(Size)
    spin_evolution, spin_samples = MCMC_higher_order(InitialSpins, h_init, J_init, Gamma_init, beta, NumberOfIterations, NumberOfSamples)
    
    C = calculate_covariance_matrix(spin_samples)
    M = np.sum(spin_samples, axis=0) / len(spin_samples)

    return Gamma_init, J_init, h_init, C, M


def Get_C_M(NumberOfItterations, NumberOfSamples, Size, beta, J_init, h_init):
    # h_init = Generate_h(std_h, mean_h, Size)
    # J_init = Generate_J(std_J, mean_J, Size)
    InitialSpins = np.ones(Size)
    spin_evolution, spin_samples = MCMC(InitialSpins, J_init, h_init, beta, NumberOfItterations, NumberOfSamples)
    C = calculate_covariance_matrix(spin_samples)
    M = np.sum(spin_samples, axis=0) / len(spin_samples)

    return C, M

def Get_C_M_higher_order(NumberOfItterations, NumberOfSamples, Size, beta, Gamma_init, J_init, h_init):
    # h_init = Generate_h(std_h, mean_h, Size)
    # J_init = Generate_J(std_J, mean_J, Size)
    InitialSpins = np.ones(Size)
    spin_samples = MCMC_higher_order(InitialSpins, h_init, J_init, Gamma_init, beta, NumberOfItterations, NumberOfSamples)
    C = calculate_covariance_matrix(spin_samples)
    M = np.sum(spin_samples, axis=0) / len(spin_samples)

    return C, M



def nMF_Reconstruction(C, M):
    length = len(M)
    try:
        J_nMF = - np.linalg.pinv(C)
    except:
        J_nMF = - scipy.linalg.pinv(C)
    h_nMF = np.zeros(length)

    # np.fill_diagonal(J_nMF, 0)
    for i in range(length):
        # print(M[i])
        # print(math.atan(M[i]))
        h_nMF[i] = math.atan(M[i]) - np.matmul(J_nMF, M)[i]

    return h_nMF, J_nMF 


def TAP_Reconstruction(C, M):
    length = len(M)
    C_inv = np.linalg.pinv(C)
    h_nMF, _ = nMF_Reconstruction(C, M)
    m_outer = np.outer(M, M)
    J_TAP = (-2 * C_inv) / (1 + (1 - 8 * C_inv * m_outer)**0.5)
    h_TAP = np.zeros(length)

    np.fill_diagonal(J_TAP, 0)
    for i in range(length):
        tmp = 0
        for j in range(length):
            tmp += M[i]*(J_TAP[i,j]**2 *(1-M[j]**2))
        # h_TAP[i] = h_nMF[i] + M[i] * np.matmul(J_TAP*J_TAP,(1-M**2))[i]
        h_TAP[i] = h_nMF[i] - tmp
    
    return h_TAP, J_TAP

def IP_Reconstruction(C, M):
    length = len(M)
    J_IP = np.zeros([length, length])
    h_IP = np.zeros(length)

    for i in range(length):
        for j in range(length):
            J_IP[i,j] = 0.25 * np.log((1+M[i]+M[j]+C[i,j]+M[i]*M[j])*(1-M[i]-M[j]+C[i,j]+M[i]*M[j])/(((1-M[i]+M[j]-C[i,j]-M[i]*M[j]) * (1+M[i]-M[j]-C[i,j]-M[i]*M[j]))))

    for i in range(length):
        h_IP[i] = 0.5 * np.log((1+M[i])/(1-M[i])) - np.matmul(J_IP, M)[i]
    
    return h_IP, J_IP


def SM_Reconstruction(C, M):
    length = len(M)
    _, J_nMF = nMF_Reconstruction(C, M)
    h_IP, J_IP = IP_Reconstruction(C, M)

    J_SM = np.zeros([length, length])

    for i in range(length):
        for j in range(length):

            J_SM[i,j] = J_nMF[i,j] + J_IP[i,j] - C[i,j]/((1-M[i]**2)*(1-M[j]**2) - C[i,j]**2)
    h_SM = h_IP

    return h_SM, J_SM


def Get_C_inv(N, mean_J, std_J, beta, fill_diagonal):
    J = Generate_J(std_J, mean_J, N)
    NumberOfIterrations = N**4
    NumberOfSamples = 10000
    InitialSpins = np.ones(N)
    spin_evolution, spin_samples = MCMC(InitialSpins, J, beta, NumberOfIterrations, NumberOfSamples)
    C = calculate_covariance_matrix(spin_samples)
    C_inv = np.linalg.pinv(C)
    if fill_diagonal==True:
        np.fill_diagonal(C_inv, 0)

    return J, C_inv, spin_samples

def Reconstruct_higher_order(N, mean_J, std_J, mean_gamma, std_gamma, beta, fill_diagonal):
    J = Generate_J(std_J, mean_J, N)
    Gamma = Generate_Gamma(std_gamma, mean_gamma, N)

    NumberOfIterrations = N**4
    NumberOfSamples = 10000
    InitialSpins = np.ones(N)
    spin_evolution, spin_samples = MCMC_higher_order(InitialSpins, J, Gamma, beta, NumberOfIterrations, NumberOfSamples)
    C = calculate_covariance_matrix(spin_samples)
    C_inv = np.linalg.pinv(C)
    if fill_diagonal==True:
        np.fill_diagonal(C_inv, 0)

    return J, C_inv, spin_samples

def Reconstruct_higher_order_with_J_and_Gamma(N, mean_J, std_J, mean_gamma, std_gamma, beta, fill_diagonal, h, J, Gamma):

    NumberOfIterrations = N**4
    NumberOfSamples = 10000
    InitialSpins = np.ones(N)

    spin_evolution, spin_samples = MCMC_higher_order(InitialSpins, h, J, Gamma, beta, NumberOfIterrations, NumberOfSamples)
    C = calculate_covariance_matrix(spin_samples)
    C_inv = np.linalg.pinv(C)
    if fill_diagonal==True:
        np.fill_diagonal(C_inv, 0)

    return C_inv, spin_samples

def Get_C_inv_With_J(N, beta, J, InitialSpins, fill_diagonal):
    # J = Generate_J(std_J, mean_J, N)
    NumberOfIterrations = N**4
    NumberOfSamples = 10000
    spin_evolution, spin_samples = MCMC(InitialSpins, J, beta, NumberOfIterrations, NumberOfSamples)
    C = calculate_covariance_matrix(spin_samples)
    C_inv = np.linalg.pinv(C)
    if fill_diagonal==True:
        np.fill_diagonal(C_inv, 0)

    return C_inv, spin_samples


def log_returns(df):
  df['log'] = np.log(df['Close'])
  df['LogReturns'] = df['log'].diff()
  df.drop(['log'], axis=1, inplace=True)
  df.dropna(inplace=True)


def Get_S_and_P_data(start_date, end_date):
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    tickers = pd.read_html(url)[0]
    sp500_tickers = tickers['Symbol'].tolist()

    # Fetch historical stock data for all S&P 500 tickers
    sp500_data = []
    
    for tick in sp500_tickers:
        try:
            data = yf.download(tick, start='1958-01-01', end='2023-12-31')
            sp500_data.append(data)
        except:
            pass

    for i in range(len(sp500_data)):
        log_returns(sp500_data[i])
    
    for i in range(len(sp500_data)):
        try:
            sp500_data[i].drop(['Open', 'High', 'Low', 'Adj Close', 'Volume'], axis=1, inplace=True)
        except:
            continue
    
    # start_date = datetime(2000, 1, 1, 0, 0, 0)
    # end_date = datetime.now()
    time_index = pd.date_range(start=start_date, end=end_date, freq='D')
    FirstColumn = 'LogReturns'+sp500_tickers[0]
    LogReturns = pd.DataFrame(index=time_index)
    LogReturns[FirstColumn] = sp500_data[0]['LogReturns']

    for i in range(500):
        try:
            CurrentMinStartingTime = min(sp500_data[i].index)
            if CurrentMinStartingTime<datetime(2000,1,1,0,0,0):
                LogReturns = pd.merge(LogReturns, sp500_data[i]['LogReturns'], left_index=True, right_index=True, suffixes=('', sp500_tickers[i]))
        except Exception as e:
            print(e)

    LogReturns.drop(['LogReturns'], axis=1, inplace=True)
    
    return LogReturns 


def RMAE(Original_J, Recontructed_J):
    Recontructed_J[abs(Recontructed_J)==0] = 1000
    np.fill_diagonal(Recontructed_J, 0)
    rmae = np.nansum(abs(Recontructed_J - Original_J))/np.sum(abs(Original_J))
    return rmae

def generate_data(gamma_variance_interval, 
                  J_variance_interval, 
                  h_variance_interval, 
                  gamma_mean_interval, 
                  J_mean_interval, 
                  h_mean_interval,
                  Size,
                  NumberOfSamples=10000,
                  beta=1):
    RMAE_df = []
    exact_Js = [] 
    nMF_Js = []
    TAP_Js = []

    NumberOfIterrations = Size**4
    counter = 0
    for mean_gamma in gamma_mean_interval:
        for variance_gamma in gamma_variance_interval:
            
            for mean_J in J_mean_interval:
                for variance_J in J_variance_interval:
                    
                    for mean_h in h_mean_interval:
                        for variance_h in h_variance_interval:
                            
                            # Gamma = Generate_Gamma(variance**0.5, mean, N)
                            # J = Generate_J(J_variance**0.5, J_mean**0.5, N)
                            # h = Generate_h(h_variance**0.5, h_mean, N)
                            Gamma_init, J_init, h_init, C, M = Get_C_M_J_higher_order(NumberOfIterrations, NumberOfSamples, Size, mean_gamma, variance_gamma**0.05, mean_J, variance_J**0.5, mean_h, variance_h**0.5, beta)
                            
                            h_nMF, J_nMF = nMF_Reconstruction(C, M) 
                            h_TAP, J_TAP = TAP_Reconstruction(C, M) 

                            nMF_RMAE = RMAE(J_init, J_nMF)
                            TAP_RMAE = RMAE(J_init, J_TAP)
                            if nMF_RMAE > 2 or TAP_RMAE > 2:
                                label = 0
                            else:
                                label = 1

                            # TAP_RMAE_df.iloc[counter] = [mean, variance ,J_mean ,J_variance, h_mean, h_variance, TAP_RMAE]
                            # nMF_RMAE_df.iloc[counter] = [mean, variance ,J_mean ,J_variance, h_mean, h_variance, nMF_RMAE]
                            
                            counter += 1

                            np.save(f'C_{counter}.npy', C)
                            np.save(f'J_TAP_{counter}.npy', J_TAP)
                            np.save(f'J_nMF_{counter}.npy', J_nMF)
                            np.save(f'J_init_{counter}.npy', J_init)
                            print(counter)

                            RMAE_df.append({"Gamma_mean":mean_gamma, "Gamma_variance":variance_gamma , "J_mean": mean_J ,"J_varinace": variance_J, "h_mean": mean_J, "h_variance": variance_h, "TAP_RMAE":TAP_RMAE, "nMF_RMAE":nMF_RMAE, "label":label})        
                            nMF_Js.append(J_nMF)
                            TAP_Js.append(J_TAP)
                            exact_Js.append(J_init)
    
    return  RMAE_df, exact_Js, nMF_Js, TAP_Js


def Get_Moments(Matrix):
    upper_triangle_indices = np.triu_indices(Matrix.shape[0], k=1)
    upper_triangle_Matrix = Matrix[upper_triangle_indices]

    mean_J = np.mean(upper_triangle_Matrix)
    var_J = np.var(upper_triangle_Matrix)
    m3_J = moment(upper_triangle_Matrix.flatten(), 3)
    m4_J = moment(upper_triangle_Matrix.flatten(), 4)
    m5_J = moment(upper_triangle_Matrix.flatten(), 5)

    Statistics_J = [mean_J, var_J, m3_J, m4_J, m5_J]

    return Statistics_J