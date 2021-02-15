# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 19:56:27 2021

@author: USER
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

Z_fun = lambda copula_corr , M , epsilon : np.sqrt(copula_corr) * M + np.sqrt(1-copula_corr) * epsilon
tau_fun = lambda Z,PD : -np.log(1-norm.cdf(Z))/PD
def one_factor_copula_simulation(port_principal_list,copula_corr,T,RR,PD,simul_num = 10000,alpha= 0.001) :
    ###############################################
    # port_principal_list is Principal Value List #
    # for example : [100,100,100,...]             #
    ###############################################
    LGD = 1-RR
    Total_num = len(port_principal_list)
    M = np.random.normal(size = (1,simul_num))
    ##########################
    # fast simulation method #
    ##########################
    half_simul = int(simul_num/2+0.5)
    e1 = np.random.normal(size = (Total_num,half_simul))
    e2 = -e1
    epsilon = np.concatenate([e1,e2],axis=1)[:,:simul_num]
    Z = Z_fun(copula_corr, M, epsilon)#np.sqrt(copula_corr) * M + np.sqrt(1-copula_corr) * epsilon
    tau = tau_fun(Z,PD)
    Simul_Loss_Matrix = np.array(port_principal_list).reshape(-1,1) * (tau<T) * LGD
    Total_Simul_Loss = Simul_Loss_Matrix.sum(0)
    X = (Simul_Loss_Matrix>0).sum(0)
    Y = Total_num
    Joint_Prob = X*(X-1)/(Y*(Y-1))
    Default_Corr = ((Joint_Prob - PD**2)/(PD * (1-PD))).mean()
    Default_Rate = X/Y
    WCDR = pd.Series(X/Y).quantile(1-alpha)
    EDR = Default_Rate.mean()
    Default_Rate_Std = Default_Rate.std()
    EL = Total_Simul_Loss.mean()
    UEL = pd.Series(Total_Simul_Loss).quantile(1-alpha)
    RC = UEL - EL
    return {'simul_result' : Total_Simul_Loss ,'expected_loss' :EL, 'unexpected_loss':UEL, 'Risk_Capital':RC, 'Default_Corr' : Default_Corr, "EDR" : EDR, "Default_Rate_Std": Default_Rate_Std, 'WCDR' : WCDR}

def Calibrate_Copula_Corr(port_principal_list,actual_default_std, T, RR,PD, simul_num = 10000) :
    #######################################
    ## Using actual std of default prob, ## 
    ## Calibrate Copula Correlation      ##
    #######################################
    DR_std = np.vectorize(lambda i : one_factor_copula_simulation(port_principal_list,i,T,RR,PD,simul_num)['Default_Rate_Std'])
    Corr_Range = np.linspace(0,0.8,20+1)
    Simulated_Std = DR_std(Corr_Range)
    Copula_Correl = Corr_Range[np.abs(Simulated_Std - actual_default_std).argmin()]
    return Copula_Correl

def Worst_Case_Default_Rate(PD,copula_corr,alpha = 0.001) :
    WCDR = norm.cdf((norm.ppf(PD) +np.sqrt(copula_corr) * norm.ppf(1-alpha))/(np.sqrt(1-copula_corr)) )
    return WCDR

def Credit_VaR_Gaussian_Copula(port_principal_list, copula_corr, RR, PD, alpha = 0.001) :
    LGD = 1-RR
    Total_Principal = np.array(port_principal_list).sum()
    WCDR = Worst_Case_Default_Rate(PD,copula_corr,alpha)
    UEL = Total_Principal * WCDR * LGD
    return {'unexpected_loss':UEL,'WCDR':WCDR}

def Credit_Risk_Capital_Gaussian_Copula(port_principal_list , copula_corr, RR , PD , alpha = 0.001) :
    Total_Principal = np.array(port_principal_list).sum()
    LGD = 1-RR
    my_dict = Credit_VaR_Gaussian_Copula(port_principal_list, copula_corr, RR, PD, alpha )
    P_range = np.arange(0.001,1,0.001)
    Default_Rate = norm.cdf((norm.ppf(PD) + np.sqrt(copula_corr) * norm.ppf(P_range))/np.sqrt(1-copula_corr))
    EDR = Default_Rate.mean()
    UEL = my_dict['unexpected_loss']
    EL = Total_Principal *EDR * (1-RR)
    RC = UEL - EL
    my_dict['expected_loss'] = EL , 
    my_dict['Risk_Capital'] = RC
    my_dict['EDR'] = EDR
    return my_dict

def copula_correlation_with_equity(equity_price_data , start_day , end_day) :
    ############################################## 
    ## Using Equity Corr > Calculate Copula Corr #
    ############################################## 
    equity_price_data.index = pd.to_datetime(equity_price_data.index)
    Price = pd.DataFrame(equity_price_data).resample('M').last()
    Return = Price.pct_change()[pd.to_datetime(start_day) : pd.to_datetime(end_day)]
    eq_corr = pd.DataFrame(np.triu(Return.corr())).applymap(lambda x : np.nan if x == 1 or x == 0 else x)
    eq_corr_ary = eq_corr.values.reshape(-1)
    corrs = pd.Series(eq_corr_ary)[pd.Series(eq_corr_ary).isna() == False].values.reshape(-1,1)
    ############################
    ## Copula Corr Straint 0 ~ 1 
    copul_corr_range = np.arange(0,1,0.02).reshape(1,-1)
    #################################
    ## (corr_equity - corr_copul)^2 #
    min_number = ((corrs - copul_corr_range)**2).sum(0).argmin()
    copul_corr_range.reshape(-1)[min_number]
    copul_corr = copul_corr_range.reshape(-1)[min_number]
    return copul_corr

def CDO_senior_mezzanin_equity(PD, copula_corr, PD_senior, PD_mezzanin , RR) :
    ########################################
    ## w_senior = 1 - WCDR(PD Senior) * LGD
    ## w_mezzanin = 1 - WCDR(PD Mezzanin) * LGD - w_senior
    ## w_equity = 1 - w_senior - w_mezzanin
    ########################################
    LGD = 1-RR
    senior_ratio = 1-(Worst_Case_Default_Rate(PD,copula_corr,alpha = PD_senior) * LGD)
    mezzanin_ratio = 1- (Worst_Case_Default_Rate(PD,copula_corr,alpha = PD_mezzanin) * LGD) - senior_ratio
    equity_ratio = 1- senior_ratio - mezzanin_ratio
    return senior_ratio, mezzanin_ratio, equity_ratio