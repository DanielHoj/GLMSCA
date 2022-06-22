import pandas as pd
import numpy as np
import itertools
import patsy.contrasts as cont
import statsmodels.api as sm
import scipy.stats as st
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from mycolorpy import colorlist as mcp
from matplotlib.lines import Line2D


'''
Fixes:
    scaling / preprocessing:
        on (mu + res) ?
    
    plot diagnostic (some type of LOF)
    
'''


# Dataclass container for residuals
@dataclass
class Residuals:
    anscombe: np.ndarray = None
    deviance: np.ndarray = None
    pearson: np.ndarray = None
    response: np.ndarray = None
    working: np.ndarray = None
    quantile: np.ndarray = None
    
# Dataclass container for SCA results
@dataclass
class Sca_results:
    anscombe: list = None
    deviance: list = None
    pearson: list = None
    response: list = None
    working: list = None
    quantile: list = None

# Dataclass container for results
@dataclass
class GLMSCA_Results:
    # Field with factory required to not make it an instance
    residuals: Residuals = field(default_factory = Residuals, init=False)
    sca_results: Sca_results = field(default_factory = Sca_results, init=False)
    dummy: np.ndarray = None
    dummy_n: list = None
    dummy_indexer: list = None
    converged: list = None
    beta: np.ndarray = None
    mu_n: list = None
    mu: np.ndarray = None
    eta: np.ndarray = None
    eta_n: np.ndarray = None
    column_names: list = None
    factor_names: list = None
    factors: int = None
        
    
# Option class 
class GLMSCA_Options:
    # Initialise inputs
    def __init__(self, scaling, n_components, residual, dist, dist_list, interaction, keep_conv_only):
        self.scaling = scaling
        self.n_components = n_components
        self.residual = residual
        self.dist = dist
        self.dist_list = dist_list
        self.interaction = interaction
        self.keep_conv_only = keep_conv_only
        
    @property
    def scaling(self):
        return self._scaling
    
    @scaling.setter
    def scaling(self, scaling):
        
        if scaling is not None:
            if not isinstance(scaling, str):
                raise ValueError('Please input processing as a string')
            elif scaling not in ['std', 'pareto']:
                raise ValueError('Please input std or pareto')
        self._scaling = scaling
    
    # Validate inputs
    @property
    def n_components(self):
        return self._n_components

    @n_components.setter
    def n_components(self, n_components):
    # Assert n_components is an int between 2-15
        if not isinstance(n_components, int):
            raise ValueError('Options n_components: Please input integer between 2 and 15')
        if not 2 <= n_components <= 15:
            raise ValueError('Options n_components: Value needs to be between 2 and 15')
        self._n_components = n_components
        
    @property
    def residual(self):
        return self._residual
    
    @residual.setter
    def residual(self, residual):
    # Assert residual is one of the following strings
        if not isinstance(residual, str):
            raise ValueError('Options residual: Please input string')
        if residual not in ['anscombe', 'deviance', 'pearson', 'response', 'working', 'quantile']:
            raise ValueError('Options residuals: Please input anscombe, deviance, pearson, response, quantile or working')
        self._residual = residual
        
    @property
    def dist(self):
        return self._dist
    
    @dist.setter
    def dist(self, dist):
    # Assert dist is one of the following strings
        if not isinstance(dist, str):
            raise ValueError('Options dist: Please input string')
        if dist not in ['Binomial', 'Poisson', 'NegativeBinomial']:
            raise ValueError("Options dist: Please choose 'Binomial', 'Poisson' or 'NegativeBinomial'")
        self._dist = dist
        
    @property
    def dist_list(self):
        return self._dist_list
    
    @dist_list.setter
    def dist_list(self, dist_list):
    # Assert dist is a list
        if not isinstance(dist_list, list) and dist_list is not None:
            raise ValueError('Options dist_list: Please input list of distributions')
        self._dist_list = dist_list
        
    @property
    def interaction(self):
        return self._interaction
    
    @interaction.setter
    def interaction(self, interaction):
    # Assert interaction is a boolean
        if not isinstance(interaction, bool):
            raise ValueError('Options interaction: Please input boolean True/False')
        self._interaction = interaction
        
    @property
    def keep_conv_only(self):
        return self._keep_conv_only
    
    @keep_conv_only.setter
    def keep_conv_only(self, keep_conv_only):
        # Assert keep_conv_only is a boolean
        if not isinstance(keep_conv_only, bool):
            raise ValueError('Options keep_conv_only: Please input boolean True/False')
        self._keep_conv_only = keep_conv_only

# Main Class    
class GLMSCA:
    # Initialise and validate inputs.
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self._x = None
        self._y = None
        self.Options = GLMSCA_Options(None, 5, 'deviance', 'Poisson', None, False, True)
        self._results = GLMSCA_Results()
    
    @property
    def X(self):
        return self._X

    @X.setter
    def X(self, dataframe):
    # Assert X is a pd.DataFrame, pd.Series or np.ndarray
        if not isinstance(dataframe, (pd.DataFrame, pd.Series, np.ndarray)):
            raise ValueError('X: Please input pandas DataFrame, Series or numpy ndarray')
        self._X = dataframe
        
    @property
    def Y(self):
        return self._Y

    @Y.setter
    # Assert Y is a pd.DataFrame or np.ndarray
    def Y(self, dataframe):
        if not isinstance(dataframe, (pd.DataFrame, np.ndarray)):
            raise ValueError('Y: Please input pandas DataFrame or numpy ndarray with more than one variable')
        self._Y = dataframe
        
    # Wrapper function. Performs algorithm 
    def fit(self, X_is_dummy = False):
        
        # Check X and Y
        self.__checkX()
        self.__checkY()
        
        # if X is not a dummy matrix, dummy matrix is created
        if X_is_dummy == True:
            self._results.dummy = self._x
            
        elif X_is_dummy == False:
            self.__dummy_function(self._x)
        
        # If Options.dist_list is provided it is used as the list of distributions
        if isinstance(self.Options.dist_list, list):
            dist = self.Options.dist_list
        # else a list of distributions are created based on .Options.dist
        else:
            dist = [self.Options.dist for k in range(self._y.shape[1])]
            
        # Iterative re-weighted Least Squares
        self.__irls(dummy = self._results.dummy, Y = self._y, dist = dist)
        # Y is partitioned based on X
        self.__partition_matrix()
        # Quantile residuals are calculated
        self.__quantile(dist)
        # Component analysis
        self.__sca()
        
    
    # Plot scores
    def plot_scores(self, factor = 0, group_by = None, components = (0, 1)):
        
        # if factor is a string (factor name) assert it is in factor_names and
        # translate to its index
        if isinstance(factor, str):
            try:
                factor_number = self._results.factor_names.index(factor)
            except:
                raise ValueError('The factor is not in factor_names')
        elif isinstance(factor, int):
            factor_number = factor
        else:
            raise ValueError('Factor input needs to be a string or an int')
        
        # Get the SCA results and raise warning if .fit() has not been called
        sca_results = getattr(self._results.sca_results, self.Options.residual)
        
        if sca_results is None:
            raise AttributeError('Please call fit() first')
            
        # Allow group_by to be a string of the column name of x
        if isinstance(group_by, str):
            try:
                group_by = getattr(self.X, group_by)
            except:
                   raise ValueError('group_by not a factor name')
        
        # Automatically assign the group_by to factor name is none is assigned
        if group_by is None:
            if self._results.factors == 1:
                group_by = pd.Series(self._x, name = self._results.factor_names[0])
            else:
                # if .Options.interactions = True, then factor_number is changed
                try:
                    X = pd.DataFrame(self._x, columns = self._results.factor_names)
                    group_by = self.X.iloc[:,factor_number]
                except:
                    if len(self._x.shape) == 1:
                        diff = self._results.factors - 1
                    else:
                        diff = self._results.factors - self._x.shape[1]
                        
                    group_by = self.X.iloc[:,factor_number-diff]
                    
        self.__plot_sca(sca_results[factor_number], d = group_by, components = components)
        
    # Plot Loadings
    def plot_loadings(self, factor = 0, components = (0, 1)):
        # if factor is a string (factor name) assert it is in factor_names and
        # translate to its index
        if isinstance(factor, str):
            try:
                factor_number = self._results.factor_names.index(factor)
            except:
                raise ValueError('The factor is not in factor_names')
                
        elif isinstance(factor, int):
            factor_number = factor
        else:
            raise ValueError('Factor input needs to be a string or an int')
        
        #  Get the SCA results and raise warning if .fit() has not been called
        sca_results = getattr(self._results.sca_results, self.Options.residual)
        if sca_results is None:
            raise AttributeError('Please call fit() first')
            
        self.__plot_sca(sca_results[factor_number], plot_type = 'loadings', components = components)
    
    # Plot raw data
    def plot_raw(self, factor = None):
        
        # Assert fit() has been called
        if self._results.factor_names is None:
            raise ValueError('Please call fit() first')
        
        # make Y as pandas DataFrame
        Y  = pd.DataFrame(self._y, columns= self._results.column_names)
        
        # If no factor to colour from
        if factor is None:
            Y.T.plot(legend = False)
        
        # Else color based on factor
        else:
            # Check if factor is string of int
            if isinstance(factor, str):
                try:
                    factor_number = self._results.factor_names.index(factor)
                    factor_name = factor
                except:
                    raise ValueError(f'{factor} not in factor_names')
            elif isinstance(factor, int):
                factor_number = factor
                factor_name = self._results.factor_names[factor]
            else:
                raise ValueError('Factor input needs to be a string or an int')
                
            if len(self._x.shape) == 1:
                factor = pd.Series(self._x, name = factor_name)
            else:
                factor = pd.Series(self._x[:,factor_number], name = factor_name)
            
            result = pd.concat([factor, Y], axis = 1)
            
            unique = factor.nunique()
            
            if unique <21:
                colors = self.__distinct_colors(unique)
            else:
                colors = mcp.gen_color(cmap ='prism', n=unique)
                
            if factor.dtype != 'int':
                codes, unique = pd.factorize(factor)
                result['codes'] = codes
                result['colors'] = result.loc[:,f'{factor.name}'].map({name:colors[codes[i]]
                                     for i, name in enumerate(result.loc[:,f'{factor.name}'])})
            else:
                result['colors'] = result.loc[:,f'{factor.name}'].map({i:colors[i-1]
                                             for i in factor})
                
            legend_elements = [Line2D([0], [0], marker='o', color='w', 
                        label=f'{factor.name}: {factor.unique()[i]}', markerfacecolor=mcolor, markersize=5)
                        for i, mcolor in enumerate(colors)]
            
            Y.T.plot(legend = False, color = result.colors)
            plt.legend(handles = legend_elements, ncol = 2)
            plt.title(f'Raw data for {factor_name}', loc = 'left')
    
    def plot_residual(self, factor = None):
        
        # Assert fit() has been called
        if self._results.factor_names is None:
            raise ValueError('Please call fit() first')
            
        res_type = self.Options.residual
        
        residual = getattr(self._results.residuals, res_type)
        
        # make Y as pandas DataFrame
        Y  = pd.DataFrame(residual, columns= self._results.column_names)
        
        # If no factor to colour from
        if factor is None:
            Y.T.plot(legend = False)
            plt.title(f'{res_type} residuals', loc = 'left')
        
        # Else color based on factor
        else:
            # Check if factor is string of int
            if isinstance(factor, str):
                try:
                    factor_number = self._results.factor_names.index(factor)
                    factor_name = factor
                except:
                    raise ValueError('The factor is not in factor_names')
            elif isinstance(factor, int):
                factor_number = factor
                factor_name = self._results.factor_names[factor]
            else:
                raise ValueError('Factor input needs to be a string or an int')
            
            if len(self._x.shape) == 1:
                factor = pd.Series(self._x, name = factor_name)
            else:
                factor = pd.Series(self._x[:,factor_number], name = factor_name)
            
            result = pd.concat([factor, Y], axis = 1)
            
            unique = factor.nunique()
            
            if unique <21:
                colors = self.__distinct_colors(unique)
            else:
                colors = mcp.gen_color(cmap ='prism', n=unique)
                
            if factor.dtype != 'int':
                codes, unique = pd.factorize(factor)
                result['codes'] = codes
                result['colors'] = result.loc[:,f'{factor.name}'].map({name:colors[codes[i]]
                                     for i, name in enumerate(result.loc[:,f'{factor.name}'])})
            else:
                result['colors'] = result.loc[:,f'{factor.name}'].map({i:colors[i-1]
                                             for i in factor})
                
            legend_elements = [Line2D([0], [0], marker='o', color='w', 
                        label=f'{factor.name}: {factor.unique()[i]}', markerfacecolor=mcolor, markersize=5)
                        for i, mcolor in enumerate(colors)]
            
            Y.T.plot(legend = False, color = result.colors)
            plt.legend(handles = legend_elements, ncol = 2)
            plt.title(f'{res_type} residuals coloured by {factor_name}', loc = 'left')
    
    '''
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    Private methods:
        
    '''
    # Set factor names and number of factor
    def __checkX(self):
        X = self.X
        if isinstance(X, pd.DataFrame):
            self._results.factor_names = list(X.columns)
            self._results.factors = X.shape[1]

            X = X.values
            self.X = pd.DataFrame(X, columns = self._results.factor_names)
            
            if len(X.shape) > 1 and X.shape[1] == 1:
                X = X.reshape(X.shape[0],)
            
        elif isinstance(X, pd.Series):
            self._results.factors = 1
            self._results.factor_names = [X.name]
            
            X = X.values
            self.X = pd.DataFrame(X, columns = self._results.factor_names)
            
        elif isinstance(X, np.ndarray):
            if len(X.shape) == 1:
                self._results.factors = 1
                self._results.factor_names = ['Factor 1']
                
                self.X = pd.Series(X, name = 'Factor 1')
                
            else:
                self._results.factors = X.shape[1]
                self._results.factor_names = [f'Factor {k+1}' 
                                          for k in range(X.shape[1])]
                
                self.X = pd.DataFrame(X, columns = self._results.factor_names)
                
        #self.X = pd.DataFrame(X, columns = self._results.factor_names)
        self._x = X
        
    def __checkY(self):
        # Set column names
        Y = self.Y
        if isinstance(Y, pd.DataFrame):
            self._results.column_names = list(Y.columns.astype('str'))
            Y = Y.values
            
        elif isinstance(Y, np.ndarray):
            self._results.column_names = [f'Variable {k+1}' 
                                          for k in range(Y.shape[1])]
            
        self.Y = pd.DataFrame(Y, columns = self._results.column_names)
        self._y = Y
            
        
    
    def __partition_matrix(self):
    
        # Get dummy matrix
        dummy = self._results.dummy
        
        # Get list of individual dummys
        dummy_n = self._results.dummy_n
        
        # Set an indexer based on dummys
        self._results.dummy_indexer = self.__dummy_index(dummy_n)
        
        # Get indexer based on dummys
        dummy_indexer = self.__dummy_index(dummy_n)
        
        # Drop intercept
        dummy_n = dummy_n[1:]
        
        # Set inverse link function based on distribution
        if self.Options.dist == 'Binomial':
            inv = self.__inverselogit
        elif self.Options.dist == 'NegativeBinomial':
            inv = self.__inverselog
        elif self.Options.dist == 'Poisson':
            inv = self.__inverselog
        
        
        # Subset beta for the reduced model
        beta_n = [self._results.beta[dummy_indexer[i],:] 
                       for i in range(1, self._results.factors + 1)]
        
        # Calculate mu for the reduced model
        eta_n = [dummy_n[i]@beta_n[i] 
                  for i in range(self._results.factors)]
        
        # Set mu based on effect
        mu_n = [None for k in range(self._results.factors)]
        
        for k in range(self._results.factors):    
            mu_n[k] = inv(eta_n[k])
            
        self._results.mu_n = mu_n
        self._results.eta_n = eta_n
            
    def __inverselog(self, z):
        # Inverse log link function
        return np.exp(z)
    
    def __inverselogit(self, z):
        # Inverse logit link function
        z = np.asarray(z)
        t = np.exp(-z)
        return 1. / (1. + t)

    def __dummy_function(self, X):
        # Factorize X
        if X.dtype != 'int':
            X = self.__factorize(X)
        
        # Prepare X
        if len(X.shape) > 1:
            rows, cols = X.shape
            X = list(X.transpose())
            n_factors = len(X)
        else:
            X = X.transpose()
            n_factors = 1
            rows = X.shape[0]
            
        #Make contrast matrix
        contrast = cont.Sum()
        intercept = np.asmatrix(np.ones(rows)).transpose()
        
        #for one factor:
        if n_factors == 1:
            
            #number of levels:
            levels = list(set(X))
            C = contrast.code_with_intercept(levels)
            
            result = C.matrix[X-1,:]
            self._results.dummy_n = [intercept] + [result[:,1:]]
        
        # for multiple factors
        else:            
            # number of levels for each factor
            levels = [list(set(X_i)) 
                      for X_i in X]
            
            # Contrast matrices
            C = [contrast.code_without_intercept(factor).matrix 
                 for factor in levels]
            
            # temp (dummy_n)
            temp = [intercept] + [C[i][X[i]-1,:] 
                    for i in range(len(C))]
            self._results.dummy_n = temp
            
            result = np.concatenate(temp, axis =1)
            
            # if Options.interaction = True
            if self.Options.interaction:
                combinations = list(itertools.combinations(self._results.factor_names,2))
                extra_names = ['*'.join(i) for i in combinations]
                extras = [a*b for a,b in itertools.combinations(temp[1:], 2)]
                
                temp_extras = temp + extras
                result = np.concatenate(temp_extras, axis =1)
                
                # Assert design rank i not higher than Y
                if result.shape[1] > self._y.shape[1]-1:
                    self.Options.interaction = False
                    raise ValueError('Design rank higher than Y rank, interaction set to False')
                self._results.factor_names += extra_names
                self._results.factors += len(extra_names)
                self._results.dummy_n = temp_extras
        
        # Set dummy matrix
        self._results.dummy = np.asarray(result)
        
    def __dummy_index(self, x):
        # Create indexer based on dummy_n
        xc = [0, *np.cumsum([i.shape[1] for i in x])]
        y = [np.asarray([q >= xc[n] and q < xc[n+1] for q in range(xc[-1])]) 
             for n in range(len(x))]
        return y
    
    def __factorize(self, X):
        # Factorize X if are strings
        if len(X.shape) == 1:
            newX = np.unique(X, return_inverse=True)[1]
        else:
            newX = np.asarray([np.unique(X[:,i], return_inverse=True)[1] 
                    for i in range(X.shape[1])]).T
        return newX

    def __irls(self, Y, dummy, dist):
        # GLM based on statsmodels GLM class
        # mu, eta and beta are calculated, with residuals
        n_rows, n_cols = Y.shape
        
        if len(dummy.shape) == 1:
            factors = 1
        else:
            factors = dummy.shape[1]
        
        beta = np.zeros([factors, n_cols])
        mu = np.zeros([n_rows, n_cols])
        eta = np.zeros([n_rows, n_cols])
        resid_anscombe = np.zeros([n_rows, n_cols])
        resid_deviance = np.zeros([n_rows, n_cols])
        resid_pearson = np.zeros([n_rows, n_cols])
        resid_response = np.zeros([n_rows, n_cols])
        resid_working = np.zeros([n_rows, n_cols])
        
        converged = np.repeat(True, n_cols)
        
        for col in range(n_cols):
            
            # Get Family
            Family = getattr(sm.families, dist[col])
            
            try:
                mdl = sm.GLM(Y[:,col], dummy, family = Family()).fit()
            except:
                print(f'Variable {col} did not converge')
                converged[col] = False
                pass
            
            if True not in converged:
                raise ValueError('All variables failed to converge')
            
            if converged[col] == True:
                beta[:,col] = mdl.params
                mu[:,col] = mdl.mu
                eta[:,col] = mdl.predict(linear = True)
                
                resid_anscombe[:,col] = mdl.resid_anscombe
                resid_deviance[:,col] = mdl.resid_deviance
                resid_pearson[:,col] = mdl.resid_pearson
                resid_response[:,col] = mdl.resid_response
                resid_working[:,col] = mdl.resid_working
        
        # Removes non converged variables
        if self._results.converged is None:
            self._results.converged = converged
            
        
        if self.Options.keep_conv_only:
            keep = converged
        else:
            keep = np.repeat(True, n_cols)
        
        self._results.beta = beta[:,keep]
        self._results.mu = mu[:,keep]
        self._results.eta = eta[:,keep]
        
        self._results.residuals.anscombe = resid_anscombe[:,keep]
        self._results.residuals.deviance = resid_deviance[:,keep]
        self._results.residuals.pearson = resid_pearson[:,keep]
        self._results.residuals.response = resid_response[:,keep]
        self._results.residuals.working = resid_working[:,keep]
        
        self._y = self._y[:,keep]
        self.Y = self.Y.iloc[:,keep]
        self.__checkY()
        
    def __quantile(self, dist):
        # Quantile residuals are calculated
        mu = self._results.mu
        y = self._y
        
        # Distribution changed to scipy.stats inputs
        mapper = {'Binomial': 'binom', 'Poisson': 'poisson', 'NegativeBinomial': 'nbinom'}
        dist_st = list(map(mapper.get, dist))
        
        # Norm quantile function
        norm_ppf = st.norm.ppf
        u = np.repeat(.5, y.shape[0])
        
        quantile_residual = np.empty(y.shape)
        
        for col in range(y.shape[1]):
            # pmf and cdf for each distribution
            pmf = getattr(st, dist_st[col]).pmf
            cdf = getattr(st, dist_st[col]).cdf
            
            # Estrimated cdf for each distribution
            if dist[col] == 'Poisson':
                e_cdf = cdf(y[:,col] - 1, mu[:,col]) + u*pmf(y[:,col], mu[:,col])
            elif dist[col] == 'Binomial':
                e_cdf = (cdf(y[:,col] - 1, p = mu[:,col], n = 1) + 
                u*pmf(y[:,col], p = mu[:,col], n = 1))
            elif dist[col] == 'NegativeBinomial':
                e_cdf = (cdf(y[:,col] - 1, mu[:,col], n = 1) + 
                u*pmf(y[:,col], mu[:,col], n = 1))
            
            # Quantile residuals for each variable
            quantile_residual[:,col] = norm_ppf(e_cdf)
        
        # Scale the quantile residuals
        ssy = np.sum(np.square(y))
        ssmu = np.sum(np.square(mu))
        sse = np.sum(np.square(quantile_residual))
        k = np.sqrt((ssy - ssmu)/sse+1e-6)
        result = k*quantile_residual
        
        # Set quantile residuals
        self._results.residuals.quantile = result
        
    def __sca(self):
        
        # Perform component analysis on each residual type
        for res_type in ['anscombe', 'deviance', 'pearson', 'response', 'working', 'quantile']:
            
            # Get residuals
            residuals = getattr(self._results.residuals, res_type)
            
            result = []
            
            # for each factor
            for k in range(self._results.factors):
                eta =  self._results.eta_n[k]
                musum = sum(self._results.mu_n[k])
                if self.Options.scaling == 'std':
                    scale = (musum + residuals).std(axis=0)
                elif self.Options.scaling == 'pareto':
                    scale = np.sqrt((musum + residuals).std(axis=0))
                else:
                    scale = 1
                
                scale += 1e-9
                
                mu_res = (eta + residuals)/scale
                
                
                #List of name of pcs:
                PCn = [f'PC{num}' for num in range(1, self.Options.n_components+1)]
                #Covariance matrix of X
                cov_mat = np.cov(eta/scale, rowvar = False)
                 
                #calculate eigenValues and eigen vectors
                eigen_values , eigen_vectors = np.linalg.eigh(cov_mat)
                 
                #Sort based on eigenvalues
                sorted_index = np.argsort(eigen_values)[::-1][:self.Options.n_components]
                sorted_eigenvalues = eigen_values[sorted_index]
                sorted_eigenvectors = eigen_vectors[:,sorted_index]
                
                #Keep n_components
                eigenvector_subset = sorted_eigenvectors[:,0:self.Options.n_components]
                
                # Calculate loadings
                loadings = pd.DataFrame(eigenvector_subset*np.sqrt(sorted_eigenvalues), 
                           index=self._results.column_names, columns = PCn).T
                
                # Caldulate scores
                scores = pd.DataFrame(np.dot((mu_res),eigenvector_subset), columns=PCn)
                
                # Calculate explained variance
                explained_variance = sorted_eigenvalues/np.sum(eigen_values)
                
                # Set factor name
                if isinstance(self._results.factor_names, list):
                    factor_name = self._results.factor_names[k]
                else:
                    factor_name = self._results.factor_names
                # 
                result.append([scores, loadings, explained_variance, factor_name])
            # Set results
            setattr(self._results.sca_results, f'{res_type}', result)
            

    def __plot_sca(self, sca_results, d = None, components = (0,1), legend = 'on', 
                plot_type ='scores'):
        
        # Get variables for speficic SCA result
        scores, loadings, explained_variance, factor_names = sca_results
        
        name = ' of ' + factor_names
            
        if plot_type == 'scores':
            scores = scores.iloc[:,[components[0],components[1]]]
            if isinstance(d, (pd.Series, np.ndarray)):
                # Create spider plot
                D_scores = pd.concat([d, scores], axis = 1)
                Centerpoints = D_scores.groupby(f'{d.name}').mean()
                D_scores = D_scores.set_index(f'{d.name}')
                D_scores.loc[:,f'Center_{D_scores.columns[0]}'] = Centerpoints.iloc[:,0]
                D_scores.loc[:,f'Center_{D_scores.columns[1]}'] = Centerpoints.iloc[:,1]
                D_scores.loc[:,f'{d.name}'] = D_scores.index
                if d.nunique() <21:
                    colors = self.__distinct_colors(d.nunique())
                else:
                    colors = mcp.gen_color(cmap ='prism', n=d.nunique())
                
                if d.dtype != 'int':
                    codes, unique = pd.factorize(d)
                    D_scores['codes'] = codes
                    D_scores['colors'] = D_scores.loc[:,f'{d.name}'].map({name:colors[codes[i]]
                                         for i, name in enumerate(D_scores.loc[:,f'{d.name}'])})
                else:
                    D_scores['colors'] = D_scores.loc[:,f'{d.name}'].map({i:colors[i-1]
                                         for i in d})
                
                fig, ax = plt.subplots(1, figsize=(8,8))
            
                plt.scatter(scores.iloc[:,0], scores.iloc[:,1], c=D_scores.colors, alpha = 0.6, s=10)
                # plot lines
                for idx, val in D_scores.iterrows():
                    x = [val[f'{scores.columns[0]}'], val[f'Center_{scores.columns[0]}']]
                    y = [val[f'{scores.columns[1]}'], val[f'Center_{scores.columns[1]}']]
                    plt.plot(x, y, c = val.colors, alpha=0.4)
                
                if legend == 'on':
                    legend_elements = [Line2D([0], [0], marker='o', color='w', 
                    label=f'{d.name}: {d.unique()[i]}', markerfacecolor=mcolor, markersize=5) 
                    for i, mcolor in enumerate(colors)]
                    plt.legend(handles=legend_elements, loc='upper right', ncol=2)
            
    
                plt.xlim(D_scores.iloc[:,0].min() -(D_scores.iloc[:,0].max() -D_scores.iloc[:,0].min())*.05,D_scores.iloc[:,0].max() +(D_scores.iloc[:,0].max() -D_scores.iloc[:,0].min())*.05)
                plt.ylim(D_scores.iloc[:,1].min() -(D_scores.iloc[:,1].max() -D_scores.iloc[:,1].min())*.05,D_scores.iloc[:,1].max() +(D_scores.iloc[:,1].max() -D_scores.iloc[:,1].min())*.05)
                
                plt.axhline(linestyle='--')
                plt.axvline(linestyle='--')    
                
                plt.title(f'Scores{name}', loc='left', fontsize=22)
                plt.xlabel(f'{scores.columns[0]} \n Explained Variance: {round(explained_variance[0]*100, 1)}%')
                plt.ylabel(f'{scores.columns[1]} \n Explained Variance: {round(explained_variance[1]*100, 1)}%')
            
            else:
                D_scores = scores
                fig, ax = plt.subplots(1, figsize=(8,8))      
                plt.scatter(scores.iloc[:,0], scores.iloc[:,1], alpha = 0.6, s=10)
                
                plt.axhline(linestyle='--')
                plt.axvline(linestyle='--')    
                
                plt.title(f'scores{name}', loc='left', fontsize=22)
                plt.xlabel(f'{scores.columns[0]} \n Explained Variance: {round(explained_variance[0]*100, 1)}%')
                plt.ylabel(f'{scores.columns[1]} \n Explained Variance: {round(explained_variance[1]*100, 1)}%')
            
                plt.xlim(scores.iloc[:,0].min() -(scores.iloc[:,0].max() -scores.iloc[:,0].min())*.05,scores.iloc[:,0].max() +(scores.iloc[:,0].max() -scores.iloc[:,0].min())*.05)
                plt.ylim(scores.iloc[:,1].min() -(scores.iloc[:,1].max() -scores.iloc[:,1].min())*.05,scores.iloc[:,1].max() +(scores.iloc[:,1].max() -scores.iloc[:,1].min())*.05)
        
        elif plot_type == 'loadings':
            title = f'Loadings{name}'
            xlabel = 'Variable'
            loadings.T.iloc[:,list(components)].plot(title = title, ylabel = xlabel)

        
        
    def __distinct_colors(self, num_colors):
        # most distinct colors for 2-20
        #max colors = 20
        colors= [
        ['#00ff00', '#0000ff'],
        ['#ff0000', '#00ff00', '#0000ff'],
        ['#ff0000', '#00ff00', '#0000ff', '#87cefa'],
        ['#ffa500', '#00ff7f', '#00bfff', '#0000ff', '#ff1493'],
        ['#66cdaa', '#ffa500', '#00ff00', '#0000ff', '#1e90ff', '#ff1493'],
        ['#808000', '#ff4500', '#c71585', '#00ff00', '#00ffff', '#0000ff', 
         '#1e90ff'],
        ['#006400', '#ff0000', '#ffd700', '#c71585', '#00ff00', '#00ffff', 
         '#0000ff', '#1e90ff'],
        ['#191970', '#006400', '#bc8f8f', '#ff4500', '#ffd700', '#00ff00', 
         '#00ffff', '#0000ff', '#ff1493'],
        ['#006400', '#00008b', '#b03060', '#ff4500', '#ffff00', '#deb887', 
         '#00ff00', '#00ffff', '#ff00ff', '#6495ed'],
        ['#8b4513', '#006400', '#4682b4', '#00008b', '#ff0000', '#ffff00', 
         '#00ff7f', '#00ffff', '#ff00ff', '#eee8aa', '#ff69b4'],
        ['#2f4f4f', '#7f0000', '#008000', '#000080', '#ff8c00', '#ffff00',
         '#00ff00', '#00ffff', '#ff00ff', '#1e90ff', '#eee8aa', '#ff69b4'],
        ['#2f4f4f', '#8b4513', '#228b22', '#000080', '#ff0000', '#ffff00',
         '#00ff00', '#00ffff', '#0000ff', '#ff00ff', '#1e90ff', '#eee8aa', 
         '#ff69b4'],
        ['#2f4f4f', '#7f0000', '#008000', '#4b0082', '#ff8c00', '#deb887', 
         '#00ff00', '#00bfff', '#0000ff', '#ff00ff', '#ffff54', '#dda0dd', 
         '#ff1493', '#7fffd4'],
        ['#2f4f4f', '#8b4513', '#006400', '#4b0082', '#ff0000', '#ffa500', 
         '#ffff00', '#00ff00', '#00ffff', '#0000ff', '#d8bfd8', '#ff00ff', 
         '#1e90ff', '#98fb98', '#ff69b4'],
        ['#2f4f4f', '#800000', '#191970', '#006400', '#bdb76b', '#48d1cc', 
         '#ff0000', '#ffa500', '#ffff00', '#0000cd', '#00ff00', '#00fa9a', 
         '#da70d6', '#d8bfd8', '#ff00ff', '#1e90ff'],
        ['#2f4f4f', '#800000', '#008000', '#bdb76b', '#4b0082', '#b03060', 
         '#48d1cc', '#ff4500', '#ffa500', '#ffff00', '#00ff00', '#00fa9a', 
         '#0000ff', '#d8bfd8', '#ff00ff', '#1e90ff', '#ee82ee'],
        ['#2f4f4f', '#7f0000', '#006400', '#7f007f', '#ff0000', '#ff8c00',
         '#ffff00', '#40e0d0', '#7fff00', '#00fa9a', '#4169e1', '#e9967a', 
         '#00bfff', '#0000ff', '#ff00ff', '#f0e68c', '#dda0dd', '#ff1493'],
        ['#808080', '#2e8b57', '#7f0000', '#808000', '#8b008b', '#ff0000',
         '#ffa500', '#ffff00', '#0000cd', '#7cfc00', '#00fa9a', '#4169e1', 
         '#00ffff', '#00bfff', '#f08080', '#ff00ff', '#eee8aa', '#dda0dd',
         '#ff1493'],
        ['#2f4f4f', '#2e8b57', '#8b0000', '#808000', '#7f007f', '#ff0000',
         '#ff8c00', '#ffd700', '#0000cd', '#00ff7f', '#4169e1', '#00ffff', 
         '#00bfff', '#adff2f', '#d8bfd8', '#ff00ff', '#f0e68c', '#fa8072',
         '#ff1493', '#ee82ee']
        ]
        try:
            color = colors[num_colors-2]
            return color
        except:
            print('Too many colors')
            
