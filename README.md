# GLMSCA

Generalised linear model - Simultaneously Component Analysis (GLMSCA) is an algorithm based on the ANOVA - Simultaneous Component Analysis [[1]](#1) [[2]](#2) algorithm (ASCA), mixed with Generalised Linear Models [[3]](#3) (GLM):

GLMSCA partitions response matrix, Y, based on linear predictors in a design 
matrix X.
Discrete distributions, such as binomial, poisson or negative binomial can be 
chosen, and effect matrices are made from generalised linear models (max 
log-likelihood).
Component analysis is performed on the effect matrices and a residual error 
term ε. 


Parameters
----------
**X** : *numpy ndarray or pandas Series/DataFrame*
Contain linear predictors so that 

$g(µ) = η = Xβ = β_0 + β_1X_{i1} + ... + β_jX_{ij}$

**Y**: *numpy ndarray or pandas DataFrame*

Matrix with response variables Y
    
Options
-------
**n_components**: *int*

Number of components in component analysis. Default is 5
    
**residual**: *str*

Type of residual to be plotted. After *.fit()* is called, results for
all residual types are calculated. Anscombe, deviance, pearson, 
response, quantile or working residuals can be chosen.

**dist**: *str*

Type of distribution used in the GLM part. *.fit()* needs to be called
if a new *.dist** has been chosen. Poisson, binomial and negative binomial
is currently implemented, and poisson is the default.
    
**dist_list**: *list*

List of distributions. It can be provide if response variables have 
different distributions. Length of the list needs to be equal to number 
of response variables in Y. If none are submitted, all variables are
assumed to be the same distribution
    
**interaction**: *boolean*

Assumes interaction effect between responses, so that interaction
between $β_1$ and $β_2$ is ( $β_1*β_2$ ). Default is false.
    
**keep_conv_only**: *boolean*

True keeps only converged variables. The variables are also deleted from 
**mdl.Y**. Default is True.

Attributes
----------
*All attributes are stored in  ._results as pseudo private attributes. All
attributes are empty (NoneType) untill .fit() is called.*

**beta**: *numpy ndarray*

Beta (β) for all response variables
    
**column_names**: *list*

List of column names for response variables, from pandas.columns. If 
none are provided, a list with '*Variable i*' is created, where *i* is the 
column number
    
**converged**: *list*

List of booleans. If variable, i, did not converge during Iterative 
Re-weighted least Squares, converged[i] is changed to False.

**dummy**: *numpy ndarray*

Dummy matrix of the linear predictor X.

Example: 

Matrix with 6 observations of 3 factors with 2 and 3 levels:

    
    [[Foo, 1, n30],
     [Foo, 2, n20],
     [Bar, 1, n10],
     [Bar, 2, n30],
     [Baz, 1, n20],
     [Baz, 2, n10]]

returns:   
    
    [[ 1.,  0.,  1., -1.,  0.,  1.],
     [ 1.,  0.,  1.,  1.,  1.,  0.],
     [ 1., -1., -1., -1., -1., -1.],
     [ 1., -1., -1.,  1.,  0.,  1.],
     [ 1.,  1.,  0., -1.,  1.,  0.],
     [ 1.,  1.,  0.,  1., -1., -1.]]
     
    
and if ```.Options.interaction == True```:

    [[ 1.,  0.,  1., -1.,  0.,  1., -0., -1.,  0.,  1., -0., -1.],
     [ 1.,  0.,  1.,  1.,  1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.],
     [ 1., -1., -1., -1., -1., -1.,  1.,  1.,  1.,  1.,  1.,  1.],
     [ 1., -1., -1.,  1.,  0.,  1., -1., -1., -0., -1.,  0.,  1.],
     [ 1.,  1.,  0., -1.,  1.,  0., -1., -0.,  1.,  0., -1., -0.],
     [ 1.,  1.,  0.,  1., -1., -1.,  1.,  0., -1., -0., -1., -1.]]
     
**dummy_indexer**: *list*

List of boolean arrays to index dummy variables.
    
**dummy_n**: *list*

List of indexed dummy variables.
    
**eta**: *numpy ndarray*

linear predictors (η):

$η = g(µ)$
    
**factor_names**: *list*

List of factor names. If none are provided, a list with '*Factor i*' is 
created, where *i* is the factor number

**factors**: *int*

number of factors

**mu**: *numpy ndarray*

Expected value of the response:

$µ = g^{−1}(η)$

**mu_n**: *list*

List of partitioned effect matrices

**residuals**: *dataclass*

Numpy ndarrays for each of the 6 residual types.

**sca_results**: *dataclass*

SCA results for each of the 6 residual types. Each of the results contain
scores, loadings, explained variance and the factor name, for each factor.
 
Methods
--------    
***fit()***:

Wrapper function. Performs algorithm
Results are stored in ._results (pseudo private attributes)
    
***plot_loadings()***:

Plots loadings of selected components for the selected factor. Default 
is component 1 and 2 and factor 0.
    
***plot_scores()***:

Plots scores of selected components for the selected factor. Default 
is component 1 and 2 and factor 0. Scores are grouped by supplied
vector (pandas Series). If none are provided the scores are grouped by
the plotted factor.

***plot_raw()***:

Plots raw data. Can be coloured according to individual factors.
    
***plot_raw()***:

Plots residuals. Can be coloured according to individual factors. Residual
to be plotted can be changed in .Options.residual

## References
<a id="1">[1]</a> 
Smilde, Age K. et al. “ANOVA-Simultaneous Component Analysis (ASCA): a New Tool for Analyzing Designed Metabolomics Data.” Bioinformatics 21.13 (2005): 3043–3048. Web.

<a id="2">[2]</a> 
Jansen, Jeroen J. et al. “ASCA: Analysis of Multivariate Data Obtained from an Experimental Design.” Journal of chemometrics 19.9 (2005): 469–481. Web.


<a id="3">[3]</a> 
Hilbe, Joseph M. “Generalized Linear Models.” The American statistician 48.3 (1994): 255–. Web.
