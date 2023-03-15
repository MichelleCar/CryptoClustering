# Crypto Market Analysis
## Case Study: Machine Learning (Unsupervised) for Cryptocurrency Market Prediction
![crypto_market jpeg](https://user-images.githubusercontent.com/115101031/225374746-42a76031-c14a-4c78-af25-7c5fd27b5db2.jpg)

Since its inception, coinciding with the international crisis of 2008 and the associated lack of confidence in the financial system, bitcoin has gained an important place in the internat 
ional financial landscape, attracting extensive media coverage, as well as the attention of regulators, government institutions, institutional and individual investors, academia, and the public in general.  However, they have also caused as many problems as they have seemingly solved.  Highly unregulated and volatile, their prices are mostly idiosyncratic, as they are mainly driven by behavioral factors and are uncorrelated with the major classes of financial assets. 

In this case study, we evaluate the efficacy of using unsupervised machine learning to analyze a dataset and determine the ability of different models to derive meaning, and drive predictability of different cryptocurrencies.

###### (Source: https://jfin-swufe.springeropen.com/articles/10.1186/s40854-020-00217-x) 

## What is Unsupervised Machine Learning?
Machine learning is a branch of artificial intelligence (AI) and computer science which focuses on the use of data and algorithms to imitate the way that humans learn, gradually improving its accuracy.

Machine learning is behind chatbots and predictive text, language translation apps, the shows Netflix suggests to you, and how your social media feeds are presented. It powers autonomous vehicles and machines that can diagnose medical conditions based on images. 

Machine learning starts with data.  In fact, machine learning is best suited for situations with lots of data — thousands or millions of examples. The data is transformed (ie. normalized) to be used as training data.  From there, programmers choose a machine learning model to use, supply the data, and let the computer model train itself to find patterns or make predictions. Over time the human programmer can also tweak the model, including changing its parameters, to help push it toward more accurate results.  Some data is held out from the training data to be used as evaluation data, which tests how accurate the machine learning model is when it is shown new data. The result is a model that can be used in the future with different sets of data.

In **unsupervised** machine learning, a program looks for patterns in unlabeled data. Unsupervised machine learning can find patterns or trends that people aren’t explicitly looking for. For example, an unsupervised machine learning program could look through online sales data and identify different types of clients making purchases.

Like any tool, it is important to be aware of the limitations and challenges it presents.  As the data analyst is important to be aware of these, mitigate them where possible, and ensure that your deliverable has protocols for including these in your reporting:
* Explainability: It can be very difficult to understanding exactly what the machine learning models are doing and how they make decisions. To mitigate this, it is important not to just accept the outcome, but instead to try and get a feeling of the ground rules that it has established, as well as ensuring you have a plan to validate the output.
* Bias and unintended outcomes: Machines are trained by humans, and human biases can be incorporated into algorithms — if biased information, or data that reflects existing inequities, is fed to a machine learning program, the program will learn to replicate it and perpetuate forms of discrimination.

###### (Source: https://mitsloan.mit.edu/ideas-made-to-matter/machine-learning-explained)

## Scope of Project

### Methodology
Essentially, after preparing the data (normalizing a dataset using the StandardScaler module from SciKit-Learn), we will test the data using the following approachto determine the best approach for determining the predictability power of the data:
1) Finding the best value for k (number of ideal clusters) using the elbow method, then clustering our cryptocurrency dataset with K-means using our original scaled DataFrame.
2) Optimizing our clusters with Principal Component Analysis, and rerunning our model with K-means using the PCA data

#### K-means & Principal Component Analysis (PCA)
The concept of "clustering" refers to the act of grouping observations, such that observations within the same group will be similar (or related) to one another and different from (or unrelated to) observations in other groups. Clustering is an unsupervised machine-learning technique. Determining clusters is done through the use of an algorithm.  

K-Means Clustering is an Unsupervised Learning iterative algorithm, which groups an unlabeled dataset into different clusters in such a way that each dataset belongs only one group that has similar properties. The main aim of this algorithm is to minimize the sum of distances between the data point and their corresponding clusters. 

Principal component analysis, or PCA, is a dimensionality-reduction method that is often used to reduce the dimensionality of large data sets, by transforming a large set of variables into a smaller one that still contains most of the information in the large set.

A fundamental step for any unsupervised algorithm is to determine the optimal number of clusters into which the data may be clustered. The Elbow Method is one of the most popular methods to determine this optimal value. 

### Results and Observations

After reading in our data and loading into a DataFrame, I generated summary statitics and plotted the data to visualize it's structure.  



In the Elbow method, we are actually varying the number of clusters (K) from 1 – 10. For each value of K, we are calculating WCSS (Within-Cluster Sum of Square). WCSS is the sum of the squared distance between each point and the centroid in a cluster. When we plot the WCSS with the K value, the plot looks like an Elbow. As the number of clusters increases, the WCSS value will start to decrease. WCSS value is largest when K = 1. When we analyze the graph, we can see that the graph will rapidly change at a point and thus creating an elbow shape. From this point, the graph moves almost parallel to the X-axis. The K value corresponding to this point is the optimal value of K or an optimal number of clusters.


https://www.analyticsvidhya.com/blog/2021/01/in-depth-intuition-of-k-means-clustering-algorithm-in-machine-learning/ 

