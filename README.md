# Crypto Market Analysis
## Case Study: Machine Learning (Unsupervised) for Cryptocurrency Market Prediction
![crypto_market jpeg](https://user-images.githubusercontent.com/115101031/225374746-42a76031-c14a-4c78-af25-7c5fd27b5db2.jpg)

Since its inception, coinciding with the international crisis of 2008 and the associated lack of confidence in the financial system, bitcoin has gained an important place in the internat 
ional financial landscape, attracting extensive media coverage, as well as the attention of regulators, government institutions, institutional and individual investors, academia, and the public in general.  However, they have also caused as many problems as they have seemingly solved.  Highly unregulated and volatile, their prices are mostly idiosyncratic, as they are mainly driven by behavioral factors and are uncorrelated with the major classes of financial assets. 
(Source: https://jfin-swufe.springeropen.com/articles/10.1186/s40854-020-00217-x) 

In this case study, we evaluate the efficacy of using unsupervised machine learning to analyze a dataset and determine the ability of different models to derive meaning, and drive predictability of different cryptocurrencies.

## What is Unsupervised Machine Learning?
Machine learning is a branch of artificial intelligence (AI) and computer science which focuses on the use of data and algorithms to imitate the way that humans learn, gradually improving its accuracy.

Machine learning is behind chatbots and predictive text, language translation apps, the shows Netflix suggests to you, and how your social media feeds are presented. It powers autonomous vehicles and machines that can diagnose medical conditions based on images. 

Machine learning starts with data.  In fact, machine learning is best suited for situations with lots of data — thousands or millions of examples. The data is transformed (ie. normalized) to be used as training data.  From there, programmers choose a machine learning model to use, supply the data, and let the computer model train itself to find patterns or make predictions. Over time the human programmer can also tweak the model, including changing its parameters, to help push it toward more accurate results.  Some data is held out from the training data to be used as evaluation data, which tests how accurate the machine learning model is when it is shown new data. The result is a model that can be used in the future with different sets of data.

In **unsupervised** machine learning, a program looks for patterns in unlabeled data. Unsupervised machine learning can find patterns or trends that people aren’t explicitly looking for. For example, an unsupervised machine learning program could look through online sales data and identify different types of clients making purchases.

Like any tool, it is important to be aware of the limitations and challenges it presents.  As the data analyst is important to be aware of these, mitigate them where possible, and ensure that your deliverable has protocols for including these in your reporting:
* Explainability: It can be very difficult to understanding exactly what the machine learning models are doing and how they make decisions. To mitigate this, it is important not to just accept the outcome, but instead to try and get a feeling of what are the ground rules that it came up with, as well as ensuring you have a plan to validate the output.
* Bias and unintended outcomes: Machines are trained by humans, and human biases can be incorporated into algorithms — if biased information, or data that reflects existing inequities, is fed to a machine learning program, the program will learn to replicate it and perpetuate forms of discrimination.

Sources:
* https://mitsloan.mit.edu/ideas-made-to-matter/machine-learning-explained
* https://www.ibm.com/topics/machine-learning

## Scope of Project

### Methodology
Essentially, after preparing the data (normalizing a dataset using the StandardScaler module from SciKit-Learn), we will test the data using the following approachto determine the best approach for determining the predictability power of the data:
1) Finding the best value for k (number of ideal clusters) using the elbow method, then clustering our cryptocurrency dataset with K-means using our original scaled DataFrame.
2) Optimizing our clusters with Principal Component Analysis, and rerunning our model with K-means using the PCA data

#### K-means & Principal Component Analysis (PCA)
The concept of "clustering" refers to the act of grouping observations, such that observations within the same group will be similar (or related) to one another and different from (or unrelated to) observations in other groups. Determining this is done through the use of an algorithm.

K-means is a centroid-based clustering algorithm that works as follows:
*  Random initialization: place k centroids randomly.
*  Cluster assignment: assign each observation to the closest cluster based on the distance to centroids.
*  Centroid update: move centroids to the means of observations of the same cluster.
*  Repeat steps 2 and 3 until convergence is reached.

The goal of PCA is to identify the most meaningful basis to re-express data, using only the most meaningful portion (features) of the data, filtering out the noise and revealing hidden structures.

Each works to decrease the dimensionality of large amounts of data. Cmbined, Principal Components Analysis (PCA) and K-means Clustering works to improve segmentation results.
