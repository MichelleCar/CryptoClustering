# Crypto Market Analysis
## Case Study: Machine Learning (Unsupervised) for Cryptocurrency Market Analysis
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

After reading in our data and loading it into a DataFrame, I generated summary statitics and plotted the data to visualize it's structure.  

<img width="884" alt="Screenshot 2023-03-15 at 7 09 14 PM" src="https://user-images.githubusercontent.com/115101031/225463567-bb38ae36-f5b6-4e56-9834-822d44952ba9.png">

<img width="1051" alt="Screenshot 2023-03-15 at 6 59 55 PM" src="https://user-images.githubusercontent.com/115101031/225461968-d95466df-788f-4793-aeb7-87f841d3bff5.png">

<img width="786" alt="Screenshot 2023-03-15 at 7 01 14 PM" src="https://user-images.githubusercontent.com/115101031/225462091-41528389-72ef-46a8-92ee-cdd30af8a17e.png">

The raw data shows a lot of variability.  I used StandardScaler to first normalize the data in preparation for our analysis.

<img width="890" alt="Screenshot 2023-03-15 at 7 08 57 PM" src="https://user-images.githubusercontent.com/115101031/225463628-68ccb364-6f19-40de-9ff4-50c1c365b927.png">

By using the Elbow method, we are actually varying the number of clusters (K) from 1 – 10. For each value of K, we are calculating WCSS (Within-Cluster Sum of Square). WCSS is the sum of the squared distance between each point and the centroid in a cluster. When we plot the WCSS with the K value, the plot looks like an Elbow. As the number of clusters increases, the WCSS value will start to decrease. WCSS value is largest when K = 1. When we analyze the graph, we can see that the graph will rapidly change at a point and thus creating an elbow shape. From this point, the graph moves almost parallel to the X-axis. The K value corresponding to this point is the optimal value of K or an optimal number of clusters.

The goal here isn’t just to make clusters, but to make good, meaningful clusters. Quality clustering is when the datapoints within a cluster are close together, and afar from other clusters.

In the elbow method, we plot the mean distance and look for the elbow point where the rate of decrease shifts. The elbow method plots the explained variation as a function of the number of clusters, and is used to choose a number of clusters when adding another cluster doesn’t improve the outcomes of modeling (inertia). Inertia is the sum of squared distance of samples to their closest cluster center. Intuitively, inertia tells how far away the points within a cluster are. We would like this number to be as small as possible.

In iterating through 10 k-values, three (3) clusters offers an inertia value of 123, four (4) clusters seems to signify a turning point in the inertia (79), and as we move towards six (6) clusters, our inertia hits 53, after which there is no appreciable change.

Settling on four (4) clusters seems the ideal choice.

When using PCA, based on the Elbow curve, the best value for the number of clusters k is 4. Ultimately, this does not differ from the original data, however, the Inertia itself varies at that point from the original data. With the original data, the inertia is 79 with k=4, and only 50 using 3 PCA. This is significant because it shows that the strength of 'k' has improved with 3 PCA. Even at k=6 inertia was 53% inertia with the original data.

By choosing k=4 with 3 PCA, we get a significant inertia reduction from 94 in k=3 to 50 in k=4. After that, reductions do not improve in more than 11 inertia units. Graphically, the elbow on k=4 is more pronounced, with subsequent points tapering off.

Comparing the elbow plots at k=4 (the number of clusters I decided on in both the original and PCA methods), it is clear that the PCA method yielded a stronger outcome, with an inertia value of 50%, compared to 79 for the original model. The lower the inertia, the better the clustering results.

![elbow_method_comparison_plot](https://user-images.githubusercontent.com/115101031/225465811-f18f6b73-0d88-4f08-b629-6a9f588a2495.png)

The original Cryptocurrencies (K-Mean Clusters=4) scatterplot vizualized a plot that was not very well defined.

![df_market_data_predictions_plot](https://user-images.githubusercontent.com/115101031/225466413-182622bf-5adf-492b-bd2c-b13edd7f0735.png)

After applying PCA, using 3 features, I observed that the highest fraction of explained variance among these variables is 37%, and the lowest one is 18%. We can also compute these fractions for subsets of variables. For instance, variables 1 and 2 together explain 72% of the total variance, and variables 1 and 3 explain 55%. Combined, all three components explain 90% of the total variance.

The concept of Explained variance is useful in assessing how important each component is. In general, the larger the variance explained by a principal component, the more important that component is. PCA is a technique used to reduce the dimensionality of data. It does this by finding the directions of maximum variance in the data and projecting the data onto those directions. The amount of variance explained by each direction is called the “explained variance.” Explained variance can be used to choose the number of dimensions to keep in a reduced dataset. It can also be used to assess the quality of a machine learning model. In general, a model with high explained variance will have good predictive power, while a model with low explained variance may not be as accurate.

When the overall sum of the two variance ratios is extremely low and the plot doesn’t give us any insight, using a 3rd Principal Component is highly beneficial.

After plotting the scattegram using the PCA results, we observe a much more well-defined clustering with clearly defined boundaries.

![df_market_data_predictions_pca_plot](https://user-images.githubusercontent.com/115101031/225466840-25e97e46-b2b3-4a33-b888-34c7e124655f.png)

### Conclusion
After visually analyzing the cluster analysis results, the impact of using fewer features to cluster the data using K-Means was clear.

1) Starting with the comparison of the elbow plots at k=4 (the number of clusters I decided on in both the original and PCA methods), it is clear that the PCA method yielded a stronger outcome, with an inertia value of 50%, compared to 79 for the original model. The lower the inertia, the better the clustering results.

For the scatter plots, the original method yielded a very cluttered result. The clusters are not well-defined and highly dispersed. Using PCA, the clusters are more defined, and allow for better analysis.

2) Using a fewer number of features (PCA) reduces the degree of inertia, and therefore, a stronger/clearer clustering effect. This happens because the reduction of dimensionality correlates to a reduction in the variance of the clustered data. In the analyzed case, we reduced the variance by using three components.

PCA achieves a higher reduction rate of the initial inertia. That means, the components are more efficient in setting up the data for clustering. In the case of PCA, the reduction was from 256 to 50 units of inertia, which is a reduction in 80% of the initial value; whereas the reduction with standarized data was from 287 to 79 units, wich is only a 72% reduction rate.

