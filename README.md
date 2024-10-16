# Predictive Analytics for Enhanced Customer Engagement: A Data-Driven Approach to Boosting Sales Performance at Imperials Ltd.

## Introduction and Background
In the competitive world of the insurance sector, companies like Imperials Ltd. are embracing innovation to expand their market presence and enhance customer relationships (Jones and Sah, 2023), (Rawat et al., 2021). This firm, a leader in its field, is leveraging the use of advanced analytics to optimize its marketing strategies. With access to a rich customer data, Imperials Ltd. has launched a project to predict which customers are most likely to invest in life insurance products. This analytics task goes beyond merely improving marketing efficiency; it represents a strategic effort to tailor product offerings to customer preferences, fostering loyalty and promoting sustained growth (Reporter, 2023).

The need for this analysis is driven by several key factors. The digital revolution has transformed consumer expectations, with customers now seeking personalized services that reflect their individual needs and life phases (Haudi, 2024). Moreover, while insurance firms gather vast quantities of data, the potential of these resources often remains untapped without sophisticated analysis. Through predictive analytics, Imperials Ltd. seeks to reveal the valuable insights hidden within its data, adopting a proactive approach to marketing (Adeoye et al., 2024).

This project lies at the nexus of Business analytics and strategic marketing, aiming to utilize both descriptive and predictive analytics techniques to navigate through historical sales information to unearth patterns and predictors of consumer behaviour. The process involves thorough data cleaning, exploration, and modelling to extract meaningful insights from complex datasets. The focal point of this analysis is the 'Purchase' variable, which signifies whether a customer has acquired a life insurance policy, thus forming the basis for targeted marketing initiatives. By accurately forecasting this outcome, Imperials Ltd. hopes to refine its marketing efforts, focusing on customer segments with a high likelihood of purchasing (Balasubramanian et al., 2021).

## Methodology

![image](https://github.com/user-attachments/assets/830e0b02-b1f7-40a1-9fd6-9b991fc8d9d9)

Our methodology is structured around the Cross-Industry Standard Process for Data Mining (CRISP-DM) framework, ensuring a systematic approach to solving Imperials Ltd.’s challenge of predicting life insurance purchases (Wirth and Hipp, 2000). This journey begins with Business Understanding, where we defined the need to leverage predictive analytics for enhancing marketing strategies. Moving into Data Understanding, we delved into the dataset, identifying key characteristics and initial data quality issues. The Data Preparation phase saw us cleaning and transforming data for analysis. Modeling involved selecting and applying suitable machine learning algorithms, followed by Evaluation to assess their performance. Though the Deployment phase extends beyond this report, it involves integrating the model into operational strategies for actionable insights.

### Descriptive Statistics and Visualizations
In the data understanding phase, our data consist of 40,000 observations for 13 different variables. Moreover, we encountered missing values in 'education', 'marriage', and 'house_owner' fields, alongside diverse categorical variables. Our descriptive analysis revealed modes for various attributes among customers who purchased life insurance, highlighting trends such as a preference for 'Professional' occupations and 'Married' status. To visualize these insights, we employed histograms to show distribution patterns, bar charts for categorical variable frequencies (Levine and Stephan, 2009), aiding in a comprehensive initial data exploration.

![image](https://github.com/user-attachments/assets/f3b31e49-519c-47b0-9d14-9caecf2c54ac)

### Machine Learning Algorithms
Given the binary nature of our target variable ('Purchased'), we selected two supervised learning algorithms for their distinct advantages:
#### Gradient Boosting: 
Chosen for its ability in handling diverse datasets and capability to improve on decision tree limitations through ensemble learning. Its strength lies in sequentially correcting errors from previous models, making it powerful for classification tasks. However, its complexity can sometimes lead to overfitting and longer training times (Konstantinov and Utkin, 2021).
#### Generalized Additive Model (GAM): 
Selected for its flexibility in modeling non-linear relationships by using smooth functions, allowing for interpretable results that directly link feature changes to outcome shifts. While highly adaptable, GAMs can be less interactive for those unfamiliar with its approach and may require careful tuning of smoothness parameters to avoid underfitting or overfitting (Baayen and Linke, 2020).
Our choice was informed by literature suggesting Gradient Boosting’s effectiveness in similar predictive tasks within the insurance domain (Jones and Sah, 2023), (Ejiyi et al., 2022),( Sahai, 2022), and GAM’s strong performance in scenarios requiring nuanced interpretation of individual feature effects. Both models undergo rigorous training, validation, and hyperparameter tuning to optimize performance, with a split dataset ensuring robust evaluation against unseen data.

### Data Pre-Processing (Data Cleaning and Preparation)
Imperials Ltd's journey towards harnessing data for strategic marketing begins with careful data pre-processing. Our first step was to determine the completeness of the data. We identified missing values across key categorical features: education, marriage, and house ownership, with the counts of missing values standing at 741, 14,027, and 3,377 respectively. We addressed these by designating an 'Unknown' category, thereby retaining these records for analysis without imputing potentially misleading values.

![image](https://github.com/user-attachments/assets/ee45de55-9cd3-4d20-90ed-402de9a7f61d)
Figure 1: Count of zero values

Next, we focused on uniqueness, where 305 duplicate records were removed to ensure the integrity of our dataset. With the validity of our data confirmed, we turned our attention to the 'child' column. To maintain consistency, we translated '0' values to 'N' for 'no children', streamlining our categorical data into a logical format.
After these foundational steps, we transformed our data to better suit the predictive models we planned to employ. Notably, we engineered a new binary variable, 'house_val_nonzero', distinguishing between customers with and without house value data.

![image](https://github.com/user-attachments/assets/65f05cb1-9690-434d-9447-591fd821d136)
Figure 2: Distribution of house_val_nonzero

Our exploratory data analysis (EDA) started with a visualization of house values, revealing a skewed distribution where most customers fall within the lower range of property valuation, crucial information for understanding economic demographics.
The pre-processing phase also includes robust encoding strategies. We transformed categorical data into a numerical format (label encoding), enabling the Generalized Additive Model (GAM) to embrace the variables. While, one hot encoding for gradient boosting Model (GBM).

### Data Analytics

#### Descriptive Statistics and Visualisations
Through statistical summaries, we differentiated the central tendencies and distributions of key variables. For instance, the analysis of the 'house_val' variable reveals a distribution with a substantial peak in the lower value range, indicating a high frequency of customers with relatively modest house valuations. This insight prompts targeted strategies for customers within this economic bracket.

![image](https://github.com/user-attachments/assets/3d1c44b9-db7c-443f-a81c-b83e95a1ef5d)
Figure 3: Distribution of House Value

Visual explorations provide further clarity. The frequency plot for education levels represents a balanced representation across different educational qualifications, yet it is the customers with a Bachelor's degree who show a higher propensity for purchasing life insurance, as illustrated in the 'Purchased Life Insurance by Education Level' histogram. Such findings hint at a correlation between education and investment in life insurance, potentially guiding the marketing focus.

![image](https://github.com/user-attachments/assets/a0c6329b-5d5b-426a-8323-5d5520069b0c)
Figure 4: Distribution of Purchase count over education level

Marital status, as seen in the 'Purchased Life Insurance by Marriage Status' chart, surfaces another layer of understanding, with married customers exhibiting higher insurance uptake. This aligns with life-stage financial planning that often accompanies marital commitments.

![image](https://github.com/user-attachments/assets/da4b8271-3555-479c-a023-86cb3b4341ec)
Figure 5: Distribution of Purchase count over marriage status

Age distribution and its relationship with life insurance purchases come to life in our visualizations, where middle-aged customer segments are spotlighted as significant contributors to policy acquisitions.

![image](https://github.com/user-attachments/assets/b2a2ac5d-5cb9-44c8-96aa-b64230c654ed)
Figure 6: Distribution of Purchase count over Age group

Family income and the presence of children are not left behind, with medium income brackets and households with children identified as segments with a greater inclination towards life insurance—a reflection of the protective instinct and financial foresight within family settings.

![image](https://github.com/user-attachments/assets/679f07e2-cdda-41f7-8a15-07b739bedda6)
Figure 7: Distribution of Purchase count over family income

![image](https://github.com/user-attachments/assets/937ac5f0-f095-4d39-9eca-e47a469f2fdf)
Figure 8: Distribution of Purchase count over having child

Mortgage status, home ownership, and regional segmentation are also visually dissected, showcasing variances in purchasing behaviour that could be pivotal in regional market segmentation and tailored marketing campaigns.

![image](https://github.com/user-attachments/assets/33d7a079-0e25-4717-9b8c-e77840ef464b)
Figure 9: Distribution of Purchase count over mortgage status

![image](https://github.com/user-attachments/assets/5510f6cf-8e91-43af-ac31-22dc1cbe9aaf)
Figure 10: Distribution of Purchase count over region

![image](https://github.com/user-attachments/assets/dec488a7-3626-4640-a177-ae9ed83c9c57)
Figure 11: Distribution of Purchase count over house ownership
Each plot and statistical measure is not just a mere representation of data; they are narratives about potential customers, guiding lights towards understanding the connection between demographics and purchasing patterns. This comprehension is essential, as it informs the predictive modelling that follows, ensuring that the foundation upon which we build our predictive analytics is powerful and insightful.

### Supervised Machine Learning
As we transition into the analytical domain of supervised machine learning, our groundwork in pre-processing ensures that our models learn from clean and structured data. This is pivotal, as the quality of input data greatly influences predictive outcomes (Tae et al., 2019).
Our dataset split adheres to the standard training and testing paradigm, allocating 80% for training our models—a Gradient Boosting Classifier and a Generalized Additive Model (GAM)—and 20% to test their predictive ability. This classification is a safeguard against overfitting and ensures our models can generalize beyond the training data (Reitermanova, 2010).

In tuning our models, we engage with hyperparameter optimization, a critical step in refining model performance (Yang and Shami, 2020). Our tuning, conducted through GridSearchCV, systematically pass through a predefined grid of hyperparameters, evaluating model performance via cross-validation. This approach not only finds the most performant parameters but also informs rigor and reproducibility to our machine learning model.

Performance evaluation highlights accuracy; we employ a collection of metrics—precision, recall, F1 score, and the Area Under the ROC Curve (AUC)—that together offer a holistic view of our models performance. Each metric highlights different aspects of predictive performance (Alpaydin, 2020).
Comparing our two chosen models, we aim to use the strengths of ensemble methods against the interpretability of GAMs, seeking not only a model that predicts with high accuracy but also one that aligns with the strategic objectives of Imperials Ltd.

## Results and Discussion

### Model Performance
The results from the Gradient Boosting and Generalized Additive Models (GAM) demonstrated parallel performance with equivalent accuracy metrics as demonstrated in the plots and summarized in the table below:

![image](https://github.com/user-attachments/assets/48a98312-2b50-4087-a52a-78a37a7030d3)
Figure 12: Performance measures of gradient boosting classifier

![image](https://github.com/user-attachments/assets/af96a0ac-aefc-42d0-9a21-b08c3af33c47)
Figure 13: Performance measures of GAM model

![image](https://github.com/user-attachments/assets/655b3c6c-8ad0-470f-9f72-609fee346ad3)

Based on the comparison of the performance metrics and plots, the Gradient Boosting Classifier slightly outperforms the Generalized Additive Model (GAM), particularly in terms of ROC AUC and precision-recall trade-off. The Gradient Boosting Classifier also demonstrates a higher peak F1 score, suggesting a better balance between precision and recall. However, the GAM shows a smaller gap between training and validation scores, which could indicate better generalization.

### Insights and Challenges:

![image](https://github.com/user-attachments/assets/5276b79d-991a-4845-98ae-3723200aed5e)
Figure 14: Feature importance plot

The feature importance plots indicated that 'house_val', 'Gender_M', 'occupation_professional', 'mortgage_1low', ‘online_Y’, ‘education_HS’, and age groups 7 and 5 are the top predictors influencing the purchase of insurance, implying that customers' socio-economic status and gender type are significant predictors of insurance purchasing behaviour.

![image](https://github.com/user-attachments/assets/794b642b-2900-48e1-9571-cff5c10e47a7)
Figure 14: Partial dependent plots

Partial Dependence Plots (PDPs) offered deeper insights into the relationship between these features and the target variable (Apley and Zhu, 2020). The top predictors, based on the noticeable changes and variations in the partial dependence plots, appear to be almost same which were highlighted from feature importance plot of our gradient boosting method.

The challenges noted were the gain of model performance after hyperparameter tuning, which may be attributed to noise in the dataset, and a ceiling on the predictive power with the given features. In this regard, the use of PDPs for interpreting the models clearly indicates the effect of each feature on the likelihood of purchasing insurance and, hence, is much richer in terms of the analytical narrative than just prediction accuracy.

## Limitations
A noteworthy limitation was the occurrence of 'Unknown' categories in features like 'education' and 'marriage' status, which may have introduced ambiguity into the models. Additionally, despite the accuracy of the models, the real-world applicability may require further validation and continuous monitoring to account for changing consumer patterns and economic conditions. The consistency in performance metrics post hyperparameter tuning also suggests a need to explore more diverse model architectures or feature engineering techniques to extract additional predictive value from the data.

## Conclusion and Recommendations

### Key Findings
The analysis has led to several key findings relevant for predicting customer purchases of insurance. Notably, the socio-economic factors such as home value, presence of a mortgage, and occupation play pivotal roles in determining insurance purchases. Additionally, marital status appears to influence the likelihood of purchasing insurance, suggesting that life stage may be correlated with insurance needs.

The models applied have demonstrated a good balance between precision and recall, indicating that the predictive framework can reliably identify the segments more or less likely to purchase insurance. Moreover, the partial dependence plots have revealed that the relationship between predictors and the target variable is not always linear, emphasizing the need for interpretation of the model outputs.

### Recommendations
Based on the insights gleaned from the analysis, the following actionable recommendations are put forward for Imperials Ltd:
1. **Targeted Marketing**: Given the strong influence of socio-economic factors, marketing efforts could be more effective if tailored to specific segments. For instance, customers with higher home values and low to medium mortgages could be targeted with premium insurance offers.
2. **Product Development**: Develop and promote insurance products that cater to the needs of different occupational groups, as the data indicates variability in insurance uptake across occupation types.
3. **Lifecycle Marketing**: Integrate marital status into customer profiles to capture life stage difference and design communication strategies for insurance products that resonate with the individual circumstances of customers.
4. **Data Enrichment**: To overcome limitations encountered in the analysis, consider augmenting the dataset with more granular socio-demographic and behavioural data to refine predictive models further (Mulligan and Wainwright, 2013).
5. **Model Deployment and Monitoring**: Implement a real-time analytics framework to deploy the developed models and continuously monitor their performance, updating them as customer patterns evolve over time (Jabbar et al., 2020).
6. **Further Initiatives**: Given the uncertainty associated with 'Unknown' categories in education and other features, there is an opportunity for customer education and engagement initiatives to better understand their needs and refine the accuracy of predictive models.

In conclusion, while the models show promise, a strategic approach combining targeted marketing, product customization, and continuous model refinement is essential for optimizing insurance sales and customer satisfaction.






## References:
1. Haudi, H., 2024. The impact of digital transformation on consumer behavior and marketing strategies. International journal of economic literature, 2(1), pp.167-179
2. Adeoye, O.B., Okoye, C.C., Ofodile, O.C., Odeyemi, O., Addy, W.A. and Ajayi-Nifise, A.O., 2024. INTEGRATING ARTIFICIAL INTELLIGENCE IN PERSONALIZED INSURANCE PRODUCTS: A PATHWAY TO ENHANCED CUSTOMER ENGAGEMENT. International Journal of Management & Entrepreneurship Research, 6(3), pp.502-511.
3. Wirth, R. and Hipp, J., 2000, April. CRISP-DM: Towards a standard process model for data mining. In Proceedings of the 4th international conference on the practical applications of knowledge discovery and data mining (Vol. 1, pp. 29-39).
4. Levine, D.M. and Stephan, D.F., 2009. Presenting data in charts and tables: categorical and numerical variables. Pearson Education
5. Konstantinov, A.V. and Utkin, L.V., 2021. Interpretable machine learning with an ensemble of gradient boosting machines. Knowledge-Based Systems, 222, p.106993.
6. Baayen, R.H. and Linke, M., 2020. An introduction to the generalized additive model. A practical handbook of corpus linguistics. New York: Springer, pp.563-591.
7. Jones, K.I. and Sah, S., 2023. The Implementation of Machine Learning In The Insurance Industry With Big Data Analytics. International Journal of Data Informatics and Intelligent Computing, 2(2), pp.21-38.
8. Tae, K.H., Roh, Y., Oh, Y.H., Kim, H. and Whang, S.E., 2019, June. Data cleaning for accurate, fair, and robust models: A big data-AI integration approach. In Proceedings of the 3rd international workshop on data management for end-to-end machine learning (pp. 1-4).
9. Ejiyi, C.J., Qin, Z., Salako, A.A., Happy, M.N., Nneji, G.U., Ukwuoma, C.C., Chikwendu, I.A. and Gen, J., 2022. Comparative analysis of building insurance prediction using some machine learning algorithms.
10. Reitermanova, Z., 2010, June. Data splitting. In WDS (Vol. 10, pp. 31-36). Prague: Matfyzpress.
11. Yang, L. and Shami, A., 2020. On hyperparameter optimization of machine learning algorithms: Theory and practice. Neurocomputing, 415, pp.295-316.
12. Alpaydin, E., 2020. Introduction to machine learning. MIT press.
13. Apley, D.W. and Zhu, J., 2020. Visualizing the effects of predictor variables in black box supervised learning models. Journal of the Royal Statistical Society Series B: Statistical Methodology, 82(4), pp.1059-1086.
14. Sahai, R., Al-Ataby, A., Assi, S., Jayabalan, M., Liatsis, P., Loy, C.K., Al-Hamid, A., Al-Sudani, S., Alamran, M. and Kolivand, H., 2022, December. Insurance Risk Prediction Using Machine Learning. In The International Conference on Data Science and Emerging Technologies (pp. 419-433). Singapore: Springer Nature Singapore.
15. Rawat, S., Rawat, A., Kumar, D. and Sabitha, A.S., 2021. Application of machine learning and data visualization techniques for decision support in the insurance sector. International Journal of Information Management Data Insights, 1(2), p.100012.
16. Jabbar, A., Akhtar, P. and Dani, S., 2020. Real-time big data processing for instantaneous marketing decisions: A problematization approach. Industrial Marketing Management, 90, pp.558-569.
17. Mulligan, M. and Wainwright, J., 2013. Modelling and model building. Environmental modelling: Finding simplicity in complexity, pp.7-26
18. Balasubramanian, R., Libarikian, A. and McElhaney, D. (2021) Insurance 2030-the impact of AI on the future of Insurance, McKinsey & Company. Available at: https://www.mckinsey.com/industries/financial-services/our-insights/insurance-2030-the-impact-of-ai-on-the-future-of-insurance (Accessed: 20 March 2024).
19. Clere, A. (2023) Machine learning and deep learning in the insurance space, InsurTech Magazine. Available at: https://insurtechdigital.com/articles/machine-learning-and-deep-learning-in-the-insurance-space (Accessed: 20 March 2024).
20. Reporter, B. (2023) Transforming the insurance industry with ai, Business Reporter. Available at: https://www.business-reporter.co.uk/finance/transforming-the-insurance-industry-with-ai (Accessed: 20 March 2024).

    
