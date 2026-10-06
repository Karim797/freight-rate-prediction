# 2-3 Minute Loom Script

Hi, I'm Karim. This is my solution for the Spotter freight-rate prediction assessment.

I started with 48,000 labeled loads covering January through October 2025. Because the final validation data is from November, I used a chronological validation split rather than a random split: January through September for training and October as the holdout. This makes the local evaluation closer to the real forecasting setup and reduces temporal leakage.

During exploration, distance was by far the strongest numerical driver of posted rate, with a correlation of about 0.91. I also found missing values in weight and market index. I handled numerical missing values using medians learned from the development data, while preserving categorical variables such as pickup, delivery and equipment. I engineered a route feature and calendar features including month, day, weekday and day of year.

I chose CatBoost regression because this dataset mixes continuous variables with several categorical freight variables, and CatBoost can model nonlinear interactions without a large one-hot encoded feature space. On the October holdout, the model achieved an MAE of about 136 dollars, RMSE of 649 dollars, R-squared of 0.82, and MAPE of about 7.15 percent. The difference between MAE and RMSE also suggests a smaller set of difficult outlier loads.

In the code, the main pipeline is in src/train_predict.py. It performs feature engineering, development-only imputation, chronological validation, final refitting on all labeled data, and then writes the 12,000 validation predictions in the required format.

For the December chart, the supplied file fixes Lexington to Fort Wayne, 360 miles, Dry Van, and 32,000 pounds, with only the date changing. The scorer validates all 31 predictions and generates the chart included in my report.

Finally, I ran the supplied scorer successfully: all 12,000 final predictions and all 31 December rows passed validation. Thank you for reviewing my solution.
