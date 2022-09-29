# OSRS Grand Exchange (GE) Price Predictor
Uses a dense neural network to predict market trends in the GE, from the popular online game old school runescape. Uses the runelite GE API to gather data and make new predictions.
Best current model has around 65% accuracy on weather the price of an item will go up or down.
# Limitations
Price prediction using machine learning is notoriously difficult. This model was only trained on (and therefore should only be used to predict the prices of) high price items, with relatively low trade volume and most importantly long timesteps, all of which reduce the effect of random unpredictible market fluctuations
