# SOLMATE

 ## Solar and Mesonet Analysis for the Tracking of Energy
 
 __SOLMATE__ is a solar modeling system based on __MESONET__ weather data and solar production report. MESONET data consists of observed weather patterns including, the measure of solar irradiance, precipitation, barometric pressure, wind speed, and temperature. Solar irradiance is the output of light energy from the sun. This data, combined with real-life solar power production data from sites in Oklahoma, allows us to track solar energy
 

 ## How the SOLMATE Application Works:

 Using the *scikit-learn* library, a Python script trains a __deep perceptron neural network using a stochastic gradient descent algorithm.__ The input data was gathered from the nearest of Oklahoma’s MESONET trackers and compared to the kilowatt hours per day of a given solar array. The neural network is then pickled for later use. Using RMarkdown, the user inputs the county location, desired month, and the average amount of their electricity bill. The algorithm then predicts the average kilowatt hours per day. We use this information to inform the user about their possible ROI, carbon savings, money saved, and price of a system to negate any costs. The R Markdown HTML file is combined with an information page and hosted on __Google Cloud Platform__ using a virtual machine. The hosted page uses the Domain.com domain “solmate.tech” to get accessed.


## SOLMATE.tech

 The original SOLMATE.tech website was created in __CSS, HTML, RMarkdown, and bootstrap__ and hosted on a __Google Cloud Platform__ virtual machine running __UBUNTU__ with __Apache2__ installed. __Python__ was embedded in __RMarkdown__ and placed in an iframe in the main __CSS/HTML__ page.
