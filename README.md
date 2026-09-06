# Thesis Demo
## Overview:
This is a Python project which validates the 2-dimensional convolutional autoencoder as an effective model for identifying structural anomalies in options-implied volatility surfaces (or in this case, 2D put-call gap surfaces), specifically the PEAD (post-earnings-announcement-drift).
It uses 2D "images" of the gap surfaces after market close (T-1) and right after market open on the next day (T0).
T-1 and T0 are fed to the model, but the CAE only tries to reconstruct T0, by trying to understand the rational behavior that transitions T-1 to T0 on control (non-earnings) days. The idea is that when given a treatment (earnings) T-1 and asked to predict T0, it will yield a high reconstruction loss because of the anomalous distortion in the gap surface caused by PEAD.
This is a precursor for my proposed HUT project idea and demonstrates the effectiveness of this ML architecture in detecting anomalies in options-implied volatility surfaces. 

## Data Layer
The market data used for this demo comes from the `massive.com` API, which gives per-minute aggregates for stock and options prices for each company across all major US markets in the past 2 years.

The data is then filtered to only include the time snapshots relevant for our T-1/T0 pairs, as follows:
    - T-1: Prices at 4:00 pm (NYSE close) with a 5 minute lookback window as a fallback
    - T0: Prices at 9:45 pm (15 minutes post-NYSE open) with a 5 minute lookback window as a fallback

The data is then separated into two buckets: control and treatment
    - Treatment days are those with earnings announcements. If the announcement was before market open, then the day before the announcement is used as T-1 and that day is then T0. Conversely, if the announcement is after market close, then the day after is used as T0 and that day is then T-1.
    - Control pairs are any other day, except for those within a 3-day buffer of the T-1/T0 for any given earnings announcement.

